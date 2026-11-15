#!/usr/bin/env python3
"""Guided sEMG set recorder for MyoMotion (talks to hardware/emg_logger/emg_logger.ino).

Guides one set with on-screen cues, flashes the sync LED at the start and end of the set,
and saves the recording so it can be aligned with the video later.

Run from the repository root:

    # live bars only, nothing saved: check each electrode before recording
    python scripts/semg/record_session.py --check

    # one guided set: 5 squat reps, normal posture, tempo 2 s down, 1 s hold, 2 s up
    python scripts/semg/record_session.py --participant P01 --exercise squat --posture normal

Set timeline: RELAX baseline, countdown 3 2 1, LED flash (SET_START), each rep's three phases
in the exercise's order (CONFIG["phase_order"]) plus a short rest, LED flash (SET_END), RELAX tail.
Ctrl+C during the set stops it and saves what was recorded.

Output folder data/raw/<participant>/S<nn>_<exercise>_<posture>/ (numbered automatically):
    emg.csv        pc_time_s, t_us, ch1, ch2, ch3        t_us exact and unwrapped; pc_time per received chunk
    events.csv     pc_time_s, t_us, source, event, rep   LED_ON t_us exact; cue t_us estimated (clock fit)
    metadata.json  settings, channel map, measured rate, missing samples, clock fit
    plot.png       every channel with the phases shaded and the LED flashes dashed

Close the Arduino IDE's Serial Monitor and Plotter first: only one program can use the port.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO

import numpy as np
import serial

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG: dict = {
    # serial link: must match hardware/emg_logger/emg_logger.ino
    "port": "/dev/ttyACM0",
    "baud_rate": 250_000,
    "read_timeout_s": 0.01,
    "sync_command": b"S",
    "nominal_rate_hz": 500.0,
    "n_channels": 3,
    "channel_pins": ("A0", "A1", "A2"),
    "adc_max": 1023,
    "micros_wrap": 2 ** 32,
    "max_line_bytes": 256,
    "stream_wait_s": 4.0,  # the Uno resets when the port opens; its bootloader takes about 2 s
    # guided set
    "default_reps": 5,
    "default_tempo": "2,1,2",
    "default_rest_between_s": 1.0,
    "default_baseline_s": 5.0,
    "tail_relax_s": 3.0,
    "countdown_from": 3,
    "countdown_tick_s": 1.0,
    # phase order per exercise; --tempo gives the seconds for each phase in this order
    "phase_order": {
        "squat": ("ECCENTRIC", "ISOMETRIC_HOLD", "CONCENTRIC"),
        "bench_press": ("ECCENTRIC", "ISOMETRIC_HOLD", "CONCENTRIC"),
        "deadlift": ("CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC"),
        "grip": ("CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC"),  # bench test: hand gripper, forearm flexors
    },
    "phase_labels": {"ECCENTRIC": "DOWN", "ISOMETRIC_HOLD": "HOLD", "CONCENTRIC": "UP", "REST": "REST"},
    "phase_labels_by_exercise": {"grip": {"CONCENTRIC": "SQUEEZE", "ECCENTRIC": "RELEASE"}},
    "default_channels": ("gluteus_maximus", "vastus_lateralis", "erector_spinae"),
    # live bars
    "bar_window_samples": 25,  # 50 ms at 500 Hz
    "bar_width_chars": 30,
    "bar_label_chars": 18,
    "bar_refresh_s": 0.1,
    # files
    "default_out_dir": REPO_ROOT / "data" / "raw",
    "participant_pattern": r"P\d{2,3}",
    "plot_dpi": 150,
    "phase_colours": {"ECCENTRIC": "#A9BFFF", "ISOMETRIC_HOLD": "#CDD2DE", "CONCENTRIC": "#1F4FE0"},
    "phase_alpha": 0.35,
    "led_colour": "#EF4136",
}

PORT_HELP = (
    "Check: the Arduino is plugged in (ls /dev/ttyACM*), --port is right, and the Arduino IDE's "
    "Serial Monitor and Serial Plotter are closed (only one program can use the port)."
)
NO_DATA_HELP = (
    "Check: emg_logger.ino is uploaded (not muscle_test.ino), the Serial Monitor and Plotter are closed, "
    "and the USB cable carries data."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EmgSample:
    """One ``D,t_us,ch1,ch2,ch3`` line from the logger."""

    pc_time_s: float  # PC seconds since the session started, shared by every sample in one received chunk
    t_us: int  # Arduino micros() when the sample was taken, unwrapped past 2**32
    values: tuple[int, ...]  # ADC readings 0..1023, one per channel


@dataclass(slots=True)
class SessionEvent:
    """One row of events.csv: a PC cue or an event reported by the Arduino."""

    pc_time_s: float
    t_us: int | None  # exact for Arduino events; None for PC cues until estimated from the clock fit
    source: str  # "pc" or "arduino"
    event: str
    rep: int | None = None


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One step of the guided set: what to show, for how long, and what to log."""

    label: str
    duration_s: float
    event: str | None = None
    rep: int | None = None
    send_sync: bool = False


@dataclass
class Recording:
    """Everything collected during one session."""

    samples: list[EmgSample] = field(default_factory=list)
    events: list[SessionEvent] = field(default_factory=list)
    clock_pairs: list[tuple[float, int]] = field(default_factory=list)  # (pc_time_s, newest t_us) per chunk
    started_at: str | None = None
    aborted_reason: str | None = None
    link_lost: bool = False


@dataclass(frozen=True)
class ClockFit:
    """Straight line from PC time to Arduino time: t_us = intercept_us + slope_us_per_s * pc_time_s."""

    intercept_us: float
    slope_us_per_s: float

    def t_us_at(self, pc_time_s: float) -> int:
        """Arduino time (µs) estimated for a PC time (s)."""
        return round(self.intercept_us + self.slope_us_per_s * pc_time_s)

    @property
    def drift_ppm(self) -> float:
        """How much faster (+) or slower (-) the Arduino clock runs than the PC clock, in parts per million."""
        return self.slope_us_per_s - 1e6


# ---------------------------------------------------------------------------
# Parsing the serial stream
# ---------------------------------------------------------------------------
class MicrosUnwrapper:
    """Turns the Arduino's 32-bit micros() counter (wraps every ~71.6 min) into a count that keeps growing."""

    def __init__(self, wrap: int) -> None:
        """Start with no wraps seen."""
        self.wrap = wrap
        self.wraps_seen = 0
        self.last_raw_t_us: int | None = None

    def unwrap(self, raw_t_us: int) -> int:
        """Return raw_t_us plus one wrap for every wrap seen; a jump back of over half the range is a wrap."""
        if self.last_raw_t_us is not None and raw_t_us < self.last_raw_t_us - self.wrap // 2:
            self.wraps_seen += 1
        self.last_raw_t_us = raw_t_us
        return raw_t_us + self.wraps_seen * self.wrap


def parse_ints(texts: list[str]) -> list[int] | None:
    """Convert every text to an int, or return None if any of them is not a whole number."""
    try:
        return [int(text) for text in texts]
    except ValueError:
        return None


class LoggerLineParser:
    """Splits the serial byte stream into lines and turns them into samples and Arduino events."""

    def __init__(self, n_channels: int, adc_max: int, micros_wrap: int, max_line_bytes: int) -> None:
        """Set the expected line format; nothing has been received yet."""
        self.n_channels = n_channels
        self.adc_max = adc_max
        self.micros_wrap = micros_wrap
        self.max_line_bytes = max_line_bytes
        self.unwrapper = MicrosUnwrapper(micros_wrap)
        self.pending = bytearray()
        self.malformed_lines = 0
        self.banner: str | None = None

    def feed(self, chunk: bytes, pc_time_s: float) -> tuple[list[EmgSample], list[SessionEvent]]:
        """Add a received chunk and return the samples and events from every line it completed."""
        self.pending.extend(chunk)
        samples: list[EmgSample] = []
        events: list[SessionEvent] = []
        for line in self.take_complete_lines():
            parsed = self.parse_line(line, pc_time_s)
            if isinstance(parsed, EmgSample):
                samples.append(parsed)
            elif isinstance(parsed, SessionEvent):
                events.append(parsed)
        return samples, events

    def take_complete_lines(self) -> list[str]:
        """Remove and return every complete line in the pending bytes; drop a runaway line with no newline."""
        lines = []
        while (end := self.pending.find(b"\n")) >= 0:
            lines.append(self.pending[:end].decode("ascii", errors="replace").strip())
            del self.pending[:end + 1]
        if len(self.pending) > self.max_line_bytes:
            self.pending.clear()
            self.malformed_lines += 1
        return lines

    def parse_line(self, line: str, pc_time_s: float) -> EmgSample | SessionEvent | None:
        """Parse one line: D (sample), E (Arduino event) or # (banner); anything else counts as malformed."""
        if not line:
            return None
        if line.startswith("#"):
            if self.banner is None:
                self.banner = line.lstrip("# ")
            return None
        fields = line.split(",")
        parsed: EmgSample | SessionEvent | None = None
        if fields[0] == "D":
            parsed = self.parse_sample(fields, pc_time_s)
        elif fields[0] == "E":
            parsed = self.parse_event(fields, pc_time_s)
        if parsed is None:
            self.malformed_lines += 1
        return parsed

    def parse_sample(self, fields: list[str], pc_time_s: float) -> EmgSample | None:
        """Turn ["D", t_us, ch1, ...] into a sample, or None if the count or a value is out of range."""
        numbers = parse_ints(fields[1:]) if len(fields) == 2 + self.n_channels else None
        if numbers is None or not 0 <= numbers[0] < self.micros_wrap:
            return None
        values = tuple(numbers[1:])
        if not all(0 <= value <= self.adc_max for value in values):
            return None
        return EmgSample(pc_time_s, self.unwrapper.unwrap(numbers[0]), values)

    def parse_event(self, fields: list[str], pc_time_s: float) -> SessionEvent | None:
        """Turn ["E", t_us, name] into an Arduino event, or None if it is malformed."""
        numbers = parse_ints(fields[1:2]) if len(fields) == 3 and fields[2] else None
        if numbers is None or not 0 <= numbers[0] < self.micros_wrap:
            return None
        return SessionEvent(pc_time_s, self.unwrapper.unwrap(numbers[0]), "arduino", fields[2])


# ---------------------------------------------------------------------------
# Live bars in the terminal
# ---------------------------------------------------------------------------
class RingMean:
    """Ring buffer of the last `size` readings with a running sum, so the moving mean costs O(1)."""

    def __init__(self, size: int) -> None:
        """Create an empty buffer that holds at most `size` readings."""
        self.readings: deque[int] = deque(maxlen=size)
        self.total = 0

    def push(self, reading: int) -> None:
        """Add a reading, dropping the oldest one when the buffer is full."""
        if len(self.readings) == self.readings.maxlen:
            self.total -= self.readings[0]
        self.readings.append(reading)
        self.total += reading

    def mean(self) -> float:
        """Mean of the readings held (0 when empty)."""
        return self.total / len(self.readings) if self.readings else 0.0


def format_bar_lines(status: str, levels_by_name: dict[str, float]) -> list[str]:
    """Build the status line plus one text bar per channel, scaled 0..adc_max."""
    width = CONFIG["bar_width_chars"]
    label_chars = CONFIG["bar_label_chars"]
    lines = [status]
    for name, level in levels_by_name.items():
        clipped = min(max(level, 0.0), CONFIG["adc_max"])
        filled = round(width * clipped / CONFIG["adc_max"])
        lines.append(f"  {name[:label_chars]:<{label_chars}} {'█' * filled}{'░' * (width - filled)} {level:6.1f}")
    return lines


class LiveBars:
    """Redraws a status line and one bar per channel (50 ms moving mean) in the terminal."""

    def __init__(self, channel_names: dict[int, str], stream: TextIO, clock: Callable[[], float]) -> None:
        """Keep one moving mean per channel in use; redraw in place only on a real terminal."""
        self.channel_names = channel_names
        self.stream = stream
        self.clock = clock
        self.means = {index: RingMean(CONFIG["bar_window_samples"]) for index in channel_names}
        self.redraw_in_place = stream.isatty()
        self.last_draw_s = -math.inf
        self.lines_on_screen = 0

    def add_samples(self, samples: list[EmgSample]) -> None:
        """Push new samples into each channel's moving mean."""
        for sample in samples:
            for index, ring in self.means.items():
                ring.push(sample.values[index])

    def show(self, status: str) -> None:
        """Redraw the bars, at most once every bar_refresh_s seconds."""
        now_s = self.clock()
        if now_s - self.last_draw_s < CONFIG["bar_refresh_s"]:
            return
        self.last_draw_s = now_s
        levels = {name: self.means[index].mean() for index, name in self.channel_names.items()}
        lines = format_bar_lines(status, levels)
        if self.redraw_in_place:
            move_up = f"\x1b[{self.lines_on_screen}F" if self.lines_on_screen else ""
            self.stream.write(move_up + "".join(f"{line}\x1b[K\n" for line in lines))
            self.lines_on_screen = len(lines)
        else:
            self.stream.write(" | ".join(lines) + "\n")
        self.stream.flush()

    def finish(self) -> None:
        """Leave the last drawing on screen; later output starts below it."""
        self.lines_on_screen = 0


# ---------------------------------------------------------------------------
# The guided set
# ---------------------------------------------------------------------------
def parse_tempo(text: str) -> tuple[float, float, float]:
    """Turn "2,1,2" into three phase durations in seconds (each above 0)."""
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(f"--tempo needs three numbers such as 2,1,2, got {text!r}")
    try:
        first, second, third = (float(part) for part in parts)
    except ValueError:
        raise ValueError(f"--tempo needs numbers, got {text!r}") from None
    if min(first, second, third) <= 0:
        raise ValueError("--tempo durations must be above 0 s")
    return first, second, third


def phase_labels_for(exercise: str) -> dict[str, str]:
    """On-screen and legend names of the phases for an exercise (e.g. SQUEEZE and RELEASE for grip)."""
    return {**CONFIG["phase_labels"], **CONFIG["phase_labels_by_exercise"].get(exercise, {})}


def build_set_plan(exercise: str, reps: int, tempo_s: tuple[float, float, float],
                   rest_between_s: float, baseline_s: float) -> list[PlanStep]:
    """List every step of one guided set in order, using the exercise's phase order from CONFIG."""
    phase_order = CONFIG["phase_order"][exercise]
    labels = phase_labels_for(exercise)
    steps = [PlanStep("RELAX: stay still (baseline)", baseline_s, event="BASELINE")]
    for number in range(CONFIG["countdown_from"], 0, -1):
        steps.append(PlanStep(f"Starting in {number}", CONFIG["countdown_tick_s"]))
    steps.append(PlanStep("GO", 0.0, event="SET_START", send_sync=True))
    for rep in range(1, reps + 1):
        for phase, duration_s in zip(phase_order, tempo_s):
            steps.append(PlanStep(labels[phase], duration_s, event=phase, rep=rep))
        steps.append(PlanStep(labels["REST"], rest_between_s, event="REST", rep=rep))
    steps.append(PlanStep("STOP", 0.0, event="SET_END", send_sync=True))
    steps.append(PlanStep("RELAX: stay still", CONFIG["tail_relax_s"], event="TAIL"))
    return steps


def describe_step(step: PlanStep, total_reps: int) -> str:
    """Status text for a step, e.g. "REP 2/5  DOWN"."""
    return step.label if step.rep is None else f"REP {step.rep}/{total_reps}  {step.label}"


class SessionRunner:
    """Reads the serial stream, runs the guided steps and collects everything into a Recording."""

    def __init__(self, link: serial.Serial, bars: LiveBars, keep_samples: bool,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        """Wrap an open serial link; samples are stored only when keep_samples is True."""
        self.link = link
        self.bars = bars
        self.keep_samples = keep_samples
        self.clock = clock
        self.parser = LoggerLineParser(CONFIG["n_channels"], CONFIG["adc_max"],
                                       CONFIG["micros_wrap"], CONFIG["max_line_bytes"])
        self.recording = Recording()
        self.t0_s = clock()

    def wait_for_stream(self, timeout_s: float) -> None:
        """Wait for the first data line (the Uno resets when the port opens), then start the session clock."""
        self.link.reset_input_buffer()
        deadline_s = self.clock() + timeout_s
        while self.clock() < deadline_s:
            samples, _ = self.parser.feed(self.link.read(self.link.in_waiting or 1), 0.0)
            if samples:
                self.parser.malformed_lines = 0  # half lines cut off when the port opened do not count
                self.t0_s = self.clock()
                self.recording.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
                return
        raise TimeoutError(f"No data from the Arduino in {timeout_s:.0f} s. {NO_DATA_HELP}")

    def read_available(self) -> None:
        """Read whatever has arrived (waiting at most the port timeout) and store it."""
        chunk = self.link.read(self.link.in_waiting or 1)
        if not chunk:
            return
        pc_time_s = self.clock() - self.t0_s
        samples, events = self.parser.feed(chunk, pc_time_s)
        self.bars.add_samples(samples)
        if self.keep_samples and samples:
            self.recording.samples.extend(samples)
            self.recording.clock_pairs.append((pc_time_s, samples[-1].t_us))
        self.recording.events.extend(events)

    def log_cue(self, event: str, rep: int | None = None) -> None:
        """Log a PC cue now; its t_us is estimated when the session is saved."""
        self.recording.events.append(SessionEvent(self.clock() - self.t0_s, None, "pc", event, rep))

    def send_sync_flash(self) -> None:
        """Ask the Arduino for one LED flash; it answers with an exact E,t_us,LED_ON line."""
        self.link.write(CONFIG["sync_command"])
        self.link.flush()

    def pump_until(self, deadline_s: float, status: str) -> None:
        """Keep reading and redrawing the bars until the clock reaches deadline_s."""
        while (now_s := self.clock()) < deadline_s:
            self.read_available()
            self.bars.show(f"{status}   {deadline_s - now_s:4.1f} s")

    def run_plan(self, plan: list[PlanStep]) -> None:
        """Run every step on a fixed timetable (step ends do not drift), logging cues and sync flashes."""
        total_reps = max((step.rep or 0 for step in plan), default=0)
        step_start_s = self.clock()
        for step in plan:
            if step.event is not None:
                self.log_cue(step.event, step.rep)
            if step.send_sync:
                self.send_sync_flash()
            step_end_s = step_start_s + step.duration_s
            self.pump_until(step_end_s, describe_step(step, total_reps))
            step_start_s = step_end_s
        self.bars.finish()

    def run_check(self) -> None:
        """Show live bars until Ctrl+C; nothing is kept."""
        while True:
            self.read_available()
            self.bars.show("CHECK: relax, then squeeze each muscle in turn (Ctrl+C to stop)")


def run_guided_set(runner: SessionRunner, plan: list[PlanStep]) -> None:
    """Run the plan; on Ctrl+C or a lost serial link, log ABORTED and note why in the recording."""
    try:
        runner.run_plan(plan)
        return
    except KeyboardInterrupt:
        runner.recording.aborted_reason = "stopped with Ctrl+C"
    except (serial.SerialException, OSError) as error:
        runner.recording.aborted_reason = f"serial link lost: {error}"
        runner.recording.link_lost = True
    runner.log_cue("ABORTED")
    runner.bars.finish()


# ---------------------------------------------------------------------------
# Quality numbers and the clock fit (own algorithms)
# ---------------------------------------------------------------------------
def fit_clock(clock_pairs: list[tuple[float, int]]) -> ClockFit:
    """Least-squares line through (pc_time_s, t_us) pairs: slope = sum(dx*dy) / sum(dx*dx).

    With fewer than two distinct PC times the clocks are assumed equal (slope 1e6 µs per s).
    """
    if not clock_pairs:
        return ClockFit(0.0, 1e6)
    pc_times_s = np.array([pair[0] for pair in clock_pairs], dtype=float)
    t_us = np.array([pair[1] for pair in clock_pairs], dtype=float)
    dx = pc_times_s - pc_times_s.mean()
    spread = float(np.sum(dx * dx))
    if spread == 0.0:
        return ClockFit(float(t_us[-1] - 1e6 * pc_times_s[-1]), 1e6)
    slope = float(np.sum(dx * (t_us - t_us.mean())) / spread)
    return ClockFit(float(t_us.mean() - slope * pc_times_s.mean()), slope)


def count_missing_samples(t_us: np.ndarray, period_us: float) -> int:
    """Count samples lost in gaps: a gap of k periods between neighbouring samples means k - 1 are missing."""
    if len(t_us) < 2:
        return 0
    periods_per_gap = np.rint(np.diff(t_us) / period_us).astype(np.int64)
    return int(np.sum(np.clip(periods_per_gap - 1, 0, None)))


def measured_rate_hz(t_us: np.ndarray) -> float:
    """Samples per second on the Arduino clock: (n - 1) intervals over the first-to-last time span."""
    if len(t_us) < 2 or t_us[-1] == t_us[0]:
        return 0.0
    return float((len(t_us) - 1) / ((t_us[-1] - t_us[0]) / 1e6))


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def normalise_label(text: str) -> str:
    """Lower-case a label and keep only a-z, 0-9 and underscores (spaces and hyphens become underscores)."""
    return re.sub(r"[^a-z0-9_]", "", re.sub(r"[\s-]+", "_", text.strip().lower()))


def channel_map(names: list[str]) -> dict[int, str]:
    """Map channel index (0 = ch1 on A0) to the muscle name given for it."""
    return dict(enumerate(names))


def channel_columns(n_channels: int) -> list[str]:
    """Column names ch1..chN."""
    return [f"ch{number}" for number in range(1, n_channels + 1)]


def next_session_dir(out_root: Path, participant: str, exercise: str, posture: str) -> Path:
    """Next free S<nn>_<exercise>_<posture> folder for this participant (numbers count across exercises)."""
    participant_dir = out_root / participant
    numbers = [int(match.group(1)) for path in participant_dir.glob("S*_*")
               if (match := re.fullmatch(r"S(\d+)_.+", path.name))]
    return participant_dir / f"S{max(numbers, default=0) + 1:02d}_{exercise}_{posture}"


def events_with_t_us(events: list[SessionEvent], clock_fit: ClockFit) -> list[SessionEvent]:
    """Copy of the events with every missing t_us estimated from the clock fit, sorted by t_us."""
    filled = [event if event.t_us is not None else replace(event, t_us=clock_fit.t_us_at(event.pc_time_s))
              for event in events]
    return sorted(filled, key=lambda event: event.t_us)


def write_emg_csv(path: Path, samples: list[EmgSample], n_channels: int) -> None:
    """Write emg.csv: one row per sample (pc_time_s, t_us, ch1..chN)."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["pc_time_s", "t_us", *channel_columns(n_channels)])
        for sample in samples:
            writer.writerow([f"{sample.pc_time_s:.6f}", sample.t_us, *sample.values])


def write_events_csv(path: Path, events: list[SessionEvent]) -> None:
    """Write events.csv: pc_time_s, t_us, source, event, rep (rep empty when not part of a rep)."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["pc_time_s", "t_us", "source", "event", "rep"])
        for event in events:
            rep = "" if event.rep is None else event.rep
            writer.writerow([f"{event.pc_time_s:.6f}", event.t_us, event.source, event.event, rep])


def build_metadata(args: argparse.Namespace, session_name: str, runner: SessionRunner, t_us: np.ndarray,
                   events: list[SessionEvent], clock_fit: ClockFit) -> dict:
    """Collect the settings and quality numbers saved in metadata.json."""
    recording = runner.recording
    period_us = 1e6 / CONFIG["nominal_rate_hz"]
    missing = count_missing_samples(t_us, period_us)
    channel_names = channel_map(args.channels)
    return {
        "participant": args.participant,
        "session": session_name,
        "exercise": args.exercise,
        "posture": args.posture,
        "started_at": recording.started_at,
        "notes": args.notes,
        "phase_order": list(CONFIG["phase_order"][args.exercise]),
        "tempo_s": list(args.tempo_s),
        "reps_planned": args.reps,
        "reps_completed": sum(1 for event in events if event.source == "pc" and event.event == "REST"),
        "rest_between_s": args.rest_between,
        "baseline_s": args.baseline,
        "tail_relax_s": CONFIG["tail_relax_s"],
        "countdown_from": CONFIG["countdown_from"],
        "aborted": recording.aborted_reason is not None,
        "abort_reason": recording.aborted_reason,
        "channels": {column: {"muscle": channel_names.get(index), "pin": CONFIG["channel_pins"][index]}
                     for index, column in enumerate(channel_columns(CONFIG["n_channels"]))},
        "port": args.port,
        "baud_rate": CONFIG["baud_rate"],
        "firmware_banner": runner.parser.banner,
        "nominal_rate_hz": CONFIG["nominal_rate_hz"],
        "measured_rate_hz": round(measured_rate_hz(t_us), 3),
        "n_samples": int(len(t_us)),
        "duration_s": round(float(t_us[-1] - t_us[0]) / 1e6, 3) if len(t_us) else 0.0,
        "missing_samples": missing,
        "missing_pct": round(100.0 * missing / (len(t_us) + missing), 3) if len(t_us) else 0.0,
        "malformed_lines": runner.parser.malformed_lines,
        "led_flashes": sum(1 for event in events if event.source == "arduino" and event.event == "LED_ON"),
        "clock_fit": {
            "intercept_us": round(clock_fit.intercept_us, 1),
            "slope_us_per_pc_s": round(clock_fit.slope_us_per_s, 3),
            "arduino_vs_pc_ppm": round(clock_fit.drift_ppm, 1),
        },
        "t_us_note": "emg.csv t_us and arduino LED_ON t_us are exact micros(); pc cue t_us are estimated "
                     "from a least-squares fit of t_us against pc_time_s per received chunk",
        "script": "scripts/semg/record_session.py",
    }


def phase_intervals(events: list[SessionEvent], end_t_us: int) -> list[tuple[str, int, int]]:
    """(phase, start_t_us, end_t_us) for every phase cue; a phase lasts until the next PC cue.

    `events` must already be sorted by t_us with every t_us filled in.
    """
    cues = [event for event in events if event.source == "pc"]
    intervals = []
    for position, cue in enumerate(cues):
        if cue.event in CONFIG["phase_colours"]:
            end = cues[position + 1].t_us if position + 1 < len(cues) else end_t_us
            intervals.append((cue.event, cue.t_us, end))
    return intervals


def save_session_plot(path: Path, t_us: np.ndarray, values: np.ndarray, channel_names: dict[int, str],
                      events: list[SessionEvent], title: str, phase_labels: dict[str, str]) -> None:
    """Save plot.png: one panel per channel in use, phases shaded, LED flashes as dashed lines."""
    import matplotlib  # imported here so the Agg backend is chosen before pyplot loads

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def seconds(t: float) -> float:
        """Seconds since the first sample."""
        return (t - t_us[0]) / 1e6

    figure, axes = plt.subplots(len(channel_names), 1, sharex=True, squeeze=False,
                                figsize=(12, 1.0 + 2.2 * len(channel_names)))
    intervals = phase_intervals(events, int(t_us[-1]))
    flashes = [event.t_us for event in events if event.source == "arduino" and event.event == "LED_ON"]
    for row, (index, name) in enumerate(channel_names.items()):
        axis = axes[row][0]
        for phase, start, end in intervals:
            axis.axvspan(seconds(start), seconds(end), color=CONFIG["phase_colours"][phase],
                         alpha=CONFIG["phase_alpha"], linewidth=0)
        for flash in flashes:
            axis.axvline(seconds(flash), color=CONFIG["led_colour"], linestyle="--", linewidth=1.2)
        axis.plot((t_us - t_us[0]) / 1e6, values[:, index], color="black", linewidth=0.6)
        axis.set_ylabel(f"ch{index + 1}\n{name}", fontsize=8)
    handles = [Patch(color=colour, alpha=CONFIG["phase_alpha"], label=phase_labels[phase])
               for phase, colour in CONFIG["phase_colours"].items()]
    handles.append(Line2D([0], [0], color=CONFIG["led_colour"], linestyle="--", label="LED flash"))
    axes[0][0].legend(handles=handles, loc="upper right", fontsize=8, ncol=len(handles))
    axes[0][0].set_title(title)
    axes[-1][0].set_xlabel("time since first sample (s, Arduino clock)")
    figure.tight_layout()
    figure.savefig(path, dpi=CONFIG["plot_dpi"])
    plt.close(figure)


def save_session(args: argparse.Namespace, runner: SessionRunner) -> tuple[Path, dict]:
    """Write emg.csv, events.csv, metadata.json and plot.png into a new session folder."""
    recording = runner.recording
    t_us = np.array([sample.t_us for sample in recording.samples], dtype=np.int64)
    values = np.array([sample.values for sample in recording.samples], dtype=np.int32)
    clock_fit = fit_clock(recording.clock_pairs)
    events = events_with_t_us(recording.events, clock_fit)
    session_dir = next_session_dir(args.out, args.participant, args.exercise, args.posture)
    session_dir.mkdir(parents=True)
    write_emg_csv(session_dir / "emg.csv", recording.samples, CONFIG["n_channels"])
    write_events_csv(session_dir / "events.csv", events)
    session_name = session_dir.name.split("_")[0]
    metadata = build_metadata(args, session_name, runner, t_us, events, clock_fit)
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    title = f"{args.participant} {session_name}  {args.exercise}  {args.posture}"
    save_session_plot(session_dir / "plot.png", t_us, values, channel_map(args.channels), events, title,
                      phase_labels_for(args.exercise))
    return session_dir, metadata


def print_summary(session_dir: Path, metadata: dict) -> None:
    """Print where the files went and the quality numbers, with warnings that matter for sync."""
    print(f"[INFO] Saved {session_dir}")
    print(f"[INFO] {metadata['n_samples']} samples over {metadata['duration_s']} s, "
          f"measured rate {metadata['measured_rate_hz']} Hz, missing {metadata['missing_samples']} "
          f"({metadata['missing_pct']}%), malformed lines {metadata['malformed_lines']}, "
          f"reps completed {metadata['reps_completed']}/{metadata['reps_planned']}")
    if metadata["led_flashes"] < 2:
        print(f"[WARN] Only {metadata['led_flashes']} LED flash(es) recorded; alignment needs the start and end flash.")
    if metadata["aborted"]:
        print(f"[WARN] Set aborted: {metadata['abort_reason']}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Define the command-line options."""
    parser = argparse.ArgumentParser(
        description="Record one guided sEMG set from emg_logger.ino (or --check to see live bars only).")
    parser.add_argument("--participant", help="participant code such as P01 (never a name)")
    parser.add_argument("--exercise", choices=sorted(CONFIG["phase_order"]), help="exercise performed")
    parser.add_argument("--posture", help="posture variation, e.g. normal, deep, shallow, leaning_forward")
    parser.add_argument("--reps", type=int, default=CONFIG["default_reps"], help="number of reps")
    parser.add_argument("--tempo", default=CONFIG["default_tempo"],
                        help="seconds per phase in the order the exercise performs them "
                             "(squat: down,hold,up; deadlift: up,hold,down)")
    parser.add_argument("--rest-between", type=float, default=CONFIG["default_rest_between_s"],
                        help="seconds of rest after each rep")
    parser.add_argument("--baseline", type=float, default=CONFIG["default_baseline_s"],
                        help="seconds of relaxed baseline before the set")
    parser.add_argument("--channels", nargs="+", default=list(CONFIG["default_channels"]), metavar="NAME",
                        help="1 to 3 muscle names for ch1 (A0), ch2 (A1), ch3 (A2)")
    parser.add_argument("--out", type=Path, default=CONFIG["default_out_dir"], help="root folder for recordings")
    parser.add_argument("--notes", default="", help="free-text notes saved in metadata.json")
    parser.add_argument("--port", default=CONFIG["port"], help="serial port of the Arduino")
    parser.add_argument("--check", action="store_true", help="show live bars only; nothing is saved")
    return parser


def check_args(args: argparse.Namespace) -> None:
    """Validate and normalise the options in place; raise ValueError with a readable message."""
    if not 1 <= len(args.channels) <= CONFIG["n_channels"]:
        raise ValueError(f"--channels takes 1 to {CONFIG['n_channels']} names")
    args.channels = [normalise_label(name) for name in args.channels]
    if not all(args.channels):
        raise ValueError("--channels names need letters or digits")
    if args.check:
        return
    missing = [flag for flag, value in (("--participant", args.participant), ("--exercise", args.exercise),
                                        ("--posture", args.posture)) if not value]
    if missing:
        raise ValueError(f"{', '.join(missing)} required (or use --check)")
    if not re.fullmatch(CONFIG["participant_pattern"], args.participant):
        raise ValueError("--participant must be a code such as P01, never a name")
    args.posture = normalise_label(args.posture)
    if not args.posture:
        raise ValueError("--posture needs letters or digits")
    if args.reps < 1:
        raise ValueError("--reps must be at least 1")
    if args.rest_between < 0 or args.baseline <= 0:
        raise ValueError("--rest-between must be 0 or more and --baseline above 0")
    args.tempo_s = parse_tempo(args.tempo)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse and validate the command line (exits with a usage message on bad options)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        check_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def run_check_mode(runner: SessionRunner) -> int:
    """Live bars until Ctrl+C; returns the exit code."""
    print("[INFO] Check mode: nothing is saved. Relax, then squeeze each muscle; its bar should jump.")
    try:
        runner.run_check()
    except KeyboardInterrupt:
        runner.bars.finish()
        print("[INFO] Check stopped.")
    return 0


def finish_session(args: argparse.Namespace, runner: SessionRunner) -> int:
    """Save the recording (if anything arrived) and print the summary; returns the exit code."""
    if not runner.recording.samples:
        print("[ERROR] No samples were recorded; nothing saved.", file=sys.stderr)
        return 1
    session_dir, metadata = save_session(args, runner)
    print_summary(session_dir, metadata)
    return 1 if runner.recording.link_lost else 0


def main(argv: list[str] | None = None) -> int:
    """Run --check or one guided set and save it; returns the exit code."""
    args = parse_args(argv)
    try:
        link = serial.Serial(args.port, CONFIG["baud_rate"], timeout=CONFIG["read_timeout_s"])
    except serial.SerialException as error:
        print(f"[ERROR] Cannot open {args.port}: {error}\n{PORT_HELP}", file=sys.stderr)
        return 1
    with link:
        bars = LiveBars(channel_map(args.channels), sys.stdout, time.perf_counter)
        runner = SessionRunner(link, bars, keep_samples=not args.check)
        try:
            runner.wait_for_stream(CONFIG["stream_wait_s"])
        except TimeoutError as error:
            print(f"[ERROR] {error}", file=sys.stderr)
            return 1
        if args.check:
            return run_check_mode(runner)
        plan = build_set_plan(args.exercise, args.reps, args.tempo_s, args.rest_between, args.baseline)
        run_guided_set(runner, plan)
    return finish_session(args, runner)


if __name__ == "__main__":
    sys.exit(main())
