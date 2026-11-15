"""Unit tests for scripts/semg/record_session.py (no hardware, no serial port)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import record_session as rs

WRAP = 2 ** 32


def make_parser() -> rs.LoggerLineParser:
    """Parser set up like the real logger: 3 channels, 10-bit ADC."""
    return rs.LoggerLineParser(n_channels=3, adc_max=1023, micros_wrap=WRAP, max_line_bytes=256)


# --- parsing -----------------------------------------------------------------
def test_parser_joins_lines_split_across_chunks() -> None:
    """A line cut between two reads is parsed once both halves have arrived."""
    parser = make_parser()
    samples_a, _ = parser.feed(b"D,1000,1,2,3\r\nD,30", 0.5)
    samples_b, _ = parser.feed(b"00,4,5,6\r\n", 0.6)
    assert [(s.t_us, s.values, s.pc_time_s) for s in samples_a + samples_b] == [
        (1000, (1, 2, 3), 0.5), (3000, (4, 5, 6), 0.6)]
    assert parser.malformed_lines == 0


def test_parser_reads_led_event_and_banner() -> None:
    """E lines become Arduino events with exact t_us; the # banner is kept, not counted as malformed."""
    parser = make_parser()
    samples, events = parser.feed(b"# emg_logger v1 rate_hz=500 channels=3\r\nE,5000,LED_ON\r\n", 1.0)
    assert samples == []
    assert [(e.t_us, e.source, e.event) for e in events] == [(5000, "arduino", "LED_ON")]
    assert parser.banner == "emg_logger v1 rate_hz=500 channels=3"
    assert parser.malformed_lines == 0


@pytest.mark.parametrize("line", [
    b"D,1000,1,2\r\n",          # too few channels
    b"D,1000,1,2,x\r\n",        # not a number
    b"D,1000,1,2,1024\r\n",     # above the 10-bit ADC range
    b"D,-5,1,2,3\r\n",          # negative time
    b"D,4294967296,1,2,3\r\n",  # beyond 32 bits
    b"E,1000\r\n",              # event without a name
    b"Q,1,2\r\n",               # unknown line type
])
def test_parser_counts_malformed_lines(line: bytes) -> None:
    """Every kind of broken line is skipped and counted."""
    parser = make_parser()
    samples, events = parser.feed(line, 0.0)
    assert samples == [] and events == []
    assert parser.malformed_lines == 1


def test_parser_drops_runaway_line_without_newline() -> None:
    """Garbage with no newline is dropped once it exceeds max_line_bytes."""
    parser = make_parser()
    parser.feed(b"x" * 300, 0.0)
    assert parser.malformed_lines == 1
    samples, _ = parser.feed(b"\nD,10,1,2,3\n", 0.1)
    assert len(samples) == 1


def test_unwrapper_continues_past_micros_overflow() -> None:
    """micros() going from near 2**32 back to a small value adds 2**32."""
    unwrapper = rs.MicrosUnwrapper(WRAP)
    raw = [WRAP - 4000, WRAP - 2000, 0, 2000]
    assert [unwrapper.unwrap(value) for value in raw] == [WRAP - 4000, WRAP - 2000, WRAP, WRAP + 2000]


# --- plan ------------------------------------------------------------------------
def phase_cues(plan: list[rs.PlanStep]) -> list[tuple[str | None, int | None, float]]:
    """(event, rep, duration) for every step that belongs to a rep."""
    return [(step.event, step.rep, step.duration_s) for step in plan if step.rep is not None]


def test_squat_plan_goes_down_hold_up_with_tempo() -> None:
    """Squat reps are eccentric first; the tempo maps onto the phases in that order."""
    plan = rs.build_set_plan("squat", reps=2, tempo_s=(2.0, 1.0, 3.0), rest_between_s=1.0, baseline_s=5.0)
    assert phase_cues(plan) == [
        ("ECCENTRIC", 1, 2.0), ("ISOMETRIC_HOLD", 1, 1.0), ("CONCENTRIC", 1, 3.0), ("REST", 1, 1.0),
        ("ECCENTRIC", 2, 2.0), ("ISOMETRIC_HOLD", 2, 1.0), ("CONCENTRIC", 2, 3.0), ("REST", 2, 1.0)]


def test_deadlift_plan_goes_up_hold_down() -> None:
    """Deadlift reps are concentric first."""
    plan = rs.build_set_plan("deadlift", reps=1, tempo_s=(2.0, 1.0, 2.0), rest_between_s=1.0, baseline_s=5.0)
    assert [event for event, _, _ in phase_cues(plan)] == ["CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC", "REST"]


def test_grip_plan_squeezes_holds_then_releases() -> None:
    """The forearm bench test shows SQUEEZE, HOLD, RELEASE and logs them as concentric, hold, eccentric."""
    plan = rs.build_set_plan("grip", reps=1, tempo_s=(2.0, 1.0, 2.0), rest_between_s=1.0, baseline_s=5.0)
    rep_steps = [(step.label, step.event) for step in plan if step.rep is not None]
    assert rep_steps == [("SQUEEZE", "CONCENTRIC"), ("HOLD", "ISOMETRIC_HOLD"), ("RELEASE", "ECCENTRIC"),
                         ("REST", "REST")]


def test_plan_frames_set_with_baseline_flashes_and_tail() -> None:
    """Baseline and countdown come first, an LED flash starts and ends the set, the relaxed tail is last."""
    plan = rs.build_set_plan("squat", reps=1, tempo_s=(2.0, 1.0, 2.0), rest_between_s=1.0, baseline_s=5.0)
    events = [step.event for step in plan if step.event is not None]
    assert events[:2] == ["BASELINE", "SET_START"]
    assert events[-2:] == ["SET_END", "TAIL"]
    assert [step.event for step in plan if step.send_sync] == ["SET_START", "SET_END"]
    countdown = [step.label for step in plan if step.event is None]
    assert countdown == ["Starting in 3", "Starting in 2", "Starting in 1"]
    assert sum(step.duration_s for step in plan) == pytest.approx(5 + 3 + 6 + 3)


@pytest.mark.parametrize("text, expected", [("2,1,2", (2.0, 1.0, 2.0)), ("1.5, 0.5, 3", (1.5, 0.5, 3.0))])
def test_parse_tempo_reads_three_durations(text: str, expected: tuple[float, float, float]) -> None:
    """Three positive numbers separated by commas."""
    assert rs.parse_tempo(text) == expected


@pytest.mark.parametrize("text", ["2,1", "2,1,2,1", "2,a,2", "2,0,2", "2,-1,2"])
def test_parse_tempo_rejects_bad_input(text: str) -> None:
    """Wrong count, non-numbers and non-positive durations are refused."""
    with pytest.raises(ValueError):
        rs.parse_tempo(text)


# --- quality numbers and the clock fit ------------------------------------------
def test_count_missing_samples_counts_each_gap() -> None:
    """One gap of 3 periods (2 missing) and one of 2 periods (1 missing)."""
    t_us = np.array([0, 2000, 8000, 10000, 14000], dtype=np.int64)
    assert rs.count_missing_samples(t_us, 2000.0) == 3
    assert rs.count_missing_samples(np.array([0, 2000, 4000]), 2000.0) == 0
    assert rs.count_missing_samples(np.array([5]), 2000.0) == 0


def test_measured_rate_uses_arduino_time_span() -> None:
    """1001 samples, 2000 µs apart = 500 Hz; jitter in single intervals does not matter."""
    t_us = np.arange(0, 1001) * 2000
    assert rs.measured_rate_hz(t_us) == pytest.approx(500.0)
    assert rs.measured_rate_hz(np.array([7])) == 0.0


def test_fit_clock_recovers_offset_and_drift() -> None:
    """An Arduino clock 300 ppm fast with a 1.5 s offset is recovered from noisy chunk pairs."""
    rng = np.random.default_rng(0)
    pc_times_s = np.linspace(0, 60, 3000)
    t_us = 1.5e6 + (1e6 + 300) * pc_times_s + rng.normal(0, 500, pc_times_s.size)
    fit = rs.fit_clock(list(zip(pc_times_s.tolist(), t_us.round().astype(int).tolist())))
    assert fit.drift_ppm == pytest.approx(300, abs=5)
    assert fit.t_us_at(30.0) == pytest.approx(1.5e6 + (1e6 + 300) * 30, abs=100)


def test_fit_clock_falls_back_without_spread() -> None:
    """A single chunk time gives equal clocks through that point."""
    fit = rs.fit_clock([(2.0, 5_000_000)])
    assert fit.slope_us_per_s == 1e6
    assert fit.t_us_at(3.0) == 6_000_000


def test_events_get_estimated_t_us_and_sorted() -> None:
    """PC cues get t_us from the fit; Arduino times stay exact; rows come out in time order."""
    fit = rs.ClockFit(intercept_us=1_000_000, slope_us_per_s=1e6)
    events = [rs.SessionEvent(0.5, None, "pc", "SET_START"),
              rs.SessionEvent(0.52, 1_400_000, "arduino", "LED_ON"),
              rs.SessionEvent(0.1, None, "pc", "BASELINE")]
    filled = rs.events_with_t_us(events, fit)
    assert [(e.event, e.t_us) for e in filled] == [
        ("BASELINE", 1_100_000), ("LED_ON", 1_400_000), ("SET_START", 1_500_000)]


def test_phase_intervals_run_until_next_cue() -> None:
    """Each phase is shaded until the next PC cue; REST and Arduino events are not shaded."""
    events = [rs.SessionEvent(0, 100, "pc", "ECCENTRIC", 1),
              rs.SessionEvent(0, 150, "arduino", "LED_ON"),
              rs.SessionEvent(0, 200, "pc", "ISOMETRIC_HOLD", 1),
              rs.SessionEvent(0, 300, "pc", "REST", 1),
              rs.SessionEvent(0, 400, "pc", "CONCENTRIC", 2)]
    assert rs.phase_intervals(events, end_t_us=500) == [
        ("ECCENTRIC", 100, 200), ("ISOMETRIC_HOLD", 200, 300), ("CONCENTRIC", 400, 500)]


# --- live bars ----------------------------------------------------------------------
def test_ring_mean_keeps_only_the_last_readings() -> None:
    """The moving mean forgets readings older than the window."""
    ring = rs.RingMean(3)
    assert ring.mean() == 0.0
    for reading in [10, 20, 30, 40]:
        ring.push(reading)
    assert ring.mean() == pytest.approx(30.0)


def test_bar_lines_scale_with_level() -> None:
    """A full-scale channel fills the bar; zero leaves it empty."""
    lines = rs.format_bar_lines("REP 1/5  DOWN", {"glute": 1023.0, "quad": 0.0})
    width = rs.CONFIG["bar_width_chars"]
    assert lines[0] == "REP 1/5  DOWN"
    assert lines[1].count("█") == width and lines[2].count("█") == 0


def test_live_bars_print_plain_lines_when_not_a_terminal() -> None:
    """Without a terminal there are no cursor codes, and redraws are rate-limited."""
    stream = io.StringIO()
    clock_s = iter([0.0, 0.05, 0.2])
    bars = rs.LiveBars({0: "glute"}, stream, lambda: next(clock_s))
    bars.add_samples([rs.EmgSample(0.0, 0, (512, 0, 0))])
    for _ in range(3):
        bars.show("CHECK")
    output = stream.getvalue()
    assert "\x1b[" not in output
    assert output.count("CHECK") == 2


# --- files and options ------------------------------------------------------------------
def test_next_session_dir_numbers_across_exercises(tmp_path: Path) -> None:
    """Numbering continues from the highest existing S<nn>, whatever the exercise."""
    assert rs.next_session_dir(tmp_path, "P01", "squat", "normal").name == "S01_squat_normal"
    (tmp_path / "P01" / "S01_squat_normal").mkdir(parents=True)
    (tmp_path / "P01" / "S07_deadlift_deep").mkdir()
    (tmp_path / "P01" / "notes").mkdir()
    assert rs.next_session_dir(tmp_path, "P01", "squat", "deep").name == "S08_squat_deep"


def test_normalise_label() -> None:
    """Spaces and hyphens become underscores; other symbols are removed."""
    assert rs.normalise_label("  Leaning Forward ") == "leaning_forward"
    assert rs.normalise_label("wide-stance!") == "wide_stance"


def test_options_require_participant_code_not_name() -> None:
    """A name instead of a P-code is refused."""
    with pytest.raises(SystemExit):
        rs.parse_args(["--participant", "Aayan", "--exercise", "squat", "--posture", "normal"])


def test_options_check_mode_needs_nothing_else() -> None:
    """--check works without participant, exercise or posture."""
    args = rs.parse_args(["--check"])
    assert args.check and args.channels == list(rs.CONFIG["default_channels"])


def test_options_default_output_is_repo_data_raw() -> None:
    """Recordings go to <repo>/data/raw by default, wherever the script is run from."""
    args = rs.parse_args(["--participant", "P01", "--exercise", "squat", "--posture", "Normal"])
    assert args.out == Path(rs.__file__).resolve().parents[2] / "data" / "raw"
    assert args.posture == "normal" and args.tempo_s == (2.0, 1.0, 2.0)


def test_options_refuse_more_than_three_channels() -> None:
    """The logger has three channels."""
    with pytest.raises(SystemExit):
        rs.parse_args(["--check", "--channels", "a", "b", "c", "d"])
