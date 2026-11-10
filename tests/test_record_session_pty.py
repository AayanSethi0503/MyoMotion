"""End-to-end tests: record_session.main() against a fake Arduino on a pseudo-terminal (no hardware)."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Callable

import pytest

import record_session as rs

WRAP = 2 ** 32


@pytest.fixture
def fast_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the fixed parts of the set so each test takes about two seconds."""
    monkeypatch.setitem(rs.CONFIG, "countdown_from", 0)
    monkeypatch.setitem(rs.CONFIG, "tail_relax_s", 0.3)
    monkeypatch.setitem(rs.CONFIG, "stream_wait_s", 2.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV file as dicts."""
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def set_args(port: str, out: Path, exercise: str, reps: int, *extra: str) -> list[str]:
    """Command line for a short guided set."""
    return ["--port", port, "--out", str(out), "--participant", "P01", "--exercise", exercise,
            "--posture", "normal", "--reps", str(reps), "--tempo", "0.2,0.1,0.2", "--rest-between", "0.1",
            "--baseline", "0.4", *extra]


def test_guided_deadlift_set_end_to_end(tmp_path: Path, fast_config: None,
                                        fake_arduino_factory: Callable) -> None:
    """A full set: files written, micros() wrap handled, no gaps, both LED flashes exact, deadlift order."""
    arduino = fake_arduino_factory(start_t_us=WRAP - 700_000)  # micros() wraps about 0.7 s in
    exit_code = rs.main(set_args(arduino.port, tmp_path, "deadlift", 2,
                                 "--channels", "gluteus_maximus", "vastus_lateralis"))
    assert exit_code == 0

    session_dir = tmp_path / "P01" / "S01_deadlift_normal"
    for name in ["emg.csv", "events.csv", "metadata.json", "plot.png"]:
        assert (session_dir / name).is_file(), name

    emg_rows = read_csv(session_dir / "emg.csv")
    t_us = [int(row["t_us"]) for row in emg_rows]
    assert list(emg_rows[0]) == ["pc_time_s", "t_us", "ch1", "ch2", "ch3"]
    assert len(t_us) > 500
    assert {later - earlier for earlier, later in zip(t_us, t_us[1:])} == {2000}
    assert t_us[0] < WRAP < t_us[-1]

    events = read_csv(session_dir / "events.csv")
    cues = [(row["event"], row["rep"]) for row in events if row["source"] == "pc"]
    rep_phases = ["CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC", "REST"]
    expected = ([("BASELINE", ""), ("SET_START", "")]
                + [(event, str(rep)) for rep in (1, 2) for event in rep_phases]
                + [("SET_END", ""), ("TAIL", "")])
    assert cues == expected
    flashes = [int(row["t_us"]) for row in events if row["source"] == "arduino" and row["event"] == "LED_ON"]
    assert [flash % WRAP for flash in flashes] == arduino.flash_t_us
    assert len(flashes) == 2

    metadata = json.loads((session_dir / "metadata.json").read_text())
    assert metadata["missing_samples"] == 0
    assert metadata["malformed_lines"] == 0
    assert metadata["measured_rate_hz"] == pytest.approx(500.0)
    assert metadata["reps_completed"] == 2 and metadata["led_flashes"] == 2
    assert metadata["aborted"] is False
    assert metadata["phase_order"] == ["CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC"]
    assert metadata["channels"]["ch2"] == {"muscle": "vastus_lateralis", "pin": "A1"}
    assert metadata["channels"]["ch3"]["muscle"] is None


def test_ctrl_c_mid_set_saves_what_was_recorded(tmp_path: Path, fast_config: None,
                                                fake_arduino_factory: Callable,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl+C during the first HOLD: files are still saved, ABORTED is logged, exit code 0."""
    arduino = fake_arduino_factory()
    original_show = rs.LiveBars.show

    def press_ctrl_c_on_hold(self: rs.LiveBars, status: str) -> None:
        """Simulate Ctrl+C as soon as the HOLD cue is on screen."""
        if "HOLD" in status:
            raise KeyboardInterrupt
        original_show(self, status)

    monkeypatch.setattr(rs.LiveBars, "show", press_ctrl_c_on_hold)
    assert rs.main(set_args(arduino.port, tmp_path, "squat", 3)) == 0

    session_dir = tmp_path / "P01" / "S01_squat_normal"
    metadata = json.loads((session_dir / "metadata.json").read_text())
    assert metadata["aborted"] is True and metadata["abort_reason"] == "stopped with Ctrl+C"
    assert metadata["reps_completed"] == 0 and metadata["led_flashes"] == 1
    cues = [row["event"] for row in read_csv(session_dir / "events.csv") if row["source"] == "pc"]
    assert cues[-2:] == ["ISOMETRIC_HOLD", "ABORTED"]
    assert len(read_csv(session_dir / "emg.csv")) > 100


def test_unplugged_mid_set_saves_and_reports_error(tmp_path: Path, fast_config: None,
                                                   fake_arduino_factory: Callable) -> None:
    """The device disappearing mid-set: what arrived is saved, the reason is recorded, exit code 1."""
    arduino = fake_arduino_factory(disconnect_after_s=1.0)
    assert rs.main(set_args(arduino.port, tmp_path, "squat", 5)) == 1
    metadata = json.loads((tmp_path / "P01" / "S01_squat_normal" / "metadata.json").read_text())
    assert metadata["aborted"] is True
    assert metadata["abort_reason"].startswith("serial link lost")


def test_second_set_gets_next_session_number(tmp_path: Path, fast_config: None,
                                             fake_arduino_factory: Callable) -> None:
    """Two sets in a row are saved as S01 and S02."""
    for _ in range(2):
        arduino = fake_arduino_factory()
        assert rs.main(set_args(arduino.port, tmp_path, "squat", 1)) == 0
    assert sorted(path.name for path in (tmp_path / "P01").iterdir()) == ["S01_squat_normal", "S02_squat_normal"]


def test_silent_port_gives_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    """A port that sends nothing (wrong sketch, plotter open) stops with advice and saves nothing."""
    if not hasattr(os, "openpty"):
        pytest.skip("needs a POSIX pseudo-terminal")
    import tty

    monkeypatch.setitem(rs.CONFIG, "stream_wait_s", 0.3)
    master_fd, slave_fd = os.openpty()
    tty.setraw(slave_fd)
    try:
        assert rs.main(set_args(os.ttyname(slave_fd), tmp_path, "squat", 1)) == 1
    finally:
        os.close(master_fd)
        os.close(slave_fd)
    assert "emg_logger.ino is uploaded" in capsys.readouterr().err
    assert not (tmp_path / "P01").exists()


def test_missing_port_gives_clear_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A port that does not exist explains what to check."""
    assert rs.main(set_args("/dev/does-not-exist", tmp_path, "squat", 1)) == 1
    assert "Serial Monitor" in capsys.readouterr().err
