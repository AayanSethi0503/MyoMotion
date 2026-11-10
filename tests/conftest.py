"""Shared test helpers: puts scripts/semg on the import path and provides a fake Arduino on a pseudo-terminal."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "semg"))

MICROS_WRAP = 2 ** 32
SAMPLE_PERIOD_US = 2000  # 500 Hz, as in emg_logger.ino
WRITE_EVERY_S = 0.002


class FakeArduino(threading.Thread):
    """Stands in for emg_logger.ino on a pseudo-terminal: D lines at 500 Hz and an E line for every 'S'.

    Sample times come from the sample index (start_t_us + index * 2000, wrapping at 2**32), so the stream has
    no gaps by construction; wall-clock time only paces how fast the lines are written.
    """

    def __init__(self, start_t_us: int, disconnect_after_s: float | None) -> None:
        """Open the pseudo-terminal; the recorder connects to `self.port`."""
        import tty  # POSIX only; imported here so collection works everywhere

        super().__init__(daemon=True)
        self.master_fd, self.slave_fd = os.openpty()
        tty.setraw(self.slave_fd)
        os.set_blocking(self.master_fd, False)
        self.port = os.ttyname(self.slave_fd)
        self.start_t_us = start_t_us
        self.disconnect_after_s = disconnect_after_s
        self.sample_index = 0
        self.flash_t_us: list[int] = []
        self.unsent = b""
        self.master_open = True
        self.stop_requested = threading.Event()

    def sample_t_us(self, index: int) -> int:
        """Raw micros() value of sample `index`."""
        return (self.start_t_us + index * SAMPLE_PERIOD_US) % MICROS_WRAP

    def sample_line(self, index: int) -> bytes:
        """One D line with a slow sine wave on each channel."""
        angle = 2 * math.pi * index / 250
        values = [300 + 100 * channel + round(80 * math.sin(angle + channel)) for channel in range(3)]
        return f"D,{self.sample_t_us(index)},{values[0]},{values[1]},{values[2]}\r\n".encode()

    def answer_sync_commands(self) -> None:
        """Queue an E,t_us,LED_ON line for every 'S' received, timed between the last and next sample."""
        try:
            received = os.read(self.master_fd, 64)
        except OSError:
            return
        for _ in range(received.count(b"S")):
            flash_t_us = (self.sample_t_us(self.sample_index) - SAMPLE_PERIOD_US // 2) % MICROS_WRAP
            self.flash_t_us.append(flash_t_us)
            self.unsent += f"E,{flash_t_us},LED_ON\r\n".encode()

    def write_unsent(self) -> None:
        """Write as much of the queued output as the pseudo-terminal accepts right now."""
        try:
            written = os.write(self.master_fd, self.unsent)
            self.unsent = self.unsent[written:]
        except BlockingIOError:
            pass

    def close_master(self) -> None:
        """Close our end once; the recorder then sees the device disappear."""
        if self.master_open:
            self.master_open = False
            os.close(self.master_fd)

    def run(self) -> None:
        """Write due samples every ~2 ms, answer sync commands, and disconnect if asked to."""
        self.unsent += b"# emg_logger v1 rate_hz=500 channels=3 (fake)\r\n"
        started_s = time.perf_counter()
        while not self.stop_requested.is_set():
            elapsed_s = time.perf_counter() - started_s
            if self.disconnect_after_s is not None and elapsed_s >= self.disconnect_after_s:
                self.close_master()
                return
            self.answer_sync_commands()
            due = int(elapsed_s * 1e6 / SAMPLE_PERIOD_US)
            while self.sample_index < due:
                self.unsent += self.sample_line(self.sample_index)
                self.sample_index += 1
            if self.unsent:
                self.write_unsent()
            time.sleep(WRITE_EVERY_S)

    def stop(self) -> None:
        """Stop the thread and close both ends of the pseudo-terminal."""
        self.stop_requested.set()
        self.join(timeout=2)
        self.close_master()
        os.close(self.slave_fd)


@pytest.fixture
def fake_arduino_factory() -> Iterator[Callable[..., FakeArduino]]:
    """Start fake Arduinos on demand and stop them all after the test."""
    if not hasattr(os, "openpty"):
        pytest.skip("needs a POSIX pseudo-terminal")
    started: list[FakeArduino] = []

    def start(start_t_us: int = 0, disconnect_after_s: float | None = None) -> FakeArduino:
        """Start one fake Arduino whose micros() begins at start_t_us."""
        arduino = FakeArduino(start_t_us, disconnect_after_s)
        arduino.start()
        started.append(arduino)
        return arduino

    yield start
    for arduino in started:
        arduino.stop()
