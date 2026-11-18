"""Temporal filters for pose-derived signals (NumPy / stdlib only, timestamp-driven).

Ported from ``scripts/cv/deadlift/deadlift_tracker.py`` (18 passing tests) [G-17]:
    raw angle -> median-3 + physiological jump gate (``AngleChannel``) -> One-Euro low-pass,
    plus a least-squares velocity estimator (``SlidingSlope``).

New here: ``KeypointSmoother``, a One-Euro filter on every landmark coordinate to remove
MediaPipe frame-to-frame jitter in the skeleton overlay, the spine proxy and the bar path.

Everything uses *seconds from frame timestamps*, never frame counts, so behaviour is identical at
15, 30 or 60 FPS and survives dropped frames.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Angle-signal conditioning constants [G-17] (values carried over from the tracker)
# ---------------------------------------------------------------------------
# One-Euro filter (Casiez, Roussel & Vogel, CHI 2012): the cut-off rises with signal speed, so it
# smooths hard while the lifter is still (kills jitter) and tracks closely while moving (low lag).
FILTER_MIN_CUTOFF_HZ = 0.5     # cut-off at zero speed
FILTER_BETA = 0.01             # cut-off gain per deg/s of speed
FILTER_D_CUTOFF_HZ = 1.0       # cut-off for the internal speed estimate

# Physiological outlier gate. Landmark swaps / mis-detections produce single-frame jumps far beyond
# any real joint motion (peak hip angular velocity in a maximal pull is a few hundred deg/s).
MAX_ANGULAR_SPEED_DEG_S = 600.0
JUMP_SLACK_DEG = 8.0           # allowance on top of speed * dt

# Angular velocity = least-squares slope over this window. Noise std of the slope falls with
# ~1/T^1.5, at the price of T/2 latency.
VELOCITY_WINDOW_S = 0.40

# A gap between samples larger than this invalidates dwell timers / velocity / filter state.
MAX_SAMPLE_GAP_S = 0.5

# ---------------------------------------------------------------------------
# Keypoint smoothing constants [G-17]. Coordinates are filtered in units of torso length, so the
# same numbers work at any image resolution or camera distance.
# ---------------------------------------------------------------------------
# Tuned by grid search on a synthetic static / ramping landmark at 30 fps (measured, not guessed):
#   min_cutoff 1.5 / beta 8.0 -> noise std x0.43 but lag only 25 ms at 0.6 torso/s  (noise inflates the cut-off)
#   min_cutoff 1.0 / beta 3.0 -> noise std x0.32 (3x less jitter), lag 57 ms at 0.6 torso/s, 84 ms at 0.3 torso/s
# A deadlift moves the bar at ~0.3-1 torso/s over ~2 s, so tens of ms of lag are irrelevant here, while jitter
# in the overlay / spine curve / bar path is not. The rep segmenter does NOT use these filtered landmarks.
KP_MIN_CUTOFF_HZ = 1.0         # cut-off at rest: strong jitter suppression
KP_BETA = 3.0                  # extra cut-off Hz per (torso length / s) of speed
KP_D_CUTOFF_HZ = 1.0


def _alpha(dt: float, cutoff_hz: float) -> float:
    """Smoothing factor of a 1st-order low-pass for sample period *dt*."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One-Euro adaptive low-pass filter driven by real timestamps."""

    def __init__(self, min_cutoff: float = FILTER_MIN_CUTOFF_HZ,
                 beta: float = FILTER_BETA,
                 d_cutoff: float = FILTER_D_CUTOFF_HZ):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t: Optional[float] = None
        self._x_raw = 0.0
        self._x_hat = 0.0
        self._dx_hat = 0.0

    def __call__(self, t: float, x: float) -> float:
        if self._t is None:
            self._t, self._x_raw, self._x_hat = t, x, x
            return x
        dt = t - self._t
        if dt <= 0.0:
            return self._x_hat
        dx = (x - self._x_raw) / dt
        a_d = _alpha(dt, self._d_cutoff)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat
        cutoff = self._min_cutoff + self._beta * abs(self._dx_hat)
        a = _alpha(dt, cutoff)
        self._x_hat = a * x + (1.0 - a) * self._x_hat
        self._t, self._x_raw = t, x
        return self._x_hat


class AngleChannel:
    """Outlier rejection + One-Euro filter for one joint angle.

    raw -> 3-point median (removes any isolated single-frame glitch, whatever its size)
        -> physiological jump gate (rejects multi-frame glitches) -> One-Euro low-pass.
    """

    def __init__(self):
        self._filter = OneEuroFilter()
        self._recent: Deque[float] = deque(maxlen=3)
        self._last_ok: Optional[float] = None
        self._last_t: Optional[float] = None

    def update(self, t: float, raw: float) -> Optional[float]:
        """Filtered angle, or None if the sample is rejected as an outlier."""
        if not math.isfinite(raw):
            return None
        self._recent.append(raw)
        value = sorted(self._recent)[len(self._recent) // 2]
        if self._last_ok is not None and self._last_t is not None:
            dt = max(t - self._last_t, 1e-3)
            limit = MAX_ANGULAR_SPEED_DEG_S * dt + JUMP_SLACK_DEG
            # The limit grows with the time since the last *accepted* sample, so a genuine
            # sustained change is re-acquired within ~0.2 s.
            if abs(value - self._last_ok) > limit:
                return None
        self._last_ok, self._last_t = value, t
        return self._filter(t, value)


class SlidingSlope:
    """Least-squares slope (units/s) of samples inside a trailing time window.

    Works with irregular timestamps (variable frame rate, dropped frames).

    Low-frame-rate rescue: normally the slope uses the samples of the last ``window_s`` seconds. If that window holds
    fewer than ``min_samples`` (the pipeline is running slowly, e.g. the heavy pose model on a laptop CPU at < 8 fps),
    and ``max_window_s`` is set, the window is stretched back over the last ``min_samples`` samples as long as they
    span no more than ``max_window_s``. At normal frame rates the behaviour is identical to the original (G-17, G-29).
    """

    def __init__(self, window_s: float = VELOCITY_WINDOW_S, min_samples: int = 4,
                 max_window_s: Optional[float] = None):
        self._window = window_s
        self._min_samples = min_samples
        self._max_window = max_window_s if max_window_s is not None else window_s
        self._buf: Deque[Tuple[float, float]] = deque()

    def clear(self) -> None:
        self._buf.clear()

    def add(self, t: float, x: float) -> None:
        self._buf.append((t, x))
        cutoff = t - self._max_window
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def slope(self) -> Optional[float]:
        if not self._buf:
            return None
        t_last = self._buf[-1][0]
        pts = [(t, x) for t, x in self._buf if t >= t_last - self._window]
        if len(pts) < self._min_samples or pts[-1][0] - pts[0][0] < 0.6 * self._window:
            if self._max_window <= self._window or len(self._buf) < self._min_samples:
                return None
            pts = list(self._buf)[-self._min_samples:]                 # stretch back to min_samples samples
            if pts[-1][0] - pts[0][0] > self._max_window:
                return None
        n = len(pts)
        mt = sum(t for t, _ in pts) / n
        mx = sum(x for _, x in pts) / n
        den = sum((t - mt) ** 2 for t, _ in pts)
        if den <= 0.0:
            return None
        return sum((t - mt) * (x - mx) for t, x in pts) / den


class KeypointSmoother:
    """One-Euro smoothing of every landmark coordinate of a ``[N, 3]`` (x_px, y_px, vis) array.

    * Filtering happens in torso-length units (``scale_px``), so tuning is resolution independent.
    * A landmark that is not reliable (visibility below the gate, or non-finite) is passed through
      untouched and its filter is reset, so stale state never drags the point when it reappears.
    * A timestamp gap larger than ``MAX_SAMPLE_GAP_S`` resets every filter.
    * Non-monotonic timestamps return the input unchanged.
    """

    def __init__(self, n_landmarks: int = 33, min_cutoff: float = KP_MIN_CUTOFF_HZ,
                 beta: float = KP_BETA, d_cutoff: float = KP_D_CUTOFF_HZ,
                 min_visibility: float = 0.5):
        self._n = n_landmarks
        self._min_vis = min_visibility
        self._filters: List[Tuple[OneEuroFilter, OneEuroFilter]] = [
            (OneEuroFilter(min_cutoff, beta, d_cutoff), OneEuroFilter(min_cutoff, beta, d_cutoff))
            for _ in range(n_landmarks)]
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        for fx, fy in self._filters:
            fx.reset()
            fy.reset()
        self._last_t = None

    def update(self, t: float, lm: np.ndarray, scale_px: float) -> np.ndarray:
        if lm.shape != (self._n, 3):
            raise ValueError(f"expected shape ({self._n}, 3), got {lm.shape}")
        if self._last_t is not None:
            if t <= self._last_t:
                return lm.copy()
            if t - self._last_t > MAX_SAMPLE_GAP_S:
                self.reset()
        self._last_t = t
        scale = max(float(scale_px), 1.0)
        out = lm.astype(float).copy()
        for i in range(self._n):
            x, y, v = lm[i]
            fx, fy = self._filters[i]
            if not (math.isfinite(x) and math.isfinite(y) and v >= self._min_vis):
                fx.reset()
                fy.reset()
                continue
            out[i, 0] = fx(t, x / scale) * scale
            out[i, 1] = fy(t, y / scale) * scale
        return out
