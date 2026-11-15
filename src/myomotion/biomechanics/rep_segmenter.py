"""Deadlift repetition segmentation and phase state machine [G-26].

    SETUP -> CONCENTRIC_PULL -> LOCKOUT -> ECCENTRIC_DESCENT -> REP_COMPLETE -> SETUP
    (STANDBY = "not in a lift": start-up, tracking loss, stalled rep)

The state machine is the one from the original ``deadlift_tracker.py`` (hysteresis zones, velocity +
displacement + dwell conditions, all timestamp-based; 18 tests cover noise, 15-60 fps, dropouts, spikes
and failed reps) [G-17]. Added on top:

  * the new phase names, with ``REP_COMPLETE`` as a short transient phase;
  * per-rep timing records (concentric / lockout / eccentric time, tempo ratio, peak lockout extension);
  * an estimate of the lift-off time (last time the hip was at its bottom turning point);
  * a causal bar-height peak/valley detector as a second opinion [P5 §3, Fig. 3]: amplitude-normalised
    trajectory, rising half-cycle = concentric pull, falling half-cycle = eccentric descent (deadlift:
    concentric first). It fills ``bar_peak_t_s`` / ``bar_rise_torso`` and never gates rep counting.

Hip angle = shoulder-hip-knee, knee angle = hip-knee-ankle (interior, 180 = fully extended). The hip is
the primary driver because the deadlift is a hip hinge and it is the only joint flexed at the bottom of
*every* variant (conventional, sumo, Romanian, stiff-leg); the knee only confirms lockout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .filters import (MAX_SAMPLE_GAP_S, AngleChannel, SlidingSlope)

# ---------------------------------------------------------------------------
# Phase-detection thresholds (carried over from the tracker; zones use separate enter/exit values so a
# lifter can never be in SETUP and LOCKOUT at once and boundary noise cannot flicker the label) [G-17]
# ---------------------------------------------------------------------------
HIP_SETUP_ENTER_DEG = 130.0    # hip at/below this = "at the bottom"
HIP_SETUP_EXIT_DEG = 140.0     # hip above this = definitely off the floor
HIP_LOCKOUT_ENTER_DEG = 160.0  # hip at/above this (and knee) = lockout zone
HIP_LOCKOUT_EXIT_DEG = 150.0   # hip below this = lockout lost
KNEE_LOCKOUT_DEG = 160.0       # knee must be this extended for lockout

# |hip velocity| below this is "stationary" (~4-5 sigma of residual velocity noise, well below real
# lifting speeds). Hysteresis comes from the displacement and dwell conditions below.
VELOCITY_MOVE_DEG_S = 8.0
ONSET_DISPLACEMENT_DEG = 4.0   # travel from the last turning point that must accompany a velocity trigger

DWELL_ONSET_S = 0.10           # start of a concentric / eccentric movement
DWELL_SETTLE_S = 0.15          # movement has stopped at the bottom
DWELL_HOLD_S = 0.10            # stationary in the lockout zone
DWELL_SETUP_S = 0.30           # standing -> stable bottom position

STALL_TIMEOUT_S = 5.0          # stationary mid-range -> abandon the rep
TRACKING_LOSS_TIMEOUT_S = 1.0  # no valid pose for this long -> STANDBY

# New in this module [G-26]
REP_COMPLETE_HOLD_S = 0.4      # how long the transient REP_COMPLETE phase is shown before SETUP
LIFTOFF_TOL_DEG = 1.5          # "still at the bottom" = within this of the bottom turning point
BAR_PROMINENCE_TORSO = 0.35    # min bar travel (torso lengths) to confirm a peak/valley (provisional)


class Phase(Enum):
    STANDBY = "Standby"
    SETUP = "Setup"
    CONCENTRIC_PULL = "Concentric Pull"
    LOCKOUT = "Lockout"
    ECCENTRIC_DESCENT = "Eccentric Descent"
    REP_COMPLETE = "Rep Complete"


@dataclass(frozen=True)
class RepTiming:
    """One counted repetition (reached lockout). Times are seconds on the input timeline.

    ``t_concentric_start`` / ``t_lockout`` / ``t_descent_start`` / ``t_end`` are *detection* times (the
    detector is causal, so each lags the physical event by a few hundred ms; all lags are of the same
    order, which keeps ``tempo_ratio`` meaningful). ``t_liftoff_s`` is the estimated physical lift-off.
    """
    rep_index: int
    t_liftoff_s: float
    t_concentric_start_s: float
    t_lockout_s: float
    t_descent_start_s: float
    t_end_s: float
    concentric_s: float            # t_lockout - t_concentric_start
    lockout_s: float               # t_descent_start - t_lockout
    eccentric_s: float             # t_end - t_descent_start
    duration_s: float              # t_end - t_concentric_start
    tempo_ratio: float             # eccentric / concentric
    peak_hip_deg: float            # peak lockout extension (largest filtered hip angle)
    peak_knee_deg: float
    bar_peak_t_s: Optional[float] = None
    bar_rise_torso: Optional[float] = None


# ---------------------------------------------------------------------------
# Bar trajectory helpers [P5]
# ---------------------------------------------------------------------------
def normalize_trajectory(h: np.ndarray, lo: float = -180.0, hi: float = 180.0) -> np.ndarray:
    """Amplitude-normalise a trajectory to [-180, 180] as in [P5 §5] (offline). Constant input -> 0."""
    h = np.asarray(h, float)
    span = float(np.nanmax(h) - np.nanmin(h))
    if not math.isfinite(span) or span < 1e-9:
        return np.zeros_like(h)
    return lo + (h - np.nanmin(h)) * (hi - lo) / span


class BarHalfCycleDetector:
    """Causal zig-zag peak/valley detector on bar height (up = +), prominence in torso lengths.

    A valley is confirmed once the height has risen ``prominence`` above the lowest value seen since the
    last peak, and a peak once it has fallen ``prominence`` below the highest value since the last valley.
    So the rising half-cycle (valley -> peak) is the deadlift's concentric pull and the falling one its
    eccentric descent [P5 Fig. 3]. Confirmation lags the extremum by the time the bar needs to travel
    ``prominence``; the returned time is the extremum's own time.
    """

    def __init__(self, prominence: float = BAR_PROMINENCE_TORSO):
        self._prom = prominence
        self.reset()

    def reset(self) -> None:
        self.extrema: List[Tuple[str, float, float]] = []   # (kind, t, height)
        self._seek: Optional[str] = None                    # "peak" | "valley" | None (undecided)
        self._lo: Optional[Tuple[float, float]] = None      # (t, h) lowest since last peak
        self._hi: Optional[Tuple[float, float]] = None      # (t, h) highest since last valley

    def update(self, t: float, h: float) -> Optional[Tuple[str, float, float]]:
        if not math.isfinite(h):
            return None
        if self._lo is None or self._hi is None:
            self._lo = self._hi = (t, h)
            return None
        if self._seek in (None, "valley") and h < self._lo[1]:      # tracking the low
            self._lo = (t, h)
        if self._seek in (None, "peak") and h > self._hi[1]:        # tracking the high
            self._hi = (t, h)
        event = None
        if self._seek in (None, "valley") and h - self._lo[1] >= self._prom:
            event = ("valley", self._lo[0], self._lo[1])
            self._seek, self._hi = "peak", (t, h)
        elif self._seek in (None, "peak") and self._hi[1] - h >= self._prom:
            event = ("peak", self._hi[0], self._hi[1])
            self._seek, self._lo = "valley", (t, h)
        if event:
            self.extrema.append(event)
        return event


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class DeadliftRepSegmenter:
    """Phase segmentation and rep counting from hip/knee angle time series (causal, timestamp driven).

    Zones (hip angle, with hysteresis):

        0 -- SETUP zone --130|140---- mid-range ----150|160-- LOCKOUT zone -- 180
             (enter <=130, exit >140)             (enter >=160 & knee, exit <150)

    Transitions (every one needs its condition to hold for a dwell time):

        STANDBY -> SETUP                hip <= 130 and not ascending, for DWELL_SETUP_S (standing
                                        upright can therefore never be labelled SETUP)
        SETUP -> CONCENTRIC_PULL        hip velocity >= +8 deg/s and hip risen >= 4 deg above the bottom
                                        turning point; or hip leaves the setup zone (> 140)
        CONCENTRIC_PULL -> LOCKOUT      lockout zone (hip >= 160, knee >= 160) and |v| < 8 deg/s
        LOCKOUT -> ECCENTRIC_DESCENT    hip velocity <= -8 deg/s and hip >= 4 deg below the top turning
                                        point; or hip drops below 150 (lockout lost)
        CONCENTRIC_PULL -> ECCENTRIC_DESCENT   same descent test (fast reversal or failed pull)
        ECCENTRIC_DESCENT -> REP_COMPLETE      hip <= 130 and no longer descending. ``rep_count``
                                        increments only if the rep reached lockout
        ECCENTRIC_DESCENT -> CONCENTRIC_PULL   mid-range reversal; does not count a rep
        REP_COMPLETE -> SETUP           after REP_COMPLETE_HOLD_S (or straight to CONCENTRIC_PULL for a
                                        touch-and-go next rep)
        any moving phase -> STANDBY     stationary mid-range for STALL_TIMEOUT_S, or tracking lost for
                                        TRACKING_LOSS_TIMEOUT_S

    ``emit_rep_complete=False`` skips the transient phase (ECCENTRIC_DESCENT goes straight to SETUP);
    the legacy tracker shim uses it to keep its original behaviour.
    """

    def __init__(self, verbose: bool = False, emit_rep_complete: bool = True,
                 on_transition: Optional[Callable[[float, Phase, Phase, "DeadliftRepSegmenter"], None]] = None):
        self.verbose = verbose
        self.emit_rep_complete = emit_rep_complete
        self._on_transition = on_transition
        self.phase: Phase = Phase.STANDBY
        self.rep_count: int = 0
        self.tracking_ok: bool = False
        self.transitions: List[Tuple[float, Phase, Phase]] = []
        self.reps: List[RepTiming] = []

        self._hip_ch = AngleChannel()
        self._knee_ch = AngleChannel()
        self._slope = SlidingSlope()
        self._hip: Optional[float] = None
        self._knee: Optional[float] = None
        self._velocity: Optional[float] = None

        self._last_t: Optional[float] = None
        self._last_valid_t: Optional[float] = None
        self._since: Dict[str, float] = {}
        self._turning_point = 0.0   # running min (SETUP/ECC/REP_COMPLETE) or max (CONC/LOCKOUT)
        self._reached_lockout = False
        self._phase_start_t = 0.0

        # per-attempt bookkeeping
        self._bottom_min = math.inf          # lowest hip angle since the bottom was entered
        self._liftoff_t: Optional[float] = None          # running estimate of the latest bottom
        self._attempt_liftoff_t: Optional[float] = None  # snapshot taken when the pull starts
        self._t_conc: Optional[float] = None
        self._t_lock: Optional[float] = None
        self._t_desc: Optional[float] = None
        self._peak_hip = -math.inf
        self._peak_knee = -math.inf

        # bar trajectory (second opinion)
        self._bar = BarHalfCycleDetector()
        self._bar_elev: Optional[float] = None
        self._bar_range = [math.inf, -math.inf]
        self._bar_peak: Optional[Tuple[float, float]] = None   # (t, rise in torso lengths) for the current rep
        self._bar_last_valley: Optional[float] = None

    # -- public API ---------------------------------------------------------
    def update(self, t: float, raw_hip: float, raw_knee: float,
               bar_height: Optional[float] = None) -> Phase:
        """Feed raw joint angles sampled at time *t* (seconds). ``bar_height`` in torso lengths, up = +."""
        if self._last_t is not None:
            if t <= self._last_t:
                return self.phase            # non-monotonic timestamp: ignore
            if t - self._last_t > MAX_SAMPLE_GAP_S:
                self._since.clear()          # stale dwell timers / velocity
                self._slope.clear()
        self._last_t = t

        if bar_height is not None and math.isfinite(bar_height):
            self._update_bar(t, bar_height)

        hip = self._hip_ch.update(t, raw_hip)
        knee = self._knee_ch.update(t, raw_knee)
        if hip is None or knee is None:
            self.report_missing(t)           # a rejected outlier == no data
            return self.phase

        self._last_valid_t = t
        self.tracking_ok = True
        self._hip, self._knee = hip, knee
        self._slope.add(t, hip)
        velocity = self._slope.slope()
        if velocity is None:                 # velocity window still warming up
            return self.phase
        self._velocity = velocity

        if self.phase in (Phase.CONCENTRIC_PULL, Phase.LOCKOUT, Phase.ECCENTRIC_DESCENT):
            self._peak_hip = max(self._peak_hip, hip)
            self._peak_knee = max(self._peak_knee, knee)

        handler = {
            Phase.STANDBY: self._step_standby,
            Phase.SETUP: self._step_setup,
            Phase.CONCENTRIC_PULL: self._step_concentric,
            Phase.LOCKOUT: self._step_lockout,
            Phase.ECCENTRIC_DESCENT: self._step_eccentric,
            Phase.REP_COMPLETE: self._step_rep_complete,
        }[self.phase]
        handler(t, hip, knee, velocity)
        return self.phase

    def report_missing(self, t: float) -> None:
        """Call when a frame yields no usable pose (dropout / low visibility)."""
        if self._last_valid_t is None:
            return
        missing = t - self._last_valid_t
        self.tracking_ok = missing <= 0.3
        if missing > TRACKING_LOSS_TIMEOUT_S and self.phase != Phase.STANDBY:
            self._reset_tracking(t)

    @property
    def hip_angle(self) -> float:
        return self._hip if self._hip is not None else 0.0

    @property
    def knee_angle(self) -> float:
        return self._knee if self._knee is not None else 0.0

    @property
    def hip_velocity(self) -> float:
        return self._velocity if self._velocity is not None else 0.0

    @property
    def liftoff_t(self) -> Optional[float]:
        """Estimated physical lift-off time. During a pull/lockout/descent this is the lift-off of that attempt;
        in SETUP / REP_COMPLETE it is the running estimate of the bottom (None before the first one)."""
        if self.phase in (Phase.CONCENTRIC_PULL, Phase.LOCKOUT, Phase.ECCENTRIC_DESCENT):
            return self._attempt_liftoff_t
        return self._liftoff_t

    @property
    def phase_start_t(self) -> float:
        return self._phase_start_t

    @property
    def bar_extrema(self) -> List[Tuple[str, float, float]]:
        return list(self._bar.extrema)

    @property
    def bar_elevation(self) -> Optional[float]:
        """Latest bar height normalised to [-180, 180] over the session so far [P5]; None without bar data."""
        return self._bar_elev

    # -- helpers ------------------------------------------------------------
    def _update_bar(self, t: float, h: float) -> None:
        lo, hi = self._bar_range
        self._bar_range = [min(lo, h), max(hi, h)]
        span = self._bar_range[1] - self._bar_range[0]
        self._bar_elev = -180.0 + 360.0 * (h - self._bar_range[0]) / span if span > 1e-9 else 0.0
        ev = self._bar.update(t, h)
        if ev is None:
            return
        kind, t_ext, h_ext = ev
        if kind == "valley":
            self._bar_last_valley = h_ext
        elif kind == "peak" and self._bar_last_valley is not None:
            self._bar_peak = (t_ext, h_ext - self._bar_last_valley)

    def _sustained(self, key: str, cond: bool, t: float, dwell: float) -> bool:
        """True once *cond* has held continuously for *dwell* seconds."""
        if not cond:
            self._since.pop(key, None)
            return False
        return t - self._since.setdefault(key, t) >= dwell

    def _enter(self, new: Phase, t: float, hip: float) -> None:
        old = self.phase
        self.phase = new
        self._since.clear()
        self._turning_point = hip
        self._phase_start_t = t
        self.transitions.append((t, old, new))
        if self._on_transition is not None:
            self._on_transition(t, old, new, self)

    def _reset_tracking(self, t: float) -> None:
        self._hip_ch = AngleChannel()
        self._knee_ch = AngleChannel()
        self._slope.clear()
        self._reached_lockout = False
        self.tracking_ok = False
        self._enter(Phase.STANDBY, t, self.hip_angle)

    def _track_bottom(self, t: float, hip: float) -> None:
        """Estimate lift-off as the last time the hip was still at its bottom turning point."""
        if hip < self._bottom_min:
            self._bottom_min = hip
        if hip <= self._bottom_min + LIFTOFF_TOL_DEG:
            self._liftoff_t = t

    def _start_attempt(self, t: float) -> None:
        self._attempt_liftoff_t = self._liftoff_t
        self._t_conc, self._t_lock, self._t_desc = t, None, None
        self._peak_hip = self._peak_knee = -math.inf
        self._bar_peak = None
        self._bottom_min = math.inf

    # -- per-phase logic ----------------------------------------------------
    def _step_standby(self, t, hip, knee, v):
        at_bottom = hip <= HIP_SETUP_ENTER_DEG and v < VELOCITY_MOVE_DEG_S
        if self._sustained("setup", at_bottom, t, DWELL_SETUP_S):
            self._bottom_min = math.inf
            self._enter(Phase.SETUP, t, hip)

    def _step_setup(self, t, hip, knee, v):
        self._turning_point = min(self._turning_point, hip)   # bottom turning point
        self._track_bottom(t, hip)
        rising = (v >= VELOCITY_MOVE_DEG_S
                  and hip - self._turning_point >= ONSET_DISPLACEMENT_DEG)
        left_zone = hip > HIP_SETUP_EXIT_DEG
        if left_zone or self._sustained("onset", rising, t, DWELL_ONSET_S):
            self._enter(Phase.CONCENTRIC_PULL, t, hip)
            self._start_attempt(t)

    def _step_rep_complete(self, t, hip, knee, v):
        self._step_setup(t, hip, knee, v)                     # touch-and-go: may start the next pull
        if self.phase == Phase.REP_COMPLETE and t - self._phase_start_t >= REP_COMPLETE_HOLD_S:
            self._enter(Phase.SETUP, t, hip)

    def _in_lockout_zone(self, hip, knee) -> bool:
        return hip >= HIP_LOCKOUT_ENTER_DEG and knee >= KNEE_LOCKOUT_DEG

    def _descent_confirmed(self, t, hip, v) -> bool:
        descending = (v <= -VELOCITY_MOVE_DEG_S
                      and self._turning_point - hip >= ONSET_DISPLACEMENT_DEG)
        return self._sustained("descent", descending, t, DWELL_ONSET_S)

    def _begin_descent(self, t, hip) -> None:
        self._t_desc = t
        self._enter(Phase.ECCENTRIC_DESCENT, t, hip)
        self._bottom_min = math.inf

    def _step_concentric(self, t, hip, knee, v):
        self._turning_point = max(self._turning_point, hip)   # top turning point
        in_zone = self._in_lockout_zone(hip, knee)
        if self._sustained("lock_zone", in_zone, t, DWELL_ONSET_S):
            self._reached_lockout = True
        holding = in_zone and abs(v) < VELOCITY_MOVE_DEG_S
        if self._sustained("hold", holding, t, DWELL_HOLD_S):
            self._reached_lockout = True
            self._t_lock = t
            self._enter(Phase.LOCKOUT, t, hip)
        elif self._descent_confirmed(t, hip, v):
            self._begin_descent(t, hip)
        elif self._sustained("stall", abs(v) < VELOCITY_MOVE_DEG_S and not in_zone,
                             t, STALL_TIMEOUT_S):
            self._enter(Phase.STANDBY, t, hip)

    def _step_lockout(self, t, hip, knee, v):
        self._turning_point = max(self._turning_point, hip)
        lost = self._sustained("lost", hip < HIP_LOCKOUT_EXIT_DEG, t, DWELL_ONSET_S)
        if lost or self._descent_confirmed(t, hip, v):
            self._begin_descent(t, hip)

    def _step_eccentric(self, t, hip, knee, v):
        self._turning_point = min(self._turning_point, hip)   # lowest point so far
        self._track_bottom(t, hip)
        settled = hip <= HIP_SETUP_ENTER_DEG and v > -VELOCITY_MOVE_DEG_S
        if self._sustained("settle", settled, t, DWELL_SETTLE_S):
            if self._reached_lockout:
                self._finish_rep(t)
                self._reached_lockout = False
            self._enter(Phase.REP_COMPLETE if self.emit_rep_complete else Phase.SETUP, t, hip)
            return
        reversing = (hip > HIP_SETUP_ENTER_DEG and v >= VELOCITY_MOVE_DEG_S
                     and hip - self._turning_point >= ONSET_DISPLACEMENT_DEG)
        if self._sustained("reverse", reversing, t, DWELL_ONSET_S):
            self._enter(Phase.CONCENTRIC_PULL, t, hip)         # no rep counted
            self._start_attempt(t)
        elif self._sustained("stall", abs(v) < VELOCITY_MOVE_DEG_S, t, STALL_TIMEOUT_S):
            self._enter(Phase.STANDBY, t, hip)

    def _finish_rep(self, t_end: float) -> None:
        self.rep_count += 1
        t_conc = self._t_conc if self._t_conc is not None else t_end
        t_lock = self._t_lock if self._t_lock is not None else (self._t_desc or t_end)
        t_desc = self._t_desc if self._t_desc is not None else t_end
        conc = max(t_lock - t_conc, 1e-3)
        ecc = max(t_end - t_desc, 1e-3)
        bar_t, bar_rise = (self._bar_peak if self._bar_peak is not None else (None, None))
        self.reps.append(RepTiming(
            rep_index=self.rep_count,
            t_liftoff_s=self._attempt_liftoff_t if self._attempt_liftoff_t is not None else t_conc,
            t_concentric_start_s=t_conc, t_lockout_s=t_lock, t_descent_start_s=t_desc, t_end_s=t_end,
            concentric_s=conc, lockout_s=max(t_desc - t_lock, 0.0), eccentric_s=ecc,
            duration_s=t_end - t_conc, tempo_ratio=ecc / conc,
            peak_hip_deg=self._peak_hip if math.isfinite(self._peak_hip) else float("nan"),
            peak_knee_deg=self._peak_knee if math.isfinite(self._peak_knee) else float("nan"),
            bar_peak_t_s=bar_t, bar_rise_torso=bar_rise))
