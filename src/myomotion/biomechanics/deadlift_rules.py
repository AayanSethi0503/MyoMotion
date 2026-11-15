"""Deterministic deadlift form rules and hip/knee dominance classification [G-23, G-24, G-25].

Evidence behind each rule (see docs/GAP_ANALYSIS.md G-23 for the full table):

  Hip/knee bias        [P8] trunk-tibia differential, +/-10 deg. P8 is a *squat* review, so applying it to
                       the deadlift is an extrapolation (G-25): it is a descriptive label, never a fault.
  Lumbar rounding      [P8] lumbar flexion / posterior pelvic tilt lowers compressive tolerance and raises
                       anterior shear; [X1] trainers' "non-neutral spine". Uses the *proxy* spine (G-24):
                       baseline-relative, confidence="proxy", thresholds provisional.
  Hips rise early      [X1] trainers: "locking out the knees too early disrupts proper lifting mechanics".
                       The "stripper deadlift" cue itself is coaching practice (no paper).
  Lockout hyperext.    coaching practice (no paper).
  Bar drift            [X1] "keep the barbell close to the shins and thighs"; thresholds are ours.

EVERY numeric threshold here is provisional and lives in :class:`DeadliftRuleConfig` so it can be tuned on
our own labelled reps. Synthetic tests prove the rule *logic*; they cannot prove real-world sensitivity.

The module never speaks or draws. It emits :class:`FormEvent` objects that the future
``coaching/cue_engine.py`` (G-04) will arbitrate.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Deque, Dict, List, Optional, Set, Tuple

from .deadlift_angles import DeadliftAngles
from .rep_segmenter import HIP_LOCKOUT_ENTER_DEG, Phase, RepTiming
from .spine_proxy import SpineBaseline, SpineDelta


class FormFlag(str, Enum):
    LUMBAR_ROUNDING = "lumbar_rounding"                 # includes posterior-pelvic-tilt / "butt wink" family
    HIPS_RISE_EARLY = "hips_rise_early"                 # "stripper deadlift"
    LOCKOUT_HYPEREXTENSION = "lockout_hyperextension"
    BAR_DRIFT = "bar_drift"


class Severity(str, Enum):
    WARNING = "warning"
    DANGER = "danger"        # safety guardrail ([P8]); highest cue priority (G-04)


class Bias(str, Enum):
    HIP_EXTENSOR = "hip_extensor"        # posterior-chain biased
    KNEE_EXTENSOR = "knee_extensor"      # quadriceps biased
    BALANCED = "balanced"


BIAS_LABEL = {Bias.HIP_EXTENSOR: "Hip-dominant (posterior chain)",
              Bias.KNEE_EXTENSOR: "Knee-dominant (quadriceps)",
              Bias.BALANCED: "Balanced hip / knee"}

FLAG_SEVERITY = {FormFlag.LUMBAR_ROUNDING: Severity.DANGER,
                 FormFlag.HIPS_RISE_EARLY: Severity.WARNING,
                 FormFlag.LOCKOUT_HYPEREXTENSION: Severity.WARNING,
                 FormFlag.BAR_DRIFT: Severity.WARNING}


@dataclass(frozen=True)
class DeadliftRuleConfig:
    """All thresholds. Source tags: [P8] paper value; [G-nn] provisional, tune on labelled data."""
    # --- Hip / knee dominance [P8 Fig. 5]: > +10 hip-extensor, < -10 knee-extensor, else balanced
    bias_threshold_deg: float = 10.0
    # --- Lumbar rounding, proxy (G-24). Deltas are tangent rotations of the proxy curve vs the set-up baseline.
    # A lumbar delta of 4 deg corresponds to ~14.5 deg change of the ear-shoulder-hip bend (lumbar = 0.28 * bend).
    lumbar_enter_deg: float = 4.0
    lumbar_exit_deg: float = 2.5
    lumbar_dwell_s: float = 0.25
    pull_init_window_s: float = 1.0          # rounding is checked at the bottom and for this long after lift-off
    baseline_per_setup: bool = False         # False: baseline = first stable set-up of the session
    # --- Hips rising faster than shoulders (G-23): over the first window after lift-off
    early_pull_window_s: float = 0.8
    early_knee_extension_deg: float = 15.0   # knee must have extended at least this much...
    early_trunk_increase_deg: float = 3.0    # ...while the trunk got MORE horizontal by at least this much
    # --- Hyperextension / lean-back at lockout (G-23). Metric = backward lean of the trunk past vertical
    lockout_backlean_enter_deg: float = 8.0
    lockout_backlean_exit_deg: float = 5.0
    lockout_dwell_s: float = 0.15
    # --- Bar / wrist drift from mid-foot, in foot lengths (G-23)
    bar_drift_enter_foot: float = 0.35
    bar_drift_exit_foot: float = 0.25
    bar_drift_dwell_s: float = 0.20
    # --- Events
    once_per_rep: bool = True                # an event is raised the first time a flag latches in each rep

    def with_overrides(self, **kw) -> "DeadliftRuleConfig":
        return replace(self, **kw)


def classify_bias(trunk_tibia_deg: float, threshold_deg: float = 10.0) -> Bias:
    """Hip vs knee dominance from the trunk-tibia differential [P8]."""
    if trunk_tibia_deg > threshold_deg:
        return Bias.HIP_EXTENSOR
    if trunk_tibia_deg < -threshold_deg:
        return Bias.KNEE_EXTENSOR
    return Bias.BALANCED


@dataclass(frozen=True)
class FormEvent:
    flag: FormFlag
    t_s: float
    rep_index: int                  # the rep this event belongs to (1-based; the upcoming rep while in set-up)
    severity: Severity
    message: str                    # short cue: joint, direction, amount (G-04 style)
    value: float                    # measured value that tripped the rule
    threshold: float
    confidence: str                 # "proxy" (spine proxy / wrist-as-bar) or "direct"
    phase: str


@dataclass(frozen=True)
class RepForm:
    """Form verdict for one counted rep."""
    rep_index: int
    bias: Bias
    trunk_tibia_at_liftoff_deg: float
    trunk_incl_at_liftoff_deg: float
    tibia_incl_at_liftoff_deg: float
    max_lumbar_delta_deg: Optional[float]
    min_lockout_trunk_deg: Optional[float]        # most backward trunk inclination during lockout (hyperextension)
    max_bar_drift_foot: Optional[float]           # largest |bar drift| during the rep
    flags: Tuple[FormFlag, ...]

    @property
    def form_ok(self) -> bool:
        return not self.flags


class _Latch:
    """Enter/exit hysteresis + dwell for a one-sided "value >= enter" condition (timestamp based, G-04)."""

    def __init__(self, enter: float, exit_: float, dwell_s: float):
        self.enter, self.exit, self.dwell = enter, exit_, dwell_s
        self.active = False
        self._since: Optional[float] = None

    def reset(self) -> None:
        self.active = False
        self._since = None

    def update(self, t: float, value: Optional[float], enabled: bool) -> bool:
        """Returns True exactly when the latch switches on at this call."""
        if not enabled:
            self.reset()
            return False
        if value is None or not math.isfinite(value):
            return False                         # no data: hold state, don't advance the timer
        if self.active:
            if value < self.exit:
                self.reset()
            return False
        if value >= self.enter:
            if self._since is None:
                self._since = t
            if t - self._since >= self.dwell:
                self.active = True
                return True
        else:
            self._since = None
        return False


class DeadliftFormAnalyzer:
    """Stateful per-frame rule evaluation + per-rep verdicts.

    Call :meth:`update` every frame with the frame's angles and the segmenter's phase, then
    :meth:`finalize_rep` when the segmenter appends a :class:`RepTiming`.
    """

    def __init__(self, cfg: Optional[DeadliftRuleConfig] = None):
        self.cfg = cfg or DeadliftRuleConfig()
        c = self.cfg
        self._latch: Dict[FormFlag, _Latch] = {
            FormFlag.LUMBAR_ROUNDING: _Latch(c.lumbar_enter_deg, c.lumbar_exit_deg, c.lumbar_dwell_s),
            FormFlag.LOCKOUT_HYPEREXTENSION: _Latch(c.lockout_backlean_enter_deg, c.lockout_backlean_exit_deg,
                                                    c.lockout_dwell_s),
            FormFlag.BAR_DRIFT: _Latch(c.bar_drift_enter_foot, c.bar_drift_exit_foot, c.bar_drift_dwell_s),
        }
        self.baseline = SpineBaseline()
        self._baseline_locked = False
        self._history: Deque[DeadliftAngles] = deque(maxlen=900)   # ~15 s at 60 fps, for lift-off look-ups
        self.events: List[FormEvent] = []
        self.rep_forms: List[RepForm] = []
        self.active: Set[FormFlag] = set()
        self._raised_this_rep: Set[FormFlag] = set()
        self._prev_phase: Phase = Phase.STANDBY
        self._early_done = False
        # per-rep accumulators
        self._max_lumbar: Optional[float] = None
        self._min_lockout_trunk: Optional[float] = None
        self._max_drift: Optional[float] = None
        self._rep_flags: Set[FormFlag] = set()
        self.spine_delta: Optional[SpineDelta] = None

    # -- helpers --------------------------------------------------------------
    @property
    def current_rep_index(self) -> int:
        return len(self.rep_forms) + 1

    def _value_at(self, t: float) -> Optional[DeadliftAngles]:
        best, best_dt = None, 1e9
        for a in self._history:
            d = abs(a.t_s - t)
            if d < best_dt:
                best, best_dt = a, d
        return best

    def _raise(self, flag: FormFlag, t: float, phase: Phase, value: float, threshold: float,
               message: str, confidence: str) -> Optional[FormEvent]:
        self._rep_flags.add(flag)
        if self.cfg.once_per_rep and flag in self._raised_this_rep:
            return None
        self._raised_this_rep.add(flag)
        ev = FormEvent(flag=flag, t_s=t, rep_index=self.current_rep_index, severity=FLAG_SEVERITY[flag],
                       message=message, value=float(value), threshold=float(threshold),
                       confidence=confidence, phase=phase.value)
        self.events.append(ev)
        return ev

    # -- per frame ------------------------------------------------------------
    def update(self, a: DeadliftAngles, phase: Phase, liftoff_t: Optional[float]) -> List[FormEvent]:
        """Evaluate every rule for one frame. Returns the events raised *by this frame*."""
        c, t = self.cfg, a.t_s
        new: List[FormEvent] = []
        self._history.append(a)
        hip = a.hip_signed_deg

        # new attempt bookkeeping
        if phase == Phase.CONCENTRIC_PULL and self._prev_phase != Phase.CONCENTRIC_PULL:
            self._early_done = False
        self._prev_phase = phase

        # ---- spine baseline (first stable set-up, or every set-up if configured)
        if phase == Phase.SETUP and a.spine is not None and (not self._baseline_locked or c.baseline_per_setup):
            self.baseline.add(a.spine)
        elif phase != Phase.SETUP and self.baseline.ready:
            self._baseline_locked = True
        self.spine_delta = self.baseline.delta(a.spine) if a.spine is not None else None

        # ---- 1. lumbar rounding / posterior pelvic tilt (proxy)
        since_liftoff = (t - liftoff_t) if liftoff_t is not None else math.inf
        rounding_phase = (phase in (Phase.SETUP, Phase.REP_COMPLETE, Phase.ECCENTRIC_DESCENT)
                          or (phase == Phase.CONCENTRIC_PULL and since_liftoff <= c.pull_init_window_s))
        lumbar = self.spine_delta.lumbar_deg if self.spine_delta is not None else None
        if lumbar is not None and rounding_phase:
            self._max_lumbar = lumbar if self._max_lumbar is None else max(self._max_lumbar, lumbar)
        if self._latch[FormFlag.LUMBAR_ROUNDING].update(t, lumbar, rounding_phase):
            ev = self._raise(FormFlag.LUMBAR_ROUNDING, t, phase, lumbar, c.lumbar_enter_deg,
                             "Chest up: possible back rounding (proxy)", "proxy")
            if ev:
                new.append(ev)

        # ---- 2. hips rising faster than shoulders (early pull)
        if (phase == Phase.CONCENTRIC_PULL and not self._early_done and liftoff_t is not None
                and 0.0 <= since_liftoff <= c.early_pull_window_s):
            ref = self._value_at(liftoff_t)
            if ref is not None:
                knee_ext = ref.knee_flexion_deg - a.knee_flexion_deg          # >0 = knee straightened
                trunk_inc = a.trunk_incl_deg - ref.trunk_incl_deg             # >0 = trunk more horizontal
                if knee_ext >= c.early_knee_extension_deg and trunk_inc >= c.early_trunk_increase_deg:
                    self._early_done = True
                    ev = self._raise(FormFlag.HIPS_RISE_EARLY, t, phase, knee_ext, c.early_knee_extension_deg,
                                     f"Hips rising too fast: knees +{knee_ext:.0f} deg, back {trunk_inc:.0f} deg "
                                     "flatter. Push the floor away, keep hips down", "direct")
                    if ev:
                        new.append(ev)
        if phase == Phase.CONCENTRIC_PULL and since_liftoff > c.early_pull_window_s:
            self._early_done = True

        # ---- 3. hyperextension / lean-back at lockout
        at_lockout = phase == Phase.LOCKOUT or (phase == Phase.CONCENTRIC_PULL and hip >= HIP_LOCKOUT_ENTER_DEG)
        lean_back = -a.trunk_incl_deg
        if at_lockout:
            self._min_lockout_trunk = (a.trunk_incl_deg if self._min_lockout_trunk is None
                                       else min(self._min_lockout_trunk, a.trunk_incl_deg))
        if self._latch[FormFlag.LOCKOUT_HYPEREXTENSION].update(t, lean_back, at_lockout):
            ev = self._raise(FormFlag.LOCKOUT_HYPEREXTENSION, t, phase, lean_back, c.lockout_backlean_enter_deg,
                             f"Lockout: leaning back {lean_back:.0f} deg. Finish tall, don't lean back",
                             "direct")
            if ev:
                new.append(ev)

        # ---- 4. bar / wrist drift from mid-foot
        drift = abs(a.bar_drift_foot) if a.bar_drift_foot is not None else None
        drift_phase = phase in (Phase.CONCENTRIC_PULL, Phase.ECCENTRIC_DESCENT)
        if drift is not None and drift_phase:
            self._max_drift = drift if self._max_drift is None else max(self._max_drift, drift)
        if self._latch[FormFlag.BAR_DRIFT].update(t, drift, drift_phase):
            fwd = (a.bar_drift_foot or 0.0) > 0
            ev = self._raise(FormFlag.BAR_DRIFT, t, phase, drift, c.bar_drift_enter_foot,
                             ("Bar drifting forward: keep it over mid-foot, close to the legs" if fwd
                              else "Bar drifting back of mid-foot: keep it over mid-foot"),
                             "proxy" if a.bar_source == "wrist" else "direct")
            if ev:
                new.append(ev)

        # Latch-based flags are active while their condition holds; "hips rise early" is a one-shot
        # verdict, shown for the rest of the pull/lockout/descent of the rep it occurred in.
        self.active = {f for f, l in self._latch.items() if l.active}
        if (FormFlag.HIPS_RISE_EARLY in self._rep_flags
                and phase in (Phase.CONCENTRIC_PULL, Phase.LOCKOUT, Phase.ECCENTRIC_DESCENT)):
            self.active.add(FormFlag.HIPS_RISE_EARLY)
        return new

    def reset_attempt(self) -> None:
        """Discard the accumulators of an attempt that did not become a counted rep (STANDBY / failed pull)."""
        self._rep_flags = set()
        self._raised_this_rep = set()
        self._max_lumbar = self._min_lockout_trunk = self._max_drift = None
        for latch in self._latch.values():
            latch.reset()
        self.active = set()

    # -- per rep --------------------------------------------------------------
    def finalize_rep(self, rep: RepTiming) -> RepForm:
        """Close the rep that the segmenter just completed and start a clean accumulator for the next."""
        ref = self._value_at(rep.t_liftoff_s)
        if ref is None:                                      # no history: fall back to the latest frame
            ref = self._history[-1]
        form = RepForm(
            rep_index=rep.rep_index, bias=classify_bias(ref.trunk_tibia_deg, self.cfg.bias_threshold_deg),
            trunk_tibia_at_liftoff_deg=ref.trunk_tibia_deg, trunk_incl_at_liftoff_deg=ref.trunk_incl_deg,
            tibia_incl_at_liftoff_deg=ref.tibia_incl_deg, max_lumbar_delta_deg=self._max_lumbar,
            min_lockout_trunk_deg=self._min_lockout_trunk, max_bar_drift_foot=self._max_drift,
            flags=tuple(sorted(self._rep_flags, key=lambda f: f.value)))
        self.rep_forms.append(form)
        self.reset_attempt()
        return form
