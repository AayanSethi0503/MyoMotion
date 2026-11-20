"""Deterministic deadlift form rules and hip/knee dominance classification [G-23, G-24, G-25].

Evidence behind each rule (see docs/GAP_ANALYSIS.md G-23 for the full table):

  Hip/knee bias        [P8] trunk-tibia differential, +/-10 deg. P8 is a *squat* review, so applying it to
                       the deadlift is an extrapolation (G-25): it is a descriptive label, never a fault.
  Lumbar rounding      [P8] lumbar flexion / posterior pelvic tilt lowers compressive tolerance and raises
                       anterior shear; [X1] trainers' "non-neutral spine". Uses the *proxy* spine (G-24):
                       baseline-relative, confidence="proxy", thresholds provisional.
  Back shape, hinged   [P8] forward trunk inclination reached by lumbar flexion (rounding) rather than by hip flexion
                       with a neutral spine; [P5] convex vs concave rounded back as error classes; [P1] the nine
                       SpinePose points the shape is measured from. ABSOLUTE region angles while the trunk is hinged
                       (G-34), so no set-up baseline and no visible legs are needed. Thresholds provisional.
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
    UPPER_BACK_ROUNDING = "upper_back_rounding"         # thoracic region convex while hinged (G-34); informational


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
                 FormFlag.BAR_DRIFT: Severity.WARNING,
                 FormFlag.UPPER_BACK_ROUNDING: Severity.WARNING}


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
    # Same rule with the real detected spine (SpinePose, G-31, G-33). Region angles come from a smooth cubic fit, which
    # is stable but under-reads a true lumbar change by ~3.5x (true +30 deg reads ~+8.5). In simulation (224 px spine,
    # 2-3 px keypoint noise) a change of 8 deg is exceeded on ~1-7% of single frames with NO true change, before the
    # One-Euro smoothing and the 0.25 s dwell; the cue fires for a true lumbar flexion change of roughly 22-28 deg or
    # more (22-25 on the synthetic lifter in tests/test_views_and_spine.py, ~28 in the noisy simulation). Lumbar is
    # the weakest region (two SpinePose models differed by ~15-30 deg on the same side-on photo), so this cue is
    # deliberately conservative. NOT validated on labelled lifts.
    lumbar_enter_deg_model: float = 8.0
    lumbar_exit_deg_model: float = 5.0
    # Back-silhouette spine (G-32): noisier than a trained keypoint model (polynomial fit to a mask edge, clothing,
    # the gluteal bulge), so more conservative. On a real 39 s recording (4 reps) its lumbar change stayed under
    # 15 deg for any sustained period, so no false warnings; sensitivity to genuine rounding is NOT validated.
    lumbar_enter_deg_contour: float = 15.0
    lumbar_exit_deg_contour: float = 10.0
    # --- Absolute back shape while hinged (G-34). Region angles of the SpinePose points (flexion positive, in the
    # spine's own frame, G-33) are read directly, with no set-up baseline and no visible legs.
    # Evidence (two recordings of ONE lifter, side-on, labelled by eye by the developer, ~25 neutral and ~80-100 visibly
    # rounded hinged frames; scratch analysis, docs/GAP_ANALYSIS.md G-34): the THORACIC angle separated them with
    # AUC 0.98 / 1.00 / 1.00 and d' 2.6 / 4.7 / 4.0 for SpinePose small / medium / large; neutral hinges read <= +5
    # (medium: <= 0), rounded ones +5..+17 (median). The LUMBAR angle did not hold up across model sizes (AUC 0.70 /
    # 0.93 / 0.59; neutral frames spread -18..+23 deg), so lumbar cues are off in the demo (``lumbar_cues``).
    # The thresholds sit between the clusters. NOT validated: another body, camera or clothing shifts the readings
    # (pronounced kyphosis may read high), which is why the cue needs the hinge, a dwell and the SpinePose source.
    back_abs_cues: bool = True
    hinge_enter_deg: float = 30.0            # trunk inclination from vertical that counts as hinged (hysteresis)
    hinge_exit_deg: float = 20.0
    thoracic_abs_enter_deg_model: float = 6.0
    thoracic_abs_exit_deg_model: float = 3.0
    lumbar_abs_enter_deg_model: float = 11.0     # only used when lumbar_cues is True (see above)
    lumbar_abs_exit_deg_model: float = 7.0
    # SpinePose-sourced LUMBAR cues (the relative-to-set-up rule and the absolute one). The library default stays True
    # so the rule logic stays testable; the demo CLI turns it off unless --lumbar-cues is given (lumbar flexion from 2-D
    # keypoints is the weakest region: P8/P5/OpenCap, and measured here, G-34). Values are still displayed and logged.
    lumbar_cues: bool = True
    # False = the geometric PROXY spine is shown but raises no back-rounding warning. On real webcam footage the proxy
    # flagged "rounding" on every rep, including while the lifter stood upright (it only sees head position relative to
    # the trunk line, G-24). Cues from detected SpinePose points are unaffected. The library default stays True so the
    # proxy rule logic stays testable; the demo CLI turns it off unless --proxy-spine-cues is given.
    proxy_spine_cues: bool = True
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
    max_lumbar_delta_deg: Optional[float]         # largest change HELD for lumbar_dwell_s (a 1-2 frame glitch is not)
    min_lockout_trunk_deg: Optional[float]        # most backward trunk inclination during lockout (hyperextension)
    max_bar_drift_foot: Optional[float]           # largest |bar drift| during the rep
    flags: Tuple[FormFlag, ...]
    peak_lumbar_abs_deg: Optional[float] = None   # largest ABSOLUTE lumbar angle held for the dwell while hinged (G-34)
    peak_thoracic_abs_deg: Optional[float] = None

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


ABS_STALE_S = 0.6        # a cue is not held across a longer gap without a SpinePose measurement (G-34)


class _HeldMax:
    """Largest value held continuously for ``dwell_s``: the minimum over a trailing window, maximised over time. This
    is the quantity a dwell latch actually sees, so a one- or two-frame keypoint glitch never appears in it."""

    def __init__(self, dwell_s: float):
        self.dwell = dwell_s
        self.value: Optional[float] = None
        self._recent: Deque[Tuple[float, float]] = deque()

    def add(self, t: float, v: float) -> None:
        self._recent.append((t, v))
        while self._recent[0][0] < t - self.dwell:
            self._recent.popleft()
        if t - self._recent[0][0] >= 0.8 * self.dwell:
            held = min(x for _, x in self._recent)
            self.value = held if self.value is None else max(self.value, held)

    def pause(self) -> None:
        """No usable value this frame: start the window again (the peak so far is kept)."""
        self._recent.clear()

    def reset(self) -> None:
        self.pause()
        self.value = None


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
        self._lumbar_model = _Latch(c.lumbar_enter_deg_model, c.lumbar_exit_deg_model, c.lumbar_dwell_s)
        self._lumbar_contour = _Latch(c.lumbar_enter_deg_contour, c.lumbar_exit_deg_contour, c.lumbar_dwell_s)
        # One set-up reference PER spine source ("proxy" / "spinepose" / "contour"): a delta is only meaningful against
        # a baseline made by the same estimator, and a one-frame fallback to another source (SpinePose missed a frame)
        # must not throw the real baseline away (found on a real clip: reps 3 and 4 lost their lumbar reading).
        self._baselines: Dict[str, SpineBaseline] = {}
        self._baseline_locked: Dict[str, bool] = {}
        # Absolute back shape while hinged (G-34): hinge state, one latch and one held-peak per region.
        self._hinged = False
        self._last_measured_t = -math.inf                        # last frame with a SpinePose measurement
        self._abs_scope = "rep"                  # "rep": inside the segmenter's reps; "bout": legs not in view
        self._lumbar_abs = _Latch(c.lumbar_abs_enter_deg_model, c.lumbar_abs_exit_deg_model, c.lumbar_dwell_s)
        self._thoracic_abs = _Latch(c.thoracic_abs_enter_deg_model, c.thoracic_abs_exit_deg_model, c.lumbar_dwell_s)
        self._lumbar_abs_held = _HeldMax(c.lumbar_dwell_s)
        self._thoracic_abs_held = _HeldMax(c.lumbar_dwell_s)
        self._history: Deque[DeadliftAngles] = deque(maxlen=900)   # ~15 s at 60 fps, for lift-off look-ups
        self.events: List[FormEvent] = []
        self.rep_forms: List[RepForm] = []
        self.active: Set[FormFlag] = set()
        self._raised_this_rep: Set[FormFlag] = set()
        self._prev_phase: Phase = Phase.STANDBY
        self._early_done = False
        # per-rep accumulators
        self._max_lumbar: Optional[float] = None
        self._lumbar_recent: Deque[Tuple[float, float]] = deque()     # (t_s, lumbar delta) over the last dwell window
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
               message: str, confidence: str, rep_flag: bool = True) -> Optional[FormEvent]:
        if rep_flag:
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
        self._abs_scope = "rep"
        self._history.append(a)
        hip = a.hip_signed_deg

        # new attempt bookkeeping
        if phase == Phase.CONCENTRIC_PULL and self._prev_phase != Phase.CONCENTRIC_PULL:
            self._early_done = False
        self._prev_phase = phase

        # ---- spine baseline (first stable set-up, or every set-up if configured). Only a spine whose curvature is
        # visible counts (not from the front/back). Each source keeps its own baseline (see __init__).
        spine = a.spine if (a.spine is not None and a.spine.curvature_visible) else None
        base = None
        if spine is not None:
            base = self._baselines.setdefault(spine.source, SpineBaseline())
            locked = self._baseline_locked.get(spine.source, False)
            if phase == Phase.SETUP and (not locked or c.baseline_per_setup):
                base.add(spine)
            elif phase != Phase.SETUP and base.ready:
                self._baseline_locked[spine.source] = True
        self.spine_delta = base.delta(spine) if base is not None else None

        # ---- 1. lumbar rounding / posterior pelvic tilt (proxy)
        since_liftoff = (t - liftoff_t) if liftoff_t is not None else math.inf
        rounding_phase = (phase in (Phase.SETUP, Phase.REP_COMPLETE, Phase.ECCENTRIC_DESCENT)
                          or (phase == Phase.CONCENTRIC_PULL and since_liftoff <= c.pull_init_window_s))
        lumbar = self.spine_delta.lumbar_deg if self.spine_delta is not None else None
        src = spine.source if spine is not None else "proxy"
        record_ok = src != "proxy" or c.proxy_spine_cues         # may the value be reported as a measurement?
        cue_allowed = record_ok and (src != "spinepose" or c.lumbar_cues)     # may it raise the lumbar cue?
        # The reported maximum (a) covers only frames from a spine source that is allowed to raise a cue (a display-only
        # proxy value must not show up as a measured lumbar change) and (b) is the largest value HELD for the dwell,
        # i.e. the same quantity the cue latch sees. On a real clip single-frame keypoint glitches read +35 and +79 deg
        # (not physiological) without ever raising the cue; the raw per-frame maximum would have reported them.
        if lumbar is not None and rounding_phase and record_ok:
            self._lumbar_recent.append((t, lumbar))
            while self._lumbar_recent[0][0] < t - c.lumbar_dwell_s:
                self._lumbar_recent.popleft()
            if t - self._lumbar_recent[0][0] >= 0.8 * c.lumbar_dwell_s:
                held = min(v for _, v in self._lumbar_recent)
                self._max_lumbar = held if self._max_lumbar is None else max(self._max_lumbar, held)
        else:
            self._lumbar_recent.clear()
        latches = {"proxy": self._latch[FormFlag.LUMBAR_ROUNDING], "spinepose": self._lumbar_model,
                   "contour": self._lumbar_contour}
        latch = latches.get(src, latches["proxy"])
        for other in latches.values():                      # only the latch of the current spine source runs
            if other is not latch:
                other.reset()
        enter = {"proxy": c.lumbar_enter_deg, "spinepose": c.lumbar_enter_deg_model,
                 "contour": c.lumbar_enter_deg_contour}.get(src, c.lumbar_enter_deg)
        msg = {"proxy": "Chest up: possible back rounding (proxy)", "spinepose": "Chest up: back rounding detected",
               "contour": "Chest up: possible back rounding (back contour)"}.get(src, "Chest up: back rounding")
        if latch.update(t, lumbar if cue_allowed else None, rounding_phase and cue_allowed):
            ev = self._raise(FormFlag.LUMBAR_ROUNDING, t, phase, lumbar, enter, msg,
                             {"proxy": "proxy", "spinepose": "direct", "contour": "contour"}.get(src, "proxy"))
            if ev:
                new.append(ev)

        # ---- 1b. absolute back shape while hinged (G-34): the cue that needs no baseline and no set-up phase
        new += self._absolute_back(t, a.trunk_incl_deg, spine, phase)

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
        if self._lumbar_model.active or self._lumbar_contour.active:
            self.active.add(FormFlag.LUMBAR_ROUNDING)
        self.active |= self._abs_active()
        if (FormFlag.HIPS_RISE_EARLY in self._rep_flags
                and phase in (Phase.CONCENTRIC_PULL, Phase.LOCKOUT, Phase.ECCENTRIC_DESCENT)):
            self.active.add(FormFlag.HIPS_RISE_EARLY)
        return new

    # -- absolute back shape while hinged (G-34) -------------------------------
    @property
    def hinged(self) -> bool:
        """True while the trunk is hinged forward (the back-shape rule is armed)."""
        return self._hinged

    def _abs_active(self) -> Set[FormFlag]:
        out: Set[FormFlag] = set()
        if self._lumbar_abs.active:
            out.add(FormFlag.LUMBAR_ROUNDING)
        if self._thoracic_abs.active:
            out.add(FormFlag.UPPER_BACK_ROUNDING)
        return out

    def _update_hinge(self, trunk_incl_deg: float) -> bool:
        """Hinged = trunk inclination from vertical >= ``hinge_enter_deg``, until it falls below ``hinge_exit_deg``."""
        c = self.cfg
        if not math.isfinite(trunk_incl_deg):
            return self._hinged
        if not self._hinged and trunk_incl_deg >= c.hinge_enter_deg:
            self._hinged = True
        elif self._hinged and trunk_incl_deg < c.hinge_exit_deg:
            self._hinged = False
            self._lumbar_abs.reset()
            self._thoracic_abs.reset()
            if self._abs_scope == "bout":                 # no reps here: one event per hinge, not one per session
                self._raised_this_rep -= {FormFlag.LUMBAR_ROUNDING, FormFlag.UPPER_BACK_ROUNDING}
        return self._hinged

    def _absolute_back(self, t: float, trunk_incl_deg: float, spine, phase: Phase) -> List[FormEvent]:
        """Lumbar and thoracic region angles read directly while the trunk is hinged. Only the SpinePose source is
        calibrated for this (the proxy cannot see curvature; the contour has its own, larger noise)."""
        c = self.cfg
        hinged = self._update_hinge(trunk_incl_deg)
        armed = bool(c.back_abs_cues and hinged)                 # the hinge arms the rule; a measurement drives it
        measured = bool(spine is not None and spine.curvature_visible and spine.source == "spinepose")
        lum = spine.lumbar_deg if (armed and measured) else None    # measured (and reported) even when its cue is off
        tho = spine.thoracic_deg if (armed and measured) else None
        if measured:
            self._last_measured_t = t
        elif armed and t - self._last_measured_t > ABS_STALE_S:
            self._lumbar_abs.reset()                              # no measurement for a while: do not hold a stale cue
            self._thoracic_abs.reset()
        in_rep = self._abs_scope == "rep"
        for held, v in ((self._lumbar_abs_held, lum), (self._thoracic_abs_held, tho)):
            if v is not None:
                held.add(t, v)
            else:
                held.pause()
        new: List[FormEvent] = []
        # A frame without a SpinePose measurement (missed detection, bridge-check reject, another source) neither
        # advances nor resets a latch: _Latch.update() holds its state for value=None while enabled.
        if self._lumbar_abs.update(t, lum, armed and c.lumbar_cues):
            ev = self._raise(FormFlag.LUMBAR_ROUNDING, t, phase, lum, c.lumbar_abs_enter_deg_model,
                             f"Chest up: lower back is rounded (lumbar {lum:+.0f} deg)", "absolute", rep_flag=in_rep)
            if ev:
                new.append(ev)
        if self._thoracic_abs.update(t, tho, armed):
            ev = self._raise(FormFlag.UPPER_BACK_ROUNDING, t, phase, tho, c.thoracic_abs_enter_deg_model,
                             f"Back rounding (upper/mid back, {tho:+.0f} deg): chest up, shoulders back", "absolute",
                             rep_flag=in_rep)
            if ev:
                new.append(ev)
        return new

    def update_back_only(self, t: float, trunk_incl_deg: float, spine,
                         phase: Phase = Phase.STANDBY) -> List[FormEvent]:
        """Back rules for a frame where the legs are not in view: no hip/knee angles, so no phases, no rep counting and
        no relative (set-up baseline) rule, but the absolute back-shape rule needs only the trunk inclination and the
        spine (G-34). Events belong to the current hinge, not to a rep."""
        self._abs_scope = "bout"
        spine = spine if (spine is not None and spine.curvature_visible) else None
        new = self._absolute_back(t, trunk_incl_deg, spine, phase)
        self.active = self._abs_active()
        return new

    def reset_baseline(self) -> None:
        """Forget every set-up reference (e.g. the body side changed, so the spine reference was the other side)."""
        self._baselines.clear()
        self._baseline_locked.clear()
        self._lumbar_model.reset()
        self._lumbar_contour.reset()
        self._latch[FormFlag.LUMBAR_ROUNDING].reset()
        self._lumbar_abs.reset()
        self._thoracic_abs.reset()
        self._hinged = False

    def reset_attempt(self) -> None:
        """Discard the accumulators of an attempt that did not become a counted rep (STANDBY / failed pull)."""
        self._rep_flags = set()
        self._raised_this_rep = set()
        self._max_lumbar = self._min_lockout_trunk = self._max_drift = None
        self._lumbar_recent.clear()
        for latch in self._latch.values():
            latch.reset()
        self._lumbar_model.reset()
        self._lumbar_contour.reset()
        self._lumbar_abs.reset()
        self._thoracic_abs.reset()
        self._lumbar_abs_held.reset()
        self._thoracic_abs_held.reset()
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
            flags=tuple(sorted(self._rep_flags, key=lambda f: f.value)),
            peak_lumbar_abs_deg=self._lumbar_abs_held.value, peak_thoracic_abs_deg=self._thoracic_abs_held.value)
        self.rep_forms.append(form)
        self.reset_attempt()
        return form
