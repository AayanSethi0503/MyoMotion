"""Per-frame deadlift pipeline: landmarks -> smoothing -> angles + spine proxy -> rep phases -> form rules.

One entry point shared by the demo CLI and the tests:

    analyzer = DeadliftAnalyzer()
    for t_s, lm in frames:                 # lm: np.ndarray [33, 3] (x_px, y_px, visibility) or None
        result = analyzer.process(t_s, lm)
    summary = analyzer.summary(meta)       # JSON-serialisable session summary

Design notes
  * The rep segmenter receives angles from the **raw** landmarks: it has its own tuned conditioning
    (median + jump gate + One-Euro + velocity window) [G-17]. Overlay, spine proxy, bar path and form rules use
    the **smoothed** landmarks (``KeypointSmoother``), so the skeleton does not jitter.
  * The body side is locked on the first valid frame so it cannot flip mid-set.
  * ``biomechanics`` stays NumPy-only: the caller converts MediaPipe output to ``[33, 3]``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .deadlift_angles import DeadliftAngles, compute_deadlift_angles
from .deadlift_rules import (Bias, DeadliftFormAnalyzer, DeadliftRuleConfig, FormEvent, FormFlag, RepForm,
                             classify_bias)
from .filters import KP_BETA, KP_D_CUTOFF_HZ, KP_MIN_CUTOFF_HZ, KeypointSmoother, OneEuroFilter
from .landmarks import choose_side
from .rep_segmenter import DeadliftRepSegmenter, Phase, RepTiming
from .spine_proxy import SpineDelta

SCHEMA = "myomotion.deadlift_session/1"

DISCLAIMERS = [
    "Spine/back-rounding cues use a geometric proxy from ear, shoulder and hip landmarks (confidence='proxy'). "
    "It is baseline-relative and cannot see true lumbar curvature; it is a placeholder for the SpinePose model.",
    "Rule thresholds are provisional (docs/GAP_ANALYSIS.md G-23/G-24) and not validated on labelled lifting data.",
    "Hip/knee dominance uses the trunk-tibia differential from a squat review [P8], applied to the deadlift (G-25).",
    "Not a medical device.",
]


@dataclass
class FrameResult:
    t_s: float
    phase: Phase
    rep_count: int
    tracking_ok: bool
    angles: Optional[DeadliftAngles] = None
    landmarks: Optional[np.ndarray] = None          # smoothed [33, 3] for drawing
    spine_delta: Optional[SpineDelta] = None
    active_flags: Set[FormFlag] = field(default_factory=set)
    new_events: List[FormEvent] = field(default_factory=list)
    bias: Optional[Bias] = None
    completed_rep: Optional[Dict[str, Any]] = None  # set on the frame a rep is counted


class DeadliftAnalyzer:
    def __init__(self, rule_cfg: Optional[DeadliftRuleConfig] = None, side: Optional[str] = None,
                 smooth: bool = True):
        self.segmenter = DeadliftRepSegmenter()
        self.rules = DeadliftFormAnalyzer(rule_cfg)
        self.smoother = KeypointSmoother() if smooth else None
        self._smooth = smooth
        self._side = side
        self._scale: Optional[float] = None                         # EMA of torso length in px
        self._bar_f = (OneEuroFilter(KP_MIN_CUTOFF_HZ, KP_BETA, KP_D_CUTOFF_HZ),
                       OneEuroFilter(KP_MIN_CUTOFF_HZ, KP_BETA, KP_D_CUTOFF_HZ))
        self.reps: List[Dict[str, Any]] = []
        self._pending: Optional[RepTiming] = None      # counted rep whose form verdict is finalised after REP_COMPLETE
        self.frames_total = 0
        self.frames_tracked = 0
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None

    # ------------------------------------------------------------------
    def process(self, t_s: float, lm: Optional[np.ndarray], bar_xy: Optional[np.ndarray] = None) -> FrameResult:
        seg = self.segmenter
        self.frames_total += 1
        self.t_first = t_s if self.t_first is None else self.t_first
        self.t_last = t_s

        raw = None
        if lm is not None:
            if self._side is None:
                self._side = choose_side(lm)
            raw = compute_deadlift_angles(lm, t_s, self._side, bar_xy)
        if raw is None:
            n_before = len(seg.transitions)
            seg.report_missing(t_s)
            completed = self._after_segmenter(n_before, len(seg.reps))
            return FrameResult(t_s, seg.phase, seg.rep_count, seg.tracking_ok,
                               active_flags=set(self.rules.active), completed_rep=completed)

        self.frames_tracked += 1
        self._scale = raw.torso_len_px if self._scale is None else 0.9 * self._scale + 0.1 * raw.torso_len_px
        lm_s = self.smoother.update(t_s, lm, self._scale) if self.smoother is not None else lm
        bar_s = self._smooth_bar(t_s, bar_xy)
        ang = compute_deadlift_angles(lm_s, t_s, self._side, bar_s) or raw

        n_before, reps_before = len(seg.transitions), len(seg.reps)
        seg.update(t_s, raw.hip_signed_deg, raw.knee_deg, bar_height=ang.bar_height_torso)
        completed = self._after_segmenter(n_before, reps_before)
        events = self.rules.update(ang, seg.phase, seg.liftoff_t)

        return FrameResult(
            t_s=t_s, phase=seg.phase, rep_count=seg.rep_count, tracking_ok=seg.tracking_ok, angles=ang,
            landmarks=lm_s, spine_delta=self.rules.spine_delta, active_flags=set(self.rules.active),
            new_events=events, bias=classify_bias(ang.trunk_tibia_deg, self.rules.cfg.bias_threshold_deg),
            completed_rep=completed)

    # ------------------------------------------------------------------
    def _smooth_bar(self, t: float, bar_xy: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if bar_xy is None or not self._smooth or self._scale is None:
            return bar_xy
        s = max(self._scale, 1.0)
        return np.array([self._bar_f[0](t, float(bar_xy[0]) / s) * s, self._bar_f[1](t, float(bar_xy[1]) / s) * s])

    def _after_segmenter(self, n_before: int, reps_before: int) -> Optional[Dict[str, Any]]:
        """React to what the segmenter just did.

        A counted rep is *not* finalised immediately: its form verdict is closed when REP_COMPLETE ends, so
        anything seen at the bottom of the descent (e.g. a butt wink) is attributed to the rep it came from.
        An attempt that never became a rep (stall / tracking loss / mid-range reversal) is discarded.
        """
        seg = self.segmenter
        if len(seg.reps) > reps_before:
            self._pending = seg.reps[-1]
        completed = None
        for _t, old, new in seg.transitions[n_before:]:
            if old == Phase.REP_COMPLETE and self._pending is not None:
                completed = self._flush_pending()
            elif new == Phase.STANDBY or (old == Phase.ECCENTRIC_DESCENT and new == Phase.CONCENTRIC_PULL):
                if self._pending is None:
                    self.rules.reset_attempt()
        return completed

    def _flush_pending(self) -> Optional[Dict[str, Any]]:
        if self._pending is None:
            return None
        form = self.rules.finalize_rep(self._pending)
        rep = _jsonable({**dataclasses.asdict(self._pending), "form": _form_dict(form)})
        self.reps.append(rep)
        self._pending = None
        return rep

    # ------------------------------------------------------------------
    def summary(self, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        seg = self.segmenter
        self._flush_pending()                       # clip ended while the last rep was still in REP_COMPLETE
        events = [dataclasses.asdict(e) for e in self.rules.events]
        out = {
            "schema": SCHEMA,
            "meta": meta or {},
            "side": self._side,
            "frames_total": self.frames_total,
            "frames_tracked": self.frames_tracked,
            "duration_s": (self.t_last - self.t_first) if self.t_first is not None else 0.0,
            "rep_count": seg.rep_count,
            "reps": self.reps,
            "events": events,
            "flag_counts": {f.value: sum(1 for e in self.rules.events if e.flag == f) for f in FormFlag},
            "phase_transitions": [{"t_s": t, "from": o.value, "to": n.value} for t, o, n in seg.transitions],
            "bar_extrema": [{"kind": k, "t_s": t, "height_torso": h} for k, t, h in seg.bar_extrema],
            "disclaimers": DISCLAIMERS,
            "rule_config": dataclasses.asdict(self.rules.cfg),
        }
        return _jsonable(out)


def _form_dict(form: RepForm) -> Dict[str, Any]:
    d = dataclasses.asdict(form)
    d["form_ok"] = form.form_ok
    return d


def _jsonable(obj: Any) -> Any:
    """Recursively convert enums / numpy / NaN into JSON-safe values (NaN and inf become null)."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k.value if isinstance(k, Enum) else k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj
