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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .deadlift_angles import DeadliftAngles, compute_deadlift_angles
from .geometry import distance, inclination_from_vertical
from .deadlift_rules import (Bias, DeadliftFormAnalyzer, DeadliftRuleConfig, FormEvent, FormFlag, RepForm,
                             classify_bias)
from .filters import KP_BETA, KP_D_CUTOFF_HZ, KP_MIN_CUTOFF_HZ, KeypointSmoother, OneEuroFilter
from .landmarks import (ANGLED_ENTER_SPAN_RATIO, ANGLED_EXIT_SPAN_RATIO, SIDE_SWITCH_HOLD_S, SIDE_SWITCH_MARGIN,
                        FramingReport, estimate_facing, estimate_facing_from_trunk, extract_midline_pose,
                        extract_side_pose, framing_report, shoulder_span_ratio, side_scores)
from .pose3d import estimate_forward, heading_span_equivalent
from .spine_contour import estimate_spine_from_mask
from .rep_segmenter import DeadliftRepSegmenter, Phase, RepTiming
from .spine_proxy import SpineDelta, SpineProxy, build_spine_from_points, build_spine_proxy

SCHEMA = "myomotion.deadlift_session/1"
MAX_SPINE_LOG_ROWS = 200_000          # ~1 h at 60 fps; a cap, not a limit anyone should reach

DISCLAIMERS = [
    "Spine source is recorded per frame. 'proxy' = a geometric curve through ear, shoulder and hip: baseline-relative, "
    "it cannot see true curvature. 'spinepose' = detected spine keypoints (docs/GAP_ANALYSIS.md G-31, unvalidated).",
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
    landmarks: Optional[np.ndarray] = None          # [33, 3] for drawing: smoothed if tracked, else the raw detection
    framing: Optional[FramingReport] = None         # why tracking is (not) possible; None = no person detected
    partial_spine: Optional[SpineProxy] = None      # proxy spine when ear/shoulder/hip are visible but the legs are not
    view: str = "side"                              # "side" (exact 2-D angles) | "angled" (3-D world angles)
    angle_source: str = "image"                     # "image" | "world"
    spine_delta: Optional[SpineDelta] = None
    active_flags: Set[FormFlag] = field(default_factory=set)
    new_events: List[FormEvent] = field(default_factory=list)
    bias: Optional[Bias] = None
    completed_rep: Optional[Dict[str, Any]] = None  # set on the frame a rep is counted
    back_only: bool = False                         # legs not in view: only the back was analysed (no reps, G-34)
    trunk_incl_deg: Optional[float] = None          # shoulder-hip inclination from vertical (back_only frames)
    hinged: bool = False                            # trunk hinged forward (back rules active)


class DeadliftAnalyzer:
    def __init__(self, rule_cfg: Optional[DeadliftRuleConfig] = None, side: Optional[str] = None,
                 smooth: bool = True):
        self.segmenter = DeadliftRepSegmenter()
        self.rules = DeadliftFormAnalyzer(rule_cfg)
        self.smoother = KeypointSmoother() if smooth else None
        self._smooth = smooth
        self._side = side
        self._side_fixed = side is not None                         # the user forced a side: never switch
        self._switch_since: Optional[float] = None
        self._facing: Optional[int] = None                          # from a decaying weighted vote over frames
        self._facing_votes = 0.0
        self._view = "side"                                         # "side" | "angled", with hysteresis
        self._span_ema: Optional[float] = None
        self._spine_smoother = KeypointSmoother(n_landmarks=9, min_visibility=0.2) if smooth else None
        self.view_frames: Dict[str, int] = {"side": 0, "angled": 0}
        self.spine_frames: Dict[str, int] = {}
        self._scale: Optional[float] = None                         # EMA of torso length in px
        self._bar_f = (OneEuroFilter(KP_MIN_CUTOFF_HZ, KP_BETA, KP_D_CUTOFF_HZ),
                       OneEuroFilter(KP_MIN_CUTOFF_HZ, KP_BETA, KP_D_CUTOFF_HZ))
        self.reps: List[Dict[str, Any]] = []
        self._pending: Optional[RepTiming] = None      # counted rep whose form verdict is finalised after REP_COMPLETE
        self.frames_total = 0
        self.frames_tracked = 0
        self.frames_back_only = 0                      # legs out of view, back still analysed (G-34)
        self.spine_log: List[Dict[str, Any]] = []      # one row per frame that had a spine: data for calibration
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None

    # ------------------------------------------------------------------
    def process(self, t_s: float, lm: Optional[np.ndarray], bar_xy: Optional[np.ndarray] = None,
                image_wh: Optional[Tuple[int, int]] = None, world: Optional[np.ndarray] = None,
                spine_points: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> FrameResult:
        """Analyse one frame.

        ``lm``: ``[33, 3]`` (x_px, y_px, visibility), or None if no person was detected.
        ``world``: optional MediaPipe world landmarks ``[33, 3]`` (metres); used for the angles when the person is not
        side-on (facing the camera, turned, left side to the camera).
        ``spine_points``: optional ``[9, 3]`` (x_px, y_px, score) detected spine points in ``SPINE_LABELS`` order
        (SpinePose); replaces the geometric proxy.
        ``mask``: optional person segmentation mask ``[H, W]`` (MediaPipe). When the person is side-on and no
        ``spine_points`` are given, the spine is estimated from the back silhouette (G-32), source "contour".

        Always returns something drawable: when the core joints are not visible the result has no angles, but carries
        the raw landmarks, a ``framing`` report saying what to fix, and a spine if one can be drawn.
        """
        seg = self.segmenter
        self.frames_total += 1
        self.t_first = t_s if self.t_first is None else self.t_first
        self.t_last = t_s

        side_now = self._update_side(lm, t_s)
        framing = framing_report(lm, side_now, image_wh=image_wh)
        view = self._update_view(lm, world)
        raw = None
        spine_in, spine_src = None, "spinepose"
        if lm is not None:
            self._vote_facing(lm, side_now, world if view == "angled" else None)
            spine_points, spine_src = self._spine_input(lm, side_now, view, spine_points, mask)
            if self._scale is None and side_now is not None:       # scale from the torso alone: legs may be out of view
                sp0 = extract_side_pose(lm, side_now, require=("shoulder", "hip"))
                if sp0 is not None:
                    self._scale = distance(sp0["shoulder"], sp0["hip"])
            spine_in = self._smooth_spine(t_s, spine_points)
            raw = compute_deadlift_angles(lm, t_s, side_now, bar_xy, facing=self._facing, world=world, view=view,
                                          spine_points=spine_in, spine_source=spine_src)
        if raw is None:
            n_before = len(seg.transitions)
            seg.report_missing(t_s)
            completed = self._after_segmenter(n_before, len(seg.reps))
            back = self._back_only(t_s, lm, side_now, view, spine_in, spine_src) if lm is not None else None
            if back is not None:                                   # legs not in view: analyse the back on its own
                return FrameResult(t_s, seg.phase, seg.rep_count, seg.tracking_ok, landmarks=back["lm"],
                                   framing=framing, partial_spine=back["spine"], active_flags=set(self.rules.active),
                                   new_events=back["events"], completed_rep=completed, view=view, back_only=True,
                                   trunk_incl_deg=back["trunk"], hinged=self.rules.hinged)
            partial = self._partial_spine(lm, framing, view, spine_points, spine_src if lm is not None else "spinepose")
            return FrameResult(t_s, seg.phase, seg.rep_count, seg.tracking_ok, landmarks=lm, framing=framing,
                               partial_spine=partial, active_flags=set(self.rules.active), completed_rep=completed,
                               view=view)

        self.frames_tracked += 1
        self.view_frames[view] = self.view_frames.get(view, 0) + 1
        self._scale = raw.torso_len_px if self._scale is None else 0.9 * self._scale + 0.1 * raw.torso_len_px
        lm_s = self.smoother.update(t_s, lm, self._scale) if self.smoother is not None else lm
        bar_s = self._smooth_bar(t_s, bar_xy)
        ang = compute_deadlift_angles(lm_s, t_s, side_now, bar_s, facing=self._facing, world=world, view=view,
                                      spine_points=self._smooth_spine(t_s, spine_points, advance=False),
                                      spine_source=spine_src) or raw
        if ang.spine is not None:
            self.spine_frames[ang.spine.source] = self.spine_frames.get(ang.spine.source, 0) + 1
            self._log_spine(t_s, ang.spine, ang.trunk_incl_deg, view, back_only=False)

        n_before, reps_before = len(seg.transitions), len(seg.reps)
        seg.update(t_s, raw.hip_signed_deg, raw.knee_deg, bar_height=ang.bar_height_torso)
        completed = self._after_segmenter(n_before, reps_before)
        events = self.rules.update(ang, seg.phase, seg.liftoff_t)

        return FrameResult(
            t_s=t_s, phase=seg.phase, rep_count=seg.rep_count, tracking_ok=seg.tracking_ok, angles=ang,
            framing=framing, landmarks=lm_s, spine_delta=self.rules.spine_delta, active_flags=set(self.rules.active),
            new_events=events, bias=classify_bias(ang.trunk_tibia_deg, self.rules.cfg.bias_threshold_deg),
            completed_rep=completed, view=view, angle_source=ang.angle_source, trunk_incl_deg=ang.trunk_incl_deg,
            hinged=self.rules.hinged)

    # ------------------------------------------------------------------
    def _back_only(self, t_s: float, lm: np.ndarray, side: Optional[str], view: str,
                   spine_pts: Optional[np.ndarray], spine_src: str) -> Optional[Dict[str, Any]]:
        """Analyse the back when the legs are not in view (knees/ankles hidden or cut off): the back rules need only the
        near-side shoulder and hip plus the spine, so a lifter standing too close to a webcam still gets back-rounding
        feedback. Reps are NOT counted and nothing leg-related is evaluated. Returns None if shoulder or hip are not
        reliable. (G-34; found on a real session where the legs were out of frame for 850 of 923 frames.)"""
        if side is None:
            return None
        sp = extract_side_pose(lm, side, require=("shoulder", "hip"), facing=self._facing)
        if sp is None:
            return None
        torso = distance(sp["shoulder"], sp["hip"])
        self._scale = torso if self._scale is None else 0.9 * self._scale + 0.1 * torso
        lm_s = self.smoother.update(t_s, lm, self._scale) if self.smoother is not None else lm
        sp = extract_side_pose(lm_s, side, require=("shoulder", "hip"), facing=self._facing) or sp
        facing = self._facing or sp.facing
        trunk = inclination_from_vertical(sp["hip"], sp["shoulder"], facing)
        spine = None
        if spine_pts is not None:
            pts = np.asarray(spine_pts, float)                     # already smoothed by the caller
            spine = build_spine_from_points(pts[:, :2], facing, pts[:, 2], curvature_visible=view == "side",
                                            source=spine_src)
        elif view == "side" and sp.has("ear"):
            spine = build_spine_proxy(sp)
        self.frames_back_only += 1
        events = self.rules.update_back_only(t_s, trunk, spine, self.segmenter.phase)
        if spine is not None:
            self.spine_frames[spine.source] = self.spine_frames.get(spine.source, 0) + 1
            self._log_spine(t_s, spine, trunk, view, back_only=True)
        return {"lm": lm_s, "spine": spine, "events": events, "trunk": trunk}

    def _log_spine(self, t_s: float, spine: SpineProxy, trunk_incl_deg: float, view: str, back_only: bool) -> None:
        """One row per frame with a spine: the raw material for calibrating the back thresholds on labelled footage."""
        if len(self.spine_log) >= MAX_SPINE_LOG_ROWS:
            return
        self.spine_log.append({
            "t_s": round(float(t_s), 3), "source": spine.source, "view": view, "back_only": back_only,
            "trunk_incl_deg": round(float(trunk_incl_deg), 2), "hinged": self.rules.hinged,
            "lumbar_deg": round(float(spine.lumbar_deg), 2), "thoracic_deg": round(float(spine.thoracic_deg), 2),
            "cervical_deg": round(float(spine.cervical_deg), 2), "pelvic_deg": round(float(spine.pelvic_deg), 2),
            "curvature_visible": bool(spine.curvature_visible), "confidence": round(float(spine.confidence), 3)})

    # ------------------------------------------------------------------
    def _update_side(self, lm: Optional[np.ndarray], t: float) -> Optional[str]:
        """Which body side faces the camera. Re-evaluated every frame (it used to lock forever, which broke as soon
        as the lifter turned around). The other side takes over only if the camera sees it clearly better for a while,
        so a flicker cannot flip it; a front view, where both are equally visible, never switches."""
        if lm is None or self._side_fixed:
            return self._side
        scores = side_scores(lm)
        cand = "right" if scores["right"] >= scores["left"] else "left"
        if self._side is None:
            self._side = cand
        elif cand != self._side and scores[cand] - scores[self._side] > SIDE_SWITCH_MARGIN:
            if self._switch_since is None:
                self._switch_since = t
            elif t - self._switch_since >= SIDE_SWITCH_HOLD_S:
                self._side, self._switch_since = cand, None
                self._facing, self._facing_votes = None, 0.0                 # the facing cues belong to the old side
                self.rules.reset_baseline()                                  # the spine reference was the other side
        else:
            self._switch_since = None
        return self._side

    def _update_view(self, lm: Optional[np.ndarray], world: Optional[np.ndarray] = None) -> str:
        """"side" (exact 2-D angles) or "angled" (3-D world angles), with hysteresis.

        The heading comes from the 3-D hip axis when world landmarks are available (it works even when the far
        shoulder is not visible in the image); otherwise from the 2-D shoulder span."""
        if lm is None:
            return self._view
        span = heading_span_equivalent(world) if world is not None else None
        if span is None:
            span = shoulder_span_ratio(lm)
        span = 0.0 if span is None else span
        self._span_ema = span if self._span_ema is None else 0.7 * self._span_ema + 0.3 * span
        if self._view == "side" and self._span_ema > ANGLED_ENTER_SPAN_RATIO:
            self._view = "angled"
        elif self._view == "angled" and self._span_ema < ANGLED_EXIT_SPAN_RATIO:
            self._view = "side"
        return self._view

    def _vote_facing(self, lm: np.ndarray, side: Optional[str], world: Optional[np.ndarray]) -> None:
        """Facing direction (+1 image-right) from a *decaying* weighted vote, so it follows the lifter if they turn
        around but one noisy frame cannot flip it. Side-on cues: toe-heel (3), nose-ear (2), knee-ankle (1). Angled
        views have no reliable 2-D cue, so the 3-D forward vector votes instead (4)."""
        if side is None:
            return
        direction, weight = estimate_facing(lm, side)
        trunk_dir, trunk_w = estimate_facing_from_trunk(lm, side)
        if trunk_w and weight < trunk_w:                 # hinged: the trunk lean beats a hanging head or hidden feet
            direction, weight = trunk_dir, trunk_w
        if world is not None:
            fw = estimate_forward(world, lm[:, 2])
            if fw is not None and abs(fw[0]) > 0.25:
                direction, weight = (1 if fw[0] > 0 else -1), 4
        self._facing_votes *= 0.95
        if weight:
            self._facing_votes += direction * weight
        if abs(self._facing_votes) > 1.0:
            self._facing = 1 if self._facing_votes > 0 else -1

    def _smooth_spine(self, t: float, pts: Optional[np.ndarray], advance: bool = True) -> Optional[np.ndarray]:
        """One-Euro smoothing of the detected spine points in torso-length units (advance=False: reuse, no update)."""
        if pts is None or self._spine_smoother is None or self._scale is None:
            return pts
        if not advance:
            return self._last_spine if getattr(self, "_last_spine", None) is not None else pts
        self._last_spine = self._spine_smoother.update(t, np.asarray(pts, float), self._scale)
        return self._last_spine

    def _spine_input(self, lm: np.ndarray, side: Optional[str], view: str, spine_points: Optional[np.ndarray],
                     mask: Optional[np.ndarray]):
        """The nine spine points to use this frame and where they came from. Detected points win; else, side-on with a
        mask, the back silhouette (G-32); else None (the geometric proxy is used downstream)."""
        if spine_points is not None:
            return spine_points, "spinepose"
        if mask is not None and view == "side" and side is not None:
            sp = extract_side_pose(lm, side, require=("ear", "shoulder", "hip"), facing=self._facing)
            if sp is not None:
                return estimate_spine_from_mask(mask, sp), "contour"
        return None, "spinepose"

    def _partial_spine(self, lm, framing, view: str, spine_points,
                       spine_src: str = "spinepose") -> Optional[SpineProxy]:
        """A drawable spine when the legs are not visible: detected points, else a proxy from ear/shoulder/hip."""
        if lm is None or framing is None:
            return None
        if spine_points is not None:
            from .spine_proxy import build_spine_from_points
            pts = np.asarray(spine_points, float)
            return build_spine_from_points(pts[:, :2], self._facing or 1, pts[:, 2], curvature_visible=view == "side",
                                           source=spine_src)
        if view == "side":
            sp = extract_side_pose(lm, framing.side, require=("ear", "shoulder", "hip"), facing=self._facing)
            return build_spine_proxy(sp) if sp is not None else None
        mid = extract_midline_pose(lm, self._facing or 1)
        return build_spine_proxy(mid, straight=True) if mid is not None else None

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
            "frames_back_monitored": self.frames_back_only,
            "frames_by_view": self.view_frames,
            "frames_by_spine_source": self.spine_frames,
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
