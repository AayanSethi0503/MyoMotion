"""MediaPipe Pose landmark indices, side-view helpers and a framing check.

``biomechanics`` never imports mediapipe or cv2. Callers (``vision/`` or a CLI script) convert a
PoseLandmarker result to ``np.ndarray [33, 3]`` = (x_px, y_px, visibility). That array is the
``body2d`` field of the Skeleton contract (ARCHITECTURE_AND_PIPELINES.md ARCH-6).

Which joints are needed (docs/DEADLIFT_TRACKING_REQUIREMENTS.md):
  CORE      shoulder, hip, knee, ankle      -> every angle and the rep count. Missing any = no tracking.
  OPTIONAL  ear                             -> proxy spine (without it: no spine cues, reps still count)
            heel, foot-index                -> facing direction, mid-foot reference for bar drift
            wrist                           -> bar-height / bar-drift proxy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# MediaPipe Pose topology (33 landmarks, BlazePose). Source: MediaPipe PoseLandmarker docs.
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32
NUM_LANDMARKS = 33

# A landmark is used only if MediaPipe is at least this confident it is visible. Below this it
# is extrapolating (existing tracker value).
MIN_LANDMARK_VISIBILITY = 0.5

SIDE_INDICES: Dict[str, Dict[str, int]] = {
    "left": dict(ear=L_EAR, shoulder=L_SHOULDER, elbow=L_ELBOW, wrist=L_WRIST, hip=L_HIP,
                 knee=L_KNEE, ankle=L_ANKLE, heel=L_HEEL, foot=L_FOOT_INDEX),
    "right": dict(ear=R_EAR, shoulder=R_SHOULDER, elbow=R_ELBOW, wrist=R_WRIST, hip=R_HIP,
                  knee=R_KNEE, ankle=R_ANKLE, heel=R_HEEL, foot=R_FOOT_INDEX),
}

# Landmarks the sagittal deadlift analysis cannot work without. Everything else degrades gracefully.
CORE_SAGITTAL: Tuple[str, ...] = ("shoulder", "hip", "knee", "ankle")
OPTIONAL_SAGITTAL: Tuple[str, ...] = ("ear", "heel", "foot", "wrist")
REQUIRED_SAGITTAL = CORE_SAGITTAL                       # kept for backwards compatibility
TRACKED_SAGITTAL: Tuple[str, ...] = ("ear", "shoulder", "hip", "knee", "ankle", "heel", "foot", "wrist")

# Heading of the person relative to the camera, from how far apart the two shoulders look in the image (as a fraction
# of the torso length): ~0 side-on, ~0.8 facing the camera. Hysteresis stops the view flickering at the boundary.
# Provisional (G-29/G-30).
ANGLED_ENTER_SPAN_RATIO = 0.30       # side -> angled, ~25 deg off side-on (use the 3-D world angles)
ANGLED_EXIT_SPAN_RATIO = 0.22        # angled -> side, ~18 deg (back to the 2-D image angles)
FRONT_SPAN_RATIO = 0.70              # reported as "front" (or back) rather than "angled"
SIDE_SWITCH_MARGIN = 0.15            # the other body side must be this much more visible to take over
SIDE_SWITCH_HOLD_S = 0.4             # ...for this long (a turn, not a flicker)
# Body (shoulder -> ankle) shorter than this fraction of the image height = too far away / too small.
MIN_BODY_HEIGHT_FRACTION = 0.30

_CORE_ADVICE = {
    "shoulder": "Shoulder not visible: keep your upper body in frame",
    "hip": "Hip not visible: stand side-on with your whole body in frame",
    "knee": "Knee not visible: step back so your legs are in frame",
    "ankle": "Ankle not visible: step back so your feet are in frame (camera ~3 m away)",
}
_OPTIONAL_NOTE = {
    "ear": "Head/ear not visible: back-rounding (spine proxy) check is off",
    "heel": "Feet not clearly visible: bar-path check is off",
    "foot": "Feet not clearly visible: bar-path check is off",
    "wrist": "Wrist not visible: bar-height tracking is off",
}


def landmark_reliable(row: np.ndarray, w: Optional[int] = None, h: Optional[int] = None,
                      min_visibility: float = MIN_LANDMARK_VISIBILITY) -> bool:
    """True if a (x, y, visibility) row is finite, visible enough and (if w/h given) in-frame."""
    if not np.all(np.isfinite(row)) or row[2] < min_visibility:
        return False
    if w is not None and h is not None:
        return 0.0 <= row[0] <= w and 0.0 <= row[1] <= h
    return True


def choose_side(lm: np.ndarray) -> str:
    """The body side facing the camera = the side with the higher mean visibility over the tracked joints."""
    scores = {}
    for side, idx in SIDE_INDICES.items():
        scores[side] = float(np.mean([lm[idx[k], 2] for k in TRACKED_SAGITTAL]))
    return "right" if scores["right"] >= scores["left"] else "left"


def extract_midline_pose(lm: np.ndarray, facing: int = 1,
                         min_visibility: float = MIN_LANDMARK_VISIBILITY) -> Optional[SidePose]:
    """Body-midline anchors (ear, shoulder, hip) as the mean of the left and right joints that are visible.

    Used when the person is not side-on: a single side's ear/shoulder/hip sit off the body's centre line in the image,
    which draws a bent line. Needs a shoulder and a hip; the ear is optional. ``side`` is reported as "mid".
    """
    def mid(a: int, b: int) -> Optional[np.ndarray]:
        pts = [lm[i, :2] for i in (a, b) if landmark_reliable(lm[i], min_visibility=min_visibility)]
        return np.mean(pts, axis=0).astype(float) if pts else None

    sh, hip, ear = mid(L_SHOULDER, R_SHOULDER), mid(L_HIP, R_HIP), mid(L_EAR, R_EAR)
    if sh is None or hip is None:
        return None
    pts = {"shoulder": sh, "hip": hip}
    vis = {"shoulder": 1.0, "hip": 1.0}
    if ear is not None:
        pts["ear"], vis["ear"] = ear, 1.0
    return SidePose(side="mid", facing=facing if facing else 1, pts=pts, vis=vis)


def side_scores(lm: np.ndarray) -> Dict[str, float]:
    """Mean visibility of the tracked joints of each body side (which side the camera sees better)."""
    return {side: float(np.mean([lm[idx[k], 2] for k in TRACKED_SAGITTAL])) for side, idx in SIDE_INDICES.items()}


def shoulder_span_ratio(lm: np.ndarray, min_visibility: float = MIN_LANDMARK_VISIBILITY) -> Optional[float]:
    """Horizontal distance between the two shoulders / torso length, or None if both are not visible.

    ~0 when side-on (they overlap), ~0.8 when facing (or turned away from) the camera. Torso length is
    shoulder-midpoint to hip-midpoint, using whichever hips are visible."""
    ls, rs = lm[L_SHOULDER], lm[R_SHOULDER]
    if not all(landmark_reliable(p, min_visibility=min_visibility) for p in (ls, rs)):
        return None
    hips = [lm[i, :2] for i in (L_HIP, R_HIP) if landmark_reliable(lm[i], min_visibility=min_visibility)]
    if not hips:
        return None
    torso = float(np.hypot(*(0.5 * (ls[:2] + rs[:2]) - np.mean(hips, axis=0))))
    return float(abs(ls[0] - rs[0]) / torso) if torso > 1e-6 else None


def view_name(span: Optional[float]) -> str:
    """"side" | "angled" | "front" from the shoulder span ratio (no hysteresis; for display)."""
    if span is None or span < ANGLED_ENTER_SPAN_RATIO:
        return "side"
    return "front" if span >= FRONT_SPAN_RATIO else "angled"


def estimate_facing(lm: np.ndarray, side: str, min_visibility: float = MIN_LANDMARK_VISIBILITY) -> Tuple[int, int]:
    """Facing direction from whatever cues are visible: ``(direction, weight)``, direction +1 = image-right.

    Strongest cue first: toe ahead of heel (weight 3), nose ahead of ear (2), knee ahead of ankle (1). Weight 0
    means no usable cue. The weight lets the caller vote over several frames (see ``DeadliftAnalyzer``).
    """
    idx = SIDE_INDICES[side]

    def ok(*keys: str) -> bool:
        return all(landmark_reliable(lm[idx[k]], min_visibility=min_visibility) for k in keys)

    if ok("heel", "foot") and abs(lm[idx["foot"], 0] - lm[idx["heel"], 0]) > 1e-6:
        return (1 if lm[idx["foot"], 0] > lm[idx["heel"], 0] else -1), 3
    nose_ear = lm[NOSE, 0] - lm[idx["ear"], 0]
    if ok("ear") and landmark_reliable(lm[NOSE], min_visibility=min_visibility) and abs(nose_ear) > 1e-6:
        return (1 if lm[NOSE, 0] > lm[idx["ear"], 0] else -1), 2
    if ok("knee", "ankle") and abs(lm[idx["knee"], 0] - lm[idx["ankle"], 0]) > 1e-6:
        return (1 if lm[idx["knee"], 0] > lm[idx["ankle"], 0] else -1), 1
    return 0, 0


def facing_direction(lm: np.ndarray, side: str) -> int:
    """+1 if the person faces image-right, -1 if image-left (best available cue; +1 if there is none)."""
    direction, _ = estimate_facing(lm, side)
    return direction or 1


@dataclass(frozen=True)
class SidePose:
    """Near-side landmarks of one frame in pixels, plus facing direction. Optional joints may be absent."""
    side: str
    facing: int
    pts: Dict[str, np.ndarray]          # name -> array([x, y]); only reliable joints are present
    vis: Dict[str, float]               # name -> visibility

    def __getitem__(self, name: str) -> np.ndarray:
        return self.pts[name]

    def has(self, name: str) -> bool:
        return name in self.pts


def extract_side_pose(lm: np.ndarray, side: Optional[str] = None,
                      min_visibility: float = MIN_LANDMARK_VISIBILITY,
                      require: Tuple[str, ...] = CORE_SAGITTAL,
                      facing: Optional[int] = None) -> Optional[SidePose]:
    """Build a :class:`SidePose` from ``lm [33, 3]``.

    Returns None unless every joint in ``require`` is reliable. Other tracked joints are included when reliable and
    simply absent otherwise. ``facing`` overrides the per-frame estimate (the analyzer locks it over time).
    """
    if lm.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected landmarks of shape (33, 3), got {lm.shape}")
    side = side or choose_side(lm)
    idx = SIDE_INDICES[side]
    for key in require:
        if not landmark_reliable(lm[idx[key]], min_visibility=min_visibility):
            return None
    pts, vis = {}, {}
    for key, i in idx.items():
        if key in require or landmark_reliable(lm[i], min_visibility=min_visibility):
            pts[key] = lm[i, :2].astype(float).copy()
            vis[key] = float(lm[i, 2])
    return SidePose(side=side, facing=facing if facing else facing_direction(lm, side), pts=pts, vis=vis)


@dataclass(frozen=True)
class FramingReport:
    """Is the person framed well enough to track? Used for live on-screen guidance."""
    side: str                           # the body side the camera sees better
    visibility: Dict[str, float]        # tracked joint -> visibility of that side's landmark
    visible: Dict[str, bool]
    ok: bool                            # all core joints visible
    problems: Tuple[str, ...]           # reasons tracking is blocked
    notes: Tuple[str, ...]              # things switched off or less accurate (not blocking)
    shoulder_span_ratio: Optional[float]
    view: str = "side"                  # "side" | "angled" | "front" (heading relative to the camera)


_VIEW_NOTE = {
    "angled": "Angled view: angles use the 3-D estimate (less exact). Turn side-on for exact angles and spine feedback",
    "front": "Facing the camera: angles use the 3-D estimate (less exact) and spine curvature cannot be seen. "
             "Turn side-on for spine feedback",
}


def framing_report(lm: Optional[np.ndarray], side: Optional[str] = None,
                   min_visibility: float = MIN_LANDMARK_VISIBILITY,
                   image_wh: Optional[Tuple[int, int]] = None) -> Optional[FramingReport]:
    """Explain what is wrong with the framing. None if no person was detected at all."""
    if lm is None:
        return None
    side = side or choose_side(lm)
    idx = SIDE_INDICES[side]
    vis = {k: float(lm[idx[k], 2]) for k in TRACKED_SAGITTAL}
    visible = {k: landmark_reliable(lm[idx[k]], min_visibility=min_visibility) for k in TRACKED_SAGITTAL}

    problems: List[str] = [_CORE_ADVICE[k] for k in CORE_SAGITTAL if not visible[k]]
    notes: List[str] = []
    for k in OPTIONAL_SAGITTAL:
        msg = _OPTIONAL_NOTE[k]
        if not visible[k] and msg not in notes:
            notes.append(msg)

    span = shoulder_span_ratio(lm, min_visibility)
    view = view_name(span)
    if view in _VIEW_NOTE:
        notes.append(_VIEW_NOTE[view])
    if image_wh is not None and visible["shoulder"] and visible["ankle"]:
        body = abs(float(lm[idx["ankle"], 1] - lm[idx["shoulder"], 1])) / image_wh[1]
        if body < MIN_BODY_HEIGHT_FRACTION:
            problems.append(f"You look small in the frame ({body:.0%} of its height): move closer or zoom in")

    return FramingReport(side=side, visibility=vis, visible=visible, ok=not problems, problems=tuple(problems),
                         notes=tuple(notes), shoulder_span_ratio=span, view=view)
