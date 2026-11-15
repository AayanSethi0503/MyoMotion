"""MediaPipe Pose landmark indices and side-view helpers.

``biomechanics`` never imports mediapipe or cv2. Callers (``vision/`` or a CLI script) convert a
PoseLandmarker result to ``np.ndarray [33, 3]`` = (x_px, y_px, visibility). That array is the
``body2d`` field of the Skeleton contract (ARCHITECTURE_AND_PIPELINES.md ARCH-6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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

# Landmarks the sagittal deadlift analysis cannot work without.
REQUIRED_SAGITTAL = ("ear", "shoulder", "hip", "knee", "ankle", "heel", "foot")


def landmark_reliable(row: np.ndarray, w: Optional[int] = None, h: Optional[int] = None,
                      min_visibility: float = MIN_LANDMARK_VISIBILITY) -> bool:
    """True if a (x, y, visibility) row is finite, visible enough and (if w/h given) in-frame."""
    if not np.all(np.isfinite(row)) or row[2] < min_visibility:
        return False
    if w is not None and h is not None:
        return 0.0 <= row[0] <= w and 0.0 <= row[1] <= h
    return True


def choose_side(lm: np.ndarray) -> str:
    """The body side facing the camera = the side with the higher mean visibility."""
    scores = {}
    for side, idx in SIDE_INDICES.items():
        scores[side] = float(np.mean([lm[idx[k], 2] for k in REQUIRED_SAGITTAL]))
    return "right" if scores["right"] >= scores["left"] else "left"


def facing_direction(lm: np.ndarray, side: str) -> int:
    """+1 if the person faces image-right, -1 if image-left (from heel -> foot-index vector).

    Falls back to ankle -> knee x-offset, then to +1."""
    idx = SIDE_INDICES[side]
    dx = lm[idx["foot"], 0] - lm[idx["heel"], 0]
    if abs(dx) > 1e-6:
        return 1 if dx > 0 else -1
    dx = lm[idx["knee"], 0] - lm[idx["ankle"], 0]
    return 1 if dx >= 0 else -1


@dataclass(frozen=True)
class SidePose:
    """Near-side landmarks of one frame in pixels, plus facing direction."""
    side: str
    facing: int
    pts: Dict[str, np.ndarray]          # name -> array([x, y])
    vis: Dict[str, float]               # name -> visibility

    def __getitem__(self, name: str) -> np.ndarray:
        return self.pts[name]


def extract_side_pose(lm: np.ndarray, side: Optional[str] = None,
                      min_visibility: float = MIN_LANDMARK_VISIBILITY) -> Optional[SidePose]:
    """Build a :class:`SidePose` from ``lm [33, 3]``; None if any required landmark is unreliable."""
    if lm.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected landmarks of shape (33, 3), got {lm.shape}")
    side = side or choose_side(lm)
    idx = SIDE_INDICES[side]
    for key in REQUIRED_SAGITTAL:
        if not landmark_reliable(lm[idx[key]], min_visibility=min_visibility):
            return None
    pts = {k: lm[i, :2].astype(float).copy() for k, i in idx.items()}
    vis = {k: float(lm[i, 2]) for k, i in idx.items()}
    return SidePose(side=side, facing=facing_direction(lm, side), pts=pts, vis=vis)
