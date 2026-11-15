"""Deadlift kinematics engine: sagittal + frontal joint/segment angles, regional spine proxy, bar path.

Pure functions on ``np.ndarray [33, 3]`` landmarks (x_px, y_px, visibility); no I/O, no cv2/mediapipe.

Sagittal (side view; the near side is chosen automatically):
    hip hinge angle            shoulder-hip-knee interior angle                       [X1 11 angles]
    knee flexion               180 deg - interior knee angle (hip-knee-ankle)         [P8 depth classes]
    trunk inclination          hip->shoulder vs vertical, forward positive            [P8]
    tibia inclination          ankle->knee   vs vertical, forward positive            [P8]
    trunk-tibia differential   trunk - tibia                                [P8], extended to the deadlift (G-25)
    regional spine proxy       cervical / thoracic / lumbar / pelvic                  [P1] heuristic (G-24)
    bar vertical trajectory    YOLO bar ROI centre if given, else the wrist           [P5], [X1]

All angles in degrees, all lengths in pixels unless the name says ``_torso`` or ``_foot`` (normalised).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import distance, inclination_from_vertical, joint_angle, wrap_deg
from .landmarks import (L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, SidePose,
                        extract_side_pose, landmark_reliable)
from .spine_proxy import SpineProxy, build_spine_proxy


@dataclass(frozen=True)
class DeadliftAngles:
    """Everything the rep segmenter, rules and overlay need for one frame (sagittal)."""
    t_s: float
    side: str
    facing: int                              # +1 faces image-right, -1 image-left
    hip_deg: float                           # interior shoulder-hip-knee angle (0-180; 180 = fully open)
    hip_signed_deg: float                    # same, but > 180 when the trunk is behind the hip-knee line
    knee_deg: float                          # interior hip-knee-ankle angle (180 = straight)
    knee_flexion_deg: float                  # 180 - knee_deg
    trunk_incl_deg: float                    # forward-lean positive, 0 = upright, <0 = leaning back
    tibia_incl_deg: float                    # knee ahead of ankle = positive
    trunk_tibia_deg: float                   # trunk - tibia  [P8]
    torso_len_px: float                      # hip -> shoulder length (normalisation scale)
    foot_len_px: float                       # heel -> foot-index length
    spine: Optional[SpineProxy]              # proxy spine (None if anchors degenerate)
    bar_xy: Optional[np.ndarray]             # bar / wrist position in px
    bar_source: Optional[str]                # "bar" (detector) | "wrist" | None
    bar_height_torso: Optional[float]        # height above the floor, torso lengths, up = +
    bar_drift_foot: Optional[float]          # horizontal offset from mid-foot in foot lengths, forward = +
    pose: SidePose                           # the near-side landmarks used (for drawing)


def compute_deadlift_angles(lm: np.ndarray, t_s: float, side: Optional[str] = None,
                            bar_xy: Optional[np.ndarray] = None) -> Optional[DeadliftAngles]:
    """Angles for one frame, or None if the near-side landmarks are unreliable / degenerate.

    ``bar_xy``: optional (x, y) px centre of a detected barbell (YOLO ROI, [X1]/[P5]). If absent the
    wrist of the near side is used as the bar proxy (arms hang straight in a deadlift).
    """
    pose = extract_side_pose(lm, side)
    if pose is None:
        return None
    sh, hip, knee, ankle = pose["shoulder"], pose["hip"], pose["knee"], pose["ankle"]
    heel, toe = pose["heel"], pose["foot"]

    hip_deg = joint_angle(sh, hip, knee)
    knee_deg = joint_angle(hip, knee, ankle)
    trunk = inclination_from_vertical(hip, sh, pose.facing)
    tibia = inclination_from_vertical(ankle, knee, pose.facing)
    torso = distance(hip, sh)
    foot = distance(heel, toe)
    if not all(math.isfinite(v) for v in (hip_deg, knee_deg, trunk, tibia)) or torso < 1e-6:
        return None

    # Signed hip angle = (inclination of hip->knee) - (trunk inclination): 85 at a deadlift bottom, 180 straight,
    # >180 when the trunk leans behind the hip-knee line. The interior angle cannot tell lean-back from flexion
    # (both read < 180), which would make the segmenter see a false descent at a hyperextended lockout.
    hip_signed = 180.0 + wrap_deg(inclination_from_vertical(hip, knee, pose.facing) - trunk - 180.0)

    spine = build_spine_proxy(pose)

    bar, source = None, None
    if bar_xy is not None and np.all(np.isfinite(bar_xy)):
        bar, source = np.asarray(bar_xy, float), "bar"
    elif pose.vis.get("wrist", 0.0) >= 0.5 and np.all(np.isfinite(pose["wrist"])):
        bar, source = pose["wrist"], "wrist"

    bar_height = bar_drift = None
    if bar is not None:
        floor_y = max(float(heel[1]), float(toe[1]))
        bar_height = (floor_y - float(bar[1])) / torso
        mid_foot_x = 0.5 * (float(heel[0]) + float(toe[0]))
        ref = foot if foot > 1e-6 else torso * 0.4          # fall back to a typical foot/torso ratio
        bar_drift = (float(bar[0]) - mid_foot_x) * pose.facing / ref

    return DeadliftAngles(
        t_s=t_s, side=pose.side, facing=pose.facing, hip_deg=hip_deg, hip_signed_deg=hip_signed,
        knee_deg=knee_deg,
        knee_flexion_deg=180.0 - knee_deg, trunk_incl_deg=trunk, tibia_incl_deg=tibia,
        trunk_tibia_deg=trunk - tibia, torso_len_px=torso, foot_len_px=foot, spine=spine,
        bar_xy=None if bar is None else np.asarray(bar, float), bar_source=source,
        bar_height_torso=bar_height, bar_drift_foot=bar_drift, pose=pose)


# ---------------------------------------------------------------------------
# Frontal plane (front-view footage only)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrontalAngles:
    shoulder_tilt_deg: float        # tilt of the L->R shoulder line from horizontal; + = image-right end lower
    hip_tilt_deg: float             # same for the hip line (pelvic obliquity proxy)
    knee_valgus_left_deg: float     # + = knee displaced toward the midline (valgus), - = varus
    knee_valgus_right_deg: float


def line_tilt_deg(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """Tilt of the line a->b from horizontal in (-90, 90]; positive when the image-right end is lower."""
    ang = math.degrees(math.atan2(p_b[1] - p_a[1], p_b[0] - p_a[0]))
    if ang > 90.0:
        ang -= 180.0
    elif ang <= -90.0:
        ang += 180.0
    return ang


def knee_valgus_deg(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, other_hip: np.ndarray) -> float:
    """Frontal-plane knee deviation: 180 - interior hip-knee-ankle angle, signed.

    Positive (valgus) when the knee sits on the midline side of the hip-ankle line, where the midline
    side is the side of the other hip. [P5] knees-inward error, [X1] "knees caved in".
    """
    interior = joint_angle(hip, knee, ankle)
    if not math.isfinite(interior):
        return float("nan")
    chord = ankle - hip
    cross_knee = chord[0] * (knee[1] - hip[1]) - chord[1] * (knee[0] - hip[0])
    cross_mid = chord[0] * (other_hip[1] - hip[1]) - chord[1] * (other_hip[0] - hip[0])
    sign = 1.0 if cross_knee * cross_mid > 0 else -1.0
    return sign * (180.0 - interior)


def compute_frontal_angles(lm: np.ndarray) -> Optional[FrontalAngles]:
    """Frontal-plane tilt / valgus angles. Only meaningful with a front-facing camera."""
    need = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)
    if not all(landmark_reliable(lm[i]) for i in need):
        return None
    p = lambda i: lm[i, :2].astype(float)       # noqa: E731
    return FrontalAngles(
        shoulder_tilt_deg=line_tilt_deg(p(L_SHOULDER), p(R_SHOULDER)),
        hip_tilt_deg=line_tilt_deg(p(L_HIP), p(R_HIP)),
        knee_valgus_left_deg=knee_valgus_deg(p(L_HIP), p(L_KNEE), p(L_ANKLE), p(R_HIP)),
        knee_valgus_right_deg=knee_valgus_deg(p(R_HIP), p(R_KNEE), p(R_ANKLE), p(L_HIP)))
