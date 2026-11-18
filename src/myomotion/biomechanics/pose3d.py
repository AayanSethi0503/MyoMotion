"""View-independent joint angles from MediaPipe's 3-D *world* landmarks (G-30).

A 2-D image of a person who is not exactly side-on (or who faces the camera) foreshortens the hip hinge, so image
angles are wrong. MediaPipe also returns ``pose_world_landmarks``: metric 3-D coordinates, origin at the hip centre,
x right, **y down**, z away from the camera. Angles measured in 3-D do not depend on where the camera is.

Checked on real footage of this project's lifter (frames from a webcam run): world y is down (shoulder above hip gives
a negative y difference), and the 3-D trunk inclination agreed with the 2-D side-view value within ~3-4 deg.
Assumes a level camera (world "up" is -y). The depth component is the model's estimate, so accuracy is lower than a
clean 2-D side view; the pipeline therefore uses this path only when the person is not side-on.

The angles use the **midline** of the body (mean of left and right joints that are visible), so the result does not
change when the lifter turns from one side to the other or faces the camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import wrap_deg
from .landmarks import (L_ANKLE, L_EAR, L_FOOT_INDEX, L_HEEL, L_HIP, L_KNEE, L_SHOULDER, NOSE, R_ANKLE, R_EAR,
                        R_FOOT_INDEX, R_HEEL, R_HIP, R_KNEE, R_SHOULDER)

UP = np.array([0.0, -1.0, 0.0])                          # MediaPipe world y points down


@dataclass(frozen=True)
class Angles3D:
    hip_deg: float                  # interior shoulder-hip-knee angle in 3-D (0-180)
    hip_signed_deg: float           # 85 at a deadlift bottom, 180 straight, > 180 hyperextended (same as 2-D, G-27)
    knee_deg: float                 # interior hip-knee-ankle angle in 3-D
    trunk_incl_deg: float           # forward-positive inclination of hip->shoulder from vertical, in the sagittal plane
    tibia_incl_deg: float           # same for ankle->knee
    forward: np.ndarray             # unit vector the person faces, in the horizontal plane (world coordinates)
    torso_len_m: float


def _mid(world: np.ndarray, vis: np.ndarray, a: int, b: int, min_vis: float) -> Optional[np.ndarray]:
    pts = [world[i] for i in (a, b) if vis[i] >= min_vis and np.all(np.isfinite(world[i]))]
    return np.mean(pts, axis=0) if pts else None


def _horizontal(v: np.ndarray) -> np.ndarray:
    h = v.copy()
    h[1] = 0.0
    return h


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at vertex b in 3-D, degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / (n1 * n2)))))


def estimate_forward(world: np.ndarray, vis: np.ndarray, min_vis: float = 0.5) -> Optional[np.ndarray]:
    """Unit horizontal vector the person faces.

    Direction line: perpendicular to the hip (else shoulder) axis, which is stable even with toe-out. Its sign is
    decided by a weighted vote of what is ahead of what: toe vs heel (3), nose vs ears (2), knee vs ankle (1).
    If neither hips nor shoulders are both visible, the toe-heel vector is used directly.
    """
    lateral = None
    for a, b in ((L_HIP, R_HIP), (L_SHOULDER, R_SHOULDER)):
        if vis[a] >= min_vis and vis[b] >= min_vis:
            lateral = _horizontal(world[b] - world[a])
            if np.linalg.norm(lateral) > 1e-6:
                break
            lateral = None

    cues = []                                            # (weight, horizontal vector pointing "ahead")
    ft = [(world[t] - world[h]) for h, t in ((L_HEEL, L_FOOT_INDEX), (R_HEEL, R_FOOT_INDEX))
          if vis[h] >= min_vis and vis[t] >= min_vis]
    if ft:
        cues.append((3.0, _horizontal(np.mean(ft, axis=0))))
    ears = [world[i] for i in (L_EAR, R_EAR) if vis[i] >= min_vis]
    if ears and vis[NOSE] >= min_vis:
        cues.append((2.0, _horizontal(world[NOSE] - np.mean(ears, axis=0))))
    kn = [(world[k] - world[a]) for k, a in ((L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE))
          if vis[k] >= min_vis and vis[a] >= min_vis]
    if kn:
        cues.append((1.0, _horizontal(np.mean(kn, axis=0))))

    if lateral is not None:
        fw = np.cross(UP, lateral / np.linalg.norm(lateral))
        fw = _horizontal(fw)
        fw /= max(float(np.linalg.norm(fw)), 1e-9)
        score = sum(w * float(np.dot(fw, c)) for w, c in cues)
        if cues and abs(score) > 1e-9:
            return fw if score > 0 else -fw
        return None
    if ft:                                               # no usable hip/shoulder axis: trust the feet
        v = cues[0][1]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else None
    return None


def heading_span_equivalent(world: np.ndarray, shoulder_width_over_torso: float = 0.72) -> Optional[float]:
    """How far from side-on the person is, on the scale of the 2-D shoulder-span ratio (0 = side-on, ~0.72 = front).

    From the hip axis (else the shoulder axis) in world coordinates: the fraction of that axis lying along the camera's
    x direction is sin(heading deviation), so a view-independent measure that does not need the far shoulder to be
    visible in the image (it often isn't at 30-60 degrees off side-on). 0.72 is shoulder width / torso length.
    """
    for a, b in ((L_HIP, R_HIP), (L_SHOULDER, R_SHOULDER)):
        lat = world[b] - world[a]
        n = math.hypot(lat[0], lat[2])
        if np.all(np.isfinite(lat)) and n > 1e-6:
            return shoulder_width_over_torso * abs(float(lat[0])) / n
    return None


def compute_angles_3d(world: np.ndarray, vis: np.ndarray, min_vis: float = 0.5) -> Optional[Angles3D]:
    """Midline hip / knee / trunk / tibia angles from world landmarks ``[33, 3]`` and visibility ``[33]``."""
    sh = _mid(world, vis, L_SHOULDER, R_SHOULDER, min_vis)
    hip = _mid(world, vis, L_HIP, R_HIP, min_vis)
    knee = _mid(world, vis, L_KNEE, R_KNEE, min_vis)
    ankle = _mid(world, vis, L_ANKLE, R_ANKLE, min_vis)
    if any(p is None for p in (sh, hip, knee, ankle)):
        return None
    fw = estimate_forward(world, vis, min_vis)
    if fw is None:
        return None

    def incl(v: np.ndarray) -> float:                    # forward-positive angle from vertical-up in the sagittal plane
        return math.degrees(math.atan2(float(np.dot(v, fw)), float(np.dot(v, UP))))

    trunk = incl(sh - hip)
    tibia = incl(knee - ankle)
    psi = incl(knee - hip)                               # direction hip -> knee
    hip_signed = 180.0 + wrap_deg(psi - trunk - 180.0)
    hip_int = _angle3(sh, hip, knee)
    knee_int = _angle3(hip, knee, ankle)
    torso = float(np.linalg.norm(sh - hip))
    if not all(math.isfinite(v) for v in (trunk, tibia, hip_signed, hip_int, knee_int)) or torso < 1e-6:
        return None
    return Angles3D(hip_deg=hip_int, hip_signed_deg=hip_signed, knee_deg=knee_int, trunk_incl_deg=trunk,
                    tibia_incl_deg=tibia, forward=fw, torso_len_m=torso)
