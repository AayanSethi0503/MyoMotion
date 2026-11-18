"""Spine estimate from the back silhouette: a light-weight, image-based alternative to SpinePose (G-32).

Why. The geometric proxy ([P1] heuristic: a curve through ear, shoulder and hip) has no image evidence, so it cannot
follow a real spine: OpenCap states that "keypoints at the hips and shoulders are insufficient for robustly determining
sagittal-plane hip, pelvis, and lumbar kinematics" [X2], and Fitness-AQA found 2-D-pose features worse than image
features for lumbar errors (F1 0.54 vs 0.61-0.63) [P5 Table 5]. A trained spine network (SpinePose [P1]) fixes it but is
a second network, with a non-commercial licence. MediaPipe's PoseLandmarker can output a person **segmentation
mask** from the model that is already running (also on phones), and in a side view the silhouette's back edge carries
the curvature of the back (surface topography: the principle behind flexicurve and rasterstereography measurements
of kyphosis and lordosis).

Method (NumPy only):
  1. Axis: hip joint -> shoulder joint. Normal pointing to the back (opposite to the facing direction).
  2. Stations along the axis; from each, march toward the back until the silhouette ends: the back surface.
  3. Robust quartic fit of the back-surface distance along the axis (clothing/hand outliers rejected, smoothed).
  4. Spine = back surface moved forward by a soft-tissue depth, between the S1 and C7 levels (the joints are NOT on
     the spine: the spine lies behind the hip and shoulder joint centres, which is why even a straight spine was drawn
     in the wrong place by the proxy). The nine labels are placed with the same vertebral-count fractions as the proxy.
  5. Cervical points continue smoothly from C7 to the ear (the head/neck silhouette is not used: hair, face).

Limits, stated plainly: side-on views only (from the front there is no sagittal back contour); loose clothing hides the
true back surface; the lowest ~0.2 of the trunk is the gluteal bulge, not lumbar spine, so lumbar curvature is the least
reliable region; the soft-tissue depth and the S1/C7 levels are provisional constants (G-32); and NOT validated against
markers or SpinePose. Outputs feed the same ``build_spine_from_points`` as SpinePose, tagged ``source="contour"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .landmarks import SidePose
from .spine_proxy import ARC_FRACTION, SPINE_LABELS

# Levels of S1 and C7 along the hip-joint -> shoulder-joint chord, as a fraction of its length (provisional, G-32).
# S1 sits a little above the hip joint; C7 a little above the shoulder joint.
T_SACRUM = 0.08
T_C7 = 1.06
# Soft tissue between the spinous processes and the skin, as a fraction of the torso length (~3 cm of a 50 cm torso).
SPINE_DEPTH_TORSO = 0.065
# How far past the axis to look for the back surface, in torso lengths.
REACH_TORSO = 0.65
# Stations: from just below the hip joint to just above the shoulder joint.
STATIONS = 28
STATION_RANGE = (-0.05, 1.12)
MIN_COVERAGE = 0.7            # fraction of stations that must find a back surface
GAP_TORSO = 0.04              # a hole in the silhouette smaller than this does not end the body
MASK_THRESHOLD = 0.5
# Polynomial degree of the back-surface profile. Chosen on a real 39 s recording (586 frames): degree 4 had the lowest
# frame-to-frame jitter (lumbar 2.29 deg vs 2.60 for cubic, 2.78 for degree 5) and the highest fit confidence (0.87 vs
# 0.82); the cubic's lumbar spike of +18 deg was an overshoot (+8 with degree 4).
FIT_DEGREE = 4
QUALITY_RESIDUAL = 0.05       # median fit residual (torso lengths) at which the confidence score reaches 0


@dataclass(frozen=True)
class BackContour:
    t: np.ndarray               # [K] station positions along the chord (fraction of its length)
    distance: np.ndarray        # [K] distance of the back surface behind the chord, torso lengths (NaN = not found)
    fit: np.ndarray             # [K] robust cubic fit of ``distance``
    coverage: float
    residual: float             # median abs residual of the fit, torso lengths (quality)
    origin: np.ndarray          # hip joint (px)
    axis: np.ndarray            # unit vector hip -> shoulder
    back: np.ndarray            # unit normal pointing to the back
    length_px: float


def _first_exit(mask: np.ndarray, start: np.ndarray, direction: np.ndarray, reach_px: float, gap_px: float,
                thr: float) -> Optional[float]:
    """Distance (px) from ``start`` along ``direction`` to where the silhouette ends; None if ``start`` is outside it.

    Holes shorter than ``gap_px`` are ignored. Marching is in 1-px steps."""
    h, w = mask.shape
    n = int(reach_px) + 1
    s = np.arange(n, dtype=float)
    xs = np.round(start[0] + direction[0] * s).astype(int)
    ys = np.round(start[1] + direction[1] * s).astype(int)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    inside = np.zeros(n, bool)
    inside[ok] = mask[ys[ok], xs[ok]] > thr
    if not inside[0]:
        return None
    last_in = 0
    gap = int(max(gap_px, 1))
    i = 0
    while i < n:
        if inside[i]:
            last_in = i
            i += 1
        else:                                            # a False run: does it end within the allowed gap?
            j = i
            while j < n and not inside[j] and j - i <= gap:
                j += 1
            if j < n and inside[j] and j - i <= gap:
                i = j
            else:
                break
    return float(last_in)


def extract_back_contour(mask: np.ndarray, hip: np.ndarray, shoulder: np.ndarray, facing: int,
                         stations: int = STATIONS) -> Optional[BackContour]:
    """Back-surface distance profile of the trunk from a person mask ``[H, W]`` (float 0-1 or bool)."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    chord = shoulder - hip
    length = float(np.hypot(*chord))
    if length < 20.0 or not facing:
        return None
    axis = chord / length
    back = np.array([-axis[1], axis[0]])
    if back[0] * facing > 0:                            # the back is opposite to the direction the person faces
        back = -back
    t = np.linspace(STATION_RANGE[0], STATION_RANGE[1], stations)
    dist = np.full(stations, np.nan)
    for i, ti in enumerate(t):
        d = _first_exit(mask, hip + axis * ti * length, back, REACH_TORSO * length, GAP_TORSO * length, MASK_THRESHOLD)
        if d is not None and d < REACH_TORSO * length - 2:     # reached the reach limit = the mask kept going: invalid
            dist[i] = d / length
    found = np.isfinite(dist)
    coverage = float(found.mean())
    if coverage < MIN_COVERAGE or found.sum() < 8:
        return None
    # robust cubic fit: two passes, dropping points further than 2.5 MAD from the first fit
    tt, dd = t[found], dist[found]
    coef = np.polyfit(tt, dd, FIT_DEGREE)
    res = dd - np.polyval(coef, tt)
    keep = np.abs(res) <= max(2.5 * 1.4826 * float(np.median(np.abs(res - np.median(res)))), 0.01)
    if keep.sum() >= 8:
        coef = np.polyfit(tt[keep], dd[keep], FIT_DEGREE)
        res = dd[keep] - np.polyval(coef, tt[keep])
    fit = np.polyval(coef, t)
    return BackContour(t=t, distance=dist, fit=fit, coverage=coverage, residual=float(np.median(np.abs(res))),
                       origin=hip.astype(float), axis=axis, back=back, length_px=length)


def spine_from_contour(contour: BackContour, ear: Optional[np.ndarray], facing: int) -> Optional[np.ndarray]:
    """Nine spine points ``[9, 3]`` (x_px, y_px, score) in ``SPINE_LABELS`` order, from a back contour and the ear.

    The score is the fit quality x coverage, in [0, 1]. The cervical points (C4, C1) are not measured: they continue the
    spine's direction at C7 and end at the ear (C1 is placed at the ear landmark)."""
    c = contour
    L = c.length_px

    def spine_at(ti: float) -> np.ndarray:
        d = float(np.interp(ti, c.t, c.fit)) - SPINE_DEPTH_TORSO     # back surface moved forward by the tissue depth
        return c.origin + c.axis * ti * L + c.back * d * L

    pts = {}
    for label in ("C7", "T3", "T8", "L1", "L3", "L5", "SACRUM"):
        g = (1.0 - ARC_FRACTION[label]) / (1.0 - ARC_FRACTION["C7"])        # 0 at the sacrum ... 1 at C7
        pts[label] = spine_at(T_SACRUM + g * (T_C7 - T_SACRUM))
    # cervical: smooth continuation of the spine direction at C7 up to the ear (quadratic Bezier, G1 at C7)
    c7, below = pts["C7"], spine_at(T_C7 - 0.12)
    up = c7 - below
    up = up / max(float(np.hypot(*up)), 1e-9)
    if ear is None:
        return None
    c1 = np.asarray(ear, float)
    ctrl = c7 + up * 0.5 * float(np.hypot(*(c1 - c7)))
    pts["C1"] = c1
    pts["C4"] = 0.25 * c7 + 0.5 * ctrl + 0.25 * c1
    quality = max(0.0, 1.0 - c.residual / QUALITY_RESIDUAL)                  # misfit of QUALITY_RESIDUAL = score 0
    score = float(min(1.0, c.coverage) * quality)
    out = np.array([[*pts[label], score] for label in SPINE_LABELS])
    return out if np.all(np.isfinite(out)) else None


def estimate_spine_from_mask(mask: np.ndarray, pose: SidePose) -> Optional[np.ndarray]:
    """Convenience: nine spine points ``[9, 3]`` from a person mask and the near-side pose, or None."""
    if not (pose.has("hip") and pose.has("shoulder") and pose.has("ear")):
        return None
    contour = extract_back_contour(mask, pose["hip"], pose["shoulder"], pose.facing)
    if contour is None:
        return None
    return spine_from_contour(contour, pose["ear"], pose.facing)


def synthetic_mask(shape, hip: np.ndarray, shoulder: np.ndarray, facing: int, back_profile, torso_depth: float = 0.30,
                   t_range=(-0.25, 1.15)) -> np.ndarray:
    """A test silhouette: a trunk whose back edge sits ``back_profile(t)`` torso lengths behind the chord and whose
    front edge is ``torso_depth`` in front of it. Used by the tests and the synthetic demo."""
    h, w = shape
    chord = shoulder - hip
    length = float(np.hypot(*chord))
    axis = chord / length
    back = np.array([-axis[1], axis[0]])
    if back[0] * facing > 0:
        back = -back
    yy, xx = np.mgrid[0:h, 0:w]
    rel = np.stack([xx - hip[0], yy - hip[1]], axis=-1).astype(float)
    t = (rel @ axis) / length
    d = (rel @ back) / length
    grid = np.linspace(t_range[0], t_range[1], 240)
    prof = np.interp(t, grid, np.array([back_profile(float(g)) for g in grid]))      # vectorised lookup table
    return ((t >= t_range[0]) & (t <= t_range[1]) & (d <= prof) & (d >= -torso_depth)).astype(np.float32)


def contour_available(mask: Optional[np.ndarray]) -> bool:
    return mask is not None and mask.size > 0 and math.isfinite(float(np.nanmax(mask)))
