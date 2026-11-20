"""Regional spine proxy from MediaPipe ear / shoulder / hip landmarks.

Deterministic geometric baseline for the SpinePose head [P1] (gap entry G-24).

Method [P1 §3.1.2]: "use shoulder and hip joints as drivers, fit two smooth curves around the
spinal region, sample initial guesses at predefined intervals assuming constant inter-vertebral
distance". Concretely here:

    anchors   ear -> C1,   shoulder -> C7,   hip -> sacral root          (near side, side view)
    curve 1   thoracolumbar: sacral root -> C7   (cubic Hermite, near-circular arc)
    curve 2   cervical:      C7 -> C1            (cubic Hermite, near-circular arc, G1 at C7)
    samples   C1, C4, C7, T3, T8, L1, L3, L5, SACRUM at fractions derived from vertebral counts

HONEST LIMITATION. With three anchors the curve's bend is a function of exactly one observable:
the ear-shoulder-hip turning angle ``beta``. The proxy therefore tracks head / upper-back drop
relative to the trunk line. It cannot see true lumbar curvature, and it cannot tell a rounded set-up
from a neutral one (all flexion measures are *relative to the athlete's own set-up baseline*).
[X2] measured 12-39 deg errors for keypoint-only trunk/pelvis kinematics. Everything produced here
is tagged ``source="proxy"`` and is to be replaced by SpinePose output (G-01).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .geometry import inclination_from_vertical, unit, wrap_deg
from .landmarks import SidePose

# Label order matches the Skeleton contract (ARCH-6) and [P1 Fig. 2]: C1, C4, C7 | T3, T8 | L1, L3, L5 | sacrum.
SPINE_LABELS: Tuple[str, ...] = ("C1", "C4", "C7", "T3", "T8", "L1", "L3", "L5", "SACRUM")

# Vertebral position counted from C1 = 1 (C1..C7 = 1..7, T1..T12 = 8..19, L1..L5 = 20..24, sacrum = 25).
# Constant inter-vertebral spacing is the assumption stated in [P1 §3.1.2]; the *ratios* below are
# our derivation from vertebral counts, not a table from the paper [G-24].
_VERTEBRA_NUMBER: Dict[str, int] = {
    "C1": 1, "C4": 4, "C7": 7, "T3": 10, "T8": 15, "L1": 20, "L3": 22, "L5": 24, "SACRUM": 25,
}
_N_GAPS = 24  # vertebral gaps between C1 and the sacrum

# Arc fraction from C1 (0.0) down to the sacrum (1.0).
ARC_FRACTION: Dict[str, float] = {k: (n - 1) / _N_GAPS for k, n in _VERTEBRA_NUMBER.items()}

_C7_FRACTION = ARC_FRACTION["C7"]             # 0.25: boundary between the cervical and thoracolumbar curves
_BETA_LIMIT_DEG = 90.0                         # |ear-shoulder-hip turning angle| beyond this is a landmark failure
_CURVE_SAMPLES = 120                           # dense polyline resolution per curve (for arc-length sampling)


@dataclass(frozen=True)
class SpineProxy:
    """Proxy spine of one frame. Angles are flexion-positive (convex-posterior = positive)."""
    points: Dict[str, np.ndarray]     # SPINE_LABELS -> [x, y] pixels
    curve: np.ndarray                 # [K, 2] polyline sacrum -> C1, for drawing
    bend_deg: float                   # beta: signed ear-shoulder-hip turning angle (forward head = +)
    cervical_deg: float               # tangent rotation C7 -> C1
    thoracic_deg: float               # tangent rotation L1 -> T3 (kyphosis proxy)
    lumbar_deg: float                 # tangent rotation SACRUM -> L1 (lumbar flexion proxy)
    pelvic_deg: float                 # inclination from vertical of SACRUM -> L5 (lumbopelvic proxy)
    source: str = "proxy"             # "proxy" (geometric, no real curvature) | "spinepose" (detected spine points)
    curvature_visible: bool = True    # False when seen from the front/back: the curve is a straight midline only
    confidence: float = 0.0           # mean keypoint score for "spinepose"; unused for the proxy

    def as_array(self) -> np.ndarray:
        """[9, 2] in SPINE_LABELS order (contract with the future SpinePose output)."""
        return np.stack([self.points[k] for k in SPINE_LABELS])


@dataclass(frozen=True)
class SpineDelta:
    """Change of each proxy measure from the athlete's own set-up baseline (flexion positive)."""
    bend_deg: float
    cervical_deg: float
    thoracic_deg: float
    lumbar_deg: float
    pelvic_deg: float


def _rotate(v: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _hermite(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, n: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n)[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def _sample_by_arclength(poly: np.ndarray, fractions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Points and (unnormalised) tangents of ``poly`` at the given arc-length fractions in [0, 1]."""
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    tang = np.gradient(poly, axis=0)
    targets = np.clip(fractions, 0.0, 1.0) * total
    idx = np.clip(np.searchsorted(s, targets, side="right") - 1, 0, len(poly) - 2)
    span = np.maximum(s[idx + 1] - s[idx], 1e-12)
    w = ((targets - s[idx]) / span)[:, None]
    pts = poly[idx] * (1 - w) + poly[idx + 1] * w
    tans = tang[idx] * (1 - w) + tang[idx + 1] * w
    return pts, tans


def build_spine_proxy(pose: SidePose, straight: bool = False) -> Optional[SpineProxy]:
    """Fit the two curves through ear / shoulder / hip and sample the 9 spine labels.

    Returns None if the anchors are degenerate (zero-length segment or an impossible fold).
    ``straight=True`` (the person is not side-on, so curvature is invisible in the image) forces a straight midline
    through the anchors, with all region angles 0 and ``curvature_visible=False``.
    """
    if not (pose.has("ear") and pose.has("shoulder") and pose.has("hip")):
        return None                                      # no ear (head cropped / hidden): no proxy spine
    hip, sh, ear = pose["hip"], pose["shoulder"], pose["ear"]
    v1, v2 = sh - hip, ear - sh                      # lower chord (sacral root -> C7), upper chord (C7 -> C1)
    l1, l2 = float(np.hypot(*v1)), float(np.hypot(*v2))
    if l1 < 1e-6 or l2 < 1e-6:
        return None
    u1, u2 = unit(v1), unit(v2)
    beta = math.atan2(u1[0] * u2[1] - u1[1] * u2[0], float(np.dot(u1, u2)))   # signed turn v1 -> v2 (image coords)
    if abs(math.degrees(beta)) > _BETA_LIMIT_DEG:
        return None
    if straight:
        beta = 0.0
        u2 = u1                                          # one straight line through hip and shoulder; ear along it
        ear = sh + u1 * l2

    # Near-circular arcs: end tangents mirror about the chord; the tangent at C7 is the bisector, so the
    # two curves join with G1 continuity.
    t_start_tl = _rotate(u1, -beta / 2.0)             # sacral root: pelvic orientation is unobservable -> mirror
    t_c7 = _rotate(u1, beta / 2.0)                    # == _rotate(u2, -beta / 2)
    t_end_cv = _rotate(u2, beta / 2.0)
    tl = _hermite(hip, sh, t_start_tl * l1, t_c7 * l1, _CURVE_SAMPLES)       # sacral root -> C7
    cv = _hermite(sh, ear, t_c7 * l2, t_end_cv * l2, _CURVE_SAMPLES)         # C7 -> C1

    # Fractions from C1 (0) to the sacrum (1) -> local arc fractions on each curve.
    tl_from_sacrum = {k: (1.0 - ARC_FRACTION[k]) / (1.0 - _C7_FRACTION)
                      for k in ("C7", "T3", "T8", "L1", "L3", "L5", "SACRUM")}
    cv_from_c7 = {k: (_C7_FRACTION - ARC_FRACTION[k]) / _C7_FRACTION for k in ("C1", "C4", "C7")}

    tl_keys = list(tl_from_sacrum)
    tl_pts, tl_tan = _sample_by_arclength(tl, np.array([tl_from_sacrum[k] for k in tl_keys]))
    cv_keys = list(cv_from_c7)
    cv_pts, cv_tan = _sample_by_arclength(cv, np.array([cv_from_c7[k] for k in cv_keys]))

    points = {k: tl_pts[i] for i, k in enumerate(tl_keys)}
    points.update({k: cv_pts[i] for i, k in enumerate(cv_keys) if k != "C7"})
    points["C7"] = sh.copy()
    tangents = {k: tl_tan[i] for i, k in enumerate(tl_keys)}
    tangents.update({"C1": cv_tan[cv_keys.index("C1")], "C7c": cv_tan[cv_keys.index("C7")]})

    facing = pose.facing

    def direction(t: np.ndarray) -> float:               # tangent direction, vertical-up = 0, forward +
        return inclination_from_vertical((0.0, 0.0), t, facing)

    cervical = wrap_deg(direction(tangents["C1"]) - direction(tangents["C7c"]))
    thoracic = wrap_deg(direction(tangents["T3"]) - direction(tangents["L1"]))
    lumbar = wrap_deg(direction(tangents["L1"]) - direction(tangents["SACRUM"]))
    pelvic = inclination_from_vertical(points["SACRUM"], points["L5"], facing)

    # Sign: forward head => convex-posterior arc => flexion positive. beta's sign depends on facing and
    # the image y-axis, so derive it from the same direction convention instead of trusting atan2 here.
    bend = wrap_deg(inclination_from_vertical(sh, ear, facing) - inclination_from_vertical(hip, sh, facing))

    curve = np.vstack([tl, cv[1:]])
    if straight:
        cervical = thoracic = lumbar = bend = 0.0
    return SpineProxy(points=points, curve=curve, bend_deg=bend, cervical_deg=cervical,
                      thoracic_deg=thoracic, lumbar_deg=lumbar, pelvic_deg=pelvic,
                      curvature_visible=not straight)


class SpineBaseline:
    """Rolling baseline of the proxy measures during the athlete's set-up (neutral reference).

    Feed it every SETUP-phase frame; call :meth:`reset` when a new SETUP segment starts so the
    baseline always describes the *current* set-up. ``delta`` returns None until ``min_samples``
    frames have been seen.
    """

    def __init__(self, max_samples: int = 90, min_samples: int = 5):
        self._buf: Deque[Tuple[float, float, float, float, float]] = deque(maxlen=max_samples)
        self._min = min_samples

    def reset(self) -> None:
        self._buf.clear()

    def add(self, p: SpineProxy) -> None:
        self._buf.append((p.bend_deg, p.cervical_deg, p.thoracic_deg, p.lumbar_deg, p.pelvic_deg))

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self._min

    def delta(self, p: SpineProxy) -> Optional[SpineDelta]:
        if not self.ready:
            return None
        med = np.median(np.asarray(self._buf), axis=0)
        cur = np.array([p.bend_deg, p.cervical_deg, p.thoracic_deg, p.lumbar_deg, p.pelvic_deg])
        d = cur - med
        return SpineDelta(*map(float, d))


# ---------------------------------------------------------------------------
# Real spine from detected keypoints (SpinePose [P1]); G-01, G-31
# ---------------------------------------------------------------------------
def _catmull_rom(pts: np.ndarray, per_segment: int = 16) -> np.ndarray:
    """Centripetal-free Catmull-Rom spline THROUGH every point (the curve passes exactly through the keypoints)."""
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1][None, :])
    return np.vstack(out)


# Region angles from detected points are estimated from a smooth fit through all nine points, not from the direction of
# the segment between neighbouring points. SpinePose places SACRUM, L5 and L3 only 5-25 px apart (the lumbar region is
# short in the image), so adjacent-segment directions are dominated by keypoint noise. Simulation (spine of known
# curvature, 224 px, noise 2 px, 6 frames averaged): adjacent segments gave a lumbar error of 13.7 deg std and
# false-alarmed (|change| > 12 deg with NO true change) in 38% of frames; the cubic fit gave 2.6 deg std and 0%. The
# price: the fit smooths local bends, so it under-reads a true lumbar change by about 3.5x (true +30 reads about +8.5
# with the chord parameter used here). Thresholds are in these units (DeadliftRuleConfig.lumbar_enter_deg_model). G-33.
REGION_FIT_DEGREE = 3
# Below this spine length (px, SACRUM -> C1) keypoint noise swamps the angles: no angles are produced.
MIN_SPINE_LENGTH_PX = 80.0


def build_spine_from_points(points: np.ndarray, facing: int, scores: Optional[np.ndarray] = None,
                            min_confidence: float = 0.25, curvature_visible: bool = True,
                            source: str = "spinepose", anterior: Optional[np.ndarray] = None,
                            min_length_px: float = MIN_SPINE_LENGTH_PX) -> Optional[SpineProxy]:
    """Region angles from the 9 detected spine points ``[9, 2]`` in ``SPINE_LABELS`` order (C1 ... SACRUM).

    Unlike the geometric proxy, these points come from an image-based method, so a curved back gives curved points.
    A weighted cubic is fitted through the nine points (weights = clipped scores) and each region angle is the rotation
    of the fitted spine direction between the region's two ends, **flexion positive**:

        lumbar    = tangent at L1 - tangent at SACRUM
        thoracic  = tangent at T3 - tangent at L1
        cervical  = tangent at C1 - tangent at C7
        pelvic    = inclination from vertical of the tangent at SACRUM
        bend      = chord(C7 -> C1) direction - chord(SACRUM -> C7) direction

    Directions are measured in the spine's own frame: the angle of a tangent from the SACRUM -> C1 chord, positive when
    it leans toward the *anterior* side. That makes the region angles independent of body orientation (upright for
    deadlifts and squats, horizontal for the bench press). ``anterior`` is the unit vector pointing to the front of the
    body in the image; if omitted it is the chord turned a quarter turn toward ``facing`` (+1 image-right, -1
    image-left), which is correct for a person standing, hinging or folded past horizontal. ``pelvic_deg`` alone uses
    the image vertical and is only meaningful for an upright body.

    Returns None if the points are not a plausible spine (low confidence, too short to measure, or not ordered along
    SACRUM -> C1). ``scores`` may exceed 1 (SimCC peak values); they are clipped to [0, 1]. ``curvature_visible=False``
    (seen from the front/back) keeps the points and curve for drawing but zeroes the angles: a side-to-side offset in
    the image is not spinal flexion. ``source`` tags where the points came from ("spinepose" or "contour", G-32).
    """
    pts = np.asarray(points, float)
    if pts.shape != (9, 2) or not np.all(np.isfinite(pts)):
        return None
    sc = np.clip(np.asarray(scores, float), 0.0, 1.0) if scores is not None else np.ones(9)
    conf = float(np.mean(sc))
    if conf < min_confidence:
        return None
    up = pts[::-1]                                        # SACRUM, L5, L3, L1, T8, T3, C7, C4, C1 (bottom to top)
    w_up = np.clip(sc[::-1], 0.05, 1.0)
    chord = up[-1] - up[0]
    length = float(np.hypot(*chord))
    if length < max(min_length_px, 1e-6):
        return None
    c_hat = chord / length
    along = (up - up[0]) @ c_hat                          # position of each point along the sacrum -> C1 chord
    # Order check on the TRUSTED points only (score >= min_confidence): one low-score wild point must be down-weighted
    # by the fit, not veto the whole frame. Fewer than 5 trusted points cannot support the cubic fit.
    trusted = sc[::-1] >= min_confidence
    if int(trusted.sum()) < 5 or np.any(np.diff(along[trusted]) < -0.02 * length):    # scrambled detection
        return None

    if anterior is None:
        # The front of the trunk is the chord turned a quarter turn toward the way the person faces. This is the same as
        # projecting (facing, 0) off the chord while the trunk leans less than 90 deg from vertical, but it stays
        # correct (and continuous) in a deep forward fold where the head is below the hips: projecting (facing, 0)
        # flips sign there and read convex rounding as extension (found on a real clip; docs/GAP_ANALYSIS.md G-34).
        a_hat = float(facing) * np.array([-c_hat[1], c_hat[0]])
    else:
        ant = np.asarray(anterior, float)
        ant = ant - (ant @ c_hat) * c_hat                 # the anterior direction perpendicular to the chord
        n_ant = float(np.hypot(*ant))
        if n_ant < 1e-6:
            return None                                   # anterior is along the spine: orientation is ambiguous
        a_hat = ant / n_ant

    def lean(v: np.ndarray) -> float:                     # angle from the chord, + toward the anterior side
        return math.degrees(math.atan2(float(v @ a_hat), float(v @ c_hat)))

    # Parametrise by position along the SACRUM -> C1 chord, NOT by cumulative path length: with path length one wild
    # point inflates the length and shifts every other point's parameter (a single 60 px outlier at weight 0.05
    # corrupted the lumbar angle by 165 deg in simulation; 1.1 deg with the chord parameter).
    t = along / along[-1]
    cx, cy = (np.polyfit(t, up[:, k], REGION_FIT_DEGREE, w=w_up) for k in (0, 1))
    dcx, dcy = np.polyder(cx), np.polyder(cy)

    def tangent(i: int) -> np.ndarray:
        return np.array([np.polyval(dcx, t[i]), np.polyval(dcy, t[i])])

    th = [lean(tangent(i)) for i in range(9)]
    if not all(math.isfinite(v) for v in th):
        return None
    lumbar = wrap_deg(th[3] - th[0])                      # tangent at L1 - tangent at SACRUM
    thoracic = wrap_deg(th[5] - th[3])                    # tangent at T3 - tangent at L1
    cervical = wrap_deg(th[8] - th[6])                    # tangent at C1 - tangent at C7
    pelvic = inclination_from_vertical((0.0, 0.0), tangent(0), facing)
    bend = wrap_deg(lean(up[8] - up[6]) - lean(up[6] - up[0]))
    if not all(math.isfinite(v) for v in (pelvic, bend)):
        return None
    if not curvature_visible:
        lumbar = thoracic = cervical = bend = pelvic = 0.0
    labelled = {label: pts[i].copy() for i, label in enumerate(SPINE_LABELS)}
    return SpineProxy(points=labelled, curve=_catmull_rom(up), bend_deg=bend, cervical_deg=cervical,
                      thoracic_deg=thoracic, lumbar_deg=lumbar, pelvic_deg=pelvic, source=source,
                      curvature_visible=curvature_visible, confidence=conf)
