"""Synthetic side-view deadlift lifter: full ``[33, 3]`` MediaPipe landmark clips with known ground truth.

Used by ``tests/test_deadlift_biomechanics.py`` and by ``scripts/demo_deadlift.py --synthetic`` so the whole
pipeline can be exercised with no camera, model or video. Scenarios: good rep, lumbar rounding, early hip
rise ("stripper"), lockout hyperextension, bar drift.

The lifter is a planar kinematic chain (px units, image y down):

    ankle -(shank, forward-leaning tau)-> knee -(thigh)-> hip -(torso, inclination theta)-> shoulder
    shoulder -(neck, inclination theta + bend)-> ear           wrist hangs under the bar (mid-foot + drift)

Joint parameters: knee flexion kappa, shank inclination tau, trunk inclination theta, head/upper-back bend
``bend`` (extra forward turn of the neck relative to the trunk line; this is the one quantity the spine
proxy can see, see spine_proxy.py). These tests validate rule *logic*, not real-world sensitivity.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np

from .landmarks import NOSE, NUM_LANDMARKS, SIDE_INDICES


def smoothstep(x: float) -> float:
    """Minimum-jerk profile 0 -> 1."""
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


@dataclass(frozen=True)
class Body:
    torso: float = 250.0       # hip -> shoulder (px)
    thigh: float = 215.0
    shank: float = 215.0
    foot: float = 130.0        # heel -> toe
    neck: float = 75.0         # shoulder -> ear
    arm: float = 300.0         # shoulder -> wrist
    ankle_h: float = 40.0
    floor_y: float = 700.0
    x0: float = 400.0          # ankle x

    def scaled(self, k: float, floor_y: float, x0: float) -> "Body":
        """Same lifter drawn ``k`` times smaller/larger, standing at (x0, floor_y) (to fit a given canvas)."""
        return Body(torso=self.torso * k, thigh=self.thigh * k, shank=self.shank * k, foot=self.foot * k,
                    neck=self.neck * k, arm=self.arm * k, ankle_h=self.ankle_h * k, floor_y=floor_y, x0=x0)


@dataclass(frozen=True)
class JointPose:
    knee_flex_deg: float
    shank_incl_deg: float
    trunk_incl_deg: float
    bend_deg: float            # head / upper-back turn relative to the trunk line, forward = +
    bar_drift_foot: float = 0.0


@dataclass(frozen=True)
class Scenario:
    name: str = "good"
    neutral_bend_deg: float = 5.0
    round_deg: float = 0.0                       # extra bend while rounding
    round_on: Tuple[str, ...] = ("pull",)        # "pull" and/or "descent"
    early_hip_rise: bool = False
    lockout_lean_deg: float = 0.0                # backward lean of the trunk at lockout
    bar_drift_foot: float = 0.0                  # peak forward bar drift (foot lengths) mid-pull


SCENARIOS: Dict[str, Scenario] = {
    "good": Scenario("good"),
    "lumbar_round": Scenario("lumbar_round", round_deg=28.0, round_on=("pull",)),
    "butt_wink": Scenario("butt_wink", round_deg=28.0, round_on=("descent",)),
    "stripper": Scenario("stripper", early_hip_rise=True),
    "hyperextension": Scenario("hyperextension", lockout_lean_deg=14.0),
    "bar_drift": Scenario("bar_drift", bar_drift_foot=0.6),
}

# Bottom / lockout postures of a conventional deadlift (hip ~85 deg, knee ~120 deg at the bottom)
BOTTOM = dict(kappa=60.0, tau=15.0, theta=50.0)
TOP = dict(kappa=0.0, tau=0.0, theta=0.0)


def _bump(s: float) -> float:
    """Rounding envelope over bar progress: ramps in over the first 8% of the bar's travel, full strength
    until s = 0.35, fading out by s = 0.6. Zero at the floor, so the set-up hold stays neutral (a rounded
    set-up would be absorbed into the baseline: see spine_proxy.py) and there are no pose jumps."""
    fade = 1.0 if s <= 0.35 else max(0.0, 1.0 - (s - 0.35) / 0.25)
    return fade * min(s / 0.08, 1.0)


def deadlift_pose(s: float, sc: Scenario, ascending: bool) -> JointPose:
    """Joint state at bar progress ``s`` (0 = floor, 1 = lockout)."""
    s = min(max(s, 0.0), 1.0)
    e = smoothstep(s)
    if sc.early_hip_rise:
        kappa = BOTTOM["kappa"] * (1.0 - smoothstep(min(s / 0.5, 1.0)))            # knees shoot straight early
        if s < 0.35:
            theta = BOTTOM["theta"] + 12.0 * (s / 0.35)                             # back gets flatter
        else:
            theta = (BOTTOM["theta"] + 12.0) * (1.0 - smoothstep((s - 0.35) / 0.65))
    else:
        kappa = BOTTOM["kappa"] * (1.0 - e)
        theta = BOTTOM["theta"] * (1.0 - e)
    tau = BOTTOM["tau"] * (1.0 - e)
    theta -= sc.lockout_lean_deg * (e ** 3)
    bend = sc.neutral_bend_deg
    if sc.round_deg and (("pull" in sc.round_on and ascending) or ("descent" in sc.round_on and not ascending)):
        bend += sc.round_deg * _bump(s)
    drift = sc.bar_drift_foot * math.sin(math.pi * s) if sc.bar_drift_foot else 0.0
    return JointPose(kappa, tau, theta, bend, drift)


def build_landmarks(p: JointPose, body: Body = Body(), facing: int = 1, near: str = "right") -> np.ndarray:
    """``[33, 3]`` (x_px, y_px, visibility). Near side visible (0.99), far side hidden (0.15)."""
    f = facing

    def vec(incl_deg: float, length: float) -> np.ndarray:       # inclination from vertical-up, forward +
        a = math.radians(incl_deg)
        return np.array([math.sin(a) * length * f, -math.cos(a) * length])

    ankle = np.array([body.x0, body.floor_y - body.ankle_h])
    knee = ankle + vec(p.shank_incl_deg, body.shank)
    hip = knee + vec(p.shank_incl_deg - p.knee_flex_deg, body.thigh)
    shoulder = hip + vec(p.trunk_incl_deg, body.torso)
    ear = shoulder + vec(p.trunk_incl_deg + p.bend_deg, body.neck)
    heel = np.array([body.x0 - f * 0.25 * body.foot, body.floor_y])
    toe = np.array([body.x0 + f * 0.75 * body.foot, body.floor_y])
    mid_foot_x = 0.5 * (heel[0] + toe[0])
    wrist = np.array([mid_foot_x + f * p.bar_drift_foot * body.foot, shoulder[1] + body.arm])
    elbow = 0.5 * (shoulder + wrist)

    lm = np.zeros((NUM_LANDMARKS, 3))
    far = "left" if near == "right" else "right"
    coords = dict(ear=ear, shoulder=shoulder, elbow=elbow, wrist=wrist, hip=hip, knee=knee, ankle=ankle,
                  heel=heel, foot=toe)
    for key, xy in coords.items():
        lm[SIDE_INDICES[near][key]] = (xy[0], xy[1], 0.99)
        lm[SIDE_INDICES[far][key]] = (xy[0] + f * 8.0, xy[1], 0.15)
    lm[NOSE] = (ear[0] + f * 30.0, ear[1] + 12.0, 0.9)
    return lm


@dataclass(frozen=True)
class ClipSpec:
    fps: float = 30.0
    n_reps: int = 1
    pre_s: float = 1.2           # bottom hold before the first pull (set-up)
    up_s: float = 2.0            # concentric
    hold_s: float = 0.8          # lockout
    down_s: float = 2.0          # eccentric
    post_s: float = 1.0          # bottom hold after each rep
    noise_px: float = 0.0        # landmark jitter (std, px) on the near side
    facing: int = 1
    seed: int = 0


def clip_duration(spec: ClipSpec) -> float:
    return spec.pre_s + spec.n_reps * (spec.up_s + spec.hold_s + spec.down_s + spec.post_s)


def rep_truth(spec: ClipSpec, rep: int) -> Dict[str, float]:
    """Ground-truth event times (s) of rep ``rep`` (0-based): lift-off, lockout start, descent start, floor."""
    t0 = spec.pre_s + rep * (spec.up_s + spec.hold_s + spec.down_s + spec.post_s)
    return dict(liftoff=t0, lockout=t0 + spec.up_s, descent=t0 + spec.up_s + spec.hold_s,
                floor=t0 + spec.up_s + spec.hold_s + spec.down_s)


def progress_at(t: float, spec: ClipSpec) -> Tuple[float, bool]:
    """(bar progress s in [0, 1], ascending?) at clip time t."""
    if t < spec.pre_s:
        return 0.0, True
    cycle = spec.up_s + spec.hold_s + spec.down_s + spec.post_s
    u = (t - spec.pre_s) % cycle
    if u < spec.up_s:
        return u / spec.up_s, True
    if u < spec.up_s + spec.hold_s:
        return 1.0, True
    if u < spec.up_s + spec.hold_s + spec.down_s:
        return 1.0 - (u - spec.up_s - spec.hold_s) / spec.down_s, False
    return 0.0, False


def generate_clip(scenario: str | Scenario = "good", spec: ClipSpec = ClipSpec(),
                  body: Body = Body()) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield ``(t_s, landmarks[33, 3])`` for the whole clip."""
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = random.Random(spec.seed)
    n = int(round(clip_duration(spec) * spec.fps))
    for i in range(n):
        t = i / spec.fps
        s, asc = progress_at(t, spec)
        lm = build_landmarks(deadlift_pose(s, sc, asc), body, facing=spec.facing)
        if spec.noise_px > 0:
            near = SIDE_INDICES["right"].values()
            for idx in near:
                lm[idx, 0] += rng.gauss(0.0, spec.noise_px)
                lm[idx, 1] += rng.gauss(0.0, spec.noise_px)
        yield t, lm


def clip_list(scenario: str | Scenario = "good", spec: ClipSpec = ClipSpec()) -> List[Tuple[float, np.ndarray]]:
    return list(generate_clip(scenario, spec))


# ---------------------------------------------------------------------------
# 3-D lifter: any viewing angle, both sides, with MediaPipe-style world landmarks
# ---------------------------------------------------------------------------
M_PER_PX = 0.002                     # body model scale: 250 px torso = 0.5 m

# lateral half-widths in metres (person's right is +)
_LATERAL_M = dict(ear=0.08, shoulder=0.18, elbow=0.20, wrist=0.20, hip=0.09, knee=0.10, ankle=0.10,
                  heel=0.10, foot=0.11)


def build_landmarks_3d(p: JointPose, body: Body = Body(), yaw_deg: float = 0.0, cx: float = 640.0,
                       floor_y: float = 700.0, spine_curve_deg: float = 0.0):
    """Planar lifter turned about the vertical axis: returns ``(image [33, 3], world [33, 3])``.

    ``yaw_deg`` is the person's heading seen from the camera: 0 = side-on facing image-right (right side to the
    camera), 90 = facing the camera, 180 = side-on facing image-left (left side to the camera), -90 = back to the
    camera. Image coordinates are an orthographic projection (px, y down); world coordinates follow MediaPipe's
    convention (metres, origin at the hip centre, x right, y DOWN, z away from the camera).
    Each side's visibility is 0.99 when it is the side closer to the camera (or equidistant), else 0.15.
    ``spine_curve_deg`` is unused here (the planar body has no spine curvature of its own).
    """
    a = math.radians(yaw_deg)
    fx, fz = math.cos(a), -math.sin(a)                  # forward in the camera's horizontal plane (x right, z away)
    rx, rz = -math.sin(a), -math.cos(a)                 # the person's right-hand direction

    def vec(incl_deg: float, length: float):            # (forward u, up h) in px
        t = math.radians(incl_deg)
        return math.sin(t) * length, math.cos(t) * length

    ankle = (0.0, body.ankle_h)
    du, dh = vec(p.shank_incl_deg, body.shank)
    knee = (ankle[0] + du, ankle[1] + dh)
    du, dh = vec(p.shank_incl_deg - p.knee_flex_deg, body.thigh)
    hip = (knee[0] + du, knee[1] + dh)
    du, dh = vec(p.trunk_incl_deg, body.torso)
    shoulder = (hip[0] + du, hip[1] + dh)
    du, dh = vec(p.trunk_incl_deg + p.bend_deg, body.neck)
    ear = (shoulder[0] + du, shoulder[1] + dh)
    heel = (-0.25 * body.foot, 0.0)
    toe = (0.75 * body.foot, 0.0)
    wrist = (0.25 * body.foot + p.bar_drift_foot * body.foot, shoulder[1] - body.arm)
    elbow = (0.5 * (shoulder[0] + wrist[0]), 0.5 * (shoulder[1] + wrist[1]))
    nose = (ear[0] + 0.09 / M_PER_PX, ear[1] - 0.03 / M_PER_PX)
    chain = dict(ear=ear, shoulder=shoulder, elbow=elbow, wrist=wrist, hip=hip, knee=knee, ankle=ankle,
                 heel=heel, foot=toe)

    def cam(point, lateral_m):
        u, h = point
        return (u * M_PER_PX * fx + lateral_m * rx, h * M_PER_PX, u * M_PER_PX * fz + lateral_m * rz)

    img = np.zeros((NUM_LANDMARKS, 3))
    world = np.zeros((NUM_LANDMARKS, 3))
    hip_c = cam(hip, 0.0)
    for key, pt in chain.items():
        pr, pl = cam(pt, +_LATERAL_M[key]), cam(pt, -_LATERAL_M[key])
        for side, c, other in (("right", pr, pl), ("left", pl, pr)):
            idx = SIDE_INDICES[side][key]
            vis = 0.99 if c[2] <= other[2] + 0.02 else 0.15
            img[idx] = (cx + c[0] / M_PER_PX, floor_y - c[1] / M_PER_PX, vis)
            world[idx] = (c[0] - hip_c[0], -(c[1] - hip_c[1]), c[2] - hip_c[2])
    n = cam(nose, 0.0)
    img[NOSE] = (cx + n[0] / M_PER_PX, floor_y - n[1] / M_PER_PX, 0.95)
    world[NOSE] = (n[0] - hip_c[0], -(n[1] - hip_c[1]), n[2] - hip_c[2])
    return img, world


def generate_clip_3d(scenario: str | Scenario = "good", spec: ClipSpec = ClipSpec(), body: Body = Body(),
                     yaw_deg: float = 0.0, world_noise_m: float = 0.0, cx: float = 640.0, floor_y: float = 700.0):
    """Yield ``(t_s, image [33, 3], world [33, 3])`` for the whole clip, seen from heading ``yaw_deg``."""
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = random.Random(spec.seed)
    n = int(round(clip_duration(spec) * spec.fps))
    for i in range(n):
        t = i / spec.fps
        s, asc = progress_at(t, spec)
        img, world = build_landmarks_3d(deadlift_pose(s, sc, asc), body, yaw_deg, cx, floor_y)
        if spec.noise_px > 0:
            img[:, 0] += [rng.gauss(0.0, spec.noise_px) for _ in range(NUM_LANDMARKS)]
            img[:, 1] += [rng.gauss(0.0, spec.noise_px) for _ in range(NUM_LANDMARKS)]
        if world_noise_m > 0:
            world += np.array([[rng.gauss(0.0, world_noise_m) for _ in range(3)] for _ in range(NUM_LANDMARKS)])
        yield t, img, world


def back_profile_for(pose: JointPose, neutral_bend_deg: float = 5.0, base: float = 0.20):
    """Back-surface distance behind the hip-shoulder chord (torso lengths) as a function of the position ``t`` along it.

    A neutral back sits ``base`` behind the chord with a gentle upper-back bulge. Rounding (extra head/upper-back bend)
    adds a convex lower-thoracic/lumbar bulge that grows with the amount of rounding; used to build test silhouettes.
    """
    rounding = max(0.0, (pose.bend_deg - neutral_bend_deg) / 28.0)

    def profile(t: float) -> float:
        bulge = math.exp(-((t - 0.38) / 0.30) ** 2)           # lower-thoracic / lumbar region
        return base + (0.03 + 0.17 * rounding) * bulge
    return profile


def generate_clip_3d_masked(scenario: str | Scenario = "good", spec: ClipSpec = ClipSpec(), body: Body = Body(),
                            yaw_deg: float = 0.0, shape=(720, 1280), cx: float = 640.0, floor_y: float = 700.0,
                            world_noise_m: float = 0.0):
    """Like :func:`generate_clip_3d` but also yields a person silhouette: ``(t_s, image, world, mask [H, W])``.

    The silhouette is a trunk polygon whose back edge follows :func:`back_profile_for` (side-on views only are
    meaningful), so the back-contour spine can be tested end to end without a camera.
    """
    from .spine_contour import synthetic_mask
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    for t, img, world in generate_clip_3d(scenario, spec, body, yaw_deg, world_noise_m, cx, floor_y):
        s, asc = progress_at(t, spec)
        pose = deadlift_pose(s, sc, asc)
        side = "right" if img[SIDE_INDICES["right"]["hip"], 2] >= img[SIDE_INDICES["left"]["hip"], 2] else "left"
        idx = SIDE_INDICES[side]
        hip, sh = img[idx["hip"], :2], img[idx["shoulder"], :2]
        facing = 1 if img[idx["foot"], 0] > img[idx["heel"], 0] else -1
        mask = synthetic_mask(shape, hip, sh, facing, back_profile_for(pose, sc.neutral_bend_deg))
        yield t, img, world, mask
