"""Both body sides, any heading (facing the camera, oblique, turning around), and the real-spine path.

Regression for the second live-webcam report: only the right side was tracked (the side was locked forever), a front
view produced a curved spine line, and the proxy spine could not follow a real curve. See GAP_ANALYSIS.md G-30/G-31.
"""

import math

import numpy as np
import pytest

from myomotion.biomechanics import DeadliftAnalyzer
from myomotion.biomechanics.landmarks import (SIDE_INDICES, extract_midline_pose, framing_report, shoulder_span_ratio,
                                              side_scores, view_name)
from myomotion.biomechanics.pose3d import compute_angles_3d, estimate_forward, heading_span_equivalent
from myomotion.biomechanics.rep_segmenter import Phase
from myomotion.biomechanics.spine_proxy import (SPINE_LABELS, build_spine_from_points, build_spine_proxy)
from myomotion.biomechanics.synthetic import (SCENARIOS, Body, ClipSpec, JointPose, build_landmarks_3d,
                                              deadlift_pose, generate_clip_3d, progress_at)
from myomotion.vision.spine_model import (CHAIN_BOTTOM_UP, SpineDetection, SpinePoseTracker, SpinePoseUnavailable,
                                          bridge_error, person_bbox, spinepose_available)

BOTTOM = JointPose(60, 15, 50, 5)                       # hip 85, knee 120, trunk +50, tibia +15


def run3d(yaw=0.0, scenario="good", world=True, n_reps=2, yaw_fn=None, **kw):
    spec = ClipSpec(n_reps=n_reps, noise_px=1.0, **kw)
    an = DeadliftAnalyzer()
    results = []
    sc = SCENARIOS[scenario]
    for t, img, w in generate_clip_3d(scenario, spec, Body(), yaw_deg=yaw, world_noise_m=0.01):
        if yaw_fn is not None:
            img, w = build_landmarks_3d(deadlift_pose(progress_at(t, spec)[0], sc, progress_at(t, spec)[1]),
                                        Body(), yaw_fn(t))
        results.append(an.process(t, img, world=w if world else None, image_wh=(1280, 720)))
    return an, results


# ---------------------------------------------------------------------------
# 3-D angles: independent of where the camera is
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("yaw", [0, 20, 45, 90, 135, 180, -45, -90, -135])
def test_3d_angles_recover_the_true_pose_from_every_heading(yaw):
    img, world = build_landmarks_3d(BOTTOM, yaw_deg=yaw)
    a = compute_angles_3d(world, img[:, 2])
    assert a.hip_signed_deg == pytest.approx(85.0, abs=0.5) and a.knee_deg == pytest.approx(120.0, abs=0.5)
    assert a.trunk_incl_deg == pytest.approx(50.0, abs=0.5) and a.tibia_incl_deg == pytest.approx(15.0, abs=0.5)


def test_3d_forward_vector_follows_the_heading_and_needs_a_cue():
    for yaw in (0, 90, 180, -90):
        img, world = build_landmarks_3d(BOTTOM, yaw_deg=yaw)
        fw = estimate_forward(world, img[:, 2])
        assert fw[0] == pytest.approx(math.cos(math.radians(yaw)), abs=0.02)
        assert fw[2] == pytest.approx(-math.sin(math.radians(yaw)), abs=0.06)   # the model foot has ~2 deg toe-out
    img, world = build_landmarks_3d(BOTTOM, yaw_deg=0)
    vis = img[:, 2].copy()
    vis[:] = 0.0
    assert compute_angles_3d(world, vis) is None                          # nothing visible


def test_heading_span_equivalent_is_zero_side_on_and_about_072_facing():
    vals = {}
    for yaw in (0, 45, 90, 180):
        img, world = build_landmarks_3d(BOTTOM, yaw_deg=yaw)
        vals[yaw] = heading_span_equivalent(world)
    assert vals[0] < 0.02 and vals[180] < 0.02
    assert vals[45] == pytest.approx(0.72 * math.sin(math.radians(45)), abs=0.02)
    assert vals[90] == pytest.approx(0.72, abs=0.02)


# ---------------------------------------------------------------------------
# Both sides, facing the camera, turning around
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("yaw, side", [(0, "right"), (180, "left")])
def test_either_side_to_the_camera_is_tracked_equally(yaw, side):
    an, results = run3d(yaw)
    assert an.segmenter.rep_count == 2 and an.frames_tracked == an.frames_total
    assert all(r.angles.side == side for r in results if r.angles)
    setup = [r.angles.trunk_incl_deg for r in results if r.angles and r.phase == Phase.SETUP]
    assert np.median(setup) == pytest.approx(50.0, abs=1.5)               # forward lean is positive for both sides


@pytest.mark.parametrize("yaw", [90, -90, 45, 135, -45])
def test_facing_the_camera_or_angled_is_tracked_with_3d_angles(yaw):
    an, results = run3d(yaw)
    assert an.segmenter.rep_count == 2 and an.frames_tracked == an.frames_total
    tracked = [r for r in results if r.angles]
    assert {r.view for r in tracked} == {"angled"} and {r.angle_source for r in tracked} == {"world"}
    setup = [r.angles.trunk_incl_deg for r in tracked if r.phase == Phase.SETUP]
    assert np.median(setup) == pytest.approx(50.0, abs=1.5)               # NOT the foreshortened 2-D value


def test_oblique_view_without_world_landmarks_is_flagged_not_silently_wrong():
    an, results = run3d(45, world=False)
    r = next(r for r in results if r.angles)
    assert r.view in ("side", "angled") and r.angle_source == "image"


def test_turning_around_mid_set_switches_side_and_keeps_counting():
    an, results = run3d(0, yaw_fn=lambda t: 0.0 if t < 6.5 else 180.0)
    assert an.segmenter.rep_count == 2
    assert an._side == "left"
    assert an._facing == -1                                               # now faces image-left
    after = [r for r in results if r.t_s > 9.0 and r.angles]
    assert after and all(r.angles.side == "left" for r in after)
    assert an.frames_tracked >= an.frames_total - 20                      # only the brief hand-over is lost


def test_a_brief_visibility_flicker_does_not_switch_sides():
    spec = ClipSpec(n_reps=1, noise_px=0.5)
    an = DeadliftAnalyzer()
    for t, img, w in generate_clip_3d("good", spec, Body(), yaw_deg=0.0):
        if 2.0 <= t < 2.2:                                                # 0.2 s: the left side looks better
            right, left = SIDE_INDICES["right"], SIDE_INDICES["left"]
            for k in right:
                img[right[k], 2], img[left[k], 2] = 0.2, 0.99
        an.process(t, img, world=w)
    assert an._side == "right"


def test_a_user_forced_side_is_never_switched():
    an = DeadliftAnalyzer(side="right")
    for t, img, w in generate_clip_3d("good", ClipSpec(n_reps=1), Body(), yaw_deg=180.0):
        an.process(t, img, world=w)
    assert an._side == "right"


def test_side_scores_and_view_helpers():
    img, _ = build_landmarks_3d(BOTTOM, yaw_deg=180)
    s = side_scores(img)
    assert s["left"] > s["right"] + 0.3
    assert [view_name(v) for v in (None, 0.1, 0.5, 0.8)] == ["side", "side", "angled", "front"]
    front, _ = build_landmarks_3d(JointPose(0, 0, 0, 5), yaw_deg=90)         # upright: no foreshortening of the torso
    assert shoulder_span_ratio(front) == pytest.approx(0.72, abs=0.03)
    hinged, _ = build_landmarks_3d(BOTTOM, yaw_deg=90)                      # hinged at the camera: torso looks short
    assert shoulder_span_ratio(hinged) > 0.9                                # (why the 3-D heading is preferred)


def test_framing_reports_the_view_without_blocking_it():
    front, _ = build_landmarks_3d(BOTTOM, yaw_deg=90)
    fr = framing_report(front, image_wh=(1280, 720))
    assert fr.ok and fr.view == "front" and any("Facing the camera" in n for n in fr.notes)


# ---------------------------------------------------------------------------
# Spine from a front/back view: never an invented curve
# ---------------------------------------------------------------------------
def test_front_view_spine_is_a_straight_midline_down_the_body_with_no_curvature_claims():
    an, results = run3d(90, scenario="lumbar_round")
    sp = next(r.angles.spine for r in results if r.angles and r.phase == Phase.CONCENTRIC_PULL)
    assert sp.source == "proxy" and not sp.curvature_visible
    assert abs(sp.lumbar_deg) < 1e-9 and abs(sp.bend_deg) < 1e-9
    xs = [sp.points[k][0] for k in SPINE_LABELS]
    height = abs(sp.points["C1"][1] - sp.points["SACRUM"][1])
    assert max(xs) - min(xs) < 0.35 * height                              # a straight line, not a bow
    assert an.summary()["flag_counts"]["lumbar_rounding"] == 0            # cannot see rounding from the front: no cue


def test_midline_anchors_sit_between_the_two_sides():
    img, _ = build_landmarks_3d(BOTTOM, yaw_deg=90)
    mid = extract_midline_pose(img)
    assert mid.pts["shoulder"][0] == pytest.approx(0.5 * (img[11, 0] + img[12, 0]), abs=1e-6)
    straight = build_spine_proxy(mid, straight=True)
    assert not straight.curvature_visible and straight.source == "proxy"


# ---------------------------------------------------------------------------
# Real spine points (SpinePose output) -> angles that follow a curved spine
# ---------------------------------------------------------------------------
def arc_points(flex_deg, length=250.0, facing=1):
    """9 points SACRUM->C1 on a convex-posterior arc, returned in SPINE_LABELS order (C1 ... SACRUM)."""
    n = 9
    th = np.radians(np.linspace(-flex_deg / 2, flex_deg / 2, n))
    seg = length / (n - 1)
    p = [np.array([0.0, 0.0])]
    for t in th[:-1] + np.diff(th) / 2 if flex_deg else th[:-1]:
        p.append(p[-1] + seg * np.array([math.sin(t) * facing, -math.cos(t)]))
    return np.array(p)[::-1]


@pytest.mark.parametrize("facing", [1, -1])
def test_spine_angles_follow_the_real_curvature_and_the_curve_passes_through_the_points(facing):
    last = None
    for flex in (0, 20, 40, 60):
        pts = arc_points(flex, facing=facing)
        sp = build_spine_from_points(pts, facing)
        assert sp.source == "spinepose" and sp.curvature_visible
        for label in SPINE_LABELS:                                        # the drawn line lies ON the detected points
            assert np.min(np.hypot(*(sp.curve - sp.points[label]).T)) < 1e-6
        if last is not None:
            assert sp.lumbar_deg > last.lumbar_deg and sp.thoracic_deg > last.thoracic_deg
        last = sp
    assert build_spine_from_points(arc_points(0), 1).lumbar_deg == pytest.approx(0.0, abs=1e-9)
    assert build_spine_from_points(arc_points(40, facing=1), 1).thoracic_deg == pytest.approx(
        build_spine_from_points(arc_points(40, facing=-1), -1).thoracic_deg, abs=1e-6)       # mirror invariant


def test_scrambled_or_low_confidence_spine_points_are_rejected():
    rng = np.random.RandomState(0)
    assert build_spine_from_points(rng.rand(9, 2) * 100, 1) is None
    assert build_spine_from_points(arc_points(20), 1, scores=np.full(9, 0.1)) is None
    assert build_spine_from_points(np.full((9, 2), np.nan), 1) is None
    assert build_spine_from_points(np.zeros((9, 2)), 1) is None


def test_spine_points_from_a_front_view_keep_the_points_but_zero_the_angles():
    sp = build_spine_from_points(arc_points(40), 1, curvature_visible=False)
    assert not sp.curvature_visible and sp.lumbar_deg == 0.0 and sp.thoracic_deg == 0.0
    assert np.min(np.hypot(*(sp.curve - sp.points["T8"]).T)) < 1e-6        # still drawn through the real points


def curved_points_for_frame(img, flex_deg):
    """Spine points on a constant-curvature arc (total tangent rotation ``flex_deg``) fitted between the synthetic
    body's hip and ear: a stand-in for what SpinePose would return for a back with that much curvature."""
    r = SIDE_INDICES["right"]
    hip, ear = img[r["hip"], :2], img[r["ear"], :2]
    local = arc_points(flex_deg, length=1.0)                              # C1 ... SACRUM, sacrum at the origin
    c_loc, c_tgt = local[0] - local[-1], ear - hip
    rot = math.atan2(c_tgt[1], c_tgt[0]) - math.atan2(c_loc[1], c_loc[0])
    scale = float(np.hypot(*c_tgt) / np.hypot(*c_loc))
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    pts = (local - local[-1]) @ R.T * scale + hip
    return np.column_stack([pts, np.full(9, 0.9)])


def run_with_model_spine(flex_at_pull):
    spec = ClipSpec(n_reps=1, noise_px=0.5)
    an = DeadliftAnalyzer()
    for t, img, w in generate_clip_3d("good", spec, Body(), yaw_deg=0.0):
        s, asc = progress_at(t, spec)
        flex = flex_at_pull * min(s / 0.10, 1.0) * (1 - min(max((s - 0.35) / 0.25, 0.0), 1.0))
        an.process(t, img, world=w, spine_points=curved_points_for_frame(img, flex))
    return an.summary()


def test_a_really_rounding_back_is_flagged_with_direct_confidence_from_detected_points():
    s = run_with_model_spine(70.0)
    assert s["reps"][0]["form"]["max_lumbar_delta_deg"] >= 12.0            # the MEASURED lumbar change, not the knob
    assert s["flag_counts"]["lumbar_rounding"] == 1
    ev = s["events"][0]
    assert ev["confidence"] == "direct" and ev["threshold"] == 12.0 and "possible" not in ev["message"]
    assert s["frames_by_spine_source"].get("spinepose", 0) > 0 and "proxy" not in s["frames_by_spine_source"]


def test_a_straight_back_from_detected_points_is_not_flagged():
    s = run_with_model_spine(0.0)
    assert s["flag_counts"]["lumbar_rounding"] == 0 and s["rep_count"] == 1


def test_a_small_real_curvature_change_below_the_threshold_is_not_flagged():
    s = run_with_model_spine(25.0)
    assert 0.0 < s["reps"][0]["form"]["max_lumbar_delta_deg"] < 12.0
    assert s["flag_counts"]["lumbar_rounding"] == 0


def test_baseline_restarts_when_the_spine_source_changes():
    an = DeadliftAnalyzer()
    spec = ClipSpec(n_reps=1, noise_px=0.3)
    seen = set()
    for i, (t, img, w) in enumerate(generate_clip_3d("good", spec, Body(), yaw_deg=0.0)):
        pts = curved_points_for_frame(img, 0.0) if t < 0.8 else None      # model for 0.8 s, then proxy
        r = an.process(t, img, world=w, spine_points=pts)
        if r.angles and r.angles.spine:
            seen.add(r.angles.spine.source)
    assert seen == {"spinepose", "proxy"}


# ---------------------------------------------------------------------------
# SpinePose wrapper (stand-in model; the real network is not installed here)
# ---------------------------------------------------------------------------
class FakeModel:
    """Mimics RTMPose: returns 37 keypoints where the spine chain ids hold a known curve."""
    def __init__(self, ids=(19, 26, 27, 28, 29, 30, 18, 35, 36)):
        self.ids, self.calls = ids, []

    def __call__(self, image, bboxes):
        self.calls.append(bboxes)
        k = np.zeros((1, 37, 2))
        sc = np.full((1, 37), 0.9)
        pts = arc_points(30)[::-1]                                        # bottom-up SACRUM ... C1
        for j, idx in enumerate(self.ids):
            k[0, idx] = pts[j] + [300, 600]
        k[0, 5], k[0, 6] = (270, 380), (330, 380)                         # coco shoulders
        k[0, 11], k[0, 12] = (280, 600), (320, 600)                       # coco hips
        return k, sc


def test_wrapper_maps_the_chain_order_to_top_to_bottom_labels_and_passes_the_box():
    tr = SpinePoseTracker(model=FakeModel())
    det = tr.detect(np.zeros((900, 700, 3), np.uint8), np.array([100.0, 50.0, 500.0, 850.0]))
    assert det.points.shape == (9, 3) and det.keypoints.shape == (37, 2)
    assert tr._model.calls[0] == [[100.0, 50.0, 500.0, 850.0]]
    assert CHAIN_BOTTOM_UP == tuple(reversed(SPINE_LABELS))
    ys = det.points[:, 1]
    assert np.all(np.diff(ys) > 0)                                        # C1 (top, smallest y) ... SACRUM (bottom)
    assert det.points[0, 1] < det.points[-1, 1]
    sp = build_spine_from_points(det.points[:, :2], 1, det.points[:, 2])
    assert sp is not None and sp.thoracic_deg > 3.0                       # the detected curve survives end to end


def test_wrapper_rejects_a_model_output_that_is_too_short():
    class Short:
        def __call__(self, image, bboxes):
            return np.zeros((1, 10, 2)), np.zeros((1, 10))
    tracker = SpinePoseTracker(model=Short())
    assert tracker.detect(np.zeros((10, 10, 3), np.uint8), np.array([0.0, 0.0, 9.0, 9.0])) is None


def test_person_bbox_and_bridge_error():
    img, _ = build_landmarks_3d(BOTTOM, yaw_deg=90, cx=640, floor_y=700)
    box = person_bbox(img, (1280, 720))
    seen = img[img[:, 2] > 0]
    assert box[0] < seen[:, 0].min() and box[2] > seen[:, 0].max() and box[3] <= 720
    assert person_bbox(np.zeros((33, 3)), (1280, 720)) is None
    k = np.zeros((37, 2))
    k[5], k[6] = img[11, :2], img[12, :2]
    k[11], k[12] = img[23, :2], img[24, :2]
    ok = SpineDetection(points=np.zeros((9, 3)), keypoints=k, scores=np.ones(37))
    assert bridge_error(ok, img) == pytest.approx(0.0, abs=1e-6)
    k2 = k + np.array([400.0, 0.0])                                       # SpinePose boxed someone else
    assert bridge_error(SpineDetection(np.zeros((9, 3)), k2, np.ones(37)), img) > 0.35


def test_missing_package_gives_a_clear_install_message():
    if spinepose_available():
        pytest.skip("spinepose is installed here")
    with pytest.raises(SpinePoseUnavailable) as e:
        SpinePoseTracker()
    assert "pip install" in str(e.value) and "--no-deps" in str(e.value)


def test_demo_refuses_a_demanded_spinepose_but_auto_falls_back_to_the_contour(capsys):
    if spinepose_available():
        pytest.skip("spinepose is installed here")
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "demo_deadlift.py")
    spec = importlib.util.spec_from_file_location("demo_spine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parse_args(["--source", "x.mp4", "--spine", "spinepose"])
    with pytest.raises(SystemExit) as e:
        mod.resolve_spine(args)
    assert "SpinePose" in str(e.value)
    auto = mod.resolve_spine(mod.parse_args(["--source", "x.mp4"]))
    assert auto == ("contour", None)                                       # auto: the contour, with a notice
    assert "back silhouette" in capsys.readouterr().out
    assert mod.resolve_spine(mod.parse_args(["--source", "x.mp4", "--spine", "proxy"])) == ("proxy", None)
    assert mod.resolve_spine(mod.parse_args(["--synthetic", "good"])) == ("proxy", None)
    assert mod.resolve_spine(mod.parse_args(["--synthetic", "good", "--spine", "contour"])) == ("contour", None)


# ---------------------------------------------------------------------------
# Proxy spine cues are opt-in in the demo (they false-alarmed on real footage)
# ---------------------------------------------------------------------------
def _demo_module():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "demo_deadlift.py")
    spec = importlib.util.spec_from_file_location("demo_proxy_opt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_proxy_rounding_cue_can_be_switched_off_but_the_spine_is_still_computed():
    from myomotion.biomechanics import DeadliftRuleConfig
    spec = ClipSpec(n_reps=1, noise_px=0.5)
    an = DeadliftAnalyzer(rule_cfg=DeadliftRuleConfig(proxy_spine_cues=False))
    seen_delta = 0.0
    for t, img, w in generate_clip_3d("lumbar_round", spec, Body(), yaw_deg=0.0):
        r = an.process(t, img, world=w)
        if r.spine_delta:
            seen_delta = max(seen_delta, r.spine_delta.lumbar_deg)
    assert an.summary()["flag_counts"]["lumbar_rounding"] == 0 and seen_delta > 4.0     # shown, not alarmed


def test_demo_defaults_to_no_proxy_cues_and_opts_in_with_a_flag(tmp_path):
    demo = _demo_module()
    assert demo.load_rule_config(None).proxy_spine_cues is False
    assert demo.load_rule_config(None, proxy_cues=True).proxy_spine_cues is True
    assert demo.parse_args(["--synthetic", "good"]).proxy_spine_cues is False
    assert demo.parse_args(["--synthetic", "good", "--proxy-spine-cues"]).proxy_spine_cues is True
