"""Tests for src/myomotion/biomechanics (deadlift demo module).

All inputs are synthetic landmark clips with known ground truth (``biomechanics/synthetic.py``): no camera,
model or video needed. They prove the *logic* of the angle maths, spine proxy, segmenter and rules. They do
not prove real-world sensitivity: that needs labelled gym footage (docs/GAP_ANALYSIS.md G-23/G-24).

Run:  python -m pytest tests/ -q
"""

import importlib.util
import json
import math
import os
import random

import numpy as np
import pytest

from myomotion.biomechanics import (Bias, DeadliftAnalyzer, DeadliftFormAnalyzer, DeadliftRepSegmenter,
                                    DeadliftRuleConfig, FormFlag, KeypointSmoother, Phase, SpineBaseline,
                                    build_spine_proxy, classify_bias, compute_deadlift_angles,
                                    compute_frontal_angles, normalize_trajectory)
from myomotion.biomechanics.deadlift_angles import knee_valgus_deg, line_tilt_deg
from myomotion.biomechanics.deadlift_rules import _Latch
from myomotion.biomechanics.geometry import inclination_from_vertical, joint_angle, wrap_deg
from myomotion.biomechanics.landmarks import SIDE_INDICES, SidePose, extract_side_pose
from myomotion.biomechanics.rep_segmenter import BarHalfCycleDetector
from myomotion.biomechanics.spine_proxy import ARC_FRACTION, SPINE_LABELS
from myomotion.biomechanics.synthetic import (SCENARIOS, Body, ClipSpec, JointPose, Scenario, build_landmarks,
                                              clip_list, rep_truth, smoothstep)

P = Phase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def analyze(scenario="good", cfg=None, **spec_kw):
    spec = ClipSpec(**{"n_reps": 1, **spec_kw})
    an = DeadliftAnalyzer(rule_cfg=cfg)
    results = [an.process(t, lm) for t, lm in clip_list(scenario, spec)]
    return an, an.summary(), results


def flags_of(summary):
    return {k for k, v in summary["flag_counts"].items() if v}


def phase_names(an):
    return [new for _, _, new in an.segmenter.transitions]


def pose_at(kappa=60.0, tau=15.0, theta=50.0, bend=5.0, drift=0.0, facing=1):
    return build_landmarks(JointPose(kappa, tau, theta, bend, drift), facing=facing)


# ---------------------------------------------------------------------------
# Geometry + angle engine
# ---------------------------------------------------------------------------
def test_geometry_conventions():
    assert math.isclose(joint_angle((1, 0), (0, 0), (0, 1)), 90.0)
    assert math.isclose(joint_angle((1, 0), (0, 0), (-1, 0)), 180.0)
    assert math.isnan(joint_angle((0, 0), (0, 0), (1, 0)))
    # image y points down: a segment going up-right is forward-leaning when facing right
    assert math.isclose(inclination_from_vertical((0, 0), (0, -1), +1), 0.0, abs_tol=1e-9)
    assert math.isclose(inclination_from_vertical((0, 0), (1, -1), +1), 45.0)
    assert math.isclose(inclination_from_vertical((0, 0), (1, -1), -1), -45.0)     # same segment, facing left
    assert math.isclose(inclination_from_vertical((0, 0), (-1, -1), -1), 45.0)
    assert math.isclose(wrap_deg(190.0), -170.0)


@pytest.mark.parametrize("facing", [1, -1])
def test_sagittal_angles_match_ground_truth(facing):
    """Hip/knee/trunk/tibia recovered exactly from a constructed pose; invariant to facing direction."""
    a = compute_deadlift_angles(pose_at(60, 15, 50, facing=facing), 0.0)
    assert a.facing == facing
    assert math.isclose(a.knee_deg, 120.0, abs_tol=1e-6)               # 180 - kappa
    assert math.isclose(a.knee_flexion_deg, 60.0, abs_tol=1e-6)
    assert math.isclose(a.hip_deg, 85.0, abs_tol=1e-6)                 # theta - (tau - kappa + 180) wrapped
    assert math.isclose(a.hip_signed_deg, 85.0, abs_tol=1e-6)
    assert math.isclose(a.trunk_incl_deg, 50.0, abs_tol=1e-6)
    assert math.isclose(a.tibia_incl_deg, 15.0, abs_tol=1e-6)
    assert math.isclose(a.trunk_tibia_deg, 35.0, abs_tol=1e-6)         # [P8] differential


def test_standing_tall_is_straight_and_vertical():
    a = compute_deadlift_angles(pose_at(0, 0, 0), 0.0)
    assert math.isclose(a.hip_deg, 180.0, abs_tol=1e-6) and math.isclose(a.knee_deg, 180.0, abs_tol=1e-6)
    assert abs(a.trunk_incl_deg) < 1e-6 and abs(a.tibia_incl_deg) < 1e-6 and abs(a.trunk_tibia_deg) < 1e-6


def test_signed_hip_exceeds_180_when_trunk_leans_back():
    """The interior angle cannot tell lean-back from flexion; the signed angle can (hyperextension at lockout)."""
    a = compute_deadlift_angles(pose_at(0, 0, -14), 0.0)
    assert math.isclose(a.hip_deg, 166.0, abs_tol=1e-6)                # interior reads "flexed"
    assert math.isclose(a.hip_signed_deg, 194.0, abs_tol=1e-6)         # signed reads "extended past straight"
    assert math.isclose(a.trunk_incl_deg, -14.0, abs_tol=1e-6)


def test_bar_height_and_drift_from_wrist():
    low = compute_deadlift_angles(pose_at(60, 15, 50, drift=0.0), 0.0)
    high = compute_deadlift_angles(pose_at(0, 0, 0, drift=0.0), 0.0)
    assert low.bar_source == "wrist" and abs(low.bar_drift_foot) < 1e-9
    assert high.bar_height_torso > low.bar_height_torso                 # bar rises during the pull
    for facing in (1, -1):
        d = compute_deadlift_angles(pose_at(drift=0.5, facing=facing), 0.0)
        assert math.isclose(d.bar_drift_foot, 0.5, abs_tol=1e-6)        # forward drift is + for either facing


def test_detector_bar_overrides_wrist():
    lm = pose_at()
    a = compute_deadlift_angles(lm, 0.0, bar_xy=np.array([123.0, 456.0]))
    assert a.bar_source == "bar" and np.allclose(a.bar_xy, [123.0, 456.0])


def test_unreliable_landmarks_give_none():
    lm = pose_at()
    lm[SIDE_INDICES["right"]["knee"], 2] = 0.2                          # low visibility
    assert compute_deadlift_angles(lm, 0.0, side="right") is None
    lm2 = pose_at()
    lm2[SIDE_INDICES["right"]["hip"], 0] = np.nan
    assert compute_deadlift_angles(lm2, 0.0, side="right") is None
    with pytest.raises(ValueError):
        extract_side_pose(np.zeros((10, 3)))


def test_side_selection_picks_visible_side():
    lm = pose_at()                                                      # right side visible by construction
    assert extract_side_pose(lm).side == "right"
    flipped = lm.copy()
    right, left = SIDE_INDICES["right"], SIDE_INDICES["left"]
    for k in right:
        flipped[[right[k], left[k]]] = flipped[[left[k], right[k]]]
    assert extract_side_pose(flipped).side == "left"


def test_frontal_angles():
    lm = np.zeros((33, 3))
    pts = {11: (300, 200), 12: (500, 200), 23: (330, 420), 24: (470, 420), 25: (325, 560), 26: (475, 560),
           27: (320, 700), 28: (480, 700)}
    for i, (x, y) in pts.items():
        lm[i] = (x, y, 0.99)
    f = compute_frontal_angles(lm)
    assert abs(f.shoulder_tilt_deg) < 1e-9 and abs(f.hip_tilt_deg) < 1e-9
    assert abs(f.knee_valgus_left_deg) < 1.0 and abs(f.knee_valgus_right_deg) < 1.0
    lm[12] = (500, 240, 0.99)                                           # image-right shoulder 40 px lower
    assert math.isclose(compute_frontal_angles(lm).shoulder_tilt_deg, math.degrees(math.atan2(40, 200)), abs_tol=1e-6)
    # knees collapse toward the midline -> positive (valgus) on both sides
    lm[25] = (380, 560, 0.99)
    lm[26] = (420, 560, 0.99)
    g = compute_frontal_angles(lm)
    assert g.knee_valgus_left_deg > 5.0 and g.knee_valgus_right_deg > 5.0
    # and bowing outward -> negative (varus)
    lm[25] = (270, 560, 0.99)
    assert compute_frontal_angles(lm).knee_valgus_left_deg < -5.0
    lm[27, 2] = 0.1
    assert compute_frontal_angles(lm) is None
    assert math.isclose(line_tilt_deg(np.array([0.0, 0.0]), np.array([-10.0, 0.0])), 0.0, abs_tol=1e-9)
    origin = np.array([0.0, 0.0])
    assert math.isnan(knee_valgus_deg(origin, origin, origin, np.array([1.0, 0.0])))


# ---------------------------------------------------------------------------
# Keypoint smoothing
# ---------------------------------------------------------------------------
def test_keypoint_smoother_kills_jitter_on_a_static_pose():
    rng = random.Random(0)
    base = pose_at()
    sm = KeypointSmoother()
    raw_err, sm_err = [], []
    for i in range(240):
        noisy = base.copy()
        noisy[:, 0] += [rng.gauss(0, 3.0) for _ in range(33)]
        noisy[:, 1] += [rng.gauss(0, 3.0) for _ in range(33)]
        out = sm.update(i / 30.0, noisy, scale_px=250.0)
        if i >= 60:
            idx = SIDE_INDICES["right"]["hip"]
            raw_err.append(noisy[idx, :2] - base[idx, :2])
            sm_err.append(out[idx, :2] - base[idx, :2])
    assert np.std(sm_err) < 0.4 * np.std(raw_err)                       # >2.5x jitter reduction at rest


def test_keypoint_smoother_tracks_motion_with_small_lag():
    sm = KeypointSmoother()
    lm = np.zeros((33, 3))
    lm[:, 2] = 0.99
    lags = []
    for i in range(120):
        t = i / 30.0
        lm[:, 0] = 100 + 150.0 * t                                      # 150 px/s ~ 0.6 torso/s
        out = sm.update(t, lm, scale_px=250.0)
        if i > 60:
            lags.append((lm[0, 0] - out[0, 0]) / 150.0)
    assert 0.0 <= np.mean(lags) < 0.15                                  # under 150 ms while moving


def test_keypoint_smoother_edge_cases():
    sm = KeypointSmoother()
    lm = pose_at()
    sm.update(1.0, lm, 250.0)
    assert np.array_equal(sm.update(0.9, lm + 50, 250.0), lm + 50)      # non-monotonic: passthrough
    low = lm.copy()
    low[5] = (999.0, 999.0, 0.1)                                        # unreliable landmark: untouched
    assert np.allclose(sm.update(1.1, low, 250.0)[5], low[5])
    jump = lm.copy()
    jump[:, :2] += 300.0
    out = sm.update(5.0, jump, 250.0)                                   # 4 s gap: state reset, no dragging
    assert np.allclose(out[:, :2][lm[:, 2] > 0.5], jump[:, :2][lm[:, 2] > 0.5])
    with pytest.raises(ValueError):
        sm.update(6.0, np.zeros((5, 3)), 250.0)


# ---------------------------------------------------------------------------
# Spine proxy [P1 heuristic]
# ---------------------------------------------------------------------------
def spine_pose(ear, sh=(0, -250), hip=(0, 0), facing=1):
    pts = {"ear": np.array(ear, float), "shoulder": np.array(sh, float), "hip": np.array(hip, float)}
    return SidePose("right", facing, pts, {k: 1.0 for k in pts})


def test_arc_fractions_follow_vertebral_counts():
    assert SPINE_LABELS == ("C1", "C4", "C7", "T3", "T8", "L1", "L3", "L5", "SACRUM")
    assert ARC_FRACTION["C1"] == 0.0 and ARC_FRACTION["SACRUM"] == 1.0
    assert math.isclose(ARC_FRACTION["C7"], 6 / 24) and math.isclose(ARC_FRACTION["T3"], 9 / 24)
    assert math.isclose(ARC_FRACTION["T8"], 14 / 24) and math.isclose(ARC_FRACTION["L1"], 19 / 24)
    assert all(a < b for a, b in zip(ARC_FRACTION.values(), list(ARC_FRACTION.values())[1:]))


def test_straight_spine_has_no_curvature_and_even_spacing():
    sp = build_spine_proxy(spine_pose(ear=(0, -320)))
    assert sp.source == "proxy"
    angles = (sp.bend_deg, sp.cervical_deg, sp.thoracic_deg, sp.lumbar_deg, sp.pelvic_deg)
    assert max(abs(a) for a in angles) < 1e-6
    ys = [sp.points[k][1] for k in SPINE_LABELS]                         # C1 .. SACRUM top to bottom
    assert all(a < b for a, b in zip(ys, ys[1:]))
    # thoracolumbar points sit at their vertebral-count fractions of the 250 px trunk
    assert math.isclose(sp.points["T8"][1], -250 * (1 - (ARC_FRACTION["T8"] - 0.25) / 0.75), abs_tol=0.5)
    assert sp.as_array().shape == (9, 2)


@pytest.mark.parametrize("facing", [1, -1])
def test_forward_head_gives_positive_flexion_split_by_region(facing):
    ear = (24 * facing, -316)                                            # head forward of the trunk line
    sp = build_spine_proxy(spine_pose(ear, facing=facing))
    assert sp.bend_deg > 15
    assert sp.cervical_deg > sp.thoracic_deg > sp.lumbar_deg > 0
    assert math.isclose(sp.thoracic_deg / sp.bend_deg, 0.555, abs_tol=0.05)    # (T3->L1 share of the arc)
    assert math.isclose(sp.lumbar_deg / sp.bend_deg, 0.278, abs_tol=0.04)      # (L1->sacrum share)
    back = build_spine_proxy(spine_pose((-24 * facing, -316), facing=facing))
    assert math.isclose(back.lumbar_deg, -sp.lumbar_deg, abs_tol=1e-6)         # head back = extension, symmetric


def test_degenerate_or_folded_anchors_give_none():
    assert build_spine_proxy(spine_pose(ear=(0, -250))) is None                # ear on the shoulder
    assert build_spine_proxy(spine_pose(ear=(0, -150))) is None                # neck folded back down the trunk
    assert build_spine_proxy(spine_pose(ear=(0, -320), sh=(0, 0))) is None     # shoulder on the hip


def test_spine_baseline_is_relative_and_needs_samples():
    base = SpineBaseline(min_samples=5)
    neutral = build_spine_proxy(spine_pose((10, -318)))
    for _ in range(4):
        base.add(neutral)
    assert not base.ready and base.delta(neutral) is None
    base.add(neutral)
    d = base.delta(neutral)
    assert base.ready and abs(d.lumbar_deg) < 1e-9 and abs(d.bend_deg) < 1e-9
    rounded = build_spine_proxy(spine_pose((60, -310)))
    assert base.delta(rounded).lumbar_deg > 3.0
    base.reset()
    assert not base.ready


# ---------------------------------------------------------------------------
# Rep segmentation
# ---------------------------------------------------------------------------
def feed_segmenter(f, total, fps=30.0, noise=1.0, seed=0, **kw):
    seg = DeadliftRepSegmenter(**kw)
    rng = random.Random(seed)
    for i in range(int(total * fps)):
        t = i / fps
        hip, knee = f(t)
        seg.update(t, hip + rng.gauss(0, noise), knee + rng.gauss(0, noise))
    return seg


def hip_knee_profile(bottom=85.0, top=178.0, kb=118.0, kt=178.0, pre=1.0, up=2.0, hold=0.8, down=2.0):
    def f(t):
        if t < pre:
            s = 0.0
        elif t < pre + up:
            s = smoothstep((t - pre) / up)
        elif t < pre + up + hold:
            s = 1.0
        elif t < pre + up + hold + down:
            s = 1.0 - smoothstep((t - pre - up - hold) / down)
        else:
            s = 0.0
        return bottom + s * (top - bottom), kb + s * (kt - kb)
    return f, pre + up + hold + down + 1.5


def test_phase_order_and_rep_record_for_one_rep():
    f, total = hip_knee_profile()
    seg = feed_segmenter(f, total)
    assert [n for _, _, n in seg.transitions] == [P.SETUP, P.CONCENTRIC_PULL, P.LOCKOUT, P.ECCENTRIC_DESCENT,
                                                  P.REP_COMPLETE, P.SETUP]
    assert seg.rep_count == 1 and len(seg.reps) == 1
    r = seg.reps[0]
    assert r.t_concentric_start_s < r.t_lockout_s < r.t_descent_start_s < r.t_end_s
    assert math.isclose(r.concentric_s + r.lockout_s + r.eccentric_s, r.t_end_s - r.t_concentric_start_s, abs_tol=1e-6)
    assert 0.6 < r.tempo_ratio < 1.6                                    # symmetric 2 s up / 2 s down
    assert r.peak_hip_deg > 170 and r.duration_s > 2.0


def test_rep_complete_is_transient_and_can_be_disabled():
    f, total = hip_knee_profile()
    seg = feed_segmenter(f, total)
    t_rc = next(t for t, _, n in seg.transitions if n == P.REP_COMPLETE)
    t_after = next(t for t, o, n in seg.transitions if o == P.REP_COMPLETE)
    assert 0.35 <= t_after - t_rc <= 0.5
    legacy = feed_segmenter(f, total, emit_rep_complete=False)
    assert P.REP_COMPLETE not in [n for _, _, n in legacy.transitions] and legacy.rep_count == 1


def test_multiple_reps_counted_and_timed():
    f1, total1 = hip_knee_profile(pre=1.0)
    n = 3

    def f(t):
        return f1(t % total1) if t >= 0 else f1(0)

    seg = feed_segmenter(f, total1 * n)
    assert seg.rep_count == n and [r.rep_index for r in seg.reps] == [1, 2, 3]


def test_liftoff_estimate_close_to_truth():
    spec = ClipSpec(n_reps=2, noise_px=1.0)
    an, summary, _ = analyze("good", n_reps=2, noise_px=1.0)
    for i, r in enumerate(summary["reps"]):
        truth = rep_truth(spec, i)["liftoff"]
        # causal + filtered, so it trails the true start by a few hundred ms but never precedes it by much
        assert truth - 0.1 <= r["t_liftoff_s"] <= truth + 0.7, (i, truth, r["t_liftoff_s"])


def test_failed_rep_without_lockout_is_not_counted():
    def f(t):
        if t < 1.0:
            return 85.0, 118.0
        if t < 2.5:
            return 85 + 40 * smoothstep((t - 1.0) / 1.5), 130.0
        if t < 3.0:
            return 125.0, 130.0
        if t < 4.5:
            return 125 - 40 * smoothstep((t - 3.0) / 1.5), 130.0
        return 85.0, 118.0
    seg = feed_segmenter(f, 6.5)
    assert seg.rep_count == 0 and seg.reps == []


def test_hyperextended_lockout_still_reaches_lockout_with_signed_hip():
    """Regression: the interior hip angle falls when the trunk leans back, which looked like a descent."""
    an, summary, _ = analyze("hyperextension")
    assert P.LOCKOUT in phase_names(an)
    assert summary["reps"][0]["peak_hip_deg"] > 185.0                   # peak lockout extension reads past straight


def test_bar_half_cycle_detector_and_normalisation():
    det = BarHalfCycleDetector(prominence=0.35)
    heights = [0.0] * 10 + list(np.linspace(0, 1.2, 40)) + [1.2] * 10 + list(np.linspace(1.2, 0, 40)) + [0.0] * 10
    events = [det.update(i / 30.0, h) for i, h in enumerate(heights)]
    kinds = [e[0] for e in events if e]
    assert kinds == ["valley", "peak"]                                   # rising half-cycle then falling half-cycle
    valley, peak = det.extrema
    assert math.isclose(valley[2], 0.0, abs_tol=1e-9) and math.isclose(peak[2], 1.2, abs_tol=1e-9)
    # jitter smaller than the prominence never creates extrema
    quiet = BarHalfCycleDetector(prominence=0.35)
    for i in range(200):
        quiet.update(i / 30.0, 0.05 * math.sin(i))
    assert quiet.extrema == []
    n = normalize_trajectory(np.array([2.0, 3.0, 4.0]))
    assert n.min() == -180.0 and n.max() == 180.0
    assert np.all(normalize_trajectory(np.ones(5)) == 0)


def test_rep_record_gets_bar_peak_from_trajectory():
    _, summary, _ = analyze("good", n_reps=2)
    for r in summary["reps"]:
        assert r["bar_peak_t_s"] is not None and r["bar_rise_torso"] > 0.35
        assert r["t_concentric_start_s"] < r["bar_peak_t_s"] < r["t_descent_start_s"] + 0.5


# ---------------------------------------------------------------------------
# Hip / knee dominance [P8]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value, expected", [
    (35.0, Bias.HIP_EXTENSOR), (10.01, Bias.HIP_EXTENSOR), (10.0, Bias.BALANCED), (0.0, Bias.BALANCED),
    (-10.0, Bias.BALANCED), (-10.01, Bias.KNEE_EXTENSOR), (-30.0, Bias.KNEE_EXTENSOR)])
def test_bias_thresholds_match_p8(value, expected):
    assert classify_bias(value) == expected


def test_bias_on_reps_reflects_posture_at_liftoff():
    _, summary, _ = analyze("good")
    form = summary["reps"][0]["form"]
    assert form["bias"] == "hip_extensor" and math.isclose(form["trunk_tibia_at_liftoff_deg"], 35.0, abs_tol=2.0)


# ---------------------------------------------------------------------------
# Form rules on whole reps
# ---------------------------------------------------------------------------
def test_good_rep_has_no_flags_and_form_ok():
    an, summary, results = analyze("good", noise_px=1.5)
    assert summary["rep_count"] == 1 and flags_of(summary) == set() and summary["events"] == []
    assert summary["reps"][0]["form"]["form_ok"] is True
    assert max(r.spine_delta.lumbar_deg for r in results if r.spine_delta) < 2.0


def test_lumbar_rounding_flagged_at_pull_initiation_with_proxy_confidence():
    an, summary, _ = analyze("lumbar_round", noise_px=1.5)
    assert flags_of(summary) == {"lumbar_rounding"}
    ev = summary["events"][0]
    assert ev["confidence"] == "proxy" and ev["severity"] == "danger" and ev["rep_index"] == 1
    assert ev["phase"] in ("Setup", "Concentric Pull")
    assert ev["value"] >= an.rules.cfg.lumbar_enter_deg
    assert summary["reps"][0]["form"]["flags"] == ["lumbar_rounding"]


def test_butt_wink_on_the_way_down_belongs_to_the_rep_it_came_from():
    an, summary, _ = analyze("butt_wink", n_reps=2)
    assert [e["rep_index"] for e in summary["events"]] == [1, 2]       # no phantom rep 3
    assert {e["phase"] for e in summary["events"]} == {"Eccentric Descent"}
    assert all(r["form"]["flags"] == ["lumbar_rounding"] for r in summary["reps"])


def test_rounded_setup_is_invisible_to_the_proxy():
    """Documented limitation (G-24): everything is relative to the athlete's own set-up baseline."""
    always_rounded = Scenario("rounded_setup", neutral_bend_deg=35.0)
    _, summary, _ = analyze(always_rounded)
    assert "lumbar_rounding" not in flags_of(summary)


@pytest.mark.parametrize("amount, expect_flag", [(6.0, False), (10.0, False), (20.0, True), (30.0, True)])
def test_lumbar_threshold_boundary(amount, expect_flag):
    _, summary, _ = analyze(Scenario("r", round_deg=amount, round_on=("pull",)))
    assert ("lumbar_rounding" in flags_of(summary)) is expect_flag


def test_hips_rising_early_flagged_only_for_stripper_pull():
    _, bad, _ = analyze("stripper", noise_px=1.5)
    assert flags_of(bad) == {"hips_rise_early"}
    ev = bad["events"][0]
    assert ev["phase"] == "Concentric Pull" and ev["confidence"] == "direct" and ev["severity"] == "warning"
    assert ev["value"] >= DeadliftRuleConfig().early_knee_extension_deg
    _, good, _ = analyze("good", noise_px=1.5)
    assert "hips_rise_early" not in flags_of(good)


def test_lockout_hyperextension_flagged_with_measured_lean():
    _, summary, _ = analyze("hyperextension", noise_px=1.5)
    assert flags_of(summary) == {"lockout_hyperextension"}
    form = summary["reps"][0]["form"]
    assert form["min_lockout_trunk_deg"] < -10.0 and form["flags"] == ["lockout_hyperextension"]
    ev = summary["events"][0]
    assert ev["value"] >= DeadliftRuleConfig().lockout_backlean_enter_deg


def test_bar_drift_flagged_and_measured_in_foot_lengths():
    _, summary, _ = analyze("bar_drift", noise_px=1.5)
    assert flags_of(summary) == {"bar_drift"}
    assert 0.5 < summary["reps"][0]["form"]["max_bar_drift_foot"] < 0.7
    assert summary["events"][0]["confidence"] == "proxy"               # wrist stands in for the bar


def test_config_override_changes_behaviour():
    strict = DeadliftRuleConfig().with_overrides(lumbar_enter_deg=20.0, lumbar_exit_deg=15.0)
    _, summary, _ = analyze("lumbar_round", cfg=strict)
    assert "lumbar_rounding" not in flags_of(summary)
    _, summary, _ = analyze("bar_drift", cfg=DeadliftRuleConfig().with_overrides(bar_drift_enter_foot=0.9))
    assert "bar_drift" not in flags_of(summary)


def test_event_raised_once_per_rep_but_flag_stays_active():
    cfg = DeadliftRuleConfig()
    _, summary, results = analyze("lumbar_round", n_reps=2)
    assert len(summary["events"]) == 2                                   # one per rep, not one per frame
    assert any(FormFlag.LUMBAR_ROUNDING in r.active_flags for r in results)
    _, many, _ = analyze("lumbar_round", cfg=cfg.with_overrides(once_per_rep=False), n_reps=1)
    assert len(many["events"]) >= 1


def test_latch_hysteresis_and_dwell():
    latch = _Latch(enter=10.0, exit_=6.0, dwell_s=0.2)
    clock = [0.0]

    def step(value, enabled=True):
        clock[0] += 0.05
        return latch.update(clock[0], value, enabled)

    # flicker around the enter threshold never fires: the exceedance is never continuous for 0.2 s
    assert not any(step(v) for v in [11, 5, 11, 5, 11, 5, 11, 8, 11, 5])
    assert not latch.active
    fired = [step(11) for _ in range(6)]                 # 0.3 s of continuous exceedance
    assert sum(fired) == 1 and latch.active              # fires exactly once, then stays on
    assert not step(8.0) and latch.active                # between exit and enter: hysteresis keeps it on
    assert not step(5.0) and not latch.active            # below exit: off
    assert not step(None) and not latch.active           # missing data never raises
    assert not any(step(20.0) for _ in range(2))         # ...and a fresh exceedance needs the dwell again
    assert not step(20.0, enabled=False) and not latch.active   # disabled phase: always reset


def test_rules_analyzer_standalone_api():
    """The rules can be driven directly without the pipeline (as the future cue engine / tests will)."""
    rules = DeadliftFormAnalyzer()
    seg = DeadliftRepSegmenter()
    a = compute_deadlift_angles(pose_at(), 0.0)
    assert rules.update(a, P.SETUP, None) == []
    assert rules.current_rep_index == 1 and rules.active == set()
    seg_rep = None
    f, total = hip_knee_profile()
    for i in range(int(total * 30)):
        seg.update(i / 30.0, *f(i / 30.0))
    seg_rep = seg.reps[0]
    form = rules.finalize_rep(seg_rep)
    assert form.rep_index == 1 and form.form_ok and rules.current_rep_index == 2


# ---------------------------------------------------------------------------
# Robustness: noise, frame rate, mirroring, dropouts
# ---------------------------------------------------------------------------
EXPECTED_FLAGS = {"good": set(), "lumbar_round": {"lumbar_rounding"}, "butt_wink": {"lumbar_rounding"},
                  "stripper": {"hips_rise_early"}, "hyperextension": {"lockout_hyperextension"},
                  "bar_drift": {"bar_drift"}}


@pytest.mark.parametrize("scenario", sorted(EXPECTED_FLAGS))
@pytest.mark.parametrize("fps", [15, 30, 60])
@pytest.mark.parametrize("facing", [1, -1])
def test_scenarios_robust_to_fps_noise_and_mirroring(scenario, fps, facing):
    for noise, seed in [(1.0, 0), (3.0, 1)]:
        _, summary, _ = analyze(scenario, n_reps=2, fps=fps, noise_px=noise, seed=seed, facing=facing)
        assert summary["rep_count"] == 2, (noise, seed)
        assert flags_of(summary) == EXPECTED_FLAGS[scenario], (noise, seed)


def test_no_false_positives_on_good_reps_across_many_seeds():
    for seed in range(12):
        _, summary, _ = analyze("good", n_reps=2, noise_px=3.0, seed=seed)
        assert summary["rep_count"] == 2 and flags_of(summary) == set(), seed


def test_short_pose_dropout_is_survived():
    spec = ClipSpec(n_reps=1, noise_px=1.0)
    an = DeadliftAnalyzer()
    for t, lm in clip_list("good", spec):
        an.process(t, None if 3.4 <= t < 3.7 else lm)                    # 0.3 s of lost pose mid-lockout
    s = an.summary()
    assert s["rep_count"] == 1 and flags_of(s) == set()
    assert s["frames_tracked"] < s["frames_total"]


def test_long_dropout_resets_to_standby_and_discards_the_attempt():
    spec = ClipSpec(n_reps=1, noise_px=1.0)
    an = DeadliftAnalyzer()
    for t, lm in clip_list("lumbar_round", spec):
        an.process(t, None if 2.0 <= t < 4.5 else lm)                    # 2.5 s lost during the pull
    assert P.STANDBY in phase_names(an)
    s = an.summary()
    assert all(r["form"]["flags"] == [] or r["rep_index"] >= 1 for r in s["reps"])
    assert an.rules.rep_forms == [] or len(an.rules.rep_forms) == s["rep_count"]


def test_standing_still_never_counts_or_flags():
    an = DeadliftAnalyzer()
    lm = pose_at(0, 0, 0)
    for i in range(300):
        an.process(i / 30.0, lm)
    s = an.summary()
    assert s["rep_count"] == 0 and flags_of(s) == set() and an.segmenter.phase == P.STANDBY


def test_never_crashes_on_missing_pose_only():
    an = DeadliftAnalyzer()
    for i in range(60):
        r = an.process(i / 30.0, None)
        assert r.angles is None and r.phase == P.STANDBY
    assert an.summary()["frames_tracked"] == 0


# ---------------------------------------------------------------------------
# Summary / JSON
# ---------------------------------------------------------------------------
def test_summary_is_strict_json_and_carries_the_proxy_disclaimer():
    _, summary, _ = analyze("lumbar_round", n_reps=2)
    text = json.dumps(summary, allow_nan=False)                         # raises if any NaN/inf slipped through
    back = json.loads(text)
    assert back["schema"] == "myomotion.deadlift_session/1" and back["rep_count"] == 2
    assert any("proxy" in d for d in back["disclaimers"])
    assert set(back["flag_counts"]) == {f.value for f in FormFlag}
    r = back["reps"][0]
    for key in ("t_liftoff_s", "concentric_s", "eccentric_s", "tempo_ratio", "peak_hip_deg", "form"):
        assert key in r
    assert back["rule_config"]["bias_threshold_deg"] == 10.0


# ---------------------------------------------------------------------------
# Demo CLI (synthetic mode: no camera, no MediaPipe model)
# ---------------------------------------------------------------------------
def _load_demo():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "demo_deadlift.py")
    spec = importlib.util.spec_from_file_location("demo_deadlift", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_cli_writes_video_and_json(tmp_path, capsys):
    cv2 = pytest.importorskip("cv2")
    demo = _load_demo()
    rc = demo.main(["--synthetic", "lumbar_round", "--reps", "1", "--no-display", "--out-dir", str(tmp_path)])
    assert rc == 0
    video = tmp_path / "synthetic_lumbar_round_annotated.mp4"
    summary = json.loads((tmp_path / "synthetic_lumbar_round_summary.json").read_text())
    assert summary["rep_count"] == 1 and summary["flag_counts"]["lumbar_rounding"] == 1
    assert summary["meta"]["frames_written"] == summary["frames_total"]
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == summary["frames_total"]
    cap.release()
    assert "[FORM]" in capsys.readouterr().out


def test_demo_cli_rule_config_override(tmp_path):
    demo = _load_demo()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"lumbar_enter_deg": 25.0, "lumbar_exit_deg": 20.0}))
    demo.main(["--synthetic", "lumbar_round", "--reps", "1", "--no-display", "--no-video", "--out-dir", str(tmp_path),
               "--config", str(cfg)])
    summary = json.loads((tmp_path / "synthetic_lumbar_round_summary.json").read_text())
    assert summary["flag_counts"]["lumbar_rounding"] == 0 and summary["rule_config"]["lumbar_enter_deg"] == 25.0


def test_scenario_registry_is_complete():
    assert set(SCENARIOS) == set(EXPECTED_FLAGS)
    body = Body().scaled(0.5, floor_y=600.0, x0=300.0)
    assert math.isclose(body.torso, 125.0) and body.floor_y == 600.0
