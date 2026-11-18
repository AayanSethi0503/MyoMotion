"""Back-silhouette spine estimate (G-32): recovers a known back profile from a mask, copes with holes and outliers,
places the spine inside the silhouette near the back edge, and feeds the pipeline as source="contour".

Real-footage evidence (a recorded webcam run) is in docs/GAP_ANALYSIS.md G-32; these tests use synthetic silhouettes.
"""

import math

import numpy as np
import pytest

from myomotion.biomechanics import DeadliftAnalyzer, DeadliftRuleConfig
from myomotion.biomechanics.landmarks import SidePose
from myomotion.biomechanics.spine_contour import (SPINE_DEPTH_TORSO, T_C7, T_SACRUM, estimate_spine_from_mask,
                                                  extract_back_contour, spine_from_contour, synthetic_mask)
from myomotion.biomechanics.spine_proxy import SPINE_LABELS, build_spine_from_points
from myomotion.biomechanics.synthetic import Body, ClipSpec, generate_clip_3d_masked

SHAPE = (600, 800)
HIP = np.array([400.0, 450.0])


def silhouette(profile, facing=1, incline_deg=0.0, length=200.0, shape=SHAPE):
    """Trunk silhouette for a hip at HIP and a shoulder ``length`` px away, leaning ``incline_deg`` forward."""
    a = math.radians(incline_deg)
    shoulder = HIP + length * np.array([math.sin(a) * facing, -math.cos(a)])
    return synthetic_mask(shape, HIP, shoulder, facing, profile), shoulder


def flat(d=0.22):
    return lambda t: d


def bulge(amp, centre=0.55, width=0.35, base=0.22):
    return lambda t: base + amp * math.exp(-((t - centre) / width) ** 2)


@pytest.mark.parametrize("facing", [1, -1])
@pytest.mark.parametrize("incline", [0.0, 35.0, 60.0])
def test_back_profile_is_recovered_from_the_silhouette(facing, incline):
    profile = bulge(0.12)
    mask, shoulder = silhouette(profile, facing, incline)
    c = extract_back_contour(mask, HIP, shoulder, facing)
    assert c is not None and c.coverage > 0.95
    mid = (c.t > 0.1) & (c.t < 1.0)
    truth = np.array([profile(t) for t in c.t])
    assert np.max(np.abs(c.fit[mid] - truth[mid])) < 0.03                  # within 3% of the torso length
    assert c.residual < 0.02


def test_unusable_masks_return_none():
    mask, shoulder = silhouette(flat())
    assert extract_back_contour(np.zeros(SHAPE, np.float32), HIP, shoulder, 1) is None          # nobody there
    assert extract_back_contour(mask, HIP, shoulder, 0) is None                                  # unknown facing
    assert extract_back_contour(mask, HIP, HIP + [3.0, -3.0], 1) is None                         # degenerate trunk
    far = mask.copy()
    far[:, :] = 0
    far[:20, :20] = 1                                                                            # silhouette elsewhere
    assert extract_back_contour(far, HIP, shoulder, 1) is None


def test_a_small_hole_in_the_silhouette_is_ignored_but_the_back_edge_still_ends_the_body():
    profile = flat(0.25)
    mask, shoulder = silhouette(profile)
    ref = extract_back_contour(mask, HIP, shoulder, 1)
    holed = mask.copy()
    holed[250:256, 330:420] = 0                                           # a 6 px hole across the trunk (3% of 200 px)
    c = extract_back_contour(holed, HIP, shoulder, 1)
    assert c is not None and np.nanmax(np.abs(c.distance - ref.distance)) < 0.05


def test_outliers_from_clothing_are_rejected_by_the_robust_fit():
    profile = flat(0.22)
    mask, shoulder = silhouette(profile)
    flap = mask.copy()
    flap[300:312, 296:322] = 1.0                                          # a small cloth flap sticking out at the back
    ref = extract_back_contour(mask, HIP, shoulder, 1)
    c = extract_back_contour(flap, HIP, shoulder, 1)
    assert c is not None
    assert np.max(np.abs(c.fit - ref.fit)) < 0.05


def test_spine_lies_inside_the_silhouette_just_in_front_of_the_back_edge():
    profile = bulge(0.15)
    mask, shoulder = silhouette(profile, 1, 40.0)
    pose = SidePose("right", 1, {"hip": HIP, "shoulder": shoulder, "ear": shoulder + [30.0, -50.0]}, {})
    pts = estimate_spine_from_mask(mask, pose)
    assert pts is not None and pts.shape == (9, 3)
    ys, xs = np.round(pts[:, 1]).astype(int), np.round(pts[:, 0]).astype(int)
    trunk = [SPINE_LABELS.index(k) for k in ("C7", "T3", "T8", "L1", "L3", "L5", "SACRUM")]
    assert np.all(mask[ys[trunk], xs[trunk]] > 0.5)                        # trunk points are inside (no neck here)
    axis = (shoulder - HIP) / np.hypot(*(shoulder - HIP))
    back = np.array([-axis[1], axis[0]])
    if back[0] > 0:
        back = -back
    for label in ("T8", "L1", "L3"):
        p = pts[SPINE_LABELS.index(label), :2]
        t = float((p - HIP) @ axis) / 200.0
        d_behind = float((p - HIP) @ back) / 200.0                          # how far behind the chord this point is
        assert d_behind == pytest.approx(profile(t) - SPINE_DEPTH_TORSO, abs=0.04)


def test_label_levels_are_ordered_from_sacrum_to_c1_with_the_ear_as_c1():
    mask, shoulder = silhouette(flat())
    ear = shoulder + np.array([28.0, -55.0])
    contour = extract_back_contour(mask, HIP, shoulder, 1)
    pts = spine_from_contour(contour, ear, 1)
    assert np.allclose(pts[SPINE_LABELS.index("C1"), :2], ear)
    chord = (pts[0, :2] - pts[-1, :2])
    along = (pts[:, :2] - pts[-1, :2]) @ (chord / np.hypot(*chord))
    assert np.all(np.diff(along) < 0)                                      # C1 ... SACRUM strictly descending
    sacrum, c7 = pts[SPINE_LABELS.index("SACRUM"), :2], pts[SPINE_LABELS.index("C7"), :2]
    assert float(np.hypot(*(c7 - sacrum))) == pytest.approx((T_C7 - T_SACRUM) * 200.0, rel=0.1)
    assert 0.0 < pts[0, 2] <= 1.0 and np.allclose(pts[:, 2], pts[0, 2])
    assert spine_from_contour(contour, None, 1) is None                    # the ear is needed for the cervical part


def test_a_more_curved_back_gives_a_larger_measured_spine_curvature_and_the_angles_mirror():
    last = -1e9
    for amp in (0.0, 0.08, 0.16, 0.24):
        results = []
        for facing in (1, -1):
            mask, shoulder = silhouette(bulge(amp), facing, 30.0)
            ear = shoulder + np.array([20.0 * facing, -55.0])
            pose = SidePose("right", facing, {"hip": HIP, "shoulder": shoulder, "ear": ear}, {})
            pts = estimate_spine_from_mask(mask, pose)
            results.append(build_spine_from_points(pts[:, :2], facing, pts[:, 2], source="contour"))
        a, b = results
        assert a.source == "contour"
        assert a.thoracic_deg == pytest.approx(b.thoracic_deg, abs=1.0)
        assert a.lumbar_deg == pytest.approx(b.lumbar_deg, abs=1.0)
        total = a.thoracic_deg + a.lumbar_deg
        assert total > last                                                # monotone in the true bulge
        last = total


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------
def run_masked(scenario, yaw=0.0, cfg=None, reps=1, with_mask=True):
    shape = (480, 640)
    body = Body().scaled(0.5, floor_y=450.0, x0=320.0)
    spec = ClipSpec(n_reps=reps, noise_px=0.4)
    an = DeadliftAnalyzer(rule_cfg=cfg or DeadliftRuleConfig())
    sources = {}
    for t, img, world, mask in generate_clip_3d_masked(scenario, spec, body, yaw_deg=yaw, shape=shape, cx=320.0,
                                                       floor_y=450.0):
        r = an.process(t, img, world=world, mask=mask if with_mask else None)
        if r.angles and r.angles.spine:
            sources[r.angles.spine.source] = sources.get(r.angles.spine.source, 0) + 1
    return an, an.summary(), sources


def test_pipeline_uses_the_contour_when_side_on_and_a_mask_is_given():
    _, summary, sources = run_masked("good")
    assert sources.get("contour", 0) > 100 and "proxy" not in sources
    assert summary["frames_by_spine_source"].get("contour", 0) > 100 and summary["rep_count"] == 1


def test_pipeline_keeps_the_proxy_without_a_mask():
    _, _, sources = run_masked("good", with_mask=False)
    assert set(sources) == {"proxy"}


def test_front_view_never_uses_the_silhouette_for_curvature():
    _, summary, sources = run_masked("lumbar_round", yaw=90.0)
    assert "contour" not in sources                                       # sagittal contour is invisible from the front
    assert summary["flag_counts"]["lumbar_rounding"] == 0


def test_detected_spine_points_take_precedence_over_the_mask():
    an = DeadliftAnalyzer()
    body = Body().scaled(0.5, floor_y=450.0, x0=320.0)
    for t, img, world, mask in generate_clip_3d_masked("good", ClipSpec(n_reps=1), body, shape=(480, 640), cx=320.0,
                                                       floor_y=450.0):
        if t > 1.0:
            break
        pts = np.column_stack([np.linspace(0, 1, 9)[::-1] * 0 + img[8, 0], np.linspace(img[8, 1] - 160, img[8, 1], 9),
                               np.full(9, 0.9)])
        r = an.process(t, img, world=world, mask=mask, spine_points=pts)
    assert r.angles.spine.source == "spinepose"


def test_rounding_in_the_silhouette_is_flagged_with_contour_confidence_and_a_good_back_is_not():
    cfg = DeadliftRuleConfig(lumbar_enter_deg_contour=8.0, lumbar_exit_deg_contour=5.0)    # synthetic bulge is mild
    _, bad, _ = run_masked("lumbar_round", cfg=cfg)
    ev = [e for e in bad["events"] if e["flag"] == "lumbar_rounding"]
    assert ev and ev[0]["confidence"] == "contour" and "back contour" in ev[0]["message"]
    assert ev[0]["threshold"] == 8.0
    _, good, _ = run_masked("good", cfg=cfg)
    assert good["flag_counts"]["lumbar_rounding"] == 0


def test_default_contour_thresholds_are_the_conservative_ones():
    c = DeadliftRuleConfig()
    assert (c.lumbar_enter_deg_contour, c.lumbar_exit_deg_contour) == (15.0, 10.0)
    assert c.lumbar_enter_deg_contour > c.lumbar_enter_deg_model
