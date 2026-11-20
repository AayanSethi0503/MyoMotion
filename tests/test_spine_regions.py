"""Region angles from detected spine points (G-33): robustness to keypoint noise, to the clustered lumbar points real
SpinePose output has, and independence of body orientation (upright for deadlifts/squats, lying for the bench press).

The "real" chains below are raw SpinePose-small output (bottom to top: hip, sp01..sp05, neck, nk02, nk03) measured on
side-on photos from Wikimedia Commons (a dumbbell row, CC BY-SA 3.0, and a landmine row, CC BY 2.0). The first formula
used (directions of neighbouring segments) produced a lumbar angle of -114 deg on the dumbbell row.
"""

import math
import os

import numpy as np
import pytest

from myomotion.biomechanics.spine_proxy import SPINE_LABELS, build_spine_from_points
from myomotion.vision.spine_model import SpinePoseTracker, spinepose_available

U = np.array([0, .042, .125, .208, .417, .625, .75, .875, 1.0])           # vertebral-count fractions, SACRUM -> C1


def curved_chain(lum=0.0, tho=20.0, cer=5.0, length=224.0, noise=0.0, seed=0):
    """Points in SPINE_LABELS order (C1 ... SACRUM), image coordinates (y down), for a spine facing image-right whose
    tangent turns by ``lum`` over the lumbar region, ``tho`` over the thoracic and ``cer`` over the cervical
    (degrees)."""
    rng = np.random.default_rng(seed)
    knots = np.array([0, .208, .625, .75, 1.0])
    theta = lambda u: np.interp(u, knots, np.cumsum([0, lum, tho, tho * 0.3, cer]))       # noqa: E731
    uu = np.linspace(0, 1, 600)
    ang = np.radians(theta(uu))
    x = np.cumsum(np.sin(ang))
    y = np.cumsum(np.cos(ang))
    s = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])).cumsum()
    x, y, arc = x / s[-1] * length, y / s[-1] * length, s / s[-1]
    pts = np.column_stack([np.interp(U, arc, x), -np.interp(U, arc, y)])                # y up -> image y down
    pts += rng.normal(0, noise, pts.shape)
    return pts[::-1]


def rotate(pts, deg, about=(0.0, 0.0)):
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (np.asarray(pts) - about) @ r.T + about


DUMBBELL_ROW = np.array([[347, 129, .79], [348, 134, .76], [353, 127, .76], [366, 122, .71], [384, 103, .85],
                         [438, 91, .87], [460, 69, .87], [464, 63, 1.0], [473, 56, 1.17]], float)       # bottom to top
LANDMINE_ROW = np.array([[415, 330, .63], [395, 333, .67], [391, 330, .70], [370, 319, .81], [339, 301, .91],
                         [287, 277, .92], [265, 255, .85], [256, 265, .78], [229, 243, .85]], float)


def from_bottom_up(chain):
    return chain[::-1, :2], chain[::-1, 2]


# ---------------------------------------------------------------------------
# Real SpinePose output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chain, facing", [(DUMBBELL_ROW, 1), (LANDMINE_ROW, -1)])
def test_real_spinepose_chains_give_finite_plausible_angles(chain, facing):
    pts, sc = from_bottom_up(chain)
    sp = build_spine_from_points(pts, facing, sc, source="spinepose")
    assert sp is not None
    for v in (sp.lumbar_deg, sp.thoracic_deg, sp.cervical_deg, sp.bend_deg):
        assert math.isfinite(v) and abs(v) < 75.0                           # the old formula gave lumbar -114 here
    assert sp.confidence <= 1.0                                              # raw scores reach 1.17 (SimCC peaks)


def adjacent_segment_lumbar(pts_top_down, facing=1):
    """The formula that was replaced: directions of neighbouring segments (kept here as the documented
    counter-example)."""
    up = pts_top_down[::-1]
    d = [math.degrees(math.atan2((up[i + 1][0] - up[i][0]) * facing, -(up[i + 1][1] - up[i][1]))) for i in range(8)]
    avg = lambda a, b: a + ((b - a + 180) % 360 - 180) / 2                   # noqa: E731
    return avg(d[2], d[3]) - d[0]


def test_the_replaced_formula_really_was_unusable_on_the_real_dumbbell_chain():
    pts, _ = from_bottom_up(DUMBBELL_ROW)
    assert abs(adjacent_segment_lumbar(pts)) > 75.0                          # documents why G-33 exists


# ---------------------------------------------------------------------------
# Noise robustness (the simulation behind G-33, as a regression test)
# ---------------------------------------------------------------------------
def test_lumbar_change_has_no_false_alarms_at_realistic_keypoint_noise():
    base = np.mean([build_spine_from_points(curved_chain(0, 20, 5, noise=2.0, seed=i), 1).lumbar_deg
                    for i in range(3)], axis=0)
    deltas, old = [], []
    for i in range(300):
        pts = curved_chain(0, 20, 5, noise=2.0, seed=100 + i)               # NO true change from the baseline
        deltas.append(build_spine_from_points(pts, 1).lumbar_deg - base)
        old.append(adjacent_segment_lumbar(pts))
    deltas = np.array(deltas)
    assert deltas.std() < 6.0                                                # simulation: 3.8 deg at 2 px
    assert np.mean(np.abs(deltas) > 8.0) < 0.10                              # the default cue threshold (single frames)
    assert np.std(old) > 12.0                                                # replaced formula: ~20 deg, useless


def test_the_fit_tracks_a_real_change_but_under_reads_it():
    base = build_spine_from_points(curved_chain(0, 20, 5), 1).lumbar_deg
    small = build_spine_from_points(curved_chain(15, 20, 5), 1).lumbar_deg - base
    big = build_spine_from_points(curved_chain(30, 20, 5), 1).lumbar_deg - base
    assert 0 < small < big < 30                                              # monotone, below the true change
    assert 0.1 < big / 30 < 0.6                                              # documented shrinkage (~0.3)


def test_a_low_score_outlier_point_barely_moves_the_angles():
    pts = curved_chain(10, 20, 5)
    sc = np.full(9, 0.9)
    clean = build_spine_from_points(pts, 1, sc)
    bad = pts.copy()
    bad[6] += [60.0, 0.0]                                                    # one wild L3 point, 60 px off
    low = sc.copy()
    low[6] = 0.05
    weighted = build_spine_from_points(bad, 1, low)
    unweighted = build_spine_from_points(bad, 1, sc)
    assert abs(weighted.lumbar_deg - clean.lumbar_deg) < 3.0                 # simulation ~1 deg (165 with arc-length)
    # the same point at full weight is either rejected as a scrambled chain or moves the angle far more
    err_w = abs(weighted.lumbar_deg - clean.lumbar_deg)
    assert unweighted is None or err_w < abs(unweighted.lumbar_deg - clean.lumbar_deg)


# ---------------------------------------------------------------------------
# Any body orientation (deadlift, squat: upright; bench press: lying)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deg", [0, 90, 180, 37, -60])
def test_region_angles_do_not_depend_on_orientation_when_the_anterior_side_is_given(deg):
    pts = curved_chain(10, 25, 6)
    ref = build_spine_from_points(pts, 1, anterior=np.array([1.0, 0.0]))
    turned = build_spine_from_points(rotate(pts, deg), 1, anterior=rotate(np.array([[1.0, 0.0]]), deg)[0])
    for name in ("lumbar_deg", "thoracic_deg", "cervical_deg", "bend_deg"):
        assert getattr(turned, name) == pytest.approx(getattr(ref, name), abs=0.5)


def test_a_lying_spine_for_the_bench_press_uses_the_chest_side_as_anterior():
    pts = curved_chain(-15, 12, 4)                                           # lumbar extension: the bench-press arch
    upright = build_spine_from_points(pts, 1)
    lying = rotate(pts, -90)                                                 # head to the left; old +x (front) is up
    assert np.allclose(rotate(np.array([[1.0, 0.0]]), -90)[0], [0.0, -1.0], atol=1e-9)   # chest up (-y in the image)
    chest_up = build_spine_from_points(lying, 1, anterior=np.array([0.0, -1.0]))
    chest_down = build_spine_from_points(lying, 1, anterior=np.array([0.0, 1.0]))
    assert chest_up.lumbar_deg == pytest.approx(upright.lumbar_deg, abs=0.5) and chest_up.lumbar_deg < 0
    assert chest_down.lumbar_deg == pytest.approx(-upright.lumbar_deg, abs=0.5)    # the sign is the anterior convention


@pytest.mark.parametrize("facing", [1, -1])
@pytest.mark.parametrize("fold_deg", [30, 60, 85, 100, 120, 150])
def test_rounding_keeps_its_sign_through_a_forward_fold_past_horizontal(fold_deg, facing):
    """Hinging forward until the head is below the hips (chord lean > 90 deg) used to flip every region angle: the
    default anterior was (facing, 0) projected off the chord, which points the wrong way there. Found on a real clip,
    where convex rounding read as extension. The default anterior is now the chord turned toward ``facing``."""
    pts = curved_chain(10, 25, 6)                                            # a rounded back: lumbar and thoracic > 0
    ref = build_spine_from_points(pts, 1)
    folded = rotate(pts, fold_deg, about=pts[-1])                            # clockwise in the image: forward for +x
    if facing == -1:
        folded = folded * np.array([-1.0, 1.0])                              # mirror: the same lifter facing left
    sp = build_spine_from_points(folded, facing)
    assert sp is not None and sp.lumbar_deg > 0 and sp.thoracic_deg > 0
    assert sp.lumbar_deg == pytest.approx(ref.lumbar_deg, abs=0.5)
    assert sp.thoracic_deg == pytest.approx(ref.thoracic_deg, abs=0.5)


def test_facing_is_the_default_anterior_and_mirrors():
    right = build_spine_from_points(curved_chain(5, 20, 5), 1)
    left = build_spine_from_points(curved_chain(5, 20, 5) * np.array([-1, 1]), -1)
    assert right.thoracic_deg == pytest.approx(left.thoracic_deg, abs=1e-6)
    assert right.lumbar_deg == pytest.approx(left.lumbar_deg, abs=1e-6)


def test_degenerate_inputs_are_rejected_not_guessed():
    pts = curved_chain(5, 20, 5)
    assert build_spine_from_points(pts, 1, anterior=np.array([0.0, -1.0])) is not None          # perpendicular: fine
    assert build_spine_from_points(pts, 1, anterior=np.array([0.0, 0.0])) is None                # no direction at all
    assert build_spine_from_points(pts * 0.2, 1) is None                              # ~45 px: below the noise floor
    assert build_spine_from_points(pts * 0.5, 1) is not None                                     # ~112 px: measurable
    assert build_spine_from_points(pts, 1, scores=np.full(9, 0.1)) is None                       # low confidence
    assert build_spine_from_points(pts[np.random.default_rng(0).permutation(9)], 1) is None      # scrambled order


def test_scores_above_one_are_clipped_into_a_confidence():
    raw = np.array([1.4, 1.3, .9, 1.46, 1.49, 1.38, 1.25, 1.43, .68])
    sp = build_spine_from_points(curved_chain(5, 20, 5), 1, scores=raw)
    assert sp is not None and 0.0 < sp.confidence <= 1.0
    assert sp.confidence == pytest.approx(np.mean(np.clip([1.4, 1.3, .9, 1.46, 1.49, 1.38, 1.25, 1.43, .68], 0, 1)))
    assert set(sp.points) == set(SPINE_LABELS)


def test_front_view_keeps_the_points_but_zeroes_every_angle():
    sp = build_spine_from_points(curved_chain(10, 30, 5), 1, curvature_visible=False)
    assert (sp.lumbar_deg, sp.thoracic_deg, sp.cervical_deg, sp.bend_deg, sp.pelvic_deg) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert not sp.curvature_visible and sp.curve.shape[1] == 2


# ---------------------------------------------------------------------------
# The real network (skipped unless the package and the cached weights are present: never downloads in a test run)
# ---------------------------------------------------------------------------
WEIGHTS = os.path.expanduser("~/.cache/spinepose/hub/checkpoints/spinepose-s_32xb256-10e_spinetrack-256x192.onnx")


@pytest.mark.skipif(not (spinepose_available() and os.path.isfile(WEIGHTS)),
                    reason="SpinePose or its cached weights absent")
def test_real_network_loads_runs_and_returns_nine_finite_points():
    tr = SpinePoseTracker(size="small")
    img = np.full((480, 640, 3), 127, np.uint8)
    det = tr.detect(img, np.array([100.0, 40.0, 400.0, 460.0]))
    assert det is not None and det.points.shape == (9, 3) and det.keypoints.shape == (37, 2)
    assert np.all(np.isfinite(det.points))                   # (no person in the image: coordinates are arbitrary)
    tr.close()
