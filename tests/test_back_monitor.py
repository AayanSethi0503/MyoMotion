"""Back analysis that does not need the legs (G-34).

Found on a real webcam session: the lifter stood close to the camera, so knees and ankles were out of frame for 850 of
923 frames, nothing was "tracked", and no back-rounding cue could ever run even though SpinePose had a spine in 919
frames. The back rules now need only the near-side shoulder and hip plus the spine, gate on the hinge (trunk
inclination), and read the THORACIC region angle directly (it separated rounded from neutral backs in every SpinePose
size on two real recordings; lumbar did not, so lumbar cues are optional).

Synthetic tests prove the rule logic only. Real sensitivity is unvalidated (docs/GAP_ANALYSIS.md G-34).
"""

import math

import numpy as np
import pytest

from myomotion.biomechanics import DeadliftAnalyzer, DeadliftRuleConfig
from myomotion.biomechanics.deadlift_rules import DeadliftFormAnalyzer, FormFlag
from myomotion.biomechanics.geometry import inclination_from_vertical
from myomotion.biomechanics.landmarks import NOSE, SIDE_INDICES, estimate_facing_from_trunk
from myomotion.biomechanics.rep_segmenter import Phase
from myomotion.biomechanics.spine_proxy import build_spine_from_points
from myomotion.biomechanics.synthetic import Body, ClipSpec, generate_clip_3d
from test_spine_regions import curved_chain, rotate
from test_views_and_spine import curved_points_for_frame

R = SIDE_INDICES["right"]
LEGS = ("knee", "ankle", "heel", "foot")


def trunk_of(img: np.ndarray) -> float:
    return inclination_from_vertical(img[R["hip"], :2], img[R["shoulder"], :2], 1)


def run_clip(flex_fn, legs=False, cfg=None, n_reps=2, spine_every=1, source_pts=curved_points_for_frame):
    """Synthetic lifter, optionally with the legs hidden; ``flex_fn(t, trunk_deg)`` gives the spine curvature."""
    spec = ClipSpec(n_reps=n_reps, noise_px=0.3)
    an = DeadliftAnalyzer(rule_cfg=cfg or DeadliftRuleConfig())
    results = []
    for i, (t, img, w) in enumerate(generate_clip_3d("good", spec, Body(), yaw_deg=0.0)):
        if not legs:
            for side in ("right", "left"):
                for n in LEGS:
                    img[SIDE_INDICES[side][n], 2] = 0.1
        pts = source_pts(img, flex_fn(t, trunk_of(img))) if i % spine_every == 0 else None
        results.append(an.process(t, img, world=w, spine_points=pts))
    return an, results


# ---------------------------------------------------------------------------
# Facing from the trunk (needs neither feet nor face)
# ---------------------------------------------------------------------------
def make_lm(sh, hip, nose=(0.0, 0.0)):
    lm = np.zeros((33, 3))
    lm[R["shoulder"]] = (*sh, 1.0)
    lm[R["hip"]] = (*hip, 1.0)
    lm[NOSE] = (*nose, 1.0)
    return lm


@pytest.mark.parametrize("facing", [1, -1])
def test_a_hinged_trunk_votes_for_the_direction_it_leans(facing):
    hip = np.array([500.0, 500.0])
    for incl in (40, 70, 90, 110):                                   # 110: folded past horizontal, head below the hips
        dx, dy = math.sin(math.radians(incl)), math.cos(math.radians(incl))
        sh = hip + 200 * np.array([facing * dx, -dy])
        assert estimate_facing_from_trunk(make_lm(sh, hip), "right") == (facing, 3)


def test_an_upright_or_unreliable_trunk_casts_no_vote():
    hip = np.array([500.0, 500.0])
    assert estimate_facing_from_trunk(make_lm(hip + [30.0, -200.0], hip), "right") == (0, 0)     # ~9 deg: upright
    lm = make_lm(hip + [150.0, -100.0], hip)
    lm[R["shoulder"], 2] = 0.2
    assert estimate_facing_from_trunk(lm, "right") == (0, 0)


def test_the_pipeline_takes_its_facing_from_the_trunk_when_feet_and_face_say_nothing_useful():
    an, _ = run_clip(lambda t, trunk: 0.0)                           # legs hidden: no heel/toe, no knee/ankle cue
    assert an._facing == 1                                           # the synthetic lifter faces image-right


# ---------------------------------------------------------------------------
# Legs not in view: the back is still analysed
# ---------------------------------------------------------------------------
def test_without_the_legs_the_back_is_analysed_but_no_reps_are_counted():
    an, res = run_clip(lambda t, trunk: 0.0)
    s = an.summary()
    assert s["rep_count"] == 0 and s["frames_tracked"] == 0
    assert s["frames_back_monitored"] > 100
    assert all(r.angles is None for r in res)                        # no leg angles without legs...
    assert any(r.back_only and r.hinged for r in res)                # ...but the hinge is recognised
    assert any(r.partial_spine is not None for r in res)             # and the spine is available for drawing


def test_a_straight_back_without_legs_raises_nothing():
    an, _ = run_clip(lambda t, trunk: 0.0)
    assert an.summary()["flag_counts"]["upper_back_rounding"] == 0 and not an.rules.events


def test_a_rounded_back_without_legs_is_flagged_once_per_hinge():
    an, res = run_clip(lambda t, trunk: 50.0 if trunk > 20 else 0.0, n_reps=2)     # rounded whenever hinged
    s = an.summary()
    assert s["rep_count"] == 0                                       # still no reps: this is the legs-out-of-view path
    ev = [e for e in s["events"] if e["flag"] == "upper_back_rounding"]
    hinges = sum(1 for a, b in zip([False] + [r.hinged for r in res], [r.hinged for r in res]) if b and not a)
    assert hinges == 3                                               # set-up, after rep 1, after rep 2
    assert len(ev) == hinges                                         # one event per hinge, not one per session
    assert all(e["confidence"] == "absolute" and e["severity"] == "warning" for e in ev)
    assert all(e["value"] >= an.rules.cfg.thoracic_abs_enter_deg_model for e in ev)
    assert any(FormFlag.UPPER_BACK_ROUNDING in r.active_flags for r in res)
    assert not [e for e in s["events"] if e["flag"] == "lumbar_rounding"] or an.rules.cfg.lumbar_cues


def test_a_mild_curve_below_the_threshold_is_not_flagged():
    an, _ = run_clip(lambda t, trunk: 15.0 if trunk > 20 else 0.0)                 # thoracic reads ~3.8 < 6
    assert an.summary()["flag_counts"]["upper_back_rounding"] == 0


def test_a_convex_spine_while_upright_is_not_flagged_only_while_hinged():
    an, res = run_clip(lambda t, trunk: 60.0 if trunk < 15 else 0.0)        # rounded when standing, neutral hinged
    assert an.summary()["flag_counts"]["upper_back_rounding"] == 0
    assert any(not r.hinged for r in res)


def test_only_the_spinepose_source_raises_the_absolute_cue():
    for src in ("contour", "proxy"):
        def pts(img, flex, src=src):
            return curved_points_for_frame(img, flex)
        spec = ClipSpec(n_reps=1, noise_px=0.3)
        an = DeadliftAnalyzer()
        for t, img, w in generate_clip_3d("good", spec, Body(), yaw_deg=0.0):
            for n in LEGS:
                img[R[n], 2] = 0.1
            flex = 60.0 if trunk_of(img) > 20 else 0.0
            sp = build_spine_from_points(pts(img, flex)[:, :2], 1, source=src)
            an.rules.update_back_only(t, trunk_of(img), sp)          # same rule entry point the pipeline uses
        assert not an.rules.events, src


def test_a_fold_past_horizontal_still_reads_as_rounding_not_extension():
    """Head below the hips (trunk > 90 deg): the region angles keep their sign, so the cue still fires."""
    rules = DeadliftFormAnalyzer(DeadliftRuleConfig(lumbar_cues=False))
    pts = rotate(curved_chain(10, 30, 6), 120, about=curved_chain(10, 30, 6)[-1])      # chord leans ~150 deg forward
    sp = build_spine_from_points(pts, 1)
    assert sp.thoracic_deg > 6
    events = []
    for i in range(30):
        events += rules.update_back_only(i / 30.0, 110.0, sp, Phase.STANDBY)
    assert [e.flag for e in events] == [FormFlag.UPPER_BACK_ROUNDING]


# ---------------------------------------------------------------------------
# Lumbar cues are optional; thoracic always works
# ---------------------------------------------------------------------------
def test_lumbar_cues_can_be_switched_off_while_the_thoracic_cue_stays_on():
    cfg = DeadliftRuleConfig(lumbar_cues=False)
    an, _ = run_clip(lambda t, trunk: 70.0 if trunk > 20 else 0.0, cfg=cfg)
    counts = an.summary()["flag_counts"]
    assert counts["lumbar_rounding"] == 0 and counts["upper_back_rounding"] >= 1


def test_the_absolute_lumbar_cue_exists_when_lumbar_cues_are_on():
    an, _ = run_clip(lambda t, trunk: 70.0 if trunk > 20 else 0.0, cfg=DeadliftRuleConfig(lumbar_cues=True))
    ev = an.summary()["events"]
    assert any(e["flag"] == "lumbar_rounding" for e in ev)


# ---------------------------------------------------------------------------
# With the legs in view the same rule runs inside the normal pipeline and lands in the rep verdict
# ---------------------------------------------------------------------------
def test_with_the_legs_in_view_the_thoracic_cue_is_part_of_the_rep_verdict():
    an, _ = run_clip(lambda t, trunk: 50.0 if trunk > 20 else 0.0, legs=True, n_reps=1)
    s = an.summary()
    assert s["rep_count"] == 1 and s["frames_tracked"] > 100 and s["frames_back_monitored"] == 0
    form = s["reps"][0]["form"]
    assert "upper_back_rounding" in form["flags"]
    assert form["peak_thoracic_abs_deg"] is not None and form["peak_thoracic_abs_deg"] >= 6.0


def test_a_neutral_rep_with_the_legs_in_view_has_no_back_flag_and_reports_its_peaks():
    an, _ = run_clip(lambda t, trunk: 0.0, legs=True, n_reps=1)
    form = an.summary()["reps"][0]["form"]
    assert not form["flags"] and form["form_ok"] is True
    assert abs(form["peak_thoracic_abs_deg"] or 0.0) < 1.0


# ---------------------------------------------------------------------------
# Summary and the calibration log
# ---------------------------------------------------------------------------
def test_the_spine_log_records_every_frame_with_a_spine_for_calibration():
    an, _ = run_clip(lambda t, trunk: 30.0 if trunk > 20 else 0.0)
    rows = an.spine_log
    assert len(rows) == an.frames_back_only > 100
    assert set(rows[0]) >= {"t_s", "source", "view", "back_only", "trunk_incl_deg", "hinged", "lumbar_deg",
                            "thoracic_deg", "cervical_deg", "pelvic_deg", "curvature_visible", "confidence"}
    assert any(r["hinged"] for r in rows) and any(not r["hinged"] for r in rows)
    assert max(r["thoracic_deg"] for r in rows) > 5.0


def test_a_missing_detection_does_not_end_the_hinge_or_repeat_the_event():
    """SpinePose misses frames (or the bridge check rejects them). A gap neither resets the dwell timer nor, once the
    cue has fired, raises it again: still exactly one event per hinge."""
    an, res = run_clip(lambda t, trunk: 50.0 if trunk > 20 else 0.0, spine_every=3, n_reps=1)    # 2/3 frames: no points
    hinges = sum(1 for a, b in zip([False] + [r.hinged for r in res], [r.hinged for r in res]) if b and not a)
    assert hinges == 2
    assert an.summary()["flag_counts"]["upper_back_rounding"] == hinges


def test_a_long_gap_without_any_measurement_drops_the_cue():
    """No SpinePose measurement for longer than ABS_STALE_S while hinged: the active cue is not held forever."""
    rules = DeadliftFormAnalyzer(DeadliftRuleConfig(lumbar_cues=False))
    sp = build_spine_from_points(curved_chain(10, 30, 6), 1)
    t = 0.0
    while t < 1.0:
        rules.update_back_only(t, 60.0, sp)
        t += 1 / 30
    assert FormFlag.UPPER_BACK_ROUNDING in rules.active
    while t < 1.2:                                                   # a short gap: held
        rules.update_back_only(t, 60.0, None)
        t += 1 / 30
    assert FormFlag.UPPER_BACK_ROUNDING in rules.active
    while t < 2.2:                                                   # a long gap: dropped
        rules.update_back_only(t, 60.0, None)
        t += 1 / 30
    assert FormFlag.UPPER_BACK_ROUNDING not in rules.active
