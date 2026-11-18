"""Partial visibility, framing guidance, low frame rate and facing fallback (what a laptop webcam actually gives).

Regression for the first live-webcam test: ear/heel/toe out of frame used to discard every frame (no skeleton, no
reps), and below ~8 fps no rep was ever counted. See docs/DEADLIFT_TRACKING_REQUIREMENTS.md.
"""

import numpy as np
import pytest

from myomotion.biomechanics import DeadliftAnalyzer, SlidingSlope
from myomotion.biomechanics.landmarks import (L_SHOULDER, NOSE, R_SHOULDER, SIDE_INDICES, estimate_facing,
                                              framing_report)
from myomotion.biomechanics.rep_segmenter import Phase
from myomotion.biomechanics.synthetic import ClipSpec, JointPose, build_landmarks, clip_list
from myomotion.coaching.overlay import OverlayRenderer, next_step_hint


def hide(*names):
    def f(lm):
        for side in ("right", "left"):
            for n in names:
                lm[SIDE_INDICES[side][n], 2] = 0.1
    return f


def run(scenario="good", mutate=None, **spec_kw):
    an = DeadliftAnalyzer()
    results = []
    for t, lm in clip_list(scenario, ClipSpec(**{"n_reps": 2, "noise_px": 1.0, **spec_kw})):
        if mutate:
            mutate(lm)
        results.append(an.process(t, lm, image_wh=(1280, 720)))
    return an, results


# ---------------------------------------------------------------------------
# Optional joints no longer block tracking
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("facing", [1, -1])
def test_feet_out_of_frame_still_counts_reps_and_draws(facing):
    an, results = run(mutate=hide("heel", "foot"), facing=facing)
    assert an.segmenter.rep_count == 2
    assert all(r.landmarks is not None for r in results)
    tracked = [r for r in results if r.angles is not None]
    assert len(tracked) == len(results)
    assert not tracked[0].angles.feet_visible and tracked[0].angles.bar_drift_foot is None
    assert an.summary()["flag_counts"]["bar_drift"] == 0                    # never guessed without feet
    bottom = next(r for r in tracked if r.phase == Phase.SETUP)
    assert bottom.angles.trunk_incl_deg > 30                                 # facing recovered from nose/ear or knee


def test_head_cropped_still_counts_reps_but_turns_the_spine_proxy_off():
    an, results = run(mutate=hide("ear"))
    assert an.segmenter.rep_count == 2
    assert all(r.angles.spine is None for r in results if r.angles)
    assert all(r.partial_spine is None for r in results)
    assert an.summary()["flag_counts"]["lumbar_rounding"] == 0
    fr = results[100].framing
    assert fr.ok and any("spine proxy" in n for n in fr.notes)               # explained, not blocking


def test_missing_ankle_blocks_tracking_but_is_still_drawable_and_explained():
    an, results = run(mutate=hide("ankle", "heel", "foot"))
    assert an.segmenter.rep_count == 0
    assert all(r.angles is None for r in results)
    assert all(r.landmarks is not None for r in results)                    # the skeleton is never withheld
    fr = results[100].framing
    assert not fr.ok and any("Ankle not visible" in p for p in fr.problems)
    assert not fr.visible["ankle"] and fr.visible["knee"]
    assert any(r.partial_spine is not None for r in results)                # spine proxy from ear/shoulder/hip alone


def test_no_person_gives_a_result_without_crashing():
    an = DeadliftAnalyzer()
    r = an.process(0.0, None)
    assert r.framing is None and r.landmarks is None and r.angles is None and r.phase == Phase.STANDBY


# ---------------------------------------------------------------------------
# Framing report
# ---------------------------------------------------------------------------
def test_framing_ok_for_a_side_view_and_notes_but_does_not_block_a_front_view():
    lm = build_landmarks(JointPose(60, 15, 50, 5))
    ok = framing_report(lm, image_wh=(1280, 720))
    assert ok.ok and ok.problems == () and ok.view == "side"
    assert ok.shoulder_span_ratio is None                      # far shoulder hidden: nothing to compare
    front = lm.copy()
    front[L_SHOULDER] = (front[R_SHOULDER, 0] + 320, front[R_SHOULDER, 1], 0.9)     # far shoulder visible and apart
    front[R_SHOULDER, 2] = 0.9
    fr = framing_report(front, image_wh=(1280, 720))
    assert fr.ok and fr.view == "front" and any("Facing the camera" in n for n in fr.notes)   # a note, not a block
    overlapped = lm.copy()
    overlapped[L_SHOULDER] = (overlapped[R_SHOULDER, 0] + 12, overlapped[R_SHOULDER, 1], 0.9)
    overlapped[R_SHOULDER, 2] = 0.9
    near = framing_report(overlapped, image_wh=(1280, 720))
    assert near.ok and near.view == "side" and near.shoulder_span_ratio < 0.1   # a true side view is not warned


def test_framing_flags_a_person_who_is_too_small_in_frame():
    lm = build_landmarks(JointPose(0, 0, 0, 5))
    assert framing_report(lm, image_wh=(1280, 720)).ok
    small = framing_report(lm, image_wh=(1280, 3000))                              # same pixels, much taller frame
    assert any("look small" in p for p in small.problems)


def test_framing_checklist_lists_every_tracked_joint():
    fr = framing_report(build_landmarks(JointPose(60, 15, 50, 5)))
    assert set(fr.visible) == {"ear", "shoulder", "hip", "knee", "ankle", "heel", "foot", "wrist"}


# ---------------------------------------------------------------------------
# Facing from weaker cues
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("facing", [1, -1])
def test_facing_cue_priority_and_fallbacks(facing):
    lm = build_landmarks(JointPose(60, 15, 50, 5), facing=facing)
    assert estimate_facing(lm, "right") == (facing, 3)                       # heel -> toe
    hide("heel", "foot")(lm)
    assert estimate_facing(lm, "right") == (facing, 2)                       # nose ahead of ear
    lm[NOSE, 2] = 0.1
    assert estimate_facing(lm, "right") == (facing, 1)                       # knee ahead of ankle at the bottom
    hide("knee")(lm)
    assert estimate_facing(lm, "right") == (0, 0)                            # no cue at all


# ---------------------------------------------------------------------------
# Low frame rate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fps", [30, 15, 10, 8, 6, 5, 4])
def test_reps_are_counted_even_at_low_processing_frame_rates(fps):
    # long tail after the last rep: a slower, stretched velocity window confirms the final descent later
    an, _ = run(fps=float(fps), noise_px=0.5, post_s=3.0)
    assert an.segmenter.rep_count == 2, fps


def test_slope_rescue_is_inert_at_normal_frame_rates_and_stretches_when_slow():
    def slope(max_window, fps, n):
        s = SlidingSlope(max_window_s=max_window)
        for i in range(n):
            s.add(i / fps, 10.0 * i / fps)                                   # exactly 10 units/s
        return s.slope()
    assert slope(None, 30, 30) == pytest.approx(10.0) and slope(1.0, 30, 30) == pytest.approx(10.0)
    assert slope(None, 5, 12) is None                                       # original behaviour: too few samples
    assert slope(1.0, 5, 12) == pytest.approx(10.0)                         # rescued
    assert slope(1.0, 2, 12) is None                                        # 4 samples would span 1.5 s: still refused


# ---------------------------------------------------------------------------
# Overlay and hints
# ---------------------------------------------------------------------------
def test_overlay_draws_in_every_state_without_crashing_and_changes_the_frame():
    cases = [run(mutate=None)[1][150], run(mutate=hide("ankle", "heel", "foot"))[1][150],
             run(mutate=hide("ear", "heel", "foot"))[1][150], DeadliftAnalyzer().process(0.0, None)]
    an = DeadliftAnalyzer()
    for res in cases:
        frame = np.zeros((720, 1280, 3), np.uint8)
        OverlayRenderer().render(frame, res, an)
        assert frame.any()
    small = np.zeros((480, 640, 3), np.uint8)                              # a typical laptop webcam frame
    OverlayRenderer().render(small, cases[1], an)
    assert small.any()


def test_overlay_warns_on_low_frame_rate():
    an, results = run()
    r = OverlayRenderer()
    frame = np.zeros((720, 1280, 3), np.uint8)
    for i, res in enumerate(results[:20]):
        res.t_s = i * 0.25                                                   # 4 fps
        r.render(frame, res, an)
    assert r._fps < 8.0


def test_next_step_hint_tells_the_user_what_the_counter_is_waiting_for():
    _, results = run()
    by_phase = {}
    for res in results:
        by_phase.setdefault(res.phase, res)
    standby = next_step_hint(by_phase[Phase.STANDBY])[0]
    assert "hinge forward" in standby or "hold still" in standby
    assert "pull to a full lockout" in next_step_hint(by_phase[Phase.SETUP])[0]
    assert "Stand fully tall" in next_step_hint(by_phase[Phase.CONCENTRIC_PULL])[0]
    assert "Lower" in next_step_hint(by_phase[Phase.LOCKOUT])[0]
    assert "Lower to the start" in next_step_hint(by_phase[Phase.ECCENTRIC_DESCENT])[0]
    _, blocked = run(mutate=hide("ankle"))
    assert "Not tracking" in next_step_hint(blocked[10])[0]
