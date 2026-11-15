"""
Deadlift Tracker (legacy camera loop) — MyoMotion FYP
=====================================================
Live / video-file deadlift tracker: MediaPipe Pose -> hip & knee angles -> phase state machine -> HUD.

The reusable logic (filters, geometry, state machine) now lives in ``src/myomotion/biomechanics/``.
This script is a thin camera loop over it and keeps its original public names (``Phase``,
``DeadliftStateMachine``, ``calculate_angle``, the zone constants) so ``tests/test_deadlift_phases.py``
passes unchanged. For the full pipeline (spine proxy, form rules, JSON summary, annotated video) use
``scripts/demo_deadlift.py``.

Legacy phases:  SETUP -> CONCENTRIC -> ISOMETRIC HOLD -> ECCENTRIC -> SETUP   (STANDBY = not in a lift)
(The package uses CONCENTRIC_PULL / LOCKOUT / ECCENTRIC_DESCENT / REP_COMPLETE; this shim maps them back.)

Input mode: toggle the VIDEO_SOURCE lines below (live camera vs video file).
Camera orientation: person facing LEFT (right side toward camera).
Press Q to quit.
"""

from __future__ import annotations

import math
import os
import sys
import time
from enum import Enum
from typing import List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

# ---------------------------------------------------------------------------
# Paths (scripts/cv/deadlift/ -> scripts/cv/ -> scripts/ -> project root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "pose_landmarker_heavy.task")

# Make ``myomotion`` importable when run directly without ``pip install -e .``
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.biomechanics.geometry import joint_angle as calculate_angle  # noqa: E402,F401
from myomotion.biomechanics.landmarks import MIN_LANDMARK_VISIBILITY  # noqa: E402
from myomotion.biomechanics.rep_segmenter import (  # noqa: E402,F401
    DeadliftRepSegmenter, Phase as _Phase, HIP_SETUP_ENTER_DEG, HIP_SETUP_EXIT_DEG,
    HIP_LOCKOUT_ENTER_DEG, HIP_LOCKOUT_EXIT_DEG, KNEE_LOCKOUT_DEG, VELOCITY_MOVE_DEG_S,
    ONSET_DISPLACEMENT_DEG, DWELL_ONSET_S, DWELL_SETTLE_S, DWELL_HOLD_S, DWELL_SETUP_S,
    STALL_TIMEOUT_S, TRACKING_LOSS_TIMEOUT_S)
from myomotion.biomechanics.filters import (  # noqa: E402,F401
    FILTER_MIN_CUTOFF_HZ, FILTER_BETA, FILTER_D_CUTOFF_HZ, MAX_ANGULAR_SPEED_DEG_S, JUMP_SLACK_DEG,
    VELOCITY_WINDOW_S, MAX_SAMPLE_GAP_S, OneEuroFilter, AngleChannel, SlidingSlope)

# ---------------------------------------------------------------------------
# Input Mode — uncomment ONE of the two lines below
# ---------------------------------------------------------------------------
VIDEO_SOURCE = 0                                # Live camera mode
# VIDEO_SOURCE = "path/to/your/video.mp4"       # Video file mode

# Landmark indices (MediaPipe Pose — 33 landmarks, right side)
RIGHT_SHOULDER = 12
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28


# ---------------------------------------------------------------------------
# Legacy phase enum (kept so the original tests and HUD colours keep working)
# ---------------------------------------------------------------------------
class Phase(Enum):
    STANDBY = "Standby"
    SETUP = "Setup"
    CONCENTRIC = "Concentric"
    ISOMETRIC_HOLD = "Isometric Hold"
    ECCENTRIC = "Eccentric"


_TO_LEGACY = {
    _Phase.STANDBY: Phase.STANDBY,
    _Phase.SETUP: Phase.SETUP,
    _Phase.CONCENTRIC_PULL: Phase.CONCENTRIC,
    _Phase.LOCKOUT: Phase.ISOMETRIC_HOLD,
    _Phase.ECCENTRIC_DESCENT: Phase.ECCENTRIC,
    _Phase.REP_COMPLETE: Phase.SETUP,
}

# BGR colours for the on-screen phase badge
PHASE_COLOURS = {
    Phase.STANDBY:        (120, 120, 120),   # dark grey — not in a lift
    Phase.SETUP:          (200, 200, 200),   # light grey — pre-pull / bottom
    Phase.CONCENTRIC:     (0, 200, 0),       # green  — ascending
    Phase.ISOMETRIC_HOLD: (0, 180, 255),     # amber  — holding at the top
    Phase.ECCENTRIC:      (0, 100, 255),     # red-orange — descending
}


class DeadliftStateMachine:
    """Legacy-API wrapper around :class:`DeadliftRepSegmenter` (no REP_COMPLETE phase, legacy names)."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._seg = DeadliftRepSegmenter(verbose=verbose, emit_rep_complete=False,
                                         on_transition=self._log if verbose else None)

    def _log(self, t, old, new, seg) -> None:
        print(f"[PHASE] t={t:7.2f}s  {_TO_LEGACY[old].value:>14} -> {_TO_LEGACY[new].value:<14}"
              f" hip={seg.hip_angle:5.1f}  reps={seg.rep_count}")

    def update(self, t: float, raw_hip: float, raw_knee: float) -> Phase:
        self._seg.update(t, raw_hip, raw_knee)
        return self.phase

    def report_missing(self, t: float) -> None:
        self._seg.report_missing(t)

    @property
    def phase(self) -> Phase:
        return _TO_LEGACY[self._seg.phase]

    @property
    def transitions(self) -> List[Tuple[float, Phase, Phase]]:
        return [(t, _TO_LEGACY[o], _TO_LEGACY[n]) for t, o, n in self._seg.transitions]

    @property
    def rep_count(self) -> int:
        return self._seg.rep_count

    @property
    def tracking_ok(self) -> bool:
        return self._seg.tracking_ok

    @property
    def hip_angle(self) -> float:
        return self._seg.hip_angle

    @property
    def knee_angle(self) -> float:
        return self._seg.knee_angle

    @property
    def hip_velocity(self) -> float:
        return self._seg.hip_velocity


# ---------------------------------------------------------------------------
# MediaPipe-object helpers (kept here: they take MediaPipe landmark objects, which
# ``myomotion.biomechanics`` deliberately never imports)
# ---------------------------------------------------------------------------
def pixel_coords(landmark, w: int, h: int) -> Tuple[float, float]:
    """Normalised landmark -> (x, y) pixel coordinates (scaling x by width and y by height keeps
    angles correct on non-square frames)."""
    return (landmark.x * w, landmark.y * h)


def landmark_reliable(landmark) -> bool:
    """True if MediaPipe is confident the landmark is visible and in-frame."""
    for attr in ("visibility", "presence"):
        value = getattr(landmark, attr, None)
        if value is not None and value < MIN_LANDMARK_VISIBILITY:
            return False
    return 0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0


def extract_joint_angles(landmarks, w: int, h: int):
    """Return (hip_deg, knee_deg, hip_xy, knee_xy) or None if unreliable."""
    needed = (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    if any(not landmark_reliable(landmarks[i]) for i in needed):
        return None
    shoulder = pixel_coords(landmarks[RIGHT_SHOULDER], w, h)
    hip = pixel_coords(landmarks[RIGHT_HIP], w, h)
    knee = pixel_coords(landmarks[RIGHT_KNEE], w, h)
    ankle = pixel_coords(landmarks[RIGHT_ANKLE], w, h)
    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)
    if not (math.isfinite(hip_angle) and math.isfinite(knee_angle)):
        return None
    return hip_angle, knee_angle, hip, knee


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------
# NOTE: OpenCV's Hershey fonts are ASCII-only; a "°" glyph renders as "??",
# so angles are written with "deg".
def draw_hud(frame, sm: DeadliftStateMachine) -> None:
    """Overlay phase badge, rep counter, and smoothed angle readouts."""
    colour = PHASE_COLOURS[sm.phase]

    cv2.putText(frame, sm.phase.value, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

    cv2.putText(
        frame,
        f"Hip: {sm.hip_angle:.0f} deg ({sm.hip_velocity:+.0f} deg/s)"
        f"   Knee: {sm.knee_angle:.0f} deg",
        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        sys.exit(
            f"[ERROR] Model not found: {MODEL_PATH}\n"
            "Download pose_landmarker_heavy.task and place it in models/."
        )

    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video source: {VIDEO_SOURCE!r}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    is_live = isinstance(VIDEO_SOURCE, int)
    frame_period_ms = 1000.0 / fps

    print(f"[INFO] Source: {VIDEO_SOURCE!r}  |  Nominal FPS: {fps:.1f}"
          f"  |  Mode: {'Live camera' if is_live else 'Video file'}")
    print("[INFO] Press Q to quit.\n")

    sm = DeadliftStateMachine()
    landmark_style = drawing_utils.DrawingSpec(
        color=(245, 117, 66), thickness=2, circle_radius=4)
    connection_style = drawing_utils.DrawingSpec(
        color=(245, 66, 230), thickness=2, circle_radius=2)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        last_ts_ms = -1
        t_start = time.monotonic()

        while True:
            loop_start = time.perf_counter()
            success, frame = cap.read()
            if not success:
                if not is_live:
                    print(f"\n[INFO] Video ended. Total reps: {sm.rep_count}")
                break

            # Frame time in seconds. Live: wall clock, so a camera that
            # delivers fewer frames than advertised is still timed correctly.
            # File: frame index / nominal FPS (assumes constant frame rate).
            t_s = (time.monotonic() - t_start) if is_live else frame_index / fps
            frame_index += 1
            # MediaPipe requires strictly increasing integer-ms timestamps.
            timestamp_ms = max(int(t_s * 1000.0), last_ts_ms + 1)
            last_ts_ms = timestamp_ms

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            joints = None
            label = None   # (hip_text, knee_text, hip_xy, knee_xy)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                drawing_utils.draw_landmarks(
                    frame, lm, pose_connections,
                    landmark_drawing_spec=landmark_style,
                    connection_drawing_spec=connection_style,
                )
                joints = extract_joint_angles(lm, w, h)
                if joints is None:
                    # Low-visibility pose: still label with the raw angles
                    # (display only — not fed to the state machine).
                    sh = pixel_coords(lm[RIGHT_SHOULDER], w, h)
                    hp = pixel_coords(lm[RIGHT_HIP], w, h)
                    kn = pixel_coords(lm[RIGHT_KNEE], w, h)
                    an = pixel_coords(lm[RIGHT_ANKLE], w, h)
                    raw_hip = calculate_angle(sh, hp, kn)
                    raw_knee = calculate_angle(hp, kn, an)
                    if math.isfinite(raw_hip) and math.isfinite(raw_knee):
                        label = (f"{raw_hip:.0f}", f"{raw_knee:.0f}", hp, kn)

            if joints is not None:
                hip_deg, knee_deg, hip_xy, knee_xy = joints
                sm.update(t_s, hip_deg, knee_deg)
                # Label the *filtered* angles next to the joints
                label = (f"{sm.hip_angle:.0f}", f"{sm.knee_angle:.0f}",
                         hip_xy, knee_xy)
            else:
                sm.report_missing(t_s)

            if label is not None:
                hip_txt, knee_txt, hip_xy, knee_xy = label
                cv2.putText(
                    frame, hip_txt,
                    (int(hip_xy[0]) - 50, int(hip_xy[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(
                    frame, knee_txt,
                    (int(knee_xy[0]) - 50, int(knee_xy[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            draw_hud(frame, sm)
            cv2.imshow("MyoMotion - Deadlift Tracker", frame)

            # File playback is paced to native FPS, net of processing time.
            if is_live:
                wait_ms = 1
            else:
                elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
                wait_ms = max(1, int(frame_period_ms - elapsed_ms))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
