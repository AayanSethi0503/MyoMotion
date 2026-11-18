"""MediaPipe PoseLandmarker (Tasks API, VIDEO mode) wrapper that returns ``np.ndarray [33, 3]``.

This is the only place the demo touches MediaPipe objects, so ``biomechanics`` never imports mediapipe/cv2.
Uses the same API calls and options as the original ``scripts/cv/deadlift/deadlift_tracker.py``; the
``mediapipe`` pin in requirements.txt is untouched. CPU delegate only (macOS and Windows safe, no CUDA).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "models", "pose_landmarker_heavy.task")


@dataclass(frozen=True)
class PoseDetection:
    image: np.ndarray                    # [33, 3] x_px, y_px, visibility
    world: Optional[np.ndarray]          # [33, 3] metres, origin at the hip centre, x right, y DOWN, z away (or None)
    mask: Optional[np.ndarray] = None    # [H, W] float32 person mask, 0-1 (only if built with masks=True)


class PoseTracker:
    """Detect one person per frame. Timestamps must be strictly increasing (enforced here)."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, min_confidence: float = 0.5, masks: bool = False):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}. Download one into models/ from "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_<size>/float16/latest/"
                "pose_landmarker_<size>.task where <size> is lite (fastest), full or heavy (most accurate).")
        import mediapipe as mp                                    # imported lazily: keeps `import myomotion` light
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        base = mp_python.BaseOptions(model_asset_path=model_path, delegate=mp_python.BaseOptions.Delegate.CPU)
        options = vision.PoseLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=min_confidence, min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence, output_segmentation_masks=masks)
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray, t_s: float) -> Optional[np.ndarray]:
        """Image landmarks as ``[33, 3]`` = (x_px, y_px, visibility), or None if no person was found."""
        det = self.detect_full(frame_bgr, t_s)
        return None if det is None else det.image

    def detect_full(self, frame_bgr: np.ndarray, t_s: float) -> Optional[PoseDetection]:
        """Image landmarks plus MediaPipe's 3-D world landmarks (for view-independent angles), or None."""
        import cv2
        h, w = frame_bgr.shape[:2]
        ts_ms = max(int(t_s * 1000.0), self._last_ts_ms + 1)      # MediaPipe needs strictly increasing int ms
        self._last_ts_ms = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.pose_landmarks:
            return None
        out = np.zeros((33, 3))
        for i, lm in enumerate(result.pose_landmarks[0][:33]):
            vis = 1.0 if lm.visibility is None else float(lm.visibility)
            if lm.presence is not None:
                vis = min(vis, float(lm.presence))
            out[i] = (lm.x * w, lm.y * h, vis)
        world = None
        wl = getattr(result, "pose_world_landmarks", None)
        if wl:
            world = np.array([[p.x, p.y, p.z] for p in wl[0][:33]], float)
        mask = None
        sm = getattr(result, "segmentation_masks", None)
        if sm:
            # copy: the view points at MediaPipe's own buffer, which is reused for the next frame
            mask = np.array(sm[0].numpy_view(), dtype=np.float32, copy=True)
            if mask.ndim == 3:
                mask = mask[..., 0]
        return PoseDetection(image=out, world=world, mask=mask)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
