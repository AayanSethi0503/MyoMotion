"""SpinePose spine keypoints [P1] as an optional, drop-in replacement for the geometric spine proxy (G-01, G-31).

Why: the geometric proxy only draws a curve through ear, shoulder and hip, so it cannot follow a real curved spine.
SpinePose is a trained keypoint model that looks at the image and predicts nine points along the spine
(C1, C4, C7, T3, T8, L1, L3, L5, sacrum).

Optional dependency. It needs ``spinepose`` (installed WITHOUT its unused ``torch`` requirement) and
``onnxruntime``; see ``INSTALL_HINT``. Model weights (ONNX, tens of MB) are downloaded on first use from the
package's own URLs (Hugging Face, ``dfki-av/spinepose``) into a cache. **Licence: CC-BY-NC-4.0, non-commercial**:
fine for the FYP, a blocker for a commercial app.

How it is used here: only the pose network is run (RTMPose, top-down, SimCC decoding), on a person box taken from
MediaPipe's landmarks, so the package's own person detector is never downloaded or run. The package's v1 helper
``_smooth_spine`` (replace every interior spine point by the midpoint of its neighbours) is deliberately NOT applied:
it would flatten exactly the curvature we want to measure.

The nine points come in the package's chain order ``SPINE_IDS = [19, 26, 27, 28, 29, 30, 18, 35, 36]``
(hip, spine_01..05, neck, neck_02, neck_03), bottom to top. By vertebra count (1 sacral, 3 lumbar, 2 thoracic,
3 cervical, [P1 Fig. 2]) that is SACRUM, L5, L3, L1, T8, T3, C7, C4, C1. This mapping is our reading of the chain
order. It is sanity-checked at run time: ``build_spine_from_points`` rejects points that are not ordered along
sacrum -> C1.

Run against the real network (small and medium, on a laptop with CoreML: 4-10 ms per inference after a ~2 s warm-up).
Scores are SimCC peak values and can exceed 1; they are clipped to [0, 1] wherever a confidence is needed.
On side-on photos the lower points (SACRUM, L5, L3) sit only 5-25 px apart, which is why region angles are estimated
from a smooth fit (spine_proxy.build_spine_from_points, G-33) and not from neighbouring points.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

INSTALL_HINT = ("pip install onnxruntime tqdm OneEuroFilter && pip install --no-deps spinepose==2.1.0   "
                "(licence CC-BY-NC-4.0, non-commercial; torch is NOT needed)")

# Chain order of spinepose's SPINE_IDS (bottom to top) -> our labels
CHAIN_BOTTOM_UP: Tuple[str, ...] = ("SACRUM", "L5", "L3", "L1", "T8", "T3", "C7", "C4", "C1")
DEFAULT_SPINE_IDS: Tuple[int, ...] = (19, 26, 27, 28, 29, 30, 18, 35, 36)
_VERSION_TO_FAMILY = {"v1": "spinetrack", "v2": "simspine"}      # v1 = the models of the SpineTrack paper [P1]

# COCO indices inside the 37 keypoints, for the MediaPipe consistency check
_SP_L_SHOULDER, _SP_R_SHOULDER, _SP_L_HIP, _SP_R_HIP = 5, 6, 11, 12
_MP_L_SHOULDER, _MP_R_SHOULDER, _MP_L_HIP, _MP_R_HIP = 11, 12, 23, 24


class SpinePoseUnavailable(RuntimeError):
    """Raised when the optional dependency or model cannot be used."""


def spinepose_available() -> bool:
    """True if both ``spinepose`` and ``onnxruntime`` can be imported (does not load any model)."""
    return all(importlib.util.find_spec(m) is not None for m in ("spinepose", "onnxruntime"))


@dataclass(frozen=True)
class SpineDetection:
    points: np.ndarray          # [9, 3] x_px, y_px, score in SPINE_LABELS order (C1 ... SACRUM, top to bottom)
    keypoints: np.ndarray       # [K, 2] all keypoints of the model (K = 37)
    scores: np.ndarray          # [K]


def person_bbox(lm: np.ndarray, image_wh: Tuple[int, int], pad: float = 0.12, min_vis: float = 0.3
                ) -> Optional[np.ndarray]:
    """xyxy box around the visible MediaPipe landmarks, padded and clipped to the image. None if too few."""
    ok = lm[:, 2] >= min_vis
    if ok.sum() < 6:
        return None
    xs, ys = lm[ok, 0], lm[ok, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = image_wh
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    box = np.array([max(0.0, x0 - px), max(0.0, y0 - py), min(float(w), x1 + px), min(float(h), y1 + py)])
    return box if (box[2] - box[0] > 8 and box[3] - box[1] > 8) else None


def bridge_error(det: SpineDetection, lm: np.ndarray) -> Optional[float]:
    """G-01 consistency check: how far apart the two models put the shoulder and hip mid-points, as a fraction of
    MediaPipe's torso length (0 = identical). Large values mean SpinePose boxed a different person or failed."""
    k = det.keypoints
    if k.shape[0] < 13:
        return None
    sp_sh, sp_hip = 0.5 * (k[_SP_L_SHOULDER] + k[_SP_R_SHOULDER]), 0.5 * (k[_SP_L_HIP] + k[_SP_R_HIP])
    mp_sh = 0.5 * (lm[_MP_L_SHOULDER, :2] + lm[_MP_R_SHOULDER, :2])
    mp_hip = 0.5 * (lm[_MP_L_HIP, :2] + lm[_MP_R_HIP, :2])
    torso = float(np.hypot(*(mp_sh - mp_hip)))
    if torso < 1e-6:
        return None
    return float(0.5 * (np.hypot(*(sp_sh - mp_sh)) + np.hypot(*(sp_hip - mp_hip))) / torso)


class SpinePoseTracker:
    """Runs the SpinePose network on a person box and returns the nine spine points.

    ``model`` may be injected (any callable ``(image, bboxes=[xyxy]) -> (keypoints [1, K, 2], scores [1, K])``) which is
    how the tests run without the real network. With ``model=None`` the real one is built from the installed package.
    """

    def __init__(self, size: str = "small", version: str = "v1", model: Optional[Any] = None,
                 spine_ids: Sequence[int] = DEFAULT_SPINE_IDS, warm_up: bool = True):
        self._ids = tuple(spine_ids)
        if len(self._ids) != 9:
            raise ValueError("expected 9 spine keypoint ids")
        if model is None:
            if not spinepose_available():
                raise SpinePoseUnavailable(f"SpinePose is not installed. {INSTALL_HINT}")
            from spinepose.pose_estimator import SpinePoseEstimator        # constants only; no detector is built
            from spinepose.tools.pose_estimation import RTMPose
            if size not in SpinePoseEstimator.MODE:
                raise ValueError(f"size must be one of {sorted(SpinePoseEstimator.MODE)}")
            cfg = SpinePoseEstimator.MODE[size]
            url = cfg["pose"]
            if "%s" in url:
                url = url % _VERSION_TO_FAMILY[version]
            model = RTMPose(url, model_input_size=cfg["pose_input_size"])
            self._ids = tuple(SpinePoseEstimator.SPINE_IDS)
            if warm_up:                          # the first call compiles the execution provider (~2 s on CoreML)
                model(np.zeros((256, 192, 3), np.uint8), bboxes=[[0.0, 0.0, 191.0, 255.0]])
        self._model = model

    def detect(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> Optional[SpineDetection]:
        """Spine points inside ``bbox_xyxy`` (px), or None if the model returned nothing usable."""
        kpts, scores = self._model(frame_bgr, bboxes=[[float(v) for v in bbox_xyxy]])
        kpts, scores = np.asarray(kpts), np.asarray(scores)
        if kpts.ndim != 3 or kpts.shape[0] < 1 or kpts.shape[1] <= max(self._ids):
            return None
        k, sc = kpts[0], scores[0]
        bottom_up = np.column_stack([k[list(self._ids)], sc[list(self._ids)]])     # SACRUM ... C1
        return SpineDetection(points=bottom_up[::-1].copy(), keypoints=k.copy(), scores=sc.copy())

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        if close is not None:
            close()
