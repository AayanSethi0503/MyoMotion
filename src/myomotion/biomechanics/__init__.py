"""Deterministic biomechanics: angles, spine proxy, rep segmentation and form rules.

Depends on NumPy and the standard library only (no cv2 / mediapipe / torch): callers convert pose
results to ``np.ndarray [33, 3]`` = (x_px, y_px, visibility).
"""

from .deadlift_angles import DeadliftAngles, FrontalAngles, compute_deadlift_angles, compute_frontal_angles
from .deadlift_pipeline import DeadliftAnalyzer, FrameResult
from .deadlift_rules import (Bias, DeadliftFormAnalyzer, DeadliftRuleConfig, FormEvent, FormFlag, RepForm,
                             Severity, classify_bias)
from .filters import AngleChannel, KeypointSmoother, OneEuroFilter, SlidingSlope
from .rep_segmenter import DeadliftRepSegmenter, Phase, RepTiming, normalize_trajectory
from .spine_proxy import SPINE_LABELS, SpineBaseline, SpineProxy, build_spine_proxy

__all__ = [
    "AngleChannel", "Bias", "DeadliftAngles", "DeadliftAnalyzer", "DeadliftFormAnalyzer", "DeadliftRepSegmenter",
    "DeadliftRuleConfig", "FormEvent", "FormFlag", "FrameResult", "FrontalAngles", "KeypointSmoother",
    "OneEuroFilter", "Phase", "RepForm", "RepTiming", "SPINE_LABELS", "Severity", "SlidingSlope",
    "SpineBaseline", "SpineProxy", "build_spine_proxy", "classify_bias", "compute_deadlift_angles",
    "compute_frontal_angles", "normalize_trajectory",
]
