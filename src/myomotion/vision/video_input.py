"""Uploaded-video input: validation and a timestamped frame reader.

The released app accepts **live camera and user-uploaded footage** (Context.md §1: "Live ... and uploaded video
(offline batch)"). This module is the upload side: ``validate_video`` gives a clear accept/reject verdict with
reasons (FR-CV-001 "frame validation"), and ``iter_video_frames`` yields ``(t_s, frame_bgr)`` with trustworthy
timestamps. The accepted format is specified in ``docs/VIDEO_INPUT_SPEC.md``; the limits below mirror it.

Timestamps come from the container (``CAP_PROP_POS_MSEC``), so variable-frame-rate phone video is timed correctly.
If the backend reports no usable time, ``frame_index / fps`` is used. Output times are always strictly increasing.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

# What phones record (verified to decode with the bundled OpenCV: mp4/mov with H.264 and MPEG-4, m4v).
RECOMMENDED_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".m4v")
# Decoded by OpenCV's FFmpeg build but not what phones produce; best effort.
BEST_EFFORT_EXTENSIONS: Tuple[str, ...] = (".avi", ".mkv", ".webm")
SUPPORTED_EXTENSIONS: Tuple[str, ...] = RECOMMENDED_EXTENSIONS + BEST_EFFORT_EXTENSIONS


@dataclass(frozen=True)
class VideoLimits:
    """Accept/reject limits. ``min_*``/``max_*`` reject; ``warn_*`` accept with a warning. Provisional (G-28)."""
    min_duration_s: float = 2.0
    max_duration_s: float = 600.0
    min_fps: float = 10.0
    max_fps: float = 120.0
    warn_fps_below: float = 24.0                 # fast pulls get under-sampled below this
    min_short_side_px: int = 240
    warn_short_side_below_px: int = 480
    warn_short_side_above_px: int = 1440         # works, but slow
    max_file_mb: float = 2048.0


@dataclass(frozen=True)
class VideoInfo:
    path: str
    extension: str
    size_mb: float
    width: int                                   # as decoded (after automatic rotation), from the first frame
    height: int
    fps: float
    frame_count: Optional[int]                   # None if the container does not say
    duration_s: Optional[float]
    orientation: str                             # "landscape" | "portrait" | "square"


@dataclass
class VideoValidation:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Optional[VideoInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings,
                "info": asdict(self.info) if self.info else None}


def _open(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)    # honour the phone's rotation metadata (portrait footage)
    return cap


def validate_video(path: str, limits: VideoLimits = VideoLimits()) -> VideoValidation:
    """Check an uploaded file before spending time on it. Never raises for a bad file; returns reasons instead."""
    errors: List[str] = []
    warnings: List[str] = []
    ext = os.path.splitext(path)[1].lower()

    if not os.path.isfile(path):
        return VideoValidation(False, [f"File not found: {path}"])
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb == 0:
        return VideoValidation(False, ["The file is empty (0 bytes)."])
    if ext not in SUPPORTED_EXTENSIONS:
        errors.append(f"Unsupported file type '{ext or '(none)'}'. Use one of: "
                      f"{', '.join(RECOMMENDED_EXTENSIONS)} (recommended) or {', '.join(BEST_EFFORT_EXTENSIONS)}.")
    elif ext in BEST_EFFORT_EXTENSIONS:
        warnings.append(f"'{ext}' is decoded on a best-effort basis; .mp4 or .mov is recommended.")
    if size_mb > limits.max_file_mb:
        errors.append(f"File is {size_mb:.0f} MB; the limit is {limits.max_file_mb:.0f} MB.")
    if errors and ext not in SUPPORTED_EXTENSIONS:
        return VideoValidation(False, errors, warnings)

    cap = _open(path)
    if not cap.isOpened():
        cap.release()
        errors.append("The video could not be opened. It may be corrupt, or use a codec this system cannot decode "
                      "(for iPhone HEVC footage, re-export as 'Most Compatible' / H.264).")
        return VideoValidation(False, errors, warnings)

    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        errors.append("The video opened but no frame could be decoded (corrupt file or unsupported codec).")
        return VideoValidation(False, errors, warnings)
    h, w = first.shape[:2]

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_ok = math.isfinite(fps) and fps > 0
    frame_count = count if count > 0 else None
    duration = (frame_count / fps) if (frame_count and fps_ok) else None
    cap.release()

    if not fps_ok:
        errors.append("The frame rate could not be determined.")
    else:
        if fps < limits.min_fps or fps > limits.max_fps:
            errors.append(f"Frame rate {fps:.1f} fps is outside the supported range "
                          f"{limits.min_fps:.0f}-{limits.max_fps:.0f} fps.")
        elif fps < limits.warn_fps_below:
            warnings.append(f"Frame rate {fps:.1f} fps is low; 24-60 fps is recommended (fast pulls may be missed).")
    if duration is None:
        warnings.append("The container does not report a length; duration limits were not checked.")
    elif duration < limits.min_duration_s:
        errors.append(f"Video is {duration:.1f} s long; at least {limits.min_duration_s:.0f} s is needed to see a rep.")
    elif duration > limits.max_duration_s:
        errors.append(f"Video is {duration / 60:.1f} min long; the limit is {limits.max_duration_s / 60:.0f} min. "
                      "Trim it to the set.")
    short = min(w, h)
    if short < limits.min_short_side_px:
        errors.append(f"Resolution {w}x{h} is too small (short side must be at least {limits.min_short_side_px} px).")
    elif short < limits.warn_short_side_below_px:
        warnings.append(f"Resolution {w}x{h} is low; 720p or 1080p is recommended.")
    elif short > limits.warn_short_side_above_px:
        warnings.append(f"Resolution {w}x{h} is very high; processing will be slow. 1080p is plenty.")

    orientation = "square" if w == h else ("landscape" if w > h else "portrait")
    info = VideoInfo(path=path, extension=ext, size_mb=round(size_mb, 2), width=w, height=h,
                     fps=round(fps, 3) if fps_ok else 0.0, frame_count=frame_count,
                     duration_s=None if duration is None else round(duration, 3), orientation=orientation)
    return VideoValidation(ok=not errors, errors=errors, warnings=warnings, info=info)


def iter_video_frames(path: str, max_seconds: Optional[float] = None) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield ``(t_s, frame_bgr)`` for a video file. ``t_s`` starts near 0 and is strictly increasing.

    Uses the container timestamp of each frame (correct for variable-frame-rate phone video); falls back to
    ``index / fps`` when the backend gives no usable time. Rotation metadata is applied automatically.
    """
    cap = _open(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    step = 1.0 / fps
    prev: Optional[float] = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if math.isfinite(pos) and (prev is None or pos > prev):
                t = pos
            else:                                   # backend gave no (or a repeated) time
                t = 0.0 if prev is None else prev + step
            prev = t
            if max_seconds is not None and t > max_seconds:
                break
            yield t, frame
    finally:
        cap.release()


def assess_tracking(frames_tracked: int, frames_total: int, rep_count: int) -> List[str]:
    """Plain-language warnings about how usable an analysed upload was (the 'enough valid frames?' check)."""
    out: List[str] = []
    if frames_total == 0:
        return ["No frames were processed."]
    frac = frames_tracked / frames_total
    if frames_tracked == 0:
        out.append("No person was detected. Film one person from the side with the whole body in frame.")
    elif frac < 0.6:
        out.append(f"The person was only tracked in {frac:.0%} of frames. Film from the side, full body in frame, "
                   "good lighting, nothing blocking the hips and legs.")
    if frames_tracked > 0 and rep_count == 0:
        out.append("No complete rep was found. Check that the clip shows set-up, the full pull to lockout and the "
                   "return to the floor.")
    return out
