"""Tests for uploaded-video validation and reading (src/myomotion/vision/video_input.py) and the demo's upload path.

Videos are generated on the fly with OpenCV's VideoWriter, so no fixtures are committed. What is *not* covered
(no ffmpeg available to make such files): HEVC/H.265 footage and rotation-metadata (portrait phone) files.
See docs/VIDEO_INPUT_SPEC.md.
"""

import importlib.util
import json
import os

import cv2
import numpy as np
import pytest

from myomotion.vision import video_input as vi
from myomotion.vision.video_input import (VideoLimits, assess_tracking, iter_video_frames, validate_video)


def make_video(path, size=(640, 360), fps=30.0, seconds=3.0, fourcc="mp4v"):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    assert w.isOpened(), f"cannot write {fourcc} to {path}"
    for i in range(int(seconds * fps)):
        f = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.circle(f, (20 + (i * 7) % (size[0] - 40), size[1] // 2), 12, (255, 255, 255), -1)
        w.write(f)
    w.release()
    return str(path)


# ---------------------------------------------------------------------------
# validate_video
# ---------------------------------------------------------------------------
def test_valid_landscape_mp4_is_accepted_with_correct_info(tmp_path):
    v = validate_video(make_video(tmp_path / "a.mp4", size=(1280, 720)))
    assert v.ok and v.errors == [] and v.warnings == []          # 720p at 30 fps is the recommended format
    i = v.info
    assert (i.width, i.height, i.orientation, i.extension) == (1280, 720, "landscape", ".mp4")
    assert abs(i.fps - 30.0) < 0.01 and abs(i.duration_s - 3.0) < 0.1 and i.frame_count == 90


def test_below_recommended_resolution_is_accepted_with_a_warning(tmp_path):
    v = validate_video(make_video(tmp_path / "small.mp4", size=(640, 360)))   # short side 360: above reject, below warn
    assert v.ok and any("Resolution 640x360 is low" in w for w in v.warnings)


def test_portrait_orientation_is_reported(tmp_path):
    v = validate_video(make_video(tmp_path / "p.mp4", size=(540, 960)))
    assert v.ok and v.info.orientation == "portrait" and v.info.height == 960


@pytest.mark.parametrize("name, fourcc", [("a.mp4", "mp4v"), ("b.mp4", "avc1"), ("c.mov", "mp4v"), ("d.mov", "avc1"),
                                          ("e.m4v", "mp4v")])
def test_recommended_containers_and_codecs_decode(tmp_path, name, fourcc):
    v = validate_video(make_video(tmp_path / name, size=(960, 540), fourcc=fourcc))
    assert v.ok, v.errors
    assert not any("best-effort" in w for w in v.warnings)       # these are the recommended types


@pytest.mark.parametrize("name, fourcc", [("e.avi", "MJPG"), ("f.mkv", "mp4v")])
def test_best_effort_containers_decode_with_a_warning(tmp_path, name, fourcc):
    v = validate_video(make_video(tmp_path / name, size=(960, 540), fourcc=fourcc))
    assert v.ok, v.errors
    assert any("best-effort" in w for w in v.warnings)


def test_missing_empty_wrong_type_and_corrupt_files_are_rejected_with_reasons(tmp_path):
    missing = validate_video(str(tmp_path / "nope.mp4"))
    assert not missing.ok and "not found" in missing.errors[0].lower()

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert not validate_video(str(empty)).ok and "empty" in validate_video(str(empty)).errors[0].lower()

    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    bad_type = validate_video(str(txt))
    assert not bad_type.ok and "Unsupported file type '.txt'" in bad_type.errors[0]

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(os.urandom(5000))
    c = validate_video(str(corrupt))
    assert not c.ok and "could not be opened" in c.errors[0] or "no frame could be decoded" in c.errors[0]


def test_limits_reject_short_tiny_and_extreme_fps_but_only_warn_on_soft_limits(tmp_path):
    assert any("at least" in e for e in validate_video(make_video(tmp_path / "short.mp4", seconds=1.0)).errors)
    assert any("too small" in e for e in validate_video(make_video(tmp_path / "tiny.mp4", size=(160, 120))).errors)
    assert any("outside the supported range" in e
               for e in validate_video(make_video(tmp_path / "slow.mp4", fps=5.0, seconds=3)).errors)
    soft = validate_video(make_video(tmp_path / "soft.mp4", size=(320, 240), fps=15.0, seconds=3))
    assert soft.ok                                               # accepted...
    assert any("Resolution" in w for w in soft.warnings)                      # ...but warned about both
    assert any("Frame rate" in w for w in soft.warnings)
    tight = VideoLimits(max_duration_s=2.0)
    assert any("limit is" in e for e in validate_video(make_video(tmp_path / "long.mp4", seconds=3), tight).errors)


def test_validation_dict_is_json_serialisable(tmp_path):
    v = validate_video(make_video(tmp_path / "a.mp4"))
    assert json.loads(json.dumps(v.to_dict()))["ok"] is True


# ---------------------------------------------------------------------------
# iter_video_frames
# ---------------------------------------------------------------------------
def test_frames_have_container_timestamps_that_are_strictly_increasing(tmp_path):
    path = make_video(tmp_path / "a.mp4", fps=30.0, seconds=2.0)
    ts = [t for t, _ in iter_video_frames(path)]
    assert len(ts) == 60 and ts[0] == pytest.approx(0.0, abs=1e-3)
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert max(abs(t - i / 30.0) for i, t in enumerate(ts)) < 1e-3


def test_max_seconds_truncates(tmp_path):
    path = make_video(tmp_path / "a.mp4", seconds=3.0)
    ts = [t for t, _ in iter_video_frames(path, max_seconds=1.0)]
    assert ts and max(ts) <= 1.0 and len(ts) < 90


def test_timestamps_fall_back_to_fps_when_backend_reports_no_time(tmp_path, monkeypatch):
    class FakeCap:
        def __init__(self):
            self.n = 0

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            self.n += 1
            return (self.n <= 5), np.zeros((10, 10, 3), np.uint8)

        def get(self, prop):
            return 25.0 if prop == cv2.CAP_PROP_FPS else 0.0     # POS_MSEC stuck at 0

        def release(self):
            pass

    monkeypatch.setattr(vi, "_open", lambda _p: FakeCap())
    ts = [t for t, _ in iter_video_frames("whatever.mp4")]
    assert ts == pytest.approx([0.0, 0.04, 0.08, 0.12, 0.16])
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_unopenable_file_raises(tmp_path):
    with pytest.raises(IOError):
        list(iter_video_frames(str(tmp_path / "nope.mp4")))


# ---------------------------------------------------------------------------
# assess_tracking
# ---------------------------------------------------------------------------
def test_tracking_assessment_messages():
    assert assess_tracking(0, 0, 0) == ["No frames were processed."]
    assert "No person was detected" in assess_tracking(0, 100, 0)[0]
    assert "only tracked in 40%" in assess_tracking(40, 100, 0)[0]
    assert any("No complete rep" in m for m in assess_tracking(90, 100, 0))
    assert assess_tracking(95, 100, 3) == []


# ---------------------------------------------------------------------------
# Demo CLI: upload path
# ---------------------------------------------------------------------------
def _demo():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "demo_deadlift.py")
    spec = importlib.util.spec_from_file_location("demo_deadlift_upload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_rejects_a_bad_upload_with_exit_code_2_and_reasons(tmp_path, capsys):
    txt = tmp_path / "clip.txt"
    txt.write_text("not a video")
    rc = _demo().main(["--source", str(txt), "--no-display", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 2 and "[REJECT]" in out and "Unsupported file type" in out and "VIDEO_INPUT_SPEC" in out
    assert not list(tmp_path.glob("*_summary.json"))                  # nothing was analysed


def test_cli_validate_only_needs_no_model_and_writes_nothing(tmp_path, capsys):
    path = make_video(tmp_path / "ok.mp4", size=(960, 540))
    rc = _demo().main(["--source", path, "--validate-only", "--out-dir", str(tmp_path / "out")])
    assert rc == 0 and "[OK   ]" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker_heavy.task")


@pytest.mark.skipif(not os.path.isfile(MODEL), reason="MediaPipe model not downloaded (models/ is gitignored)")
def test_cli_analyses_an_upload_and_warns_when_no_person_is_found(tmp_path, capsys):
    path = make_video(tmp_path / "empty_scene.mp4", size=(960, 540), seconds=2.0)
    rc = _demo().main(["--source", path, "--no-display", "--no-video", "--out-dir", str(tmp_path)])
    assert rc == 0
    summary = json.loads((tmp_path / "empty_scene_summary.json").read_text())
    assert summary["meta"]["input"] == "uploaded_video" and summary["meta"]["video"]["ok"] is True
    assert summary["frames_tracked"] == 0
    assert any("No person was detected" in w for w in summary["meta"]["quality_warnings"])
    assert "[WARN ] No person was detected" in capsys.readouterr().out
