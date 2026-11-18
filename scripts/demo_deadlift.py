"""MyoMotion deadlift demo (40% prototype): pose -> spine proxy -> angles -> rep phases -> form cues.

Examples (run from the repository root):

    python scripts/demo_deadlift.py --source path/to/deadlift.mp4        # UPLOADED video file (validated first)
    python scripts/demo_deadlift.py --source path/to/deadlift.mp4 --validate-only   # just check the file
    python scripts/demo_deadlift.py --source 0                           # live webcam 0
    python scripts/demo_deadlift.py --synthetic lumbar_round --reps 3    # no camera / model needed
    python scripts/demo_deadlift.py --synthetic good --yaw 180           # left side to camera (90 = facing it)

Spine: ``--spine auto`` (default) uses SpinePose (trained spine keypoints) when installed, else the BACK CONTOUR: the
person-mask silhouette of MediaPipe (no extra network, side view only, G-32). ``--spine proxy`` is the old geometric
curve that cannot follow a real spine. SpinePose install: see myomotion/vision/spine_model.py (INSTALL_HINT).

Writes ``<out-dir>/<name>_annotated.mp4`` and ``<out-dir>/<name>_summary.json``. Press Q in the window to stop.
Accepted upload format and filming guidance: docs/VIDEO_INPUT_SPEC.md (side view, whole body, one person, mp4/mov).
Exit codes: 0 = ran, 2 = the uploaded video was rejected (reasons are printed).

Thin CLI only: the logic lives in ``src/myomotion/`` (vision/pose.py, biomechanics/, coaching/overlay.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:                       # lets the script run without `pip install -e .`
    sys.path.insert(0, _SRC)

from myomotion import __version__                                              # noqa: E402
from myomotion.biomechanics import DeadliftAnalyzer, DeadliftRuleConfig          # noqa: E402
from myomotion.biomechanics.synthetic import (SCENARIOS, Body, ClipSpec, generate_clip_3d,  # noqa: E402
                                              generate_clip_3d_masked)
from myomotion.coaching.overlay import OverlayRenderer                          # noqa: E402
from myomotion.vision.video_input import (SUPPORTED_EXTENSIONS, assess_tracking, iter_video_frames,  # noqa: E402
                                          validate_video)

CANVAS_W, CANVAS_H = 1280, 720
# (t_s, image, image landmarks [33,3] or None, world landmarks [33,3] or None, person mask [H,W] or None)
Frame = Tuple[float, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="uploaded video file path, or an integer webcam index (e.g. 0)")
    src.add_argument("--synthetic", choices=sorted(SCENARIOS), help="run a synthetic lifter instead of a camera")
    ap.add_argument("--reps", type=int, default=3, help="reps for --synthetic (default 3)")
    ap.add_argument("--noise-px", type=float, default=1.5, help="landmark jitter for --synthetic (default 1.5 px)")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="--synthetic heading: 0 right side to camera, 180 left side, 90 facing the camera, -90 away")
    ap.add_argument("--model", default=None,
                    help="pose_landmarker_*.task path (default models/pose_landmarker_heavy.task)")
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "output"), help="where to write the video and JSON")
    ap.add_argument("--name", default=None, help="output file stem (default: derived from the source)")
    ap.add_argument("--side", choices=["left", "right"], default=None,
                    help="body side facing the camera (default: auto)")
    ap.add_argument("--no-display", action="store_true", help="do not open a window (headless / batch)")
    ap.add_argument("--no-video", action="store_true", help="do not write the annotated video")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop after this many seconds of input")
    ap.add_argument("--config", default=None, help="JSON file overriding DeadliftRuleConfig fields")
    ap.add_argument("--spine", choices=["auto", "spinepose", "contour", "proxy"], default="auto",
                    help="spine source: SpinePose keypoints, the back silhouette contour (side view, no extra model), "
                         "or the geometric proxy that cannot follow a real curve (auto: SpinePose if installed, "
                         "else contour)")
    ap.add_argument("--spine-size", choices=["small", "medium", "large"], default="small",
                    help="SpinePose model size (small is fastest)")
    ap.add_argument("--spine-every", type=int, default=1, help="run SpinePose every N frames (reuse in between)")
    ap.add_argument("--proxy-spine-cues", action="store_true",
                    help="also raise back-rounding warnings from the geometric proxy spine (off by default: on real "
                         "footage it false-alarms; SpinePose cues are always on)")
    ap.add_argument("--validate-only", action="store_true",
                    help="check an uploaded video against docs/VIDEO_INPUT_SPEC.md and exit (no analysis)")
    return ap.parse_args(argv)


def load_rule_config(path: Optional[str], proxy_cues: bool = False) -> DeadliftRuleConfig:
    cfg = DeadliftRuleConfig(proxy_spine_cues=proxy_cues)
    if path:
        with open(path) as fh:
            overrides = json.load(fh)
        cfg = cfg.with_overrides(**overrides)
    return cfg


def synthetic_frames(scenario: str, reps: int, noise_px: float, yaw: float,
                     masks: bool = False) -> Tuple[Iterator[Frame], float]:
    spec = ClipSpec(n_reps=reps, noise_px=noise_px, seed=1)
    body = Body().scaled(0.62, floor_y=CANVAS_H - 70, x0=CANVAS_W * 0.45)

    def gen() -> Iterator[Frame]:
        kw = dict(yaw_deg=yaw, world_noise_m=0.005, cx=CANVAS_W * 0.45, floor_y=CANVAS_H - 70)
        source = (generate_clip_3d_masked(scenario, spec, body, shape=(CANVAS_H, CANVAS_W), **kw) if masks
                  else ((t, lm, w, None) for t, lm, w in generate_clip_3d(scenario, spec, body, **kw)))
        for t, lm, world, mask in source:
            img = np.full((CANVAS_H, CANVAS_W, 3), 32, np.uint8)
            cv2.putText(img, f"SYNTHETIC: {scenario}, heading {yaw:g} deg", (CANVAS_W - 520, CANVAS_H - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
            yield t, img, lm, world, mask
    return gen(), spec.fps


def webcam_frames(index: str, model: Optional[str], masks: bool = False) -> Tuple[Iterator[Frame], float, str]:
    """Live frames from a webcam, timed with the wall clock (a camera that delivers fewer frames than it
    advertises is still timed correctly). MediaPipe landmarks are attached here."""
    from myomotion.vision.pose import DEFAULT_MODEL_PATH, PoseTracker
    cap = cv2.VideoCapture(int(index))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open webcam {index!r}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    tracker = PoseTracker(model or DEFAULT_MODEL_PATH, masks=masks)

    def gen() -> Iterator[Frame]:
        t0 = time.monotonic()
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                t_s = time.monotonic() - t0
                det = tracker.detect_full(img, t_s)
                yield t_s, img, (det.image if det else None), (det.world if det else None), (det.mask if det else None)
        finally:
            cap.release()
            tracker.close()
    return gen(), fps, "webcam" + index


def uploaded_frames(path: str, fps: float, model: Optional[str], max_seconds: Optional[float],
                    masks: bool = False) -> Iterator[Frame]:
    """Frames of an already-validated uploaded file, timed from the container (correct for variable-frame-rate
    phone video), with MediaPipe landmarks attached."""
    from myomotion.vision.pose import DEFAULT_MODEL_PATH, PoseTracker
    tracker = PoseTracker(model or DEFAULT_MODEL_PATH, masks=masks)

    def gen() -> Iterator[Frame]:
        try:
            for t_s, img in iter_video_frames(path, max_seconds=max_seconds):
                det = tracker.detect_full(img, t_s)
                yield t_s, img, (det.image if det else None), (det.world if det else None), (det.mask if det else None)
        finally:
            tracker.close()
    return gen()


def check_upload(path: str) -> "tuple[int, object]":
    """Validate an uploaded file and print the verdict. Returns (exit_code, VideoValidation)."""
    v = validate_video(path)
    for w in v.warnings:
        print(f"[WARN ] {w}")
    if not v.ok:
        print(f"[REJECT] {os.path.basename(path)} cannot be analysed:")
        for e in v.errors:
            print(f"         - {e}")
        print(f"         Accepted types: {', '.join(SUPPORTED_EXTENSIONS)}. See docs/VIDEO_INPUT_SPEC.md.")
        return 2, v
    i = v.info
    dur = f"{i.duration_s:.1f} s" if i.duration_s is not None else "length unknown"
    print(f"[OK   ] {i.extension} {i.width}x{i.height} {i.orientation}, {i.fps:.1f} fps, {dur}, {i.size_mb:.1f} MB")
    return 0, v


def resolve_spine(args):
    """Decide the spine source. Returns ``(mode, tracker)``: mode is "spinepose" | "contour" | "proxy"; tracker is the
    SpinePoseTracker only for "spinepose". ``auto`` = SpinePose if installed, else the back contour. Exits if SpinePose
    was demanded but is missing. The synthetic lifter has no image for a network to read, so it uses the proxy unless
    ``--spine contour`` asks for its synthetic silhouette."""
    if args.spine in ("proxy", "contour"):
        return args.spine, None
    if args.synthetic:
        return "proxy", None
    from myomotion.vision.spine_model import INSTALL_HINT, SpinePoseTracker, SpinePoseUnavailable, spinepose_available
    if not spinepose_available():
        if args.spine == "spinepose":
            sys.exit(f"[ERROR] --spine spinepose needs the SpinePose package.\n        {INSTALL_HINT}")
        print("[INFO] SpinePose is not installed: estimating the spine from the back silhouette (side view only, "
              "no extra model).\n       For a trained spine network (non-commercial licence): " + INSTALL_HINT)
        return "contour", None
    try:
        print(f"[INFO] loading SpinePose ({args.spine_size}); the model is downloaded on first use ...")
        return "spinepose", SpinePoseTracker(size=args.spine_size)
    except (SpinePoseUnavailable, OSError, RuntimeError) as exc:
        if args.spine == "spinepose":
            sys.exit(f"[ERROR] could not start SpinePose: {exc}")
        print(f"[WARN ] could not start SpinePose ({exc}); using the back silhouette instead")
        return "contour", None


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_rule_config(args.config, proxy_cues=args.proxy_spine_cues)
    spine_mode, spine_tracker = resolve_spine(args)
    use_masks = spine_mode == "contour"
    if args.synthetic:
        frames, fps = synthetic_frames(args.synthetic, args.reps, args.noise_px, args.yaw, masks=use_masks)
        stem, source_desc, is_live = f"synthetic_{args.synthetic}", f"synthetic:{args.synthetic}", False
    elif args.source.isdigit():
        frames, fps, stem = webcam_frames(args.source, args.model, masks=use_masks)
        source_desc, is_live = args.source, True
    else:                                                    # an uploaded file: validate before doing any work
        code, validation = check_upload(args.source)
        if code or args.validate_only:
            return code
        upload_info = validation.info
        fps = upload_info.fps
        frames = uploaded_frames(args.source, fps, args.model, args.max_seconds, masks=use_masks)
        stem, source_desc, is_live = os.path.splitext(os.path.basename(args.source))[0], args.source, False
    if args.validate_only:
        print("[INFO] --validate-only applies to an uploaded video file (--source path/to/file.mp4)")
        return 0
    stem = args.name or stem

    os.makedirs(args.out_dir, exist_ok=True)
    video_path = os.path.join(args.out_dir, f"{stem}_annotated.mp4")
    json_path = os.path.join(args.out_dir, f"{stem}_summary.json")

    analyzer = DeadliftAnalyzer(rule_cfg=cfg, side=args.side)
    renderer = OverlayRenderer()
    spine_pts = None
    spine_bad = 0
    writer = None
    n = 0
    last_wall = time.perf_counter()
    last_problems: tuple = ()
    print(f"[INFO] source={source_desc} nominal_fps={fps:.1f} -> {video_path if not args.no_video else '(no video)'}")
    try:
        for t_s, img, lm, world, mask in frames:
            if args.max_seconds is not None and t_s > args.max_seconds:
                break
            if spine_tracker is not None and lm is not None and n % max(1, args.spine_every) == 0:
                from myomotion.vision.spine_model import bridge_error, person_bbox
                box = person_bbox(lm, (img.shape[1], img.shape[0]))
                det = spine_tracker.detect(img, box) if box is not None else None
                err = bridge_error(det, lm) if det is not None else None
                if det is not None and (err is None or err < 0.35):      # G-01: both models must agree on the body
                    spine_pts = det.points
                else:
                    spine_pts = None
                    spine_bad += det is not None
            elif spine_tracker is not None and lm is None:
                spine_pts = None
            res = analyzer.process(t_s, lm, image_wh=(img.shape[1], img.shape[0]), world=world,
                                   spine_points=spine_pts, mask=mask if use_masks else None)
            problems = res.framing.problems if res.framing else ("No person detected",)
            if problems != last_problems:                       # say what is wrong once per change, not per frame
                for msg in problems:
                    print(f"[FRAME] t={t_s:6.2f}s {msg}")
                if not problems and last_problems:
                    print(f"[FRAME] t={t_s:6.2f}s framing OK: tracking")
                last_problems = problems
            for ev in res.new_events:
                print(f"[FORM] t={ev.t_s:6.2f}s rep {ev.rep_index} {ev.severity.value.upper():7s} {ev.message}")
            if res.completed_rep:
                r = res.completed_rep
                print(f"[REP ] #{r['rep_index']} pull {r['concentric_s']:.2f}s down {r['eccentric_s']:.2f}s "
                      f"tempo {r['tempo_ratio']:.2f} {r['form']['bias']} flags={r['form']['flags'] or 'none'}")
            renderer.render(img, res, analyzer)
            h, w = img.shape[:2]
            if writer is None and not args.no_video:
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    print("[WARN] could not open a video writer; continuing without saving video")
                    args.no_video, writer = True, None
            if writer is not None:
                writer.write(img)
            n += 1
            if not args.no_display:
                cv2.imshow("MyoMotion - Deadlift demo", img)
                wait = 1 if is_live else max(1, int(1000.0 / fps - (time.perf_counter() - last_wall) * 1000.0))
                last_wall = time.perf_counter()
                if cv2.waitKey(wait) & 0xFF == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
        if spine_tracker is not None:
            spine_tracker.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    uploaded = not args.synthetic and not args.source.isdigit()
    meta = {"tool": "myomotion.demo_deadlift", "version": __version__, "source": source_desc, "nominal_fps": fps,
            "frames_written": n, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "annotated_video": None if args.no_video else video_path}
    quality = []
    if uploaded:
        quality = assess_tracking(analyzer.frames_tracked, analyzer.frames_total, analyzer.segmenter.rep_count)
        meta["input"] = "uploaded_video"
        meta["video"] = validation.to_dict()
        meta["quality_warnings"] = quality
    elif args.synthetic:
        meta["input"] = "synthetic"
    else:
        meta["input"] = "live_camera"
    meta["spine_mode"] = spine_mode
    meta["spine_rejected_by_bridge_check"] = spine_bad
    summary = analyzer.summary(meta)
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[DONE] {summary['rep_count']} rep(s) in {summary['duration_s']:.1f}s "
          f"({summary['frames_tracked']}/{summary['frames_total']} frames tracked)")
    raised = {k: v for k, v in summary["flag_counts"].items() if v}
    print(f"       flags: {raised or 'none'}")
    print(f"       summary: {json_path}")
    if not args.no_video:
        print(f"       video:   {video_path}")
    for w in quality:
        print(f"[WARN ] {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
