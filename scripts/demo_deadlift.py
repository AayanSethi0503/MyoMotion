"""MyoMotion deadlift demo (40% prototype): pose -> spine proxy -> angles -> rep phases -> form cues.

Examples (run from the repository root):

    python scripts/demo_deadlift.py --source path/to/deadlift.mp4        # video file
    python scripts/demo_deadlift.py --source 0                           # webcam 0
    python scripts/demo_deadlift.py --synthetic lumbar_round --reps 3    # no camera / model needed

Writes ``<out-dir>/<name>_annotated.mp4`` and ``<out-dir>/<name>_summary.json``. Press Q in the window to stop.
Film from the SIDE, full body in frame, camera ~2-3 m away, ideally at hip height.

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
from myomotion.biomechanics.synthetic import SCENARIOS, Body, ClipSpec, generate_clip  # noqa: E402
from myomotion.coaching.overlay import OverlayRenderer                          # noqa: E402

CANVAS_W, CANVAS_H = 1280, 720
Frame = Tuple[float, np.ndarray, Optional[np.ndarray]]      # (t_s, image, landmarks or None if the tracker should run)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="video file path, or an integer webcam index (e.g. 0)")
    src.add_argument("--synthetic", choices=sorted(SCENARIOS), help="run a synthetic lifter instead of a camera")
    ap.add_argument("--reps", type=int, default=3, help="reps for --synthetic (default 3)")
    ap.add_argument("--noise-px", type=float, default=1.5, help="landmark jitter for --synthetic (default 1.5 px)")
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
    return ap.parse_args(argv)


def load_rule_config(path: Optional[str]) -> DeadliftRuleConfig:
    cfg = DeadliftRuleConfig()
    if path:
        with open(path) as fh:
            overrides = json.load(fh)
        cfg = cfg.with_overrides(**overrides)
    return cfg


def synthetic_frames(scenario: str, reps: int, noise_px: float) -> Tuple[Iterator[Frame], float]:
    spec = ClipSpec(n_reps=reps, noise_px=noise_px, seed=1)
    body = Body().scaled(0.62, floor_y=CANVAS_H - 70, x0=CANVAS_W * 0.45)

    def gen() -> Iterator[Frame]:
        for t, lm in generate_clip(scenario, spec, body):
            img = np.full((CANVAS_H, CANVAS_W, 3), 32, np.uint8)
            cv2.putText(img, f"SYNTHETIC: {scenario}", (CANVAS_W - 420, CANVAS_H - 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (90, 90, 90), 2, cv2.LINE_AA)
            yield t, img, lm
    return gen(), spec.fps


def camera_frames(source: str, model: Optional[str]) -> Tuple[Iterator[Frame], float, str]:
    """Frames from a webcam or a video file, with MediaPipe landmarks attached lazily in the main loop."""
    from myomotion.vision.pose import DEFAULT_MODEL_PATH, PoseTracker
    is_live = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_live else source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video source: {source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    tracker = PoseTracker(model or DEFAULT_MODEL_PATH)

    def gen() -> Iterator[Frame]:
        t0, idx = time.monotonic(), 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                # Live: wall clock (a camera that delivers fewer frames than advertised is still timed right).
                # File: frame index / nominal fps (assumes a constant frame rate).
                t_s = (time.monotonic() - t0) if is_live else idx / fps
                idx += 1
                yield t_s, img, tracker.detect(img, t_s)
        finally:
            cap.release()
            tracker.close()
    return gen(), fps, ("webcam" + source) if is_live else os.path.splitext(os.path.basename(source))[0]


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_rule_config(args.config)
    if args.synthetic:
        frames, fps = synthetic_frames(args.synthetic, args.reps, args.noise_px)
        stem, source_desc, is_live = f"synthetic_{args.synthetic}", f"synthetic:{args.synthetic}", False
    else:
        frames, fps, stem = camera_frames(args.source, args.model)
        source_desc, is_live = args.source, args.source.isdigit()
    stem = args.name or stem

    os.makedirs(args.out_dir, exist_ok=True)
    video_path = os.path.join(args.out_dir, f"{stem}_annotated.mp4")
    json_path = os.path.join(args.out_dir, f"{stem}_summary.json")

    analyzer = DeadliftAnalyzer(rule_cfg=cfg, side=args.side)
    renderer = OverlayRenderer()
    writer = None
    n = 0
    last_wall = time.perf_counter()
    print(f"[INFO] source={source_desc} nominal_fps={fps:.1f} -> {video_path if not args.no_video else '(no video)'}")
    try:
        for t_s, img, lm in frames:
            if args.max_seconds is not None and t_s > args.max_seconds:
                break
            res = analyzer.process(t_s, lm)
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
        if not args.no_display:
            cv2.destroyAllWindows()

    meta = {"tool": "myomotion.demo_deadlift", "version": __version__, "source": source_desc, "nominal_fps": fps,
            "frames_written": n, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "annotated_video": None if args.no_video else video_path}
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
