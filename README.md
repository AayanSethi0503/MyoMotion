# MyoMotion (Still in the Development Phase)

Final Year Project 


MyoMotion is an advanced biomechanical coaching platform that fuses computer vision and electromyography by deploying YOLO, MediaPipe pose tracking, and spine aware feature extraction on a standard 2D camera feed, utilising sEMG calibrated deep learning architectures, to perform high precision, phase segmented muscle activation estimation (%MVC) and deliver real-time causal corrective feedback.



**Team:** Ahmed Nawaz · Aayan Sethi · Wajid Ali  


---

## Project Structure

```
MyoMotion/
├── src/myomotion/             # Python package: vision/, biomechanics/, coaching/, (semg/, sync/, models/ planned)
├── scripts/
│   ├── demo_deadlift.py       # deadlift demo CLI
│   ├── cv/                    # Legacy trackers
│   │   └── deadlift/
│   │       └── deadlift_tracker.py
│   ├── semg/                  # sEMG acquisition, MVC normalization, serial logger
│   └── sync/                  # Video–sEMG synchronization (LED/timestamp alignment)
├── training/                  # Model training code, configs, evaluation scripts
├── models/                    # Downloaded model weights (not committed to git)
│   └── pose_landmarker_heavy.task
├── data/                      # Datasets (not committed to git)
│   ├── raw/                   # Per-participant synchronized recordings
│   └── processed/             # Cleaned, feature-extracted, ready for training
├── notebooks/                 # Exploratory Jupyter notebooks
├── app/                       # End-user application (mobile / web)
├── hardware/                  # Arduino sketches, wiring diagrams, sEMG module docs
├── diagrams/                  # UML, ERD, architecture (.puml, .png, .svg)
├── docs/                      # Context.md, SRS, literature
├── CLAUDE.md                  # Agent context file
├── ROADMAP.md                 # Development phases
├── requirements.txt           # Python dependencies
└── .gitignore
```

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the MediaPipe model (if not present)
#    Place pose_landmarker_heavy.task into models/
```

## Running

```bash
# Always run from the project root

# Deadlift demo: pose + spine proxy + angles + rep phases + form cues. Writes output/*.mp4 and output/*.json
python scripts/demo_deadlift.py --source path/to/side_view_deadlift.mp4              # uploaded video (validated first)
python scripts/demo_deadlift.py --source path/to/side_view_deadlift.mp4 --validate-only   # check the file only
python scripts/demo_deadlift.py --source 0                  # live webcam (add --save-raw to keep the untouched frames)
python scripts/demo_deadlift.py --synthetic lumbar_round    # no camera or model needed

# Legacy minimal tracker (hip/knee angles + phase only)
python scripts/cv/deadlift/deadlift_tracker.py

# Tests
python -m pytest tests -q
```

Either body side, any heading (side-on is best; facing the camera works via a 3-D estimate). MediaPipe tracks the body; the **spine** comes from SpinePose when it is installed (`--spine auto`, G-31, G-33) and otherwise from your back silhouette (side-on, no extra model, G-32). SpinePose is an optional, **non-commercially licensed** install:

```bash
pip install -r requirements-spine.txt
pip install --no-deps spinepose==2.1.0   # weights download on first use to ~/.cache/spinepose
python scripts/demo_deadlift.py --source path/to/video.mp4 --spine spinepose   # demand it (fails clearly if absent)
```

What the tracker needs to count a rep (camera, pose, joints): `docs/DEADLIFT_TRACKING_REQUIREMENTS.md`.
Uploaded footage: `.mp4` / `.mov` / `.m4v` (H.264), 720p-1080p, 24-60 fps, side view, whole body in frame; full spec in `docs/VIDEO_INPUT_SPEC.md`.
Filming: **side** view, full body in frame, camera about 2-3 m away. Press **Q** to quit a live window.
The back is monitored even when your legs are out of frame (no reps are counted then); the back-rounding cue reads the upper/mid-back angle of the SpinePose spine while you are hinged (`--lumbar-cues` adds the noisier lumbar cue). A per-frame spine log is written to `output/<name>_spine.csv`. None of the spine thresholds is validated against ground truth yet (G-33, G-34).

## Tech Stack

| Layer | Tool |
|---|---|
| Pose estimation | MediaPipe Pose Landmarker (Tasks API) |
| Computer vision | OpenCV |
| EMG hardware | Arduino Uno + 3× AD8226 modules |
| ML target | %MVC regression (scikit-learn / PyTorch / Keras Tensorflow) |

