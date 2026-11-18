# Video input specification (uploaded footage)

MyoMotion takes **two kinds of input**: a **live camera** and an **uploaded video file** (offline batch). Neither needs any sensor on the person. This page is the contract for uploaded files. The checks are implemented in `src/myomotion/vision/video_input.py` (`validate_video`), and a file that fails them is rejected with reasons before any analysis runs.

## 1. File format

| | Accepted | Notes |
|---|---|---|
| **Container (recommended)** | `.mp4`, `.mov`, `.m4v` | what iPhone and Android cameras record |
| **Container (best effort)** | `.avi`, `.mkv`, `.webm` | decodes, but accepted with a warning |
| **Video codec** | H.264 / AVC, MPEG-4 Part 2 | **verified** to decode on this project's OpenCV 5.0.0 build (FFmpeg + AVFoundation) |
| **HEVC / H.265** | depends on the OpenCV FFmpeg build | **not verified**. iPhone "High Efficiency" mode records HEVC. If a file is rejected as undecodable, re-export as **H.264** (iPhone: Settings → Camera → Formats → *Most Compatible*) |
| **Audio** | ignored | may be present |
| **Not supported** | images, streaming URLs, DRM-protected files, multi-file sets | one set per file |

## 2. Technical limits

| Property | Recommended | Accepted (warning) | Rejected |
|---|---|---|---|
| Resolution (short side) | 720 p – 1080 p (1280×720 or 1920×1080) | 240 – 479 px; above 1440 px (slow) | below 240 px |
| Frame rate | 24 – 60 fps | 10 – 23 fps | below 10 or above 120 fps |
| Duration | 5 – 90 s (one set) | up to 10 min | under 2 s, over 10 min |
| File size | under 500 MB | up to 2 GB | over 2 GB |
| Orientation | landscape or portrait | square | n/a |
| Frame timing | constant or variable frame rate | n/a | undeterminable frame rate |

- **Variable frame rate (phone video):** frame times are read from the container, so timing stays correct.
- **Portrait footage:** OpenCV is asked to apply the file's rotation metadata automatically. This is requested in code but **not verified**, because no ffmpeg was available to build a test file with rotation metadata.
- **Low frame rate** (< 24 fps) under-samples a fast pull, and phase boundaries get coarser. Prefer 30 or 60 fps.

## 3. Content requirements (what to film)

The pipeline reads one side of the body in a 2-D image, so framing decides whether the result is usable.

1. **Side view.** The camera is at 90° to the lifter. The side facing the camera is analysed, and left or right is chosen automatically. A front or rear view is **not** supported for the deadlift (the frontal-plane angles exist in code but are not wired into the demo).
2. **Whole body in frame in every frame**, head to feet, including the bar and plates at the floor. Leave margin at the top for the lockout.
3. **One person.** Only one pose is tracked. Other people may appear but should not overlap the lifter.
4. **Static camera.** Use a tripod or stable prop, with no panning, zooming or hand-held shake. Roughly hip height, 2–3 m away.
5. **Unedited.** No crop, filter, slow-motion, time-lapse or stabilisation. Mirrored video is fine.
6. **Good lighting and a plain background.** Avoid hips and legs hidden behind the bar, plates, a rack or loose clothing.
7. **Show the whole set.** The clip should start with the lifter in the set-up position and include full pulls to lockout and returns to the floor. A clip with no complete rep is analysed, but warns.

## 4. What the system tells you after an upload

- **Rejected** (`exit code 2`): the file breaks a hard limit or can't be decoded. Each reason is printed. Nothing is analysed.
- **Accepted with warnings:** low resolution, low frame rate, best-effort container. Analysis runs.
- **Quality warnings after analysis** (stored in the JSON under `meta.quality_warnings`):
  - no person detected;
  - the person was tracked in under 60% of frames;
  - no complete rep found.

## 5. How to check a file without analysing it

```bash
python scripts/demo_deadlift.py --source path/to/video.mp4 --validate-only
```

Prints either `[OK   ]` with the detected format (type, size, orientation, fps, length) or `[REJECT]` with reasons. It needs no model download.

## 6. Status of these limits

The numeric limits are provisional product choices, logged as G-28 in `docs/GAP_ANALYSIS.md`. No paper specifies them. What is verified: that the listed containers decode, that H.264 works, and that timestamps are read from the container. What is not verified: HEVC, rotation metadata, and detection quality on real footage of lifters.
