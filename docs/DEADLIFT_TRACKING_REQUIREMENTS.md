# What the deadlift tracker needs: camera, pose, and what counts as a rep

Read this before filming or using the webcam. Every number here is a constant in `src/myomotion/biomechanics/` (`rep_segmenter.py`, `landmarks.py`), so this page matches what the code does. For the format of uploaded video files see `docs/VIDEO_INPUT_SPEC.md`.

**Hip angle** below means the angle at your hip between your torso and your thigh. 180° is standing straight; the smaller it is, the more you are folded forward. The screen shows it as `Hip`.

---

## 1. Camera and framing

| Requirement | Detail |
|---|---|
| **Heading** | **Side-on (90° to the camera) is best.** Either side (left or right to the camera) and either direction works, and the app follows you if you turn around. **Angled or facing the camera also works**: it then switches to a 3-D estimate (MediaPipe world landmarks), which is less exact, and **spine curvature cannot be seen** from the front or back, so there are no back-rounding cues in that view |
| **Whole leg in frame, always** | To **count reps** and compute hip/knee angles the camera must see your **shoulder, hip, knee and ankle** (on the side facing it) in **every frame**, including the bottom of the lift. Ideally the whole body, head to feet. **If the legs are out of frame the back is still monitored** (shoulder, hip and the spine are enough, section 7), but no reps are counted |
| **Distance** | Far enough for that. A laptop webcam needs roughly **3 m** (about 10 ft). Phone cameras: about 2–3 m. If the feet are cut off, step back |
| **Camera position** | **Static** (tripod, shelf, stack of books). Around **hip height**, level, not tilted. No handheld, no zoom, no panning |
| **One person** | Only one pose is tracked. Keep others out of the shot |
| **Visibility** | Good light, plain background, nothing hiding your hip, knee or ankle (a rack, plates, loose long coat, another person) |
| **Frame rate** | At least **8 fps** on the `fps` readout (top left). Below that the counter becomes unreliable (see section 5) |

## 2. Which joints matter

| Joint (the side facing the camera) | Needed? | If it is missing |
|---|---|---|
| shoulder, hip | **Required for everything**, including the back monitor | nothing is analysed |
| knee, ankle | **Required for reps and leg angles** | `BACK MONITOR` instead of `NOT TRACKING`: no reps, no hip/knee angles, but the back cue still runs. The screen says which joint is missing |
| ear / head | optional | the spine proxy is off (SpinePose, if installed, does not need it). Reps still count |
| heel + toe | optional | the bar-path check is off; facing direction is taken from other cues. Reps still count |
| wrist | optional | bar-height and bar-drift tracking are off |

The red/green checklist on the left of the screen shows each joint live. **Green dot** = seen. **Red ring** = MediaPipe is not confident about that joint.

## 3. The pose and movement that registers one rep

The rep counter is a sequence of stages. **Every stage must happen, in this order, for a rep to be counted.**

| # | Stage (shown on screen) | What you do | What the code checks |
|---|---|---|---|
| 1 | `STANDBY` → `SETUP` | **Hinge forward into your starting position and hold still.** Bar on the floor, or hands hanging low for a bodyweight or empty-bar test | hip angle **≤ 130°** and moving slower than 8°/s, for **0.3 s** |
| 2 | `CONCENTRIC PULL` | Start to stand up | hip rising at **≥ 8°/s** and **≥ 4°** above the start, held for 0.1 s (or hip passes 140°) |
| 3 | still `CONCENTRIC PULL`, then `LOCKOUT` | **Stand all the way up, hips and knees fully extended** | hip angle **≥ 160° AND knee angle ≥ 160°**, for at least **0.1 s**. If you then stop moving (< 8°/s for 0.1 s) the phase becomes `LOCKOUT` |
| 4 | `ECCENTRIC DESCENT` | Lower back down | hip falling at ≥ 8°/s and ≥ 4° below the top, for 0.1 s (or hip drops below 150°) |
| 5 | `REP COMPLETE` | **Return to the starting position and pause briefly.** This is when the rep is counted | hip angle **≤ 130°** and no longer moving down (faster than −8°/s), for **0.15 s** |

After `REP COMPLETE` (0.4 s) it goes back to `SETUP`. You can start the next pull straight away (touch-and-go) or pause.

### Typical angles, for reference
- Conventional deadlift set-up: hip ≈ 60–100°, knee ≈ 100–140° (comfortably under the 130° hip limit).
- Romanian / stiff-leg deadlift: hip ≈ 80–120° at the bottom, knees stay ≥ 160° throughout. Counts normally.
- Lockout: hip and knee both ≈ 170–180°.

### What does NOT count as a rep
- **Starting from standing** without hinging down: there is no set-up, so nothing starts. Bend into the start position first.
- **Not folding forward enough** at the bottom (hip stays above 130°): a shallow hinge or a top-half pull never registers a set-up or a finish.
- **Not standing fully tall** (hip or knee stays under 160°): half reps, partial lockouts and hang-pulls are not counted. They are treated as a failed attempt.
- **Not returning** to hip ≤ 130° before the next pull: the rep is not closed.
- **Stalling** more than 5 s partway up or down: the attempt is abandoned and the counter returns to `STANDBY`.
- **Losing tracking** for more than 1 s: also returns to `STANDBY` (a short dropout under 1 s is ignored).

### Pace
No particular speed is needed, but the transitions need short stable moments: **settle for about half a second at the start**, and don't bounce through the bottom. A 2–3 s up and 2–3 s down lift is ideal.

## 4. Stance and set-up (what to physically do)

- **Feet**: hip-width (conventional) or wide (sumo). Both work, because only the hip and knee angles seen from the side are used. Feet flat, and the ankle must be visible.
- **Hands/bar**: hanging straight down. A real bar is not needed: a **broomstick or just your hands** is fine for testing the tracking. The wrist stands in for the bar.
- **Spine**: neutral. The back-rounding cue compares you to **your own set-up**, so set up the way you want to lift. A set-up that is already rounded is invisible to this check.
- **Clothing**: fitted around the hip, knee and ankle so the joints are visible.

## 5. Reading the screen

| You see | Meaning | Do this |
|---|---|---|
| `BACK MONITOR` + `Not tracking reps: legs out of frame` | knee or ankle not visible, so reps are not counted; the back is still analysed (trunk angle, spine regions, rounding cue) | step back ~3 m to count reps; for back feedback alone you can stay |
| `NOT TRACKING` + a red `MISSING hip` etc. | shoulder or hip not visible | follow the advice line (usually: step back) |
| `HINGED: back cue armed` | your trunk is >= 30 deg from vertical, so the back-rounding rule is active | n/a |
| `BACK ROUNDING - chest up, shoulders back` | the upper/mid-back (thoracic) region of the SpinePose spine is convex (>= +6 deg) while hinged | lift the chest, pull the shoulders back |
| `View: angled (3D estimate)` | you are not side-on, so angles come from the 3-D estimate | fine for counting; turn side-on for exact angles and spine feedback |
| `Spine: curve not visible from this angle` | front/back view: curvature cannot be seen | turn side-on to get spine feedback |
| `Spine (proxy: display only)` | the spine curve is only geometry through ear/shoulder/hip and **cannot follow a real curved back** (shown only with `--spine proxy`, or for a few frames when neither SpinePose nor the silhouette is available) | use the default (`--spine auto`); SpinePose install: `requirements-spine.txt` |
| `You look small in the frame` | your body is under 30% of the frame height | move closer or zoom in |
| `LOW FRAME RATE (n fps)` | the computer is too slow for reliable counting | close other apps, use a lighter pose model (`--model models/pose_landmarker_lite.task`) |
| Bottom line: `START: hinge forward…` | waiting for stage 1 | fold forward to ≤ 130° and hold still |
| Bottom line: `Stand fully tall…` | waiting for stage 3 | stand all the way up (hip ≥ 160°, knee ≥ 160°) |
| Bottom line: `Lower to the start…` | waiting for stage 5 | return to ≤ 130° and pause |
| `Reps: n` goes up | a rep was counted | n/a |

## 6. Webcam versus phone video

A laptop webcam is a hard case: the lens is low and wide, so getting the whole body in frame needs about 3 m of clear space, and webcams are often 640×480 at low frame rates. For the best results **film with a phone** (side-on, tripod, 1080p, 30 fps) and **upload the file** (`--source file.mp4`); see `docs/VIDEO_INPUT_SPEC.md`.

## 7. The spine line: what it is and is not

- **`(back contour)`** (cyan, the default): the spine is estimated from the **silhouette of your back** (MediaPipe's person mask), so the line follows a curved back. Needs a **side-on view**; loose or baggy clothing hides the true back surface and makes it less accurate; the lower back is the least reliable region. Warnings from it are conservative and labelled "possible". Not yet validated against markers.
- **`(SpinePose)`** (blue, used automatically when installed): nine points **detected from the image** by a trained model, used **only for the spine**; MediaPipe still tracks every other joint. Works from any view (from the front/back you get a vertical midline only, no curvature). Region angles (lumbar / thoracic / cervical) come from a smooth fit through the nine points; they under-read a true change by about 3x. **The back-rounding cue uses the thoracic (upper/mid-back) angle, absolute, only while your trunk is hinged**; the **lumbar angle is displayed but raises no cue by default** (it disagreed between SpinePose sizes and false-alarmed on neutral reps; `--lumbar-cues` turns it on). Default model size is `medium` (about 9 ms per frame on a laptop; not measured on a phone), with a non-commercial licence (D-05). Facing from the side is required for curvature: from the front there is no back-rounding cue.
- **`(proxy)`** (magenta, `--spine proxy` only): a smooth curve through your ear, shoulder and hip. It has no information about the real spine between them, so it can look curved when you are straight and straight when you are curved. It is **display only**; warnings from it are off unless you pass `--proxy-spine-cues`.
- **Front/back view**: a straight muted midline, labelled "curve not visible here".

## 8. Status

The thresholds (130° / 160° / 8°/s and the dwell times) were chosen for the original tracker and are covered by synthetic tests only. They have **not** been validated on a labelled set of real lifts. The framing and view thresholds (angled view from ~25° off side-on, small-in-frame 30%) are provisional (G-29, G-30 in `docs/GAP_ANALYSIS.md`). The 3-D path was checked against real footage only for the trunk angle (within ~3-4° of the 2-D value on 4 frames); the back-contour spine was checked on one real recording only (no ground truth, no false alarms in 4 reps); the SpinePose path was run on that same recording (4 reps, no rounding cue, single-frame keypoint glitches of +35 and +79 deg were filtered out by the 0.25 s hold) but that recording has an old overlay drawn into it and **no ground truth exists**, so its accuracy is unknown. The SpinePose upper/mid-back cue (thoracic >= +6 deg while hinged) was calibrated **by eye on two recordings of one lifter** (about 25 neutral and 80-100 rounded frames) and is **not validated**; it can false-alarm on people with a naturally rounded upper back. See G-34 in `docs/GAP_ANALYSIS.md` for the evidence, the model-size comparison and what the papers do and do not provide. The per-frame spine angles are written to `<name>_spine.csv`, and `--save-raw` keeps the untouched camera frames, so a session can be re-analysed and the thresholds tuned on footage the team has labelled.
