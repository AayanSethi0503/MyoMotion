# MyoMotion — Gap Analysis Register

_v1.0 (2026-11-15). This file lists every method that does **not** come from papers [P1]–[P8] (keys in `docs/METHODOLOGY_SPEC.md`). Rule: no external method enters the code without an entry here. Each entry gives the gap, why the papers don't close it, the chosen solution, the interface to paper-derived modules, and how it gets validated._

The brief named four expected gaps (G-01…G-04). Reading the papers turned up more genuine gaps (G-05…G-22). They're logged too rather than silently filled.

Status: **Decided** = implement as written. **Provisional** = implement behind a config flag and validate. **Open** = needs a team decision.

---

## Index

| ID | Gap | Status |
|---|---|---|
| G-01 | MediaPipe 33-landmark ↔ SpinePose 37-keypoint bridge | Decided |
| G-02 | Monocular 3D root-centring and scale normalisation | Decided |
| G-03 | Multi-rate video ↔ sEMG synchronisation, ring buffers, serial budget | Decided |
| G-04 | Coaching cue hysteresis and arbitration | Decided |
| G-05 | Hardware outputs an envelope, but the paper DSP assumes raw EMG | **Open** (bench check) |
| G-06 | 3-channel hardware means each block measures a different muscle subset | Decided |
| G-07 | MIA's unmasked attention vs live (causal) coaching | Provisional |
| G-08 | Transformer hyperparameters not given in MIA main text | Provisional |
| G-09 | Content of the athlete conditioning tensor; concat dimension | Provisional |
| G-10 | Transplanting [P3] classification transfer learning to regression | Provisional |
| G-11 | MVC protocol (isometric trials) | Decided |
| G-12 | "200 ms window, 10 ms overlap" ambiguity | Decided |
| G-13 | Monocular adaptation of [P6] correction; polynomial order; FK rendering | Provisional |
| G-14 | SpinePose smoothness-loss $\alpha_s$, $T$ not given | Provisional |
| G-15 | lr scaling when the effective batch < 1024 | Decided |
| G-16 | Regional spine angle / pelvic tilt definitions; butt-wink thresholds | **Open** (needs data) |
| G-17 | Landmark signal conditioning (median, jump gate, One-Euro) | Decided (implemented) |
| G-18 | Notch Q; zero-phase vs causal filtering | Decided |
| G-19 | Feature thresholds, bands, wavelet histogram; rectified vs non-rectified input | Decided |
| G-20 | Statistical test for the fatigue trend | Provisional |
| G-21 | ICC form | Decided |
| G-22 | −1 padding collides with legal feature values | Decided |
| G-23 | Deadlift rule set is not in [P1]–[P8] (only [X1] names deadlift error classes) | Provisional |
| G-24 | Spine-proxy curve construction, vertebral-count arc fractions, baseline-relative flexion measures | Provisional |
| G-25 | Applying [P8]'s squat trunk–tibia thresholds to the deadlift | Provisional |
| G-26 | Rep segmentation: legacy state machine kept, bar half-cycle detector added, `REP_COMPLETE` phase | Decided |
| G-27 | Signed hip angle (> 180 deg) so a hyperextended lockout is not read as a descent | Decided |
| G-28 | Uploaded-video acceptance limits and timestamping | Provisional |
| G-29 | Core vs optional joints, framing guidance, facing vote, low-fps velocity rescue | Provisional |
| G-30 | Both body sides, any heading: dynamic side, 3-D world-landmark angles, view hysteresis | Provisional |
| G-31 | Real spine: SpinePose keypoints as the spine source (optional dependency, CC-BY-NC) | **Installed and run on real footage 2026-11-19; not validated against ground truth** |
| G-32 | Spine from the back silhouette (MediaPipe mask): the default now, no extra network | Provisional; checked on one real recording only |
| G-33 | Region angles from detected spine points: weighted cubic fit in the spine's own frame (any body orientation) | Provisional; shrinks true change ~3.5x; lumbar weakest |
| G-34 | Back rounding without visible legs: hinge-gated ABSOLUTE thoracic angle from SpinePose, lumbar cues optional, anterior-sign fix for deep folds, facing from the trunk | **Provisional; calibrated by eye on two recordings of one lifter; not validated** |
| D-01 | Tier 2 sensors on the end user vs sensor-free released app | **Resolved 2026-11-15, wording corrected 2026-11-18** |
| D-02…D-04 | Product decisions that conflict between the brief and `Context.md` | **Open** |
| D-05 | Commercial use of SpinePose (CC-BY-NC-4.0 code and weights) | **Open, parked by the user 2026-11-19; not a blocker** |

---

## G-01 — MediaPipe 33 ↔ SpinePose 37 bridge

- **Gap.** [P1] extends an RTMPose body skeleton (COCO/Halpe topology, 37 output keypoints). MediaPipe PoseLandmarker uses its own 33-landmark BlazePose topology and ships as a sealed `.task` bundle, so its head can't be expanded or distilled the way [P1] does. No paper bridges the two.
- **Why external.** [P1] assumes access to the body-pose network's heatmap head. The MediaPipe Tasks API exposes only landmarks.
- **Solution.** Run both models on the same YOLO person crop and fuse at the keypoint level:
  1. Map shared joints: MediaPipe {11 L-shoulder, 12 R-shoulder, 23 L-hip, 24 R-hip, 25/26 knees, 27/28 ankles} ↔ the corresponding COCO indices in SpinePose output. **Load the SpinePose name→index table from the released model metadata.** Never hard-code it from paper prose (METHODOLOGY_SPEC V2).
  2. **Consistency check per frame:** mean pixel distance between paired shared joints, normalised by torso length. If > 0.15 (provisional), the spine is marked unreliable for that frame (`spine_source="none"`).
  3. **Rigid alignment:** least-squares 2D similarity transform (Umeyama) from SpinePose shared joints to MediaPipe shared joints. Apply it to the 9 spine points, so spine and body live in one consistent frame even under small crop/resize mismatches.
  4. Output `Skeleton` (ARCH-6) with `body2d/body3d` from MediaPipe and `spine2d` re-ordered to `SPINE_LABELS = (C1, C4, C7, T3, T8, L1, L3, L5, SACRUM)`.
  5. **Fallback** when SpinePose can't run: [P1] two-curve heuristic from shoulders/hips, tagged `spine_source="heuristic"`. Lumbar/pelvic rules are disabled for heuristic frames.
- **Interfaces.** `vision/spine.py → vision/bridge.py → biomechanics/angles.py`.
- **Validation.** Shared-joint residual distribution on our videos. Lumbar MAE vs spinous markers ≤ 5° (SRS). Ablation: proxy-spine vs SpinePose for both activation RMSE and form F1.

## G-02 — Monocular 3D root-centring and scale normalisation

- **Gap.** [P2] used VIBE 3D keypoints normalised to a bounding box. [P6] used two-camera DLT. We have one camera and no mesh model.
- **Solution.** MediaPipe `pose_world_landmarks` (metric, hip-centred, provided by the Tasks API). Recompute the root as the mid-hip, then divide by the athlete's median torso length from the first standing second. Angles are taken from the image plane matching the camera view (side = sagittal, front = frontal), with world coordinates as a cross-check.
- **Why not VIBE/SMPL.** Too heavy for ≤ 33 ms live inference, and the brief excludes stereo.
- **Interfaces.** `vision/normalize.py` → `biomechanics/angles.py` (keypoint features), `coaching/correction.py` (FK rendering).
- **Validation.** Angle error vs goniometer on static poses. Test–retest of torso-normalised segment ratios (CV < 5%).

## G-03 — Multi-rate synchronisation and ring buffers

- **Gap.** [P2] aligns by device timestamps (method in its unavailable supplementary). No paper specifies host-side sync between a 30–60 fps camera and a 333–2000 Hz serial ADC with a drifting MCU clock.
- **Facts that drive the design.**
  - The Arduino Uno's ATmega328P runs from a **ceramic resonator** (~±0.5%), so the true sample rate can differ by up to ~5 ms per second. **Never assume nominal fs.**
  - USB-serial delivery is bursty (host-side jitter of a few ms or more). Per-packet arrival times are noisy.
  - Camera exposure→`grab()` latency is tens to 100+ ms and differs per device. It must be measured, not assumed.
- **Solution.**
  1. **MCU side** (`hardware/semg_logger`): timer-ISR sampling at fixed fs. A frame is `[0xA5 0x5A][uint16 counter][3×uint16 adc][uint8 crc8]` = 11 B. At 3 ch × 2 kHz that is 22 kB/s, which is too close to 250 000 baud (≈ 25 kB/s), so use **1 000 000 baud** (exact divisor at 16 MHz with U2X). Envelope mode at 1 kHz fits 250 000 baud. The ADC budget (~9.6 kSPS at prescaler 128) covers 6 kSPS. The LED/beep GPIO edge is written into the stream as a flag bit in the counter frame.
  2. **Host side:** a reader thread timestamps each chunk with `time.monotonic_ns()`. Counter unwrapping gives the sample index *n*. `sync/clock.py` fits $t_{host}(n)=a+b\,n$ by robust regression on the **lower envelope** of arrival times (minimum-delay points), giving offset *a* and true period *b*.
  3. Frame grabber thread: `t_host` at grab. **LED onset frames** (`sync/led_sync.py`, brightness step in an ROI) vs **LED flag in the sEMG stream** directly measure camera latency. Two pulses per session (start/end) also check drift.
  4. Lock-free single-producer/single-consumer **ring buffers** (numpy, preallocated, drop-oldest) per stream. Consumers read by timestamp range, not by index.
  5. `sync/align.py` resamples both streams onto the model grid (10 Hz default) by timestamp interpolation (linear for angles, area-averaging for envelopes).
- **Interfaces.** Producers: `semg/acquisition.py`, the camera loop. Consumers: `data/dataset.py` (offline), the real-time models tick (online).
- **Validation.** LED test: residual offset after correction ≤ 33 ms (SRS) over a 10-minute recording. Report drift ppm.

## G-04 — Coaching cue hysteresis and arbitration

- **Gap.** Papers give *what* is wrong ([P6] classes, [P8] rules) but not how to present it without flicker or nagging.
- **Solution** (`coaching/cue_engine.py`), reusing the tracker's proven pattern (enter/exit thresholds, dwell timers, timestamps not frame counts):
  - each cue has `enter_threshold`, `exit_threshold` (exit nearer to normal), `min_dwell_s` (0.3 s), `cooldown_s` (5 s), `max_per_rep` (1 spoken)
  - priority: **safety** ([P8] lumbar/pelvic, valgus) > **form class** ([P6]) > **activation targeting** ([P2] edit)
  - one active spoken cue at a time. The visual overlay may show the top 2.
  - cue text = joint + direction + amount ("Lift chest ~10°"), legible from 2 m (SRS)
- **Validation.** Synthetic-signal tests (as in `tests/test_deadlift_phases.py`): no toggling under ±noise at threshold, cooldown respected, priority preemption.

## G-05 — Envelope-output hardware vs raw-signal DSP (OPEN)

- **Gap.** [P3]/[P7] DSP (notch, 20–400 Hz Butterworth, rectification, frequency features, MNF fatigue) needs **raw** EMG sampled ≥ 1 kHz. `Context.md` §4.2 records that the team's **AD8226 modules output a rectified, smoothed envelope**.
- **Consequence if confirmed.** Notch/bandpass/rectify are skipped (done in analogue). The 6 frequency-domain features and the wavelet feature are unavailable. Fatigue = RMS/MAV trends only. Y(t) is still valid as %MVC.
- **Solution.** Two acquisition modes (`raw`, `envelope`) per METHODOLOGY_SPEC §3.1, recorded in `meta.json`. All feature code checks the mode and raises if a frequency feature is requested in envelope mode.
- **Decision needed.** Scope the module output (bipolar ≈ raw, unipolar slow ≈ envelope). If raw is wanted, a raw-capable front-end (e.g. a MyoWare RAW/SIG pin) and fs ≥ 1 kHz are needed.

## G-06 — Muscle-subset (masked) supervision

- **Gap.** [P2] recorded 8 muscles at once. Our hardware records **3 per block**, with different muscles per block/exercise. Unmeasured muscles have no label.
- **Solution.** Global muscle vocabulary $\mathcal M$ (configs). Y stores NaN for unmeasured muscles, plus a mask $\mu\in\{0,1\}^{M}$ per window. Loss is masked MSE (METHODOLOGY_SPEC §4.2). Metrics are computed only where μ = 1. The UI shows **only muscles measured for that exercise** (`Context.md` §10).
- **Validation.** Unit test: gradients are zero for masked muscles. Per-muscle sample counts are reported with every metric.

## G-07 — Non-causal attention vs live coaching

- **Gap.** [P2] uses "no attention masking" over a 3 s window, i.e. bidirectional. Live coaching can only use the past.
- **Solution.** Live: feed only the trailing window, so the model is causal w.r.t. the future by construction. Use smoothed predictions for the last steps and report a ~200 ms display lag (ARCH-5). **Ablation:** causal-masked variant of the same encoder. Pick whichever has lower LOSO RMSE *on the newest step*.
- **Bi-GRU [P6]** runs per completed rep, so it needs no change.

## G-08 — Unspecified Transformer details

- **Gap.** [P2] main text omits FFN width, dropout, positional encoding, lr, batch size and epochs (supplementary not in repo).
- **Solution.** d_model 128 (given), 4 layers / 8 heads (given), FFN 512, dropout 0.1, sinusoidal PE, Adam lr 1e-4, batch 64, early stopping on inner-fold val (patience 20). All in `configs/models/motion2muscle.yaml`. **If the MIA supplementary is obtained, replace these defaults and update this entry.**

## G-09 — Athlete conditioning tensor

- **Gap.** [P2] §4.4 concatenates a per-subject $y\in\mathbb R^{2\times T}$ but its content is in the supplementary. A subject-ID code can't generalise to a new athlete.
- **Solution.** $y$ = computable athlete descriptors (anthropometric ratios, calibration-rep per-phase %MVC summaries, log MVC amplitudes), tiled over T. Concatenate to $d_t$, then `cond_proj: Linear(128+C_cond → 128)` initialised as [I | 0], so Tier 2 equals Tier 1 at initialisation. Missing descriptors are zero-filled, with an indicator channel.
- **Validation.** Ablation over the descriptor groups. Confirm the identity at init with a unit test.

## G-10 — [P3] transfer protocol on a regression task

- **Gap.** [P3] studies **classification** with sEMG as input. MyoMotion fine-tunes a **vision→sEMG regressor**.
- **Solution.** Keep [P3]'s procedure (pretrained init, retrain the last layer or first layer only, same hyperparameters, early stopping on 10% val, hold out reps) and swap the loss for masked MSE. Gate deployment on held-out-rep improvement (METHODOLOGY_SPEC §4.4), because [P3] shows fine-tuning can hurt with very few reps.

## G-11 — MVC protocol

- **Gap.** [P7] took MVC from peak/mean RMS during dynamic exercises. %MVC across subjects needs a repeatable reference.
- **Solution.** Isometric MVC per muscle, 3 × 5 s, SENIAM-style test positions (Hermens et al. 2000, cited by [P2] as its ref. [17] for placement), max 200 ms RMS. Also store [P7] dynamic peak/mean for comparison. Redo every session.

## G-12 — Window step

- **Gap.** [P3] text: "200 ms sliding window with 10 ms overlap".
- **Solution.** Default `window_ms=200, step_ms=10` (100 Hz feature rate). Configurable. `step_ms=190` reproduces the literal reading.

## G-13 — [P6] correction under one camera

- **Gap.** [P6] curve-fits stereo-DLT 3D coordinates. It gives no polynomial order and no FK/rendering method.
- **Solution.** Fit per channel on normalised time, order chosen by BIC over 2…8. Time-scale per phase. Intercept shift per [P6]. Render targets via planar forward kinematics using the athlete's measured segment lengths (G-02), so the target skeleton keeps the athlete's proportions.

## G-14 — SpinePose smoothness parameters

- **Gap.** [P1] Eq. 7 needs sigmoid sharpness $\alpha_s$ and threshold $T$, and gives neither.
- **Solution.** $T$ = 1.5 × the median ground-truth inter-vertebral distance in heatmap pixels. $\alpha_s$ = 10 / T. Grid-search on val.

## G-15 — lr scaling with a smaller batch

- **Solution.** [P1] scales batch to keep effective batch 1024. If we can't, use gradient accumulation to 1024. If that's still impossible, use linear lr scaling $lr = 4\times10^{-3}\cdot B/1024$ (Goyal et al. 2017).

## G-16 — Spine and pelvis angle definitions; butt-wink threshold (OPEN)

- **Gap.** [P1] gives points, not clinical angles. [P8] gives the qualitative rule (lumbar flexion / posterior pelvic tilt at end-range hip flexion is harmful) but **no numeric threshold**.
- **Solution.** Definitions in METHODOLOGY_SPEC §2.5: 3-point regional bends, SACRUM→hip-centre pelvic line, Δ from the athlete's standing baseline. Signed with flexion positive using facing direction. **Thresholds**: set from our marker-validated dataset (e.g. 95th percentile of Δ in expert-labelled good reps). Until then, rules run in **report-only mode** and never trigger spoken cues.

## G-17 — Landmark signal conditioning

- **Solution (implemented).** 3-sample median, physiological jump gate (600 °/s + 8°), One-Euro filter (Casiez et al., CHI 2012), least-squares velocity over 0.4 s, all timestamp-driven. Ported unchanged from `scripts/cv/deadlift/deadlift_tracker.py` to `biomechanics/filters.py` (`tests/test_deadlift_phases.py` still passes). **New:** `KeypointSmoother` applies One-Euro to every landmark in torso-length units, tuned by measurement (min cut-off 1.0 Hz, beta 3.0: about 3x less jitter at 57 ms lag at 0.6 torso/s; see the comment in `filters.py`). It feeds the overlay, spine proxy, bar path and form rules. The rep segmenter deliberately keeps its own raw-angle conditioning.

## G-18 — Filter implementation details

- **Solution.** Notch `iirnotch(f0=mains_hz, Q=30)`. Butterworth as SOS. Offline `sosfiltfilt` (zero phase; effective order doubles). Online `sosfilt` with carried `zi` (causal; group delay is reported). Assert `f_high < 0.45·fs`.

## G-19 — Feature definitions

- **Gap.** [P3] lists the 18 features by name only (follows its ref. [28]).
- **Solution.** Formulas in METHODOLOGY_SPEC §3.5. Threshold ε = 3 × the rest-baseline std of the filtered signal. FR bands: low 20–150 Hz / high 150–400 Hz (raw mode, fs ≥ 1 kHz). PSD: Welch, Hann, 200 ms window. Wavelet histogram: `db4` DWT level 4, 9-bin histogram of the detail-coefficient amplitudes per level, summarised by the entropy of the histogram (one scalar). ZC/WAMP/SSC and spectra use the **filtered, non-rectified** signal. Amplitude features use |x|.

## G-20 — Fatigue trend test

- **Solution.** Per-rep medians. OLS slope over ≥ 4 reps. One-sided t-test on the slope, p < 0.05 for each indicator. Fatigue = RMS↑ ∧ MAV↑ ∧ MNF↓ ([P7]). Envelope mode: RMS↑ ∧ MAV↑, flagged as "amplitude only".

## G-21 — ICC form

- **Solution.** Report ICC(A,1) (two-way random, absolute agreement, single rater; McGraw & Wong 1996) as primary, since [P7] says "absolute agreement". Also report the one-way ICC(1) matching [P7] Eq. 2.

## G-22 — Padding sentinel

- **Gap.** [P6] pads with −1. Signed angles and normalised coordinates can legally be −1.
- **Solution.** Keep −1 as the pad value (paper-faithful), but always carry a boolean `valid_mask [B, T]` and use packed sequences / masked attention. Never infer padding from values.

## G-23 — Deadlift rule set

- **Gap.** [P8] is a squat review and [P6] is squat classification. Neither covers the deadlift. [X1] (Ko et al.) reports deadlift trainer-defined errors: **non-neutral spine / rounding of the lower back**, **improper barbell position (keep the bar close to shins and thighs)**, and **locking out the knees too early**, and classifies deadlifts into correct / non-neutral spine / arms too wide / arms too narrow. It gives no angle thresholds.
- **Solution.** `biomechanics/deadlift_rules.py` implements four flags. Each cites what supports it, and **every numeric threshold is provisional** (a field of `DeadliftRuleConfig`, to be tuned on our own labelled reps):
  | Flag | Support | Thresholds |
  |---|---|---|
  | Lumbar rounding / posterior pelvic tilt | [P8] mechanism (lumbar flexion → lower compressive tolerance, more anterior shear), [X1] non-neutral spine | baseline-relative Δ, G-24 |
  | Hips rising faster than shoulders ("stripper") | [X1] "locking out the knees too early"; coaching practice | Δknee-extension and Δtrunk-inclination over the first 0.8 s after lift-off |
  | Hyperextension / lean-back at lockout | coaching practice (no paper) | trunk lean-back, 8° enter / 5° exit |
  | Bar drift from mid-foot | [X1] "bar close to shins and thighs"; coaching practice | fraction of foot length, 0.35 enter / 0.25 exit |
- **Interfaces.** Input `DeadliftAngles` (angles + spine proxy + bar) and the rep segmenter's phase. Output `FormEvent`. The future `coaching/cue_engine.py` (G-04) consumes events. The rules module does not speak or draw.
- **Validation.** Synthetic tests prove the rule *logic* (good rep passes; each fault triggers). They don't prove real-world sensitivity. That needs labelled gym/lab video.

## G-24 — Spine-proxy construction (deterministic baseline)

- **Gap.** [P1] bootstraps pseudo-labels with a heuristic ("shoulder and hip joints as drivers, fit two smooth curves, sample at predefined intervals assuming constant inter-vertebral distance") but gives no equations, ratios or code. The user brief also adds the ear as a cervical anchor.
- **Solution.** `biomechanics/spine_proxy.py`:
  1. Anchors (side view, near side): ear → **C1**, shoulder → **C7**, hip → **sacral root**.
  2. Two cubic Hermite curves, G¹-continuous at C7: a cervical curve (C7→C1) and a thoracolumbar curve (sacral root→C7). The tangent at C7 is the bisector of the two chords. The tangent at the sacral root is the chord direction (pelvic orientation is not observable).
  3. Sampling fractions come from **vertebral counts** with constant inter-vertebral spacing, as in [P1]. Counting from C1 = 1: C1 1, C4 4, C7 7, T3 10, T8 15, L1 20 (T12 = 19), L3 22, L5 24, sacrum 25, so arc fractions (n−1)/24 = 0, .125, .25, .375, .583, .792, .875, .958, 1. These ratios are **our derivation**, not a table from [P1].
  4. Region angles are tangent-direction changes along the curve (flexion positive): cervical C1→C7, thoracic (kyphosis) T3→L1, lumbar L1→SACRUM, plus a pelvic-tilt proxy from the sacral end tangent.
  5. Every flexion measure is reported **relative to the athlete's own setup baseline** (Δ from neutral setup), never as an absolute anatomical angle.
- **Hard limitation (be honest in the UI and thesis).** With only ear, shoulder and hip, the curve's bend is a function of one observable, the ear–shoulder–hip angle. The proxy therefore detects *head/upper-back drop relative to the trunk line*. It cannot see true lumbar curvature, and a lifter who set up already rounded has a rounded baseline. [X2] measured 12–39° errors for keypoint-only trunk/pelvis kinematics, and `Context.md` §3.5 says the same. All proxy outputs carry `confidence="proxy"`. They're replaced by SpinePose (G-01) when available.
- **Validation.** Lumbar-angle MAE vs spinous-process markers (SRS ≤ 5°), proxy vs SpinePose ablation.

## G-25 — [P8] trunk–tibia thresholds on the deadlift

- **Gap.** [P8] derives ±10° from Barrack et al. squat regressions and evaluates at **peak knee flexion**. No paper gives deadlift thresholds.
- **Solution.** Reuse ±10° unchanged as a *descriptive label* (hip-extensor / balanced / knee-extensor biased), evaluated at the **lift-off posture**, the deadlift's deepest hip/knee flexion (the analogue of peak knee flexion). It is a muscle-bias descriptor only. It never triggers a fault. Expect most conventional deadlifts to read hip-biased. Sumo and trap-bar variants will read differently. State this in any report.

## G-26 — Rep segmentation (deadlift)

- **Decision.** Keep the existing hysteresis state machine from `scripts/cv/deadlift/deadlift_tracker.py` (18 passing tests: noise, 15–60 fps, dropouts, spikes, failed reps) and move it to `biomechanics/rep_segmenter.py`. Changes: phase names `SETUP → CONCENTRIC_PULL → LOCKOUT → ECCENTRIC_DESCENT → REP_COMPLETE` (+ `STANDBY`), a transient `REP_COMPLETE` phase (0.4 s, then `SETUP`), per-rep timing records, and the time of the bottom turning point (lift-off estimate).
- **Added from [P5].** An optional causal bar-height peak/valley detector on the equipment/wrist trajectory, amplitude-normalised to [−180, 180]. The rising half-cycle is the concentric pull and the falling half-cycle the eccentric descent (deadlift: concentric first). It is a **second opinion**: it fills `bar_peak_t_s` and `bar_rise_torso` in each rep record. It does not gate rep counting.
- **Compatibility.** The legacy script keeps its old public names (`Phase.ISOMETRIC_HOLD`, `DeadliftStateMachine`) through a shim, so `tests/test_deadlift_phases.py` passes unchanged.

---

## G-30 — Both body sides and any heading (second live-webcam report)

- **Gap.** (1) The body side was locked on the first valid frame, so as soon as the lifter's left side faced the camera (or they turned around) every frame used the occluded far side: 0 of 189 frames got angles in a reproduction. (2) The 2-D side-view maths assumes the person is exactly side-on; real footage of this lifter was ~38° off side-on (heel-to-toe direction measured from the 3-D landmarks), and facing the camera foreshortens the hip hinge completely. (3) The proxy spine used one side's ear/shoulder/hip, which in a front view sits off the body's centre line and draws a bent line.
- **Solution.**
  - `DeadliftAnalyzer._update_side`: side re-chosen every frame; the other side takes over only if it is clearly more visible (> 0.15) for 0.4 s; both-equal (front view) never switches; a user-forced `--side` is fixed. A switch resets the facing vote and the spine baseline.
  - `_vote_facing`: decaying weighted vote (toe-heel 3, nose-ear 2, knee-ankle 1, 3-D forward 4); follows a turn within ~1 s.
  - `pose3d.py`: MediaPipe `pose_world_landmarks` (metres, y down) give view-independent midline hip/knee/trunk/tibia angles. Up = -y (verified on real frames); forward from the hip axis with the sign from foot/nose/knee cues. Used only when the person is not side-on.
  - View = "side" (exact 2-D angles, the validated path) or "angled" (3-D), chosen from the 3-D hip-axis heading (works when the far shoulder is not visible), with hysteresis (enter ~25° off side-on, leave ~18°).
  - Spine from a non-side view: straight midline through mid-ear / mid-shoulder / mid-hip, `curvature_visible=False`, no rounding rules (curvature cannot be seen from the front).
  - Bar drift is disabled when not side-on.
- **Limits.** The depth in world landmarks is a model estimate: lower accuracy than a clean 2-D side view. Assumes a level camera. Checked on real footage only for the trunk angle (4 frames, within ~3-4° of 2-D). Thresholds provisional.
- **Tests.** `tests/test_views_and_spine.py` (3-D lifter from every heading incl. left side, facing, away, turning around mid-set).

## G-32 — Spine from the back silhouette (no extra network; the mobile-friendly default)

- **Why not just the papers' method.** Of the eight papers only P1, P8 and P5 (plus the supporting X1/X2) discuss the spine at all; P2, P3, P4, P6, P7 do not mention it. None gives a spine estimator that avoids a trained model or image evidence: **P1** offers the heuristic that failed here, plus the trained SpinePose (teacher-student); **X2 (OpenCap)** states that "keypoints at the hips and shoulders are insufficient for robustly determining sagittal-plane hip, pelvis, and lumbar kinematics" and fixes it with an LSTM marker augmenter trained on motion-capture data (needs 3-D triangulated input and mocap data we do not have); **P5 (Fitness-AQA)** finds image features beat 2-D pose features for the lumbar error (F1 0.61-0.63 vs 0.54, Table 5); **P8** gives the mechanics (forward trunk lean = hip flexion or lumbar flexion) but pelvic orientation is not observable from landmarks. So an *image-based* signal is needed. A learned landmark-to-spine model (the X2/P1 route) is the right long-term path but needs labelled data; see the roadmap below.
- **Gap.** A spine estimate that lies on a curved back, runs on a phone, and needs no non-commercial model.
- **Solution (external method, logged here per rule 2).** `biomechanics/spine_contour.py`: MediaPipe's person **segmentation mask** (same model, `output_segmentation_masks`) -> back-surface profile by ray-casting from the hip-shoulder axis toward the back -> robust quartic fit -> spine = back surface moved forward by a soft-tissue depth, between the S1 and C7 levels, nine labels at the vertebral-count fractions (G-24) -> `build_spine_from_points(..., source="contour")`. The principle (back-surface curvature approximates spinal curvature) is standard in surface topography (flexicurve, rasterstereography); it is not from P1-P8.
- **Found on the user's real recording (586 frames, 1080p):** the old proxy's line ran through the front of the torso (the joint centres are not on the spine) and bowed on an upright back; the contour lies on the back edge, is straight when upright, and its lumbar change relative to set-up goes toward extension at lockout (-12 to -22 deg) and returns to ~0 on descent, which is the expected deadlift trajectory. Fit quartic over cubic: lowest jitter (lumbar 2.29 deg vs 2.60), highest confidence (0.87 vs 0.82); the cubic's +18 deg lumbar spike was an overshoot (+8 with quartic). No false rounding cues in the 4 reps (the proxy raised 5).
- **Cost (measured, laptop CPU, 1080p):** mask +2.1 ms/frame on MediaPipe heavy (12.2 -> 14.4 ms), contour maths 0.71 ms/frame. No second network, no new dependency.
- **Limits.** Side-on only (from the front there is no sagittal back edge). Loose clothing hides the true back surface (the recorded lifter wears a loose kurta). The lowest ~0.2 of the trunk is the gluteal bulge, so **lumbar curvature is the least reliable region**. The soft-tissue depth (0.065 torso), S1/C7 levels (0.08 / 1.06) and the 15/10 deg cue thresholds are provisional. **Not validated against markers or SpinePose.** Cervical points are a smooth continuation to the ear, not measured.
- **Status of cues.** Contour-based rounding warnings are on, with conservative thresholds (15 / 10 deg) and the message "possible back rounding (back contour)". Region angles are baseline-relative as before.
- **SpinePose vs this method (decision aid).**
  | | Back contour (G-32) | SpinePose (G-31) | Proxy (G-24) |
  |---|---|---|---|
  | follows a curved spine | yes, side-on, from the silhouette | yes, trained on spine labels, any view | no |
  | extra network | none | yes (P1 Table 1: -s 5.98 M params / 0.72 GFLOPs, -m 14.3 M / 1.98, -l 28.7 M / 4.22) | none |
  | measured cost here | ~3 ms/frame | 4-10 ms/frame, laptop CoreML (G-31) | ~0 |
  | loose clothing | hurts | handled by training (unverified here) | n/a |
  | views | side-on only | any | n/a |
  | licence | none | **CC-BY-NC-4.0** (non-commercial) | none |
  | status | checked on 1 real recording | run on the same recording and on photos (G-31); no ground truth | known to fail |
  **Recommendation (written before the 2026-11-19 decision to integrate SpinePose; still the sound plan if the licence or phone cost blocks it):** ship the contour as the phone baseline. Use SpinePose **offline, on a PC, as a reference to measure the contour's error** (research use; allowed by the licence) and as the teacher for the roadmap below. Move SpinePose on-device only if that measurement shows the contour is not good enough *and* the licence is cleared; if it is, SpinePose-s every 3rd frame next to MediaPipe lite/full is the realistic phone configuration.
- **Roadmap (P1 teacher-student + X2 augmenter, mobile).** (1) Record a labelled set of side-view lifts (set-up, pull, lockout, and deliberately rounded reps, different clothing). (2) Offline, run SpinePose as the teacher to produce spine labels on those frames; compare the contour's angles against them (this gives the first real error numbers). (3) Train a tiny sequence model (GRU/LSTM, ~100 k parameters) mapping MediaPipe landmarks (+ contour features) to the teacher's spine points, as OpenCap's augmenter does from keypoints; it runs in microseconds on a phone. Needs consent for the footage and the licence question answered first.

## G-31 — Real spine from SpinePose keypoints

- **Gap.** The proxy cannot follow a curved spine (G-24); the user confirmed on real footage that its points do not lie on the spine. Only an image-based model can. Planned since G-01.
- **Decision (user, 2026-11-19).** Install and integrate SpinePose **for the nine spine points only** (so spinal curvature can be tracked in deadlifts, squats, bench press and similar). MediaPipe stays the source of every other joint. The CC-BY-NC licence is parked as an open decision (D-05), not a blocker.
- **Solution.** `vision/spine_model.py` runs the SpinePose pose network ([P1]; RTMPose, SimCC decode, the paper's "spinetrack" weights) on a person box derived from MediaPipe's landmarks (the package's own detector is never run), returns nine points in `SPINE_LABELS` order, and the pipeline uses them instead of the proxy/contour (`build_spine_from_points`, G-33). G-01 consistency check: SpinePose and MediaPipe shoulder/hip mid-points must agree within 0.35 torso lengths, else the detection is dropped for that frame. The package's v1 `_smooth_spine` (midpoint of neighbours) is bypassed because it flattens curvature. Scores are SimCC peak values and can exceed 1, so they are clipped to [0, 1] before use as weights/confidence. The first call takes ~2 s (graph build), so the tracker warms up on load.
- **Install (done 2026-11-19, in the project `.venv`):** `pip install -r requirements-spine.txt && pip install --no-deps spinepose==2.1.0` (`torch`, `onnx`, `anytree` are declared by the package but never imported). Weights (ONNX, small 22.8 MB) download on first use from Hugging Face `dfki-av/spinepose` into `~/.cache/spinepose/`. Not in `requirements.txt` on purpose (optional; licence).
- **Licence: CC-BY-NC-4.0 (code and weights).** See D-05.
- **Chain-order mapping** `SACRUM, L5, L3, L1, T8, T3, C7, C4, C1` = package chain ids `[19, 26, 27, 28, 29, 30, 18, 35, 36]` (hip, spine_01..05, neck, neck_02, neck_03). Checked on real output: the cumulative spacing along the chain matches the vertebral-count fractions of [P1 Fig. 2].
- **Measured here (MacBook, CoreML provider, SpinePose-small):** 4-10 ms per detection after warm-up. On the user's 586-frame recording the whole analysis (decode + MediaPipe heavy + SpinePose every frame + overlay + video write) took ~17 s wall-clock. The G-01 bridge check dropped 16 of 417 tracked frames. **Laptop numbers only; nothing was measured on a phone.**
- **What real output looks like (side-on photos, Wikimedia Commons, CC licences).** The detected points follow a curved back. The lower points are bunched (SACRUM, L5, L3 only 5-25 px apart in a 224 px spine), which is why G-33 exists. Front-view photos give a clean vertical midline (curvature is invisible there; `curvature_visible=False`, nothing invented). **Small and medium models disagreed on the lumbar angle of the same landmine-row photo (-10 deg vs +19 deg): the lumbar region is the weakest.**
- **Real clip, honest reading.** 4 reps counted (same as with the contour). The lumbar change relative to the set-up, held for 0.25 s, was +8.6, -17.1, -6.9, -8.0 deg per rep (negative = extension), and no rounding cue fired. Single-frame readings reached +35 and +79 deg: not physiological, so keypoint glitches (the rep summary therefore reports the *held* value, not the per-frame maximum). **That recording has an earlier version of our own overlay drawn into it, so frame-level behaviour on it is not an accuracy measurement.** No marker or reference ground truth exists yet.
- **Bugs found by this run (fixed, tested):** (1) a one-frame fallback to another spine source wiped the set-up baseline, so later reps had no lumbar reading at all; each source now keeps its own baseline. (2) The rep summary's lumbar maximum took raw per-frame values, including display-only proxy frames. (3) Fit parametrisation, see G-33.
- **Cost knob.** `--spine-every N` reuses the last detection between runs; `--spine-size small|medium|large`.

## G-33 — Region angles from detected spine points (cubic fit in the spine's own frame)

- **Gap.** [P1] gives the nine points, not a recipe for lumbar/thoracic/cervical angles. The first formula (direction of neighbouring segments: lumbar = direction(L1-L3) - direction(L3-SACRUM), etc.) gave **-114 deg lumbar** on a real dumbbell-row chain, because SpinePose places SACRUM, L5 and L3 only a few pixels apart and the segment directions are then pure keypoint noise.
- **Simulation (spine of known curvature, 224 px, Gaussian keypoint noise; scratch script, not committed).** Adjacent-segment lumbar angle: false alarm (|delta| above the cue threshold with *no* true change) in **56% of frames at 2 px noise**. Weighted cubic fit: no-change lumbar delta std 3.8 deg at 2 px / 5.5 deg at 3 px, P(|delta| > 10 deg) 0.00 / 0.07.
- **Solution (external method, logged per rule 1).** `spine_proxy.build_spine_from_points`: fit a cubic x(t), y(t) through the nine points, weights = clipped scores (floor 0.05), with t = position along the SACRUM-C1 chord (not path length); region angle = rotation of the fitted tangent between the region's two ends, flexion positive (lumbar = tangent(L1) - tangent(SACRUM), thoracic = tangent(T3) - tangent(L1), cervical = tangent(C1) - tangent(C7)). Directions are measured in the spine's own frame (angle from the chord, positive toward the anterior side), so the result does not depend on body orientation: the same code serves upright lifts (deadlift, squat) and the supine bench press (`anterior=` argument).
- **Why the chord parameter.** With path-length parametrisation one wild point stretches the parameter of every other point: a single 60 px outlier at weight 0.05 moved the lumbar angle by ~165 deg; with the chord it moves it by 1.1-1.6 deg (same noise sensitivity, slightly more shrinkage). Order check: only trusted points (score >= 0.25) must be ordered along the chord, so one low-score wild point is down-weighted rather than rejecting the frame; fewer than 5 trusted points returns None.
- **The price: shrinkage.** The smooth fit under-reads a true lumbar change by about 3.5x (true +30 deg reads ~+8.5; a synthetic lifter in the tests reads +9.2 for a true +25). Region angles are therefore **relative indicators, not calibrated degrees**.
- **Threshold consequence.** `lumbar_enter_deg_model` 8 / exit 5 deg (was 12 / 8 for the abandoned formula): the cue fires for a true lumbar flexion change of roughly 22-28 deg or more, and a no-change signal exceeds 8 deg on 1-7% of single frames before smoothing and the 0.25 s dwell. Provisional; **not validated against labelled lifts**.
- **Needs.** A recording with a marker-based motion-capture (or equivalent) reference, with deliberately rounded and neutral reps, to turn the shrinkage factor and the threshold into measured numbers.
- **Tests.** `tests/test_spine_regions.py`: real dumbbell-row and landmine-row chains, noise regression, orientation invariance (rotated 0/90/180 deg, bench-press orientation, sign flip with the opposite anterior), outlier down-weighting, degenerate input, score clipping, and a real-network smoke test (skipped unless SpinePose and its cached weights are present).

## G-34 — Back rounding that works at 90 degrees without visible legs

- **What the user reported (third live test).** Facing the camera looked right; side-on, back rounding was never detected; the end of the run said `15/923 frames tracked`.
- **Diagnosis (replaying the recording frame by frame; scratch scripts, not committed).** `923` is simply how many frames the camera delivered in 62 s at ~15 fps. `tracked` means shoulder, hip, knee **and ankle** were all seen, which the rep counter and every rule used to require. The lifter stood close to a laptop webcam: a person was found in 919 of 923 frames and SpinePose returned a spine in all 919, but the ankle was not visible in 850 frames and the knee in 687, so almost nothing ran. Of 154 deeply bent side-on frames, 154 had a SpinePose spine and 154 were side-on, but the ankle was visible in 65 and all four core joints in 7. **The back was seen; the legs gate hid it.** (Replaying the same recording through the current code: 69 frames fully tracked, plus 634 where only the back was analysed.)
- **A second, silent bug (fixed).** The spine's "front" direction was `(facing, 0)` projected off the shoulder-hip chord. That is wrong once the head is below the hips (chord leaning more than 90 deg from vertical, e.g. the bottom of a stiff-leg deadlift or a toe touch): every region angle changed sign, so convex rounding read as extension. Now the front is the chord turned a quarter turn toward `facing` (continuous through the fold). Regression test: `test_rounding_keeps_its_sign_through_a_forward_fold_past_horizontal` fails on the old convention for folds of 85 deg and more.
- **What the papers give for this (read from `docs/resarchpapers/`; none gives a geometric detector).**
  - [P1] SpinePose: the nine spine points (3 cervical, 5 thoracolumbar, 1 sacral), their training losses, and the remark that the points enable "torsional and flexion angles". **No angle recipe, no threshold.** Its own validation (OpenSim IK from a calibrated multi-camera rig) is not available here.
  - [P8] The fault: forward trunk inclination reached by lumbar flexion (rounding) instead of hip flexion with a neutral spine; posterior pelvic tilt at end-range hip flexion. Qualitative: no numbers. Used for: **gate on the hinge** (trunk inclination) and treat lumbar as the safety region.
  - [P5] Fitness-AQA: the error classes **convex vs concave rounded back** (the sign convention we use: flexion positive = convex), errors judged per static frame / half-cycle, and the finding that image features beat 2-D pose features for lumbar errors (F1 0.61-0.63 vs 0.54). Its detectors are **trained classifiers**; we have no labelled data.
  - [X1] Ko et al.: a "non-neutral spine" class per concentric/eccentric phase, again a trained classifier on joint coordinates and angles.
  - [X2] OpenCap: hip/shoulder keypoints are "insufficient" for lumbar kinematics; the fix is a trained marker augmenter plus multi-view inverse kinematics.
  So the thresholds below are **ours**, from our own recordings; the papers motivate the quantity and the gate, not the number.
- **Solution.** (1) `DeadliftAnalyzer._back_only`: when the legs are not in view, the back is still analysed from the near-side shoulder and hip plus the spine (no hip/knee angles, no phases, no reps; the overlay says so). (2) `DeadliftFormAnalyzer._absolute_back` / `update_back_only`: **hinged** = trunk inclination >= 30 deg (until < 20 deg); while hinged, the SpinePose **thoracic** region angle (absolute, flexion positive) >= **+6 deg** (exit +3) for 0.25 s raises `UPPER_BACK_ROUNDING`, once per hinge. No set-up baseline is needed (a lifter who sets up already rounded is no longer invisible to this cue). The same rule also runs inside the normal pipeline and lands in the rep verdict (`peak_thoracic_abs_deg`). (3) A frame without a SpinePose measurement neither advances nor resets the latch (stale after 0.6 s). (4) Facing: a hinged trunk votes for the way it leans (`estimate_facing_from_trunk`), because the face hangs and the feet are out of frame exactly when this matters. (5) `--spine-size` default is now `medium`. (6) Every frame with a spine is logged (`<name>_spine.csv`) and `--save-raw` keeps the untouched frames, so thresholds can be recalibrated on labelled footage.
- **Evidence for the thoracic choice (small, one lifter, labelled by eye by the developer from a handful of frames; treat as a pilot).** Hinged side-on frames, ~25 neutral (bottoms of reps in an earlier clip) vs ~80-100 visibly rounded (the new session), raw per-frame angles:

  | SpinePose size | lumbar AUC | thoracic AUC | thoracic d' | thoracic >= 6 deg: rounded / neutral frames |
  |---|---|---|---|---|
  | small | 0.70 | 0.98 | 2.6 | 68% / 5% (a single glitch frame at +11) |
  | medium | 0.93 | 1.00 | 4.7 | 86% / 0% |
  | large | 0.59 | 1.00 | 4.0 | 96% / 0% |

  Lumbar did **not** hold up across model sizes (neutral hinges spread from -18 to +23 deg), consistent with P8/P5/X2 and with the earlier small-vs-medium disagreement on a photo. Hence **lumbar cues are off in the demo** (`DeadliftRuleConfig.lumbar_cues`, `--lumbar-cues` to enable); lumbar is still displayed and logged. With the final rules the earlier clip raised no back-rounding cue; the new session raised four (about 31 s, 38 s, 42 s and 52-57 s), which match visibly hunched backs in the frames, but **nobody has confirmed them against ground truth**.
- **Cost.** SpinePose on a laptop (CoreML): small 6.7 ms, medium 8.7 ms, large 10.7 ms per detection (median). Laptop only; nothing measured on a phone (medium is 2.7x the FLOPs of small, [P1] Table 1).
- **Limits.** One lifter, one camera, one kurta. Natural thoracic kyphosis varies between people, so an absolute threshold will false-alarm on some bodies; per-user calibration from the spine log is the fix. The cue is thoracic ("upper/mid back"), not the lumbar fault P8 cares most about. Needs a side-on view (curvature is not visible from the front). Concave rounding (P5's CCRB, an excessive arch) is not detected. Hinge threshold (30/20 deg) and the staleness window are provisional.
- **Open (do not close silently).** G-16 (lumbar / butt-wink threshold) stays open. Whether a lumbar cue returns depends on a labelled recording with a motion-capture or equivalent reference.
- **Tests.** `tests/test_back_monitor.py` (facing from the trunk, legs-out-of-view analysis, thoracic cue once per hinge, mild curve not flagged, upright convex spine not flagged, only SpinePose raises it, fold past horizontal, lumbar cues optional, rep-verdict integration, dropout handling, staleness, spine log); `tests/test_spine_regions.py::test_rounding_keeps_its_sign_through_a_forward_fold_past_horizontal`.

## G-29 — Partial visibility, framing guidance and low frame rate (found in the first live-webcam test)

- **Gap.** The first webcam run drew no skeleton and counted no rep. Two causes, both mine: (1) `extract_side_pose` required ear, heel and toe as well as shoulder/hip/knee/ankle, so any one out of frame discarded the whole frame (no skeleton, no reps, no explanation); (2) the velocity estimator needed 4 samples in 0.4 s, so below ~8 fps (the heavy pose model on a laptop CPU) the state machine never advanced.
- **Solution.** `landmarks.py`: CORE joints (shoulder, hip, knee, ankle) are required; ear, heel, toe, wrist are optional and switch off only their own feature (spine proxy / facing / bar path). `framing_report` explains what blocks tracking (missing core joint, facing the camera, too small in frame). `estimate_facing` falls back toe-heel (weight 3) → nose-ear (2) → knee-ankle (1) and `DeadliftAnalyzer` locks facing by a weighted vote. Bar drift is disabled without a visible heel+toe rather than guessed. `FrameResult` always carries the raw landmarks and a proxy spine (if ear/shoulder/hip are seen), so something is always drawn. `SlidingSlope(max_window_s=1.0)` stretches the velocity window back over the last 4 samples when the normal 0.4 s window is too sparse: identical at >= 10 fps, and reps are counted down to 4 fps in tests.
- **Provisional thresholds.** Front-view warning when the shoulders are further apart than 0.55 x torso; "small" below 30% of frame height. Not validated on real footage.
- **Tests.** `tests/test_tracking_robustness.py`. User-facing requirements: `docs/DEADLIFT_TRACKING_REQUIREMENTS.md`.

## G-28 — Uploaded-video acceptance and timestamping

- **Gap.** Context.md §1 requires uploaded video (offline batch), SRS FR-CV-001 requires frame validation, and Context.md §6 lists "Validate Video" and "enough valid frames?" steps. No paper specifies input limits.
- **Solution.** `vision/video_input.py`: `validate_video` returns accept / warn / reject with reasons (type, size, decodability, fps, duration, resolution); `iter_video_frames` yields `(t_s, frame)` using the container's timestamp for each frame (right for variable-frame-rate phone video), falling back to `index / fps`, and always strictly increasing; `assess_tracking` warns after analysis when too few frames had a person or no rep was found. The limits are in `VideoLimits` and documented in `docs/VIDEO_INPUT_SPEC.md`.
- **Provisional.** The numeric limits are product choices. Verified: mp4/mov/m4v/avi/mkv/webm with H.264 and MPEG-4 decode with exact timestamps. **Not verified:** HEVC, rotation-metadata (portrait phone) files.
- **Tests.** `tests/test_video_input.py`.

## G-27 — Signed hip angle

- **Gap.** The interior shoulder-hip-knee angle is capped at 180 deg and reads *less* than 180 when the trunk leans **back** past the hip-knee line, exactly as it does for flexion. Fed to the segmenter, a hyperextended lockout looked like the hip closing: a false descent, and LOCKOUT was never entered. Found by running the synthetic scenarios, not by the papers.
- **Solution.** `DeadliftAngles.hip_signed_deg = 180 + wrap(incl(hip->knee) - trunk_incl - 180)`: 85 at a deadlift bottom, 180 straight, > 180 hyperextended. The segmenter and the lockout rule use it; the interior `hip_deg` is still reported and is what the legacy tracker uses. `RepTiming.peak_hip_deg` therefore reads past 180 for a hyperextended lockout ("peak lockout extension").
- **Test.** `test_hyperextended_lockout_still_reaches_lockout_with_signed_hip` (fails if the segmenter is fed the interior angle).

## Resolved decisions

**D-01 (resolved 2026-11-15, team instruction; wording corrected 2026-11-18): the released app is sensor-free.** *Correction:* the first wording said "100% camera-only", which wrongly implied live-camera-only. The app takes **live camera and uploaded video**; what it never uses is an sEMG or other body sensor. This matches `Context.md` §1/§6.1 (one actor; no sensors in the use-case or deployment diagrams).
- sEMG hardware is used **strictly** during **dataset collection** and **FYP evaluation/validation**.
- The Tier-2 athlete protocol (MVC → 1–5 calibration reps → freeze-all-but-head fine-tune, [P3]) is therefore a **research/evaluation experiment** run by the team on consenting test participants. It answers "does personalisation help?" and is reported in the thesis. It is **not** a released-app feature, and the app needs no sEMG UI, serial/BLE code or MVC screens.
- `ROADMAP.md` Phase 5's "Hybrid Model (Hardware-Calibrated)" variant should be reworded as "research/evaluation variant" to match.
- If a personalised model is later shipped, sEMG-derived conditioning channels (log MVC, per-phase %MVC summaries) must be zero-masked with their indicator channel set, leaving only video-derived anthropometric channels (G-09). That would be a new decision, not an extension of this one.
- Code consequence: `calibration/` stays under `src/myomotion/` but is imported only by research scripts, never by the app/runtime path.

## Open product decisions (conflicts between the brief and `Context.md`)

| ID | Conflict | Needs |
|---|---|---|
| ~~D-01~~ | ~~Tier 2 puts sEMG sensors on the end user~~ | **Resolved 2026-11-15** — see "Resolved decisions" below |
| D-02 | The brief targets YOLO barbell tracking. `Context.md` lists bodyweight-only vs load as an open decision | Decide whether loaded lifts are in scope; affects [P5] phase sync |
| D-03 | The exercise list is not final (`Context.md` §2). [P8] rules cover only the squat | Pick exercises; each needs a cited rule file or `source: TBD` |
| D-04 | Name/term conventions: the brief says "Personalized/Athlete"; earlier team docs say "Personalised mode" | Code uses `tier="global" | "athlete"`; UI wording is the team's choice |
| D-05 | SpinePose is CC-BY-NC-4.0 (code **and** weights), so a commercial release cannot ship it as is. The user (2026-11-19): it is unknown whether the product will be rolled out commercially, so the licence is not a concern for now | Parked, not resolved. Before any public/commercial release: ask the authors for a commercial licence (the Hugging Face organisation is `dfki-av`), or fall back to the back-contour spine (G-32), or train our own model (G-32 roadmap). Until then keep it an optional install (`requirements-spine.txt`), never in `requirements.txt` |
