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
| D-01 | Tier 2 sensors on the end user vs camera-only released app | **Resolved 2026-11-15** |
| D-02…D-04 | Product decisions that conflict between the brief and `Context.md` | **Open** |

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

## G-27 — Signed hip angle

- **Gap.** The interior shoulder-hip-knee angle is capped at 180 deg and reads *less* than 180 when the trunk leans **back** past the hip-knee line, exactly as it does for flexion. Fed to the segmenter, a hyperextended lockout looked like the hip closing: a false descent, and LOCKOUT was never entered. Found by running the synthetic scenarios, not by the papers.
- **Solution.** `DeadliftAngles.hip_signed_deg = 180 + wrap(incl(hip->knee) - trunk_incl - 180)`: 85 at a deadlift bottom, 180 straight, > 180 hyperextended. The segmenter and the lockout rule use it; the interior `hip_deg` is still reported and is what the legacy tracker uses. `RepTiming.peak_hip_deg` therefore reads past 180 for a hyperextended lockout ("peak lockout extension").
- **Test.** `test_hyperextended_lockout_still_reaches_lockout_with_signed_hip` (fails if the segmenter is fed the interior angle).

## Resolved decisions

**D-01 (resolved 2026-11-15, team instruction): the released app runs 100% camera-only.** This matches `Context.md` §1/§6.1 (one actor; no sensors in the use-case or deployment diagrams).
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
