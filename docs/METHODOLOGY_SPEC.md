# MyoMotion — Methodology Specification

_Status: governing spec, v1.0 (2026-11-15). Every method below cites its source. Code that contradicts this file is a bug in the code or in this file; fix one of them in the same PR._

**Citation keys**

| Key | Paper (PDF in `docs/Research Papers/`) |
|---|---|
| **[P1]** | Khan, Krauß, Stricker — *Towards Unconstrained 2D Pose Estimation of the Human Spine* (SpineTrack / SpinePose), arXiv:2504.08110v1 — `2504.08110v1.pdf` |
| **[P2]** | Chiquier, Vondrick — *Muscles in Action* (MIA), arXiv:2212.02978v3 — `2212.02978v3.pdf` |
| **[P3]** | Lehmler, Saif-ur-Rehman, Glasmachers, Iossifidis — *Deep Transfer-Learning for Patient Specific Model Re-calibration: Application to sEMG-Classification*, arXiv:2112.15019v1 — `2112.15019v1.pdf` |
| **[P4]** | Peng et al. — *Towards Video-based Activated Muscle Group Estimation in the Wild* (MuscleMap / TransM³E), ACM MM '24 — `3664647.3680630.pdf` |
| **[P5]** | Parmar, Gharat, Rhodin — *Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment* (Fitness-AQA), ECCV 2022 — `136980104.pdf` |
| **[P6]** | Chariar et al. — *AI Trainer: Autoencoder Based Approach for Squat Analysis and Correction*, IEEE Access 2023 — `AI_Trainer_...pdf` |
| **[P7]** | Bawa, Banitsas — *Design Validation of a Low-Cost EMG Sensor…*, Sensors 2022, 22, 5799 — `Bawa_Banitsas_paper.pdf` |
| **[P8]** | Straub, Powers — *A Biomechanical Review of the Squat Exercise*, IJSPT 2024, 19(4):490–501 — `ijspt_2024_19_4_94600 (2).pdf` |
| [X1] | Ko, Nasridinov, Park — *Real-Time AI Posture Correction for Powerlifting Exercises Using YOLOv5 and MediaPipe*, IEEE Access 2024 (supporting, in folder) |
| [X2] | Uhlrich et al. — *OpenCap*, PLOS Comput Biol 2023 (supporting, in folder) |
| [G-nn] | Engineering gap filled by an external method — see `docs/GAP_ANALYSIS.md` |

---

## 0. Verification notes (read first)

The project brief that commissioned this spec paraphrased the papers. I checked every hyperparameter against the PDF text. These are the places where the brief and the papers **differ**. The paper wins unless a gap entry says otherwise.

| # | Brief said | Paper actually says | Consequence |
|---|---|---|---|
| V1 | SpineTrack-Unreal has label noise; prefer `train-real-coco` / `train-real-yoga` | [P1] says nothing about Unreal label noise or those split names. It says Real was built by active learning from COCO + YogaPose and that **training on SpineTrack alone is sufficient** | Prefer Real for fine-tuning as a *project choice* (in-the-wild realism). Don't cite [P1] for the noise claim. Confirm split names from the released dataset |
| V2 | Spine_01..03 = lumbar, Spine_04..05 = thoracic | [P1] Fig. 2 gives names and vertebrae (C1, C4, C7, T3, T8, L1, L3, L5, sacrum) but the extracted text doesn't fix the name→vertebra order | Code refers to **vertebral labels** (`C1…L5`, `SACRUM`). The name→index map is loaded from the released model metadata, never hard-coded from prose ([G-01]) |
| V3 | MIA conditioning $y\in\mathbb{R}^{C_{cond}\times T}$ (MVC + anthropometry) | [P2] §4.4: "a unique tensor $y\in\mathbb{R}^{2\times T}$ per subject"; contents in the (unavailable) supplementary | Extending $y$ to MVC + anthropometry channels is our extension ([G-09]) |
| V4 | MIA Transformer captures eccentric/concentric force–gravity asymmetry | Not stated in [P2]. [P2] says it captures "long range patterns over time", **no attention masking** | Don't cite [P2] for asymmetry. The non-causal attention needs a live-mode policy ([G-07]) |
| V5 | Lehmler: 200 ms window, **10 ms step** | [P3]: "200 ms sliding window with **10 ms overlap**" | Ambiguous: a literal 10 ms overlap means a 190 ms step. We use a **10 ms step** (190 ms overlap) so features come out at 100 Hz; `step_ms` is configurable and the choice is logged ([G-12]) |
| V6 | Freezing intermediate layers boosts accuracy ~5–12% in 22% fewer epochs | [P3]: fine-tuning gains **+5 points over the un-fine-tuned pretrained model** and **+12 points over subject-specific models**, in **22% fewer epochs**, averaged over all settings. Separately: retraining **only the first or only the last layer** is as good as retraining all layers | Freezing is justified by "no loss vs all-layers", not by the 5–12% gain |
| V7 | [P3] maps to our regression | [P3] is **classification** (NinaPro hand gestures) with sEMG as the **input**. In MyoMotion, sEMG is the **target** | The [P3] DSP chain and features apply to *our sEMG labels/fatigue analysis*. The [P3] transfer protocol is transplanted onto a vision→sEMG regressor (extrapolation, [G-10]) |
| V8 | [P7] filters low-cost sensor 20–400/450 Hz + notch | [P7]: **commercial** system 4th-order Butterworth **20–450 Hz**. **Low-cost** sensor (MyoWare, **333 Hz** sampling) "bandpass Butterworth at 45–55 Hz" + notch for 50 Hz | A 20–450 Hz band needs fs ≥ 1 kHz. The filter design must check `f_high < 0.45·fs` ([§3.1]) |
| V9 | [P7] MVC from isometric/dynamic MVC trials | [P7]: MVC "occurred during the exercises performed", computed from the **RMS signal at peak and mean level** | We record dedicated isometric MVC trials (SENIAM practice, [G-11]) **and** report [P7]-style dynamic peak/mean RMS as a secondary normaliser |
| V10 | [P6] Bi-GRU trained with categorical CE + Adam | [P6] text: Luong dot-product attention, dropout, **1000 epochs, lr 0.001**, pad to **300 frames with −1**, 80:20 split, **94%** val accuracy. Optimiser/loss are in Table 2 (an image, not machine-readable) | Adam + categorical CE is the default; **verify against Table 2 of the PDF** before claiming paper-faithfulness |
| V11 | [P6] polynomial correction on keypoints | [P6] fits curves per coordinate of **stereo-DLT 3D** keypoints (19 of 33 MediaPipe points). The order *n* isn't given | Monocular adaptation and polynomial order selection are logged ([G-13]) |
| V12 | [P5] YOLO | [P5] used **custom YOLOv3** for barbells; [X1] used **YOLOv5s** for person + equipment | We follow [X1] for the detector family and [P5] for trajectory processing |

---

## 1. Problem formulation

Symbols below are fixed project-wide (matches the glossary in `Context.md`).

- $X(t)\in\mathbb{R}^{F}$: per-frame **kinematic feature vector** (joint angles, segment inclinations, regional spine angles; optionally root-centred normalised keypoints). On a sequence: $x\in\mathbb{R}^{T\times F}$.
- $Y(t)\in[0,\infty)^{M}$: **measured** sEMG activation in **%MVC** for $M$ muscles. $\hat Y(t)$ = predicted.
- Rep phases: `ECCENTRIC`, `ISOMETRIC_HOLD`, `CONCENTRIC`, `REST` (`SETUP`/`STANDBY` are tracker states, see `scripts/cv/deadlift/deadlift_tracker.py`).
- Subjects $s\in\mathcal{S}$, exercises $e\in\mathcal{E}$, muscles $\mathcal{M}$; each recording block measures a subset $\mathcal{M}_b\subset\mathcal{M}$, $|\mathcal{M}_b|=3$ (hardware: 3 channels).

Learning problems:

1. **Motion → Muscle** $\hat m=E_\theta(x)$ [P2 Eq. 1].
2. **Muscle → Motion** $\hat x=D_\omega(m)$ [P2 Eq. 1].
3. **Form-error classification** per rep [P6], with rule-based guardrails [P8].
4. **Spine keypoint estimation** extending the body skeleton [P1].

---

## 2. Vision and spine methodology

### 2.1 Person and equipment detection — [X1], [P5]

- Detector: **YOLOv5s**, fine-tuned for classes `{person, barbell_plate, dumbbell, kettlebell}` [X1 §III-C: YOLOv5s was chosen for real-time suitability].
- The person box gives the ROI crop for pose estimation [X1 Fig. 1: YOLOv5 → person ROI → MediaPipe].
- The equipment box centre $c_t=(u_t,v_t)$ feeds the **trajectory synchroniser** (§2.6) [P5 §3, Fig. 3].

### 2.2 Base skeleton — MediaPipe PoseLandmarker (Tasks API)

- 33 landmarks. Use `pose_landmarks` (normalised image coords, with `visibility`/`presence`) **and** `pose_world_landmarks` (metres, origin at the hip centre) [G-02].
- Model: `models/pose_landmarker_heavy.task` for offline/upload work. For live work, use `full`/`lite` if the latency budget (§ARCH-5) is exceeded.
- Reliability gate: landmark used only if `visibility ≥ 0.5` and `presence ≥ 0.5` and inside the image (current `landmark_reliable()` in the deadlift tracker).
- MediaPipe has **no spine landmarks**. Its only trunk points are 11/12 (shoulders) and 23/24 (hips). Spine comes from §2.3.

### 2.3 Regional spine extension — [P1]

**Topology [P1 §3.1, Fig. 2]:** 9 spinal keypoints — cervical **C1, C4, C7**; thoracic **T3, T8**; lumbar **L1, L3, L5**; **sacrum** (root, "Hip"). Also 2 sternoclavicular joints and 6 foot points. SpinePose outputs **37 keypoints** in total [P1 Table 1].

**MyoMotion macro regions** (used by biomechanics; built from the 9 points):

| Region | Points | Used for |
|---|---|---|
| Cervical | C1, C4, C7 | neck flexion (head-forward faults) |
| Upper thoracic | C7, T3 | upper-back rounding |
| Mid thoracic | T3, T8, L1 | thoracic kyphosis proxy |
| Lumbar | L1, L3, L5 | lumbar lordosis/flexion proxy |
| Sacral / lumbopelvic | L5, SACRUM (+ MediaPipe hip centre) | pelvic tilt proxy, "butt wink" |

**Inference path (default):** run a released **SpinePose** checkpoint (RTMPose family, top-down, input 256×192) on the YOLO person crop. Then fuse with MediaPipe through the bridge in [G-01].

**Bootstrap heuristic [P1 §3.1.2]** (pseudo-labels and fallback): use shoulder and hip joints as drivers, fit **two smooth curves** around the spinal region, and sample keypoints at predefined intervals assuming **constant inter-vertebral distance**. It is only used (a) to seed pseudo-labels for active-learning refinement on our own gym frames, and (b) as an explicitly flagged low-confidence fallback when SpinePose is unavailable. **A curve driven only by shoulders and hips contains no curvature information.** Never use fallback output for lumbar-flexion rules ([X2]: keypoint-only trunk/pelvis kinematics gave 12–39° larger errors; `Context.md` §3.5).

**Teacher–student training [P1 §3.2]** (only if we train our own spine model, e.g. a smaller mobile student or adding gym data):

- Spawn the student $S$ from the pretrained teacher $T$ (body expert). $\mathcal{K}_{student}=\mathcal{K}_{body}\cup\mathcal{K}_{spine}$, $\mathcal{K}_{body}\cap\mathcal{K}_{spine}=\emptyset$ [P1 Eq. 1].
- **Head expansion:** copy pretrained weights into the original rows. Initialise **new rows from a normal distribution with mean/std computed from the existing weights**, *not zeros* [P1 §3.2].
- Losses [P1 Eqs. 2–9]:
  $$\mathcal{L}_{pos}=\sum_{k\in\mathcal{K}_{student}}\mathrm{KL}(p_{s,k},g_k)\qquad \mathcal{L}_{distill}=\sum_{k\in\mathcal{K}_{body}}\mathrm{KL}(p_{s,k},p_{t,k})$$
  $$\Delta\phi_b=\operatorname{atan2}\big(\sin(\phi^p_b-\phi^g_b),\cos(\phi^p_b-\phi^g_b)\big),\qquad \mathcal{L}_{structure}=\frac{1}{\pi|\mathcal{B}|}\sum_{b\in\mathcal{B}}|\Delta\phi_b|$$
  ($\phi_b$ = bone angle w.r.t. the horizontal axis)
  $$\tilde s_1=s_1,\quad \tilde s_i=\tilde s_{i-1}+w_i(s_i-\tilde s_{i-1}),\quad w_i=1-0.5\,\sigma\big(\alpha_s(\lVert s_i-\tilde s_{i-1}\rVert-T)\big),\quad \mathcal{L}_{spine}=\mathrm{MSE}(\{s_i\},\{\tilde s_i\})$$
  ($s_i$ = soft-argmax spine positions; $\alpha_s$ = sigmoid sharpness, distinct from the loss weight $\alpha$; $T$ = distance threshold. Neither $\alpha_s$ nor $T$ is given in [P1] → config, tune on val, [G-14].)
  $$\boxed{\mathcal{L}_{total}=\alpha\mathcal{L}_{pos}+\beta\mathcal{L}_{distill}+\gamma_1\mathcal{L}_{structure}+\gamma_2\mathcal{L}_{spine},\quad \alpha=5,\ \beta=2.5,\ \gamma_1=0.1,\ \gamma_2=0.5}$$
  [P1 Table 2 caption].
- Schedule [P1 §4]: start from **Body8-pretrained RTMPose**; **10 epochs**; **AdamW, lr = 4×10⁻³**; **linear warm-up for 1000 iterations**, then **cosine annealing to 5% of the initial lr**; effective batch **1024** (scale lr linearly if smaller [G-15]); input **256×192**; flip test at evaluation.
- Ablation expectation [P1 Table 2]: full loss gives SpineTrack AP 88.4 / COCO 75.2 (SpinePose-l). Distillation guards COCO/Halpe AP; $\mathcal{L}_{spine}$ gives the largest spine AP gain.
- Data: SpineTrack-Real preferred for gym realism (see V1). Ground truth for **our** accuracy claim: spinous-process markers (C7, T12, L3, S1) filmed side-on, target **lumbar angle MAE ≤ 5°** (SRS acceptance criterion).

**Metrics [P1 §4]:** COCO-style OKS **AP/AR**, reported separately for body, feet, spine and overall.

### 2.4 Coordinate normalisation (monocular) — [G-02]

[P2] used VIBE 3D keypoints normalised to a bounding box. [P6] used stereo DLT. We have one camera:

1. Take `pose_world_landmarks` (hip-centred, metres).
2. Root-centre at the hip midpoint (already the origin; recompute for robustness).
3. Scale-normalise by **torso length** $L_{torso}=\lVert \text{mid-shoulder}-\text{mid-hip}\rVert$, using the median over the athlete's first standing second.
4. Angles (§2.5) are computed from **image-plane** coordinates for the camera-facing plane (side view = sagittal, front view = frontal). Image angles are what MediaPipe localises best. World coordinates are used for out-of-plane sanity checks and for keypoint features.

### 2.5 Biomechanical angles

All angles are in degrees. Interior joint angle at vertex $b$ is $\operatorname{atan2}(|\mathbf{v}_1\times\mathbf{v}_2|,\mathbf{v}_1\cdot\mathbf{v}_2)$, which is stable near 0°/180° (existing `calculate_angle`). Signed angles use the facing direction (nose/toe vs hip) so that **flexion is positive** [G-16].

| Feature | Definition | Source |
|---|---|---|
| Hip angle | ∠(shoulder, hip, knee) | [X1] (11 angles), tracker |
| Knee angle / **knee flexion** | ∠(hip, knee, ankle); flexion $=180°-$ angle | [P8] depth classes |
| Ankle angle | ∠(knee, ankle, foot_index) | [X1] |
| Elbow, shoulder angles | ∠(shoulder, elbow, wrist), ∠(elbow, shoulder, hip) | [X1] (bench press) |
| **Trunk inclination** $\theta_{trunk}$ | sagittal angle of SACRUM→C7 vs vertical (forward +); fallback mid-hip→mid-shoulder (flag `spine_source=proxy`) | [P8] |
| **Tibia inclination** $\theta_{tibia}$ | sagittal angle of ankle→knee vs vertical (forward +) | [P8] |
| **Trunk–tibia angle** | $\Delta_{TT}=\theta_{trunk}-\theta_{tibia}$, **evaluated at peak knee flexion** | [P8] |
| Regional spine bends | $\kappa=180°-\angle(P_a,P_b,P_c)$, signed: cervical (C1,C4,C7), thoracic (C7,T8,L1), lumbar (L1,L3,L5), lumbopelvic (L3,L5,SACRUM) | [P1] points; definition [G-16] |
| Pelvic tilt proxy | inclination of SACRUM→hip-centre line; reported as Δ from the athlete's standing baseline (posterior +) | [P8] concept; definition [G-16] |
| Stance width ratio | inter-ankle distance / shoulder width (front view) | [P8] |
| Foot rotation | heel→foot_index angle vs forward axis (front/top-ish view; low confidence in 2D) | [P8] |
| Knee valgus proxy | frontal-plane knee medial deviation from the hip–ankle line, normalised by femur length (front view) | [P5] KIE, [X1] "knees caved in" |

Signal conditioning before any rule/model: 3-sample median → physiological jump gate (600 °/s + 8° slack) → One-Euro filter ($f_{c,min}=0.5$ Hz, $\beta=0.01$, $f_{c,d}=1$ Hz). Velocity: least-squares slope over 0.40 s [G-17; already implemented and tested in the deadlift tracker].

### 2.6 Repetition and phase segmentation — [P5], tracker

1. **Loaded lifts:** track the equipment box centre $v_t$ (vertical). Smooth it, then **amplitude-normalise the trajectory to [−180, 180]** [P5 §5 impl.]. The roughly parabolic trajectory gives one rep per cycle. The **first half-cycle** and **second half-cycle** [P5 Fig. 3] are mapped to phases **per exercise**: squat/bench = ECCENTRIC then CONCENTRIC; deadlift = CONCENTRIC then ECCENTRIC. [P5] demonstrated this on the back squat only, so the mapping lives in each exercise config.
2. **Bodyweight / no equipment detected:** angle-driven hysteresis state machine (existing `DeadliftStateMachine` pattern: enter/exit zones, velocity ≥ 8 °/s, ≥ 4° displacement, dwell times, all timestamp-based).
3. `ISOMETRIC_HOLD` = |velocity| < threshold inside the end-range zone for ≥ dwell. The deadlift module names the phases `SETUP → CONCENTRIC_PULL → LOCKOUT → ECCENTRIC_DESCENT → REP_COMPLETE` (+ `STANDBY`); `LOCKOUT` is the deadlift's `ISOMETRIC_HOLD`. The segmenter is driven by the **signed** hip angle so a hyperextended lockout stays in the lockout zone (G-26, G-27).
4. Acceptance: phase boundaries within **±150 ms** of hand-labelled ground truth (SRS).

### 2.7 Self-supervised form representations — [P5] (optional, Phase 2)

- **CVCSPC:** quasi-synchronise videos by normalised equipment elevation. Anchor/positive = frames at similar elevation across subjects/views; negative = elevation difference **> δ = 30** (on the [−180,180] scale). Loss = **distance-ratio loss** [P5 Eq. 1]:
  $$\mathcal{L}=-\log\frac{e^{-\lVert\phi_{anc}-\phi_{pos}\rVert_2}}{e^{-\lVert\phi_{anc}-\phi_{pos}\rVert_2}+e^{-\lVert\phi_{anc}-\phi_{neg}\rVert_2}}$$
  Backbone ResNet-18, lr 1e-4, 100 epochs, batch 25 [P5 §5].
- **Motion Disentangling (MD):** anchor = first half-cycle; positive = augmented anchor; negative = second half-cycle. Randomly **temporally reverse** either {anchor, positive} or {negative} so global motion matches and local anomalies (e.g. knees inward) stand out. Backbone R(2+1)D-18, same loss, lr 1e-4, 20 epochs, batch 5, 32-frame clips [P5 §3.2, §5].
- Downstream: freeze/fine-tune → per-error binary heads. Metric **F1** (imbalanced) [P5 §5].

---

## 3. sEMG methodology

### 3.1 Acquisition modes

| Mode | When | Chain |
|---|---|---|
| `raw` | Front-end outputs bipolar raw EMG (e.g. MyoWare RAW pin, commercial) **and** fs ≥ 1 kHz | full §3.2 chain |
| `envelope` | Front-end outputs a hardware-rectified/smoothed envelope. `Context.md` §4.2 says the team's **AD8226 modules do this** | §3.3 chain; frequency features disabled |

The mode is a per-session field in `meta.json`. **Bench-verify the AD8226 output on a scope before data collection**: a raw signal is bipolar around Vref, an envelope is unipolar and slow ([G-05]).

Sampling rule: the bandpass upper edge must satisfy $f_{high}<0.45\,f_s$. At $f_s$ = 333 Hz [P7 low-cost], 20–400 Hz is **impossible**. Arduino Uno target: **3 ch × 2 kHz** (raw) or 3 ch × 1 kHz (envelope), timer-interrupt sampling, binary serial ≥ 250 000 baud ([G-03] bandwidth budget).

### 3.2 Raw-mode DSP chain — [P3], [P7]

Order (per channel):

1. **Standardise** (z-score using the session's rest-baseline statistics) [P3].
2. **Notch** IIR at mains frequency — **50 Hz** (Pakistan) / 60 Hz elsewhere — and harmonics optional [P7]. Q = 30 default ([G-18]: Q isn't given).
3. **Band-pass, 4th-order Butterworth, 20–400 Hz** [P3] (or 20–450 Hz [P7 commercial]). Implemented as second-order sections. Offline = zero-phase `sosfiltfilt`. Online = causal `sosfilt` with persistent state [G-18].
4. **Full-wave rectification** $|x|$ [P3, P7].
5. **Windowing: 200 ms window, 10 ms step** (see V5) [P3].
6. **Envelope** $=$ RMS per window ($\text{RMS}=\sqrt{\tfrac1N\sum x_i^2}$) [P7 Eq. 3] → 100 Hz envelope stream.
7. **Training only:** random **shuffling of window order** inside the training set [P3]. This is for feature-based models only. Sequence models keep temporal order and shuffle **sequences** instead.

### 3.3 Envelope-mode chain

Hardware envelope → anti-alias low-pass (≤ 0.45·fs of the target grid) → rest-baseline subtraction → MVC normalisation (§3.4) → resample to the model grid. No notch/bandpass/rectify (already done in analogue). Fatigue uses amplitude indicators only (§3.6) ([G-05]).

### 3.4 MVC normalisation — [P7], [G-11]

- Per muscle, per **session** (electrodes move between sessions, so redo MVC every session): 3 × 5 s isometric MVC trials (SENIAM-style positions [G-11]), 60 s rest.
- $\text{MVC}_m=\max_{\text{trials}}\ \max_{\text{windows}}\ \text{RMS}_{200\text{ms}}$ (raw) or the envelope peak (envelope mode).
- Also compute the [P7]-style **dynamic peak and mean RMS** during the exercise block. Store both; `%MVC` uses the isometric MVC.
- $Y_m(t)=100\cdot\text{env}_m(t)/\text{MVC}_m$ → **%MVC** (can exceed 100 in dynamic work; don't clip, but flag > 150).

### 3.5 Eighteen handcrafted features — [P3]

Computed per 200 ms window. Inputs: $x$ = filtered **non-rectified** window for sign/spectrum-based features, $|x|$ for amplitude features ([G-19]: [P3] lists features after rectification, but zero crossings/WAMP/spectra are undefined or distorted on a rectified signal). Thresholds/bands aren't given in [P3]; defaults in [G-19].

*Time domain (11):* VAR $=\frac{1}{N-1}\sum x_i^2$; RMS; IEMG $=\sum|x_i|$; MAV $=\frac1N\sum|x_i|$; LOG $=\exp(\frac1N\sum\log|x_i|)$; WL $=\sum|x_{i+1}-x_i|$; AAC $=\text{WL}/N$; DASDV $=\sqrt{\frac{1}{N-1}\sum(x_{i+1}-x_i)^2}$; ZC (sign changes with $|x_i-x_{i+1}|\ge\epsilon$); WAMP $=\sum[|x_i-x_{i+1}|\ge\epsilon]$; MYOP $=\frac1N\sum[|x_i|\ge\epsilon]$.

*Frequency domain (6),* on the periodogram/Welch PSD $P_j$ at $f_j$: FR (low-band / high-band power); MNF $=\sum f_jP_j/\sum P_j$ [P7 Eq. 5]; MDF (frequency splitting total power in half); PKF $=\arg\max P_j$; MNP $=\frac1M\sum P_j$; TTP $=\sum P_j$.

*Wavelet (1):* wavelet histogram ([G-19] definition).

**Use in MyoMotion:** (a) signal-quality gating and SNR checks; (b) fatigue indicators (§3.6); (c) the athlete calibration summary in the Tier-2 conditioning vector (§4.4); (d) the feature-MLP baseline for [P3] reproduction. They are **not** inputs to the camera-only models (no sensors at runtime).

### 3.6 Fatigue — [P7]

Within a set (and across sets in a session), fatigue is indicated by **RMS ↑ and MAV ↑ together with MNF ↓** [P7 §3.2]. Implementation: per rep, compute the median RMS/MAV/MNF over the concentric phase. Fit a linear trend over reps. Flag fatigue when slope(RMS) > 0 and slope(MAV) > 0 and slope(MNF) < 0, each significant (one-sided, p < 0.05, over ≥ 4 reps) [G-20 for the test]. Envelope mode: amplitude indicators only; report "frequency indicator unavailable".

Protocol consequence (also `Context.md` §3.4): fatigue, load and tempo change amplitude. Posture-variation comparisons must hold **load and tempo constant**, **randomise order**, and include rest.

### 3.7 Sensor validation metrics — [P7]

Against a reference system (or test–retest): **ICC** (two-way, absolute agreement, single measure, ICC(A,1); [P7] reports "absolute agreement" and gives the one-way form $S_b^2/(S_b^2+S_w^2)$, so we report both, [G-21]) and **Spearman ρ** $=1-\frac{6\sum d_i^2}{n(n^2-1)}$ [P7 Eq. 1]. Interpretation bands per [P7]: 0.90–1.00 very high, 0.70–0.89 high, 0.50–0.69 moderate, 0.26–0.49 low. Reference values: peak-level ICC 0.74–0.92 (mean 0.83), Spearman 0.71–0.85.

---

## 4. Motion-to-muscle models

### 4.1 Data windowing and synchronisation — [P2]

- [P2]: sEMG at 10 Hz, video 29.97 fps resampled to the sEMG rate, aligned by timestamps. **T = 30 frames = 3 s** in and out. Our default: **model grid 10 Hz, T = 30** (paper-faithful). Option: 30 Hz grid with T = 90 (same 3 s) as an ablation.
- Video features $X(t)$ (30–60 fps) and envelope $Y(t)$ (100 Hz) are both resampled onto the grid by timestamp interpolation (`sync/`, [G-03]).
- **Splits — our primary protocol is stricter than [P2]:** [P2] randomly assigned 20% of 3 s intervals per exercise per subject to test (same-subject leakage). We report **LOSO** [P3] as the headline. The [P2] protocol is reported only as "MIA-comparable".

### 4.2 Encoder $E_\theta$ — [P2 §4.3]

$E(x)=f_{\theta_2}(g_{\theta_1}(x))$.

- $g_{\theta_1}$: temporal convolution. The kernel's spatial extent spans **all F input channels**, temporal extent **≈ 1 s** ($k=\text{round}(1.0\cdot f_{grid})$, forced odd → 11 at 10 Hz, 'same' padding), **128 output channels** → $d_1..d_T\in\mathbb{R}^{128}$.
- $f_{\theta_2}$: **Transformer encoder, 4 layers, 8 heads, d_model = 128, no attention mask** (offline/upload). FFN width, dropout and positional encoding aren't specified in [P2] → 512 / 0.1 / sinusoidal [G-08].
- Head: **FC 128 → M** per time step → $\hat m_t\in\mathbb{R}^M$.
- Loss: **MSE** [P2 Eq. 2], **masked to the muscles measured in the sample's recording block** ([G-06]: our 3-channel hardware measures different muscle subsets per block):
  $$\mathcal{L}_{E}=\frac{\sum_{t,m}\mu_{m}\,(\hat m_{t,m}-m_{t,m})^2}{\sum_{t,m}\mu_m},\quad \mu_m=\mathbb 1[m\in\mathcal M_b]$$
- Optimiser: **Adam** [P2 §4.1]. lr isn't specified in the main text → 1e-4 default, tuned on inner LOSO folds [G-08].

### 4.3 Decoder $D_\omega$ and muscle-goal editing — [P2 §4.2–4.3, §6]

- Same factorisation: $g_{\omega_1}$ (conv spanning all M muscle channels, ≈ 1 s) → 4-layer/8-head Transformer → FC 128 → F.
- Loss MSE on $x$. Joint objective $\min_{\theta,\omega}\mathbb E[\mathcal L(E_\theta(x),m)+\mathcal L(D_\omega(m),x)]$ [P2 Eq. 2].
- **Editing** [P2 Eq. 3–4]: $\bar m^k=s\cdot\hat m^k$, $\bar m^j=\hat m^j\ \forall j\ne k$; $\bar x=D_\omega(\bar m)$. [P2 §6] edits one *or more* muscles. MyoMotion goal vector: **raise the target muscle** ($s>1$) and **hold or lower compensators** (e.g. lumbar erector spinae in the squat), because more total activation isn't automatically better (`Context.md` facts).
- **Safety post-filter** (mandatory): $\bar x$ is rejected if it violates any [P8] guardrail (§5.1), e.g. a deeper squat that induces posterior pelvic tilt.
- **Anthropometric rendering:** $\bar x$ (angles) → forward kinematics with the athlete's measured segment lengths → overlay skeleton ([G-13]). Then [P6] polynomial smoothing/time-alignment (§5.3).

### 4.4 Tier 1 vs Tier 2 — [P2 §4.4], [P3]

**Tier 1 — Global ("Average Joe")**: non-conditional $E_\theta$ trained on all training subjects. Input = camera-derived $x$ only. Output = %MVC.

**Tier 2 — Athlete (personalised)**:

1. **Conditioning tensor** [P2 §4.4]: $y\in\mathbb{R}^{C_{cond}\times T}$, a per-athlete vector tiled over time and concatenated to $d_1..d_T$ before the Transformer. [P2] used $C_{cond}=2$. Ours (extension, [G-09]): 
   - anthropometrics from video: femur/tibia/torso/humerus ratios to height-proxy (4)
   - per-muscle calibration summaries from the calibration reps: mean & peak %MVC per phase, normalised (≤ 2·M)
   - MVC absolute amplitude (log, per muscle, M), for sensor/skin variance [P3 intro: electrode placement, skin, fat]
2. **Concat-then-project:** $[d_t;y_t]\in\mathbb{R}^{128+C_{cond}}\to\mathbb{R}^{128}$ via a linear `cond_proj`. Initialise it so the 128 base columns are the identity and the conditioning columns are **zero**, so a freshly created Tier-2 model reproduces Tier 1 exactly [G-09].
3. **Fine-tuning protocol [P3]:** initialise from Global weights. **Freeze everything except the last layer** (FC head) **plus `cond_proj`**. Ablation: first layer ($g_{\theta_1}$) only. [P3] found first-only ≈ last-only ≈ all-layers. Train on the **1–5 calibration reps** with **early stopping on a 10% validation split** of calibration windows (min. 1 window) [P3 §Approach]. Keep the hyperparameters of global training [P3].
4. After calibration the sensors come off. Runtime = $E_\theta(x;\,y_{athlete})$, camera only.

Expected effect sizes (for planning, not guaranteed for regression): [P3] +5 points vs un-fine-tuned and +12 vs subject-specific, 22% fewer epochs. Gains shrink quickly with fewer reps, and raw-input models can **degrade** with 1 rep [P3 Table 3]. So Tier 2 must be gated: deploy the personalised head only if its held-out-rep RMSE beats Tier 1 on that athlete's held-out calibration rep.

### 4.5 Multimodal token fusion — [P4] (optional, Phase 2)

If an RGB/visual branch is added beside the skeleton branch:

- **MCT:** C learnable classification tokens. Prediction $y=\mathrm{SoftMax}(P_\alpha(\frac1C\sum_i cls_i))$ [P4 Eq. 1]. For a skeleton GCN branch, concatenate node features with classification-MCT and KD-MCT, project, split, and merge nodes into cls via $\hat{cls}_m=P_s(z^*_{GCN})+cls^*_m$ [P4 Eqs. 2–3].
- **MCTKD:** **KL-divergence** between the KD-MCTs of the two branches **after each feature-map reduction block**, averaged over blocks, added **with equal weight** to the BCE loss [P4 Eq. 4].
- **MCTF:** mixed self/cross attention between classification and KD MCTs [P4 Eqs. 5–9], $cls_f = cls_a + CLS_f(LN(cls_r),LN(cls_m))$, then $cls_f := cls_a + DP(M_\theta(LN(cls_f)))$ [P4 Eq. 11]. Test-time = average of both branches.
- Schedule: 80 epochs MCT-only per branch, then 80 epochs full; AdamW lr 1e-4 [P4 §5.1].
- For MyoMotion regression the "classes" are muscles. BCE is replaced by our masked MSE. KL distillation on tokens is kept.

---

## 5. Form assessment and correction

### 5.1 Deterministic biomechanical rules — [P8]

| Rule | Condition | Output |
|---|---|---|
| Hip/knee bias | $\Delta_{TT}>+10°$ → **hip-extensor bias**; $\Delta_{TT}<-10°$ → **knee-extensor (quadriceps) bias**; otherwise **neutral**. Evaluated at peak knee flexion. (Regression basis: $\Delta_{TT}=-8°$ ⇔ hip/knee flexion-moment ratio 1.0) | bias label + target-muscle cue |
| Depth class | knee flexion 0–90° partial/shallow; 90–110° medium/parallel; 110–135° full/deep | depth label |
| Lumbar-flexion guardrail | forward trunk inclination achieved through **lumbar flexion** instead of hip flexion with a neutral spine → reduced tolerance to compressive load, more anterior shear | safety fault (highest priority) |
| Posterior pelvic tilt ("butt wink") | near end-range hip flexion, posterior pelvic tilt Δ with coupled lumbar flexion Δ (vs standing baseline) exceeds thresholds | safety fault. **No numeric threshold in [P8]** → data-driven [G-16] |
| Stance width | narrow 75–100%, medium 100–150%, wide 150–200% of shoulder width; > 150% → elevated knee valgus moment (+23%) | stance label, valgus watch |
| Foot rotation | ~30° toe-out → knee valgus moment −50%, varus +80%; no change in quad/hamstring/gastrocnemius activation | cue context |
| Activation priors | medium/wide stance: gluteus maximus +13–61%, gastrocnemius −18%; shallow→medium depth: glute max +65%; stance doesn't change quads/hamstrings; lumbar erector spinae doesn't rise beyond 90° knee flexion | sanity priors for $D_\omega$ edits and model QA |

Rules are **exercise-scoped** ([P8] is squat-only). Every other exercise needs its own rule file with its own citations, or `source: TBD`.

### 5.2 Temporal form classifier — [P6]

- Input: per-rep feature sequence (angles + normalised keypoints; [P6] used 19 of 33 MediaPipe points in 3D). **Pad to 300 frames with −1** [P6 §II-C]. **Always pass an explicit boolean mask** too ([G-22]: −1 is a legal value for signed angles and normalised coordinates).
- Architecture: **stacked Bi-GRU** encoder + **Luong (dot-product) attention** $e_{ij}=s_{i-1}^\top h_j$ (or $s_{i-1}^\top W h_j$ if dims differ), $\alpha_{ij}=\mathrm{softmax}_j(e_{ij})$, $c_i=\sum_j\alpha_{ij}h_j$ [P6 Eqs. 3–7], **dropout**, softmax output [P6 Eq. 9].
- Training: **lr 0.001**, **1000 epochs** (keep best val checkpoint), Adam + categorical cross-entropy (V10: verify).
- Classes: exercise-specific taxonomy in `configs/exercises/<exercise>.yaml`. Seed for squat: good, bending forward (BF), toes lifting (TL), knees inward [P5 KIE, X1], knees forward [P5 KFE], shallow [P5 SS], non-neutral spine [X1], butt wink [P6 future work].
- Reference result: **94.0%** over 7 squat classes, better than Bi-LSTM+attn and Bi-RNN (77.5%) [P6]. [P6] used a non-subject-wise 80:20 split, so our number must be **LOSO**.
- It runs **once per completed rep**. Bidirectionality is therefore fine in live mode.

### 5.3 Anthropometry-preserving trajectory correction — [P6]

For each coordinate (or angle) channel of a canonical good rep, fit $y=k_0+k_1\tau+\dots+k_n\tau^n$ on normalised time $\tau\in[0,1]$ [P6 Eq. 10]. For a user rep:

1. **Time-scale** $\tau$ to the user's rep (phase-wise, so eccentric and concentric durations are matched separately).
2. **Shift the intercept** $k_0$ to the user's own offset (start-frame value), preserving their limb lengths/proportions.
3. Order $n$: [P6] doesn't give it → choose per channel by BIC in 2…8 on the reference set [G-13].
4. Output = target trajectory for the overlay ("Aim here" line) and per-joint deviation (deg) for cues.

Combined with §4.3: $D_\omega$ proposes *what to change* (muscle-goal-driven). [P6] curve fitting makes it *temporally aligned and anthropometrically consistent*. [P8] rules veto unsafe targets.

### 5.4 Coaching cues

Cue text names **joint, direction and amount** (`Context.md`, AIFit style). Priority: safety ([P8] guardrails) > form class ([P6]) > activation targeting ([P2] edit). State machine with enter/exit hysteresis, minimum dwell, per-cue cooldown, ≤ 1 spoken cue per rep ([G-04]).

---

## 6. Evaluation protocol

### 6.1 Splits (leakage-proof)

| Split | Definition | Source | Reported as |
|---|---|---|---|
| **LOSO** | train on N−1 subjects, test on the held-out subject. All normalisers fit on train folds | [P3] | **headline** activation metric |
| **Known vs new exercise** | `known_val/test`: exercises seen in training (held-out subjects/reps); `new_val/test`: exercises absent from training. With few exercises → **leave-one-exercise-out** (as [P2] OOD: one model per held-out exercise) | [P4] §3.3, [P2] §5.2 | generalisation |
| Tuning subjects | hyperparameter search on a disjoint subject subset (as [P3] used DB2's first 10 subjects) or nested LOSO | [P3] | — |
| Tier-2 reps | calibration reps ≠ evaluation reps of the same athlete | [P3] | personalisation gain |
| MIA-comparable | random 80/20 of 3 s intervals per subject×exercise | [P2] | appendix only |

### 6.2 Metrics

| Component | Metric | Source / target |
|---|---|---|
| Spine keypoints | OKS AP/AR (body/feet/spine/overall) | [P1]; SpinePose-s spine AP 89.6 |
| Spine angles | lumbar angle MAE vs markers | SRS: ≤ 5° |
| Activation regression | **RMSE** (%MVC) per muscle and exercise | [P2] (their units: 8.0–20.4 in-distribution) |
| | **Pearson r** per muscle (LOSO) | SRS tiered target: r ≥ 0.80 prime movers (ambitious; report tiers) |
| | **ICC(A,1)**, **Spearman ρ** predicted vs measured | [P7] |
| | mAP on thresholded activation (active/inactive) | [P4], for comparability only |
| Form errors | **F1** per error (imbalanced) | [P5] |
| | accuracy, confusion matrix | [P6] |
| Phases | boundary error (ms) | SRS ±150 ms |
| Sync | measured LED/beep offset and drift | SRS ≤ 33 ms |
| Runtime | per-frame latency p50/p95, FPS | SRS ≤ 33 ms |
| Sensor | ICC, Spearman vs reference | [P7] |

### 6.3 Baselines (must be reported alongside)

- Retrieval and conditional retrieval (nearest neighbour) [P2 §5.1].
- Random Forest on [X1]'s 11 angles (posture) [X1].
- [P3] feature-MLP for the sEMG-side checks.
- Trig/proxy spine vs SpinePose spine (ablation that justifies the spine module).
