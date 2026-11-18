# Change Log: spec documents + deadlift biomechanics module (2026-11-15)

Written for: the MyoMotion team, as a code-level record of everything Claude changed in this repository.

Scope: two work sessions. **Session 1** produced the governing documents. **Session 2** did the housekeeping and implemented the deadlift biomechanics module. All of it is in commit `3786785` ("Deadlift biomechanics module, plus demo scripts") except `CLAUDE.md` and `Context.md`, which are gitignored on purpose.

**Not covered here:** `scripts/semg/record_session.py`, `hardware/…`, `tests/conftest.py`, `tests/test_record_session*.py` and the `requirements.txt` pyserial line. That is Aayan's sEMG recorder work (merge `1a0af04`), not part of these sessions.

Every number below was read back from the files on disk. Line counts are from `wc -l`.

---

## 0. File inventory

| File | Status | Lines | Purpose |
|---|---|---|---|
| `CLAUDE.md` | new, gitignored | 194 | project rules, tree, guardrails, module notes |
| `Context.md` | copied, gitignored | n/a | team context, copied from `~/Desktop/Ahmed/Context.md` |
| `docs/METHODOLOGY_SPEC.md` | new | 351 | equations, hyperparameters, splits, metrics, cited to papers P1–P8 |
| `docs/ARCHITECTURE_AND_PIPELINES.md` | new | 258 | module map, 5 pipelines, interfaces, migration table |
| `docs/GAP_ANALYSIS.md` | new | 244 | G-01…G-27 register, D-01…D-04 decisions |
| `pyproject.toml` | new | 21 | package metadata and pytest path |
| `README.md` | modified | +15/−2 | run commands, project tree |
| `.gitignore` | modified | +1 | `docs/Research Papers/` (was already modified before the sessions) |
| `src/myomotion/__init__.py` | new | 3 | package, `__version__ = "0.1.0"` |
| `src/myomotion/biomechanics/geometry.py` | new | 66 | angle and inclination primitives |
| `src/myomotion/biomechanics/landmarks.py` | new | 97 | MediaPipe indices, side and facing selection |
| `src/myomotion/biomechanics/filters.py` | new | 207 | ported filters + `KeypointSmoother` |
| `src/myomotion/biomechanics/spine_proxy.py` | new | 195 | Paper 1 curve-fitting spine heuristic |
| `src/myomotion/biomechanics/deadlift_angles.py` | new | 153 | sagittal/frontal angles, bar path |
| `src/myomotion/biomechanics/rep_segmenter.py` | new | 457 | phase state machine, rep timing, bar peak/valley |
| `src/myomotion/biomechanics/deadlift_rules.py` | new | 333 | form rules, bias classification |
| `src/myomotion/biomechanics/deadlift_pipeline.py` | new | 195 | per-frame orchestration, JSON summary |
| `src/myomotion/biomechanics/synthetic.py` | new | 206 | synthetic lifter with ground truth |
| `src/myomotion/biomechanics/__init__.py` | new | 21 | public exports |
| `src/myomotion/vision/pose.py` (+`__init__`) | new | 66 | MediaPipe wrapper |
| `src/myomotion/coaching/overlay.py` (+`__init__`) | new | 213 | OpenCV overlay |
| `scripts/demo_deadlift.py` | new | 186 | demo CLI |
| `scripts/cv/deadlift/deadlift_tracker.py` | rewritten | 670 → 335 | thin camera loop plus compatibility shim |
| `tests/test_deadlift_biomechanics.py` | new | 645 | 95 tests |
| `tests/test_deadlift_phases.py` | untouched | 275 | 18 original tests, still passing |

Result: `python -m pytest tests -q` → 113 passed (18 original + 95 new) at the end of session 2. `flake8` with `--max-line-length=120` was clean on `scripts/ src/ tests/` at that point.

---

# Part A: Session 1, the governing documents

These are documents, not code, so they're summarised briefly. The code-level material starts in Part B.

**A1. Research.** The ten PDFs in `docs/Research Papers/` were read (text extracted with `pypdf` in a throw-away venv in the scratchpad, outside the repo). Every hyperparameter in the original brief was checked against the paper text.

**A2. `docs/METHODOLOGY_SPEC.md`.** Maps each method to papers [P1]–[P8]. Section 0 lists 12 places where the brief and the papers disagree. The ones that matter for code:

| # | Brief said | Paper says | Consequence in code |
|---|---|---|---|
| V5 | 200 ms window, 10 ms step | [P3]: "10 ms overlap" | `step_ms` is configurable; default 10 ms step |
| V6 | freezing gives +5–12% | +5 pts vs un-fine-tuned, +12 pts vs subject-specific; first-only ≈ last-only ≈ all layers | justify freezing by "no loss", not by the gain |
| V7 | [P3] is our regression | [P3] is classification with sEMG as *input* | transfer protocol is an extrapolation (G-10) |
| V8 | low-cost sensor 20–400 Hz | MyoWare sampled at 333 Hz | assert `f_high < 0.45·fs` |
| V3/V4 | conditioning = MVC + anthropometry; transformer captures eccentric asymmetry | [P2]: `y ∈ R^{2×T}`, no attention mask | extension logged as G-09; live causality as G-07 |

**A3. `docs/ARCHITECTURE_AND_PIPELINES.md`.** Module map, five Mermaid/ASCII pipelines (collection and sync, SpinePose distillation, motion↔muscle training, athlete calibration, real-time loop), the `Skeleton` interface contract, and the migration table.

**A4. `docs/GAP_ANALYSIS.md`.** Register of every non-paper method. Session 1 created G-01…G-22 and D-01…D-04.

**A5. `CLAUDE.md`.** Rules, tensor conventions, leakage rules, anti-patterns, commands.

**A6. Memory.** Added `context-md-location.md` under the Claude memory directory (outside the repo), later rewritten in session 2.

---

# Part B: Session 2, housekeeping

## B1. `Context.md` sync

```bash
cp ~/Desktop/Ahmed/Context.md ./Context.md
cmp ~/Desktop/Ahmed/Context.md ./Context.md        # byte-identical
git check-ignore -v Context.md CLAUDE.md           # .gitignore:43 and :42 → both ignored
```

No `.gitignore` change was needed: lines 42–43 already ignore both files.

## B2. D-01 resolution (Tier 2 vs camera-only)

`docs/GAP_ANALYSIS.md`:
- The D-01 row in the open-decisions table was struck through and now points to a new **"Resolved decisions"** section.
- That section states: the released app is 100% camera-only; sEMG hardware is used only for dataset collection and FYP evaluation; the Tier-2 protocol is a research experiment, not an app feature; if a personalised model ever ships, sEMG-derived conditioning channels are zero-masked with the indicator channel set (G-09).

`CLAUDE.md`:
- Rule 3 now lists only D-02…D-04 as open. Rule 4 became "**The released app is 100% camera-only**…".
- The dual-tier table gained a "Where it exists" row (Tier 2 = research / FYP-evaluation protocol only).

`docs/ARCHITECTURE_AND_PIPELINES.md` ARCH-4 was retitled "Athlete calibration (Tier 2): research / FYP-evaluation protocol only", got a blockquote citing D-01, and its last sequence step changed from "camera-only coaching" to "camera-only evaluation of the personalised model".

**Mistake caught here:** my first `Edit` inserted the new tier table but left the old one in place, so `CLAUDE.md` had two "Dual-tier deployment contract" sections, one of which contradicted D-01 ("deploy only if…"). I found it by reading the file back and removed the stale copy with:

```python
a = s.index("## Dual-tier deployment contract", s.index("## Dual-tier deployment contract") + 10)
b = s.index("## Repository tree (authoritative)")
s = s[:a] + s[b:]
assert s.count("## Dual-tier deployment contract") == 1
```

## B3. Directory guardrails in `CLAUDE.md`

The old "Directory layout (target)" section was replaced with an authoritative tree plus seven numbered guardrails. The three you asked for are rules 1–3:

1. NEVER create a duplicate top-level folder for a package module (`src/models/`, `src/biomechanics/`, `biomechanics/` at root). Each name exists once, under `src/myomotion/`.
2. NEVER scatter loose prototype scripts in the repo root or `src/`.
3. ALWAYS inspect the tree and `pyproject.toml` / `requirements.txt` before creating, moving or renaming files or adding a dependency.

Rules 4–7 I added: new subpackages need approval; exercise code is named `<exercise>_*.py` inside the owning module; `biomechanics/` is NumPy + stdlib only; run pytest before and after any move.

## B4. `pyproject.toml` (new)

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "myomotion"
version = "0.1.0"
requires-python = ">=3.12"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- **Why it exists:** the repo had no `pyproject.toml`, and `src/myomotion` must be importable by tests.
- **`pythonpath = ["src"]`:** makes pytest import `myomotion` without installing it.
- **`dynamic dependencies`:** reads `requirements.txt` so versions are never duplicated. The `mediapipe` pin is untouched.
- **Scripts:** `demo_deadlift.py` and the legacy tracker insert `<root>/src` into `sys.path` themselves, so no `pip install -e .` is required.

## B5. Smaller doc edits

- `README.md`: "Running" section now lists the demo commands; the project tree gained `src/myomotion/` and `demo_deadlift.py`.
- `docs/ARCHITECTURE_AND_PIPELINES.md`: ARCH-0 module map shows the real `biomechanics/` file names; the ARCH-7 migration rows are marked **DONE**.
- `docs/METHODOLOGY_SPEC.md` §2.6: phase naming and the signed-hip note.
- `docs/GAP_ANALYSIS.md`: added G-23…G-27 (section G below).

---

# Part C: Session 2, code, file by file

Conventions used by all the modules:
- Image y points **down**. Angles are in **degrees**.
- Inclinations are measured from vertical-up, **forward positive** in the direction the person faces.
- Landmarks are passed as `np.ndarray [33, 3]` = `(x_px, y_px, visibility)`.
- `biomechanics/` imports only NumPy and the standard library.

## C1. `biomechanics/geometry.py`

Pure 2-D helpers.

```python
def joint_angle(a, b, c) -> float:
    v1x, v1y = a[0]-b[0], a[1]-b[1]
    v2x, v2y = c[0]-b[0], c[1]-b[1]
    if math.hypot(v1x, v1y) < 1e-6 or math.hypot(v2x, v2y) < 1e-6:
        return float("nan")
    cross = v1x*v2y - v1y*v2x
    dot   = v1x*v2x + v1y*v2y
    return math.degrees(math.atan2(abs(cross), dot))
```
- Interior angle at vertex `b`, 0–180°. It uses `atan2(|cross|, dot)` instead of `acos` because `acos` loses precision near 0° and 180°, which is exactly where full extension sits.
- It is ported unchanged from the original tracker's `calculate_angle`. The legacy tracker re-exports it under that name.
- Returns NaN for a zero-length segment, so callers must check `isfinite`.

```python
def inclination_from_vertical(lower, upper, facing) -> float:
    dx    = (upper[0] - lower[0]) * facing
    dy_up = lower[1] - upper[1]          # image y is down, so "up" is lower.y - upper.y
    return math.degrees(math.atan2(dx, dy_up))
```
- `facing` is +1 (person faces image-right) or −1 (image-left). Multiplying `dx` by `facing` makes "forward" positive either way, so the same segment gives +45° facing right and −45° facing left, and mirrored poses give identical numbers.
- 0° = vertical, 90° = horizontal, negative = leaning backward.
- `direction_deg` is an alias used for spine tangents. `wrap_deg` wraps to (−180, 180]. `distance` is Euclidean. `unit` normalises and returns the zero vector unchanged.

## C2. `biomechanics/landmarks.py`

- **Indices:** `NOSE=0`, ears 7/8, shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, knees 25/26, ankles 27/28, heels 29/30, foot-index 31/32.
- **`SIDE_INDICES["left"|"right"]`:** maps names (`ear, shoulder, elbow, wrist, hip, knee, ankle, heel, foot`) to those indices.
- **`REQUIRED_SAGITTAL`:** `("ear", "shoulder", "hip", "knee", "ankle", "heel", "foot")`.
- **`landmark_reliable(row, w=None, h=None, min_visibility=0.5)`:** finite and `visibility ≥ 0.5`; optionally inside the image.
- **`choose_side(lm)`:** mean visibility over the seven required landmarks per side; the higher side wins (ties go to right). In a side view the camera-facing limb is visible and the far limb is extrapolated, so this picks the trustworthy one.
- **`facing_direction(lm, side)`:** sign of `foot_index.x − heel.x`; falls back to the knee-vs-ankle x offset, then +1.
- **`SidePose`:** frozen dataclass `{side, facing, pts: name→[x,y], vis: name→float}` with `__getitem__`.
- **`extract_side_pose(lm, side=None)`:** returns `None` if any required landmark is unreliable; raises `ValueError` if the shape is not `(33, 3)`.

## C3. `biomechanics/filters.py`

### Ported unchanged from the tracker
`OneEuroFilter`, `AngleChannel`, `SlidingSlope` and their constants (`FILTER_MIN_CUTOFF_HZ=0.5`, `FILTER_BETA=0.01`, `FILTER_D_CUTOFF_HZ=1.0`, `MAX_ANGULAR_SPEED_DEG_S=600`, `JUMP_SLACK_DEG=8`, `VELOCITY_WINDOW_S=0.40`, `MAX_SAMPLE_GAP_S=0.5`).

- **One-Euro:** a first-order low-pass whose cut-off rises with the filtered speed: `cutoff = min_cutoff + beta·|dx̂|`. It smooths hard when still and tracks closely when moving.
- **`AngleChannel`:** raw → 3-point median (removes any single-frame glitch) → jump gate (reject if `|Δ| > 600·dt + 8°`, where `dt` is since the last *accepted* sample) → One-Euro.
- **`SlidingSlope`:** least-squares slope over a trailing 0.4 s window with irregular timestamps.

### Changed
`OneEuroFilter` gained `reset()` (the constructor now calls it), needed by the keypoint smoother.

### New: `KeypointSmoother`
Applies One-Euro to every coordinate of a `[N, 3]` landmark array.

```python
def update(self, t, lm, scale_px):
    if self._last_t is not None:
        if t <= self._last_t:            return lm.copy()      # non-monotonic: passthrough
        if t - self._last_t > MAX_SAMPLE_GAP_S: self.reset()   # gap: drop stale state
    ...
    scale = max(float(scale_px), 1.0)
    for i in range(self._n):
        x, y, v = lm[i]
        if not (isfinite(x) and isfinite(y) and v >= self._min_vis):
            fx.reset(); fy.reset(); continue                   # unreliable: leave untouched
        out[i,0] = fx(t, x/scale) * scale
        out[i,1] = fy(t, y/scale) * scale
```
- **Torso-length units:** coordinates are divided by `scale_px` before filtering. One-Euro's `beta` multiplies a *speed*, so without normalisation the same constants behave differently at 480p and 1080p.
- **Unreliable landmarks** are passed through and their filter reset, so stale state never drags the point when it reappears.
- **Visibility is not modified.**

### Parameter tuning (measured)
The first values (`min_cutoff=1.5, beta=8.0`) failed my own test: only ~2× jitter reduction. A 1-D grid search on synthetic noise at 30 fps gave:

| min_cutoff | beta | noise std ratio (3 px) | lag @ 0.6 torso/s |
|---|---|---|---|
| 1.5 | 8.0 | 0.43 | 25 ms |
| **1.0** | **3.0** | **0.32** | **57 ms** (84 ms @ 0.3) |

Adopted `KP_MIN_CUTOFF_HZ = 1.0`, `KP_BETA = 3.0`, `KP_D_CUTOFF_HZ = 1.0`. With `beta=8` the noise itself inflated the cut-off. A deadlift moves the bar at ~0.3–1 torso/s over ~2 s, so tens of milliseconds of lag don't matter, while jitter in the overlay and spine curve does. The measurement and rationale are in a comment above the constants.

## C4. `biomechanics/spine_proxy.py`

Implements the [P1] "fit two smooth curves through shoulder/hip drivers, sample at constant inter-vertebral spacing" heuristic, plus the ear as the cervical anchor you asked for.

### Sampling fractions (derived, not copied from the paper)
Vertebral position counted from C1 = 1: C1=1, C4=4, C7=7, T3=10, T8=15, L1=20, L3=22, L5=24, sacrum=25. There are 24 gaps, so:

```python
ARC_FRACTION = {k: (n - 1) / 24 for k, n in _VERTEBRA_NUMBER.items()}
# C1 0.0 | C4 0.125 | C7 0.25 | T3 0.375 | T8 0.5833 | L1 0.7917 | L3 0.875 | L5 0.9583 | SACRUM 1.0
```
[P1] states the constant-spacing assumption but gives no ratios; these come from vertebral counts and are recorded as our derivation (G-24).

### Curve construction (`build_spine_proxy(pose)`)
Anchors: hip = sacral root, shoulder = C7, ear = C1.

```python
v1, v2 = sh - hip, ear - sh
beta   = atan2(cross(u1,u2), dot(u1,u2))        # signed turn from lower to upper chord
if abs(degrees(beta)) > 90: return None         # impossible fold = landmark failure
t_start_tl = rotate(u1, -beta/2)                # sacral tangent: mirror of the C7 tangent
t_c7       = rotate(u1, +beta/2)                # bisector, shared by both curves
t_end_cv   = rotate(u2, +beta/2)
tl = hermite(hip, sh,  t_start_tl*l1, t_c7*l1, 120)   # sacral root -> C7
cv = hermite(sh,  ear, t_c7*l2,      t_end_cv*l2, 120)  # C7 -> C1
```
- Both curves are cubic Hermite splines.
- **Near-circular arcs:** the end tangents mirror about the chord, so each curve's chord direction is the mean of its end tangents.
- **G¹ continuity at C7:** the tangent there is the bisector of the two chords, so the curves join smoothly.
- **Pelvic tangent:** pelvic orientation is unobservable, so the sacral tangent is a mirror assumption.
- **Sampling:** `_sample_by_arclength` resamples each dense polyline at the arc fractions above, rebased per curve. The cervical curve covers fractions 0–0.25 and the thoracolumbar curve 0.25–1.0.

### Region angles (flexion positive)
```python
cervical = wrap(dir(tan C1) - dir(tan C7))      # C7 -> C1
thoracic = wrap(dir(tan T3) - dir(tan L1))      # kyphosis proxy
lumbar   = wrap(dir(tan L1) - dir(tan SACRUM))  # lumbar flexion proxy
pelvic   = inclination_from_vertical(SACRUM, L5, facing)
bend     = wrap(incl(sh->ear) - incl(hip->sh))  # beta in the facing convention
```
Going up a convex-posterior (kyphotic) arc, the tangent rotates *forward*, so rotation upward = flexion.

Checked numerically before building on it: a 20° forward-head bend gives cervical 19.8°, thoracic 11.0°, lumbar 5.5°, pelvic −9.4°; a straight spine gives zeros; mirroring (facing −1) gives the same values; head-back gives the exact negatives. This ratio (thoracic ≈ 0.55·β, lumbar ≈ 0.28·β) follows from the arc-length shares, and the tests assert it.

### The limitation (honest)
All region angles are fixed fractions of one observable, β. The proxy tracks head and upper-back drop relative to the trunk line; it cannot see true lumbar curvature.

### `SpineBaseline`
Rolling buffer (`max_samples=90`, `min_samples=5`) of `(bend, cervical, thoracic, lumbar, pelvic)` during set-up. `delta(p)` returns `SpineDelta` = current − **median** of the buffer, or `None` until ready. Everything downstream uses these baseline-relative deltas, never absolute values.

`SpineProxy` (frozen) carries `points`, `curve`, the five angles and `source="proxy"`; `as_array()` returns `[9, 2]` in `SPINE_LABELS` order, the future SpinePose contract.

## C5. `biomechanics/deadlift_angles.py`

`compute_deadlift_angles(lm, t_s, side=None, bar_xy=None) -> Optional[DeadliftAngles]`:

| Field | Computation |
|---|---|
| `hip_deg` | `joint_angle(shoulder, hip, knee)` (interior, ≤180) |
| `hip_signed_deg` | see below |
| `knee_deg`, `knee_flexion_deg` | `joint_angle(hip, knee, ankle)`; `180 − knee_deg` |
| `trunk_incl_deg` | `inclination_from_vertical(hip, shoulder, facing)` |
| `tibia_incl_deg` | `inclination_from_vertical(ankle, knee, facing)` |
| `trunk_tibia_deg` | `trunk − tibia` ([P8] differential) |
| `torso_len_px`, `foot_len_px` | `|hip→shoulder|`, `|heel→toe|` |
| `spine` | `build_spine_proxy(pose)` |
| `bar_xy`, `bar_source` | detector box centre (`"bar"`) if given, else near-side wrist if visibility ≥ 0.5 (`"wrist"`) |
| `bar_height_torso` | `(floor_y − bar_y)/torso`, `floor_y = max(heel.y, toe.y)`; up = + |
| `bar_drift_foot` | `(bar_x − mid_foot_x)·facing / foot_len`; forward = +; falls back to `0.4·torso` if the foot length is ~0 |

Returns `None` if the side pose is unreliable or any of hip/knee/trunk/tibia is non-finite or the torso length is zero.

### Signed hip angle (G-27)
```python
hip_signed = 180.0 + wrap_deg(inclination_from_vertical(hip, knee, facing) - trunk - 180.0)
```
85° at a deadlift bottom, 180° straight, >180° when the trunk leans behind the hip–knee line. The interior angle can't tell lean-back from flexion (both read <180°). See bug 3 in Part H.

### Frontal plane (front-view footage only)
- `line_tilt_deg(a, b)`: tilt of the line from horizontal, in (−90, 90], positive when the image-right end is lower.
- `knee_valgus_deg(hip, knee, ankle, other_hip)`: `180 − interior angle`, signed positive (valgus) when the knee sits on the same side of the hip–ankle line as the other hip (the midline side). Computed from the sign of two 2-D cross products.
- `compute_frontal_angles(lm)` returns `FrontalAngles(shoulder_tilt, hip_tilt, knee_valgus_left, knee_valgus_right)` or `None`. It is covered by tests but not used by the side-view demo.

## C6. `biomechanics/rep_segmenter.py`

### Phase enum and constants
`Phase`: `STANDBY, SETUP, CONCENTRIC_PULL, LOCKOUT, ECCENTRIC_DESCENT, REP_COMPLETE`.

The tracker's thresholds are carried over unchanged (hysteresis zones so SETUP and LOCKOUT can never both be true):

| Constant | Value | Meaning |
|---|---|---|
| `HIP_SETUP_ENTER/EXIT_DEG` | 130 / 140 | at the bottom / definitely off the floor |
| `HIP_LOCKOUT_ENTER/EXIT_DEG` | 160 / 150 | lockout zone enter / lost |
| `KNEE_LOCKOUT_DEG` | 160 | knee must be this extended |
| `VELOCITY_MOVE_DEG_S` | 8 | below this = stationary |
| `ONSET_DISPLACEMENT_DEG` | 4 | travel from the last turning point |
| `DWELL_ONSET/SETTLE/HOLD/SETUP_S` | 0.10 / 0.15 / 0.10 / 0.30 | condition must hold this long |
| `STALL_TIMEOUT_S`, `TRACKING_LOSS_TIMEOUT_S` | 5.0, 1.0 | abandon / go to STANDBY |

New: `REP_COMPLETE_HOLD_S = 0.4`, `LIFTOFF_TOL_DEG = 1.5`, `BAR_PROMINENCE_TORSO = 0.35` (provisional).

### Transitions
Every one needs its condition to hold for its dwell:

```
STANDBY → SETUP                  hip ≤ 130 and v < +8, for 0.30 s      (standing upright is never SETUP)
SETUP → CONCENTRIC_PULL          v ≥ +8 and hip ≥ 4° above bottom point, or hip > 140
CONCENTRIC_PULL → LOCKOUT        hip ≥ 160 and knee ≥ 160 and |v| < 8, for 0.10 s
LOCKOUT/CONC → ECCENTRIC_DESCENT v ≤ −8 and hip ≥ 4° below top point; or hip < 150 (lockout lost)
ECCENTRIC → REP_COMPLETE         hip ≤ 130 and v > −8 for 0.15 s; rep counted only if lockout was reached
REP_COMPLETE → SETUP             after 0.4 s (or straight to CONCENTRIC_PULL for touch-and-go)
ECCENTRIC → CONCENTRIC_PULL      mid-range reversal; no rep counted
moving phase → STANDBY           stalled 5 s mid-range, or tracking lost 1 s
```

### What changed from the tracker's `DeadliftStateMachine`
1. New phase names; `ISOMETRIC_HOLD` is now `LOCKOUT`.
2. **`REP_COMPLETE`** is a transient phase: `_step_rep_complete` calls `_step_setup` first (so a touch-and-go next rep can start), then returns to SETUP once `t − phase_start ≥ 0.4`.
3. **`emit_rep_complete: bool = True`.** When `False` the machine goes ECCENTRIC → SETUP directly. The legacy shim passes `False`.
4. **`on_transition` callback** replaces the tracker's built-in `print`.
5. **`RepTiming`** records per rep:

```python
@dataclass(frozen=True)
class RepTiming:
    rep_index; t_liftoff_s; t_concentric_start_s; t_lockout_s; t_descent_start_s; t_end_s
    concentric_s = t_lockout − t_concentric_start
    lockout_s    = t_descent_start − t_lockout
    eccentric_s  = t_end − t_descent_start
    duration_s   = t_end − t_concentric_start
    tempo_ratio  = eccentric_s / concentric_s
    peak_hip_deg; peak_knee_deg; bar_peak_t_s; bar_rise_torso
```
The `t_*` values are *detection* times. The detector is causal, so each lags the physical event by a few hundred ms, and the lags are of similar size, which keeps `tempo_ratio` meaningful.

6. **Lift-off estimate.** `_track_bottom` keeps a running minimum hip angle and records the last time the hip was within `LIFTOFF_TOL_DEG` of it:

```python
def _track_bottom(self, t, hip):
    if hip < self._bottom_min: self._bottom_min = hip
    if hip <= self._bottom_min + LIFTOFF_TOL_DEG: self._liftoff_t = t
```
`_start_attempt` snapshots it into `_attempt_liftoff_t`. The public `liftoff_t` property returns the snapshot during a pull/lockout/descent and the running value otherwise. (This snapshot is bug fix 1 in Part H.) The estimate trails the true start by ≈0.35 s because of the One-Euro lag at low speed.

### Bar peak/valley detector (Paper 5 idea)
`normalize_trajectory(h)` rescales to [−180, 180] (constant input → zeros). `BarHalfCycleDetector` is a causal zig-zag detector:

```python
if self._seek in (None, "valley") and h < self._lo[1]: self._lo = (t, h)   # track the low
if self._seek in (None, "peak")   and h > self._hi[1]: self._hi = (t, h)   # track the high
if self._seek in (None, "valley") and h - self._lo[1] >= prom:  → emit ("valley", *lo); seek = "peak"
elif self._seek in (None, "peak") and self._hi[1] - h >= prom:  → emit ("peak",   *hi); seek = "valley"
```
A valley is confirmed once the bar has risen `prominence` above its lowest point since the last peak, and vice versa. The rising half-cycle (valley → peak) is the concentric pull. It is a second opinion only: it fills `bar_peak_t_s` and `bar_rise_torso` and **never gates rep counting**. While drafting I left a dead no-op block at the top of `update`; it was removed before any test ran.

## C7. `biomechanics/deadlift_rules.py`

### Configuration
All thresholds live in a frozen `DeadliftRuleConfig` (with `with_overrides(**kw)`):

| Field | Default | Source |
|---|---|---|
| `bias_threshold_deg` | 10.0 | [P8] |
| `lumbar_enter/exit_deg`, `lumbar_dwell_s` | 4.0 / 2.5 / 0.25 | G-24, provisional |
| `pull_init_window_s` | 1.0 | G-23 |
| `baseline_per_setup` | False | baseline = first stable set-up of the session |
| `early_pull_window_s`, `early_knee_extension_deg`, `early_trunk_increase_deg` | 0.8 / 15 / 3 | G-23 |
| `lockout_backlean_enter/exit_deg`, `lockout_dwell_s` | 8 / 5 / 0.15 | G-23 |
| `bar_drift_enter/exit_foot`, `bar_drift_dwell_s` | 0.35 / 0.25 / 0.20 | G-23 |
| `once_per_rep` | True | one event per flag per rep |

A lumbar Δ of 4° corresponds to ≈14.5° of ear–shoulder–hip bend change (`lumbar = 0.278·bend`).

### Bias
```python
def classify_bias(d, thr=10.0):
    return HIP_EXTENSOR if d > thr else KNEE_EXTENSOR if d < -thr else BALANCED
```
Strict inequalities: exactly ±10° is `BALANCED`. A descriptive label only; it never raises a fault (G-25).

### `_Latch`: hysteresis plus dwell
A one-sided "value ≥ enter" detector, timestamp-based:
- `enabled=False` resets it; `None`/non-finite values hold state and don't advance the timer.
- Once active, it stays active until `value < exit`.
- It fires (returns `True` once) only after `value ≥ enter` has held continuously for `dwell` seconds.

### `DeadliftFormAnalyzer.update(a, phase, liftoff_t)` runs four rules per frame

| Rule | Active when | Condition |
|---|---|---|
| **Lumbar rounding** (`LUMBAR_ROUNDING`, severity danger, confidence `proxy`) | `SETUP`, `REP_COMPLETE`, `ECCENTRIC_DESCENT`, or the first 1.0 s after lift-off | proxy `lumbar_deg` Δ ≥ 4.0° vs baseline, latched |
| **Hips rise early** (`HIPS_RISE_EARLY`, warning) | `CONCENTRIC_PULL`, within 0.8 s of lift-off, once per attempt | knee extended ≥ 15° **and** trunk inclination increased ≥ 3° since the lift-off frame (looked up in a 900-frame history) |
| **Lockout hyperextension** (`LOCKOUT_HYPEREXTENSION`, warning) | `LOCKOUT`, or `CONCENTRIC_PULL` with signed hip ≥ 160° | `−trunk_incl ≥ 8°`, latched with exit at 5° |
| **Bar drift** (`BAR_DRIFT`, warning; `proxy` when the wrist stands in for the bar) | `CONCENTRIC_PULL`, `ECCENTRIC_DESCENT` | `|bar_drift_foot| ≥ 0.35`, exit 0.25 |

- **Baseline handling:** SETUP frames feed `SpineBaseline` until the baseline is ready and a non-SETUP phase is seen, which locks it (unless `baseline_per_setup`).
- **`_raise`** adds the flag to the rep's flag set, then suppresses a duplicate event if `once_per_rep` and the flag was already raised this rep.
- **Accumulators per rep:** `_max_lumbar`, `_min_lockout_trunk`, `_max_drift`, `_rep_flags`.
- **`active`** = latched flags, plus `HIPS_RISE_EARLY` for the rest of the rep once flagged.

### `finalize_rep(rep)` and `reset_attempt()`
`finalize_rep` looks up the frame nearest `rep.t_liftoff_s` in the history, builds an immutable `RepForm` (bias, trunk/tibia/trunk-tibia at lift-off, max lumbar Δ, min lockout trunk, max bar drift, flags), appends it, then calls `reset_attempt()`. `reset_attempt()` clears the accumulators and latches. The pipeline calls it for attempts that never became a rep.

`FormEvent` fields: `flag, t_s, rep_index, severity, message, value, threshold, confidence, phase`.

## C8. `biomechanics/deadlift_pipeline.py`

`DeadliftAnalyzer.process(t_s, lm, bar_xy=None) -> FrameResult` is the single entry point.

Per frame:
1. Lock the body side on the first valid frame (`choose_side`), so it cannot flip mid-set.
2. `raw = compute_deadlift_angles(lm, …)`. If `None`: `segmenter.report_missing(t)`, run `_after_segmenter`, return a result with no angles.
3. Update `_scale` = EMA (0.9/0.1) of the torso length.
4. `lm_s = KeypointSmoother.update(t, lm, scale)`; smooth `bar_xy` with two One-Euro filters in torso units.
5. `ang = compute_deadlift_angles(lm_s, …)`, falling back to `raw` if that returns `None`.
6. `segmenter.update(t, raw.hip_signed_deg, raw.knee_deg, bar_height=ang.bar_height_torso)`: **raw landmarks and the signed hip** drive the segmenter, whose own conditioning was tuned for them.
7. `_after_segmenter`, then `rules.update(ang, phase, segmenter.liftoff_t)`: **smoothed** data drives the rules and overlay.

### Deferred rep finalisation
A counted rep is not finalised immediately; it is held in `_pending` and finalised when `REP_COMPLETE` ends:

```python
def _after_segmenter(self, n_before, reps_before):
    if len(seg.reps) > reps_before: self._pending = seg.reps[-1]
    for _t, old, new in seg.transitions[n_before:]:
        if old == Phase.REP_COMPLETE and self._pending is not None:
            completed = self._flush_pending()
        elif new == Phase.STANDBY or (old == ECCENTRIC_DESCENT and new == CONCENTRIC_PULL):
            if self._pending is None: self.rules.reset_attempt()    # abandoned attempt
```
This is so that anything seen at the bottom of the descent is attributed to the rep it came from (bug 4 in Part H). `summary()` flushes any pending rep first, for clips that end during `REP_COMPLETE`.

### `summary(meta)`
Returns a strict-JSON dict. Keys: `schema="myomotion.deadlift_session/1"`, `meta`, `side`, `frames_total/tracked`, `duration_s`, `rep_count`, `reps`, `events`, `flag_counts`, `phase_transitions`, `bar_extrema`, `disclaimers`, `rule_config`.

`_jsonable` recursively converts enums to values, NumPy scalars/arrays to Python, and **NaN/inf to `null`**, so `json.dumps(..., allow_nan=False)` succeeds (a test asserts this). The `disclaimers` list states that spine cues are a proxy, thresholds are unvalidated, and the bias label is a squat-paper extrapolation.

## C9. `biomechanics/synthetic.py`

A planar kinematic lifter (px units, `Body`: torso 250, thigh 215, shank 215, foot 130, neck 75, arm 300, ankle height 40) used for tests and `--synthetic`.

Joint chain, with `vec(incl, length) = [sin(incl)·length·f, −cos(incl)·length]` and `f = facing`:

```python
knee     = ankle + vec(shank_incl,              shank)
hip      = knee  + vec(shank_incl − knee_flex,  thigh)
shoulder = hip   + vec(trunk_incl,              torso)
ear      = shoulder + vec(trunk_incl + bend,    neck)      # bend = the head/upper-back turn
wrist    = (mid_foot_x + f·drift·foot,  shoulder.y + arm)
```
- **Visibility:** the near side is 0.99; the far side is a copy offset by 8 px at 0.15 visibility.
- **Bottom pose** (κ=60, τ=15, θ=50) gives hip 85°, knee 120°, trunk–tibia +35°. **Top** is all zeros.
- **Scenarios** (`SCENARIOS`): `good`, `lumbar_round` (28° extra bend on the pull), `butt_wink` (28° on the descent), `stripper` (knees extend early and the trunk gets flatter), `hyperextension` (14° back-lean at lockout, ∝ e³), `bar_drift` (0.6 foot-lengths of forward drift mid-pull).
- **`_bump(s)`** is the rounding envelope: it ramps in over the first 8% of bar travel, holds to s = 0.35, and fades to zero by 0.6. It is zero at the floor, so the set-up hold stays neutral and there are no pose jumps (bug 2 in Part H).
- **`ClipSpec`** holds fps, reps, phase durations, noise (std px on the near side), facing and seed. `progress_at(t, spec)` returns `(s, ascending)`. `rep_truth(spec, rep)` returns ground-truth event times. `generate_clip` yields `(t, lm)`. `Body.scaled(k, floor_y, x0)` shrinks the lifter to fit a canvas.

## C10. `vision/pose.py` (`PoseTracker`)

Wraps the MediaPipe Tasks API exactly as the original tracker used it: `PoseLandmarkerOptions`, `RunningMode.VIDEO`, `num_poses=1`, CPU delegate, no CUDA. `mediapipe` is imported lazily inside `__init__`, so importing `myomotion` stays light.

```python
ts_ms = max(int(t_s*1000.0), self._last_ts_ms + 1)     # strictly increasing int ms is mandatory
...
vis = 1.0 if lm.visibility is None else float(lm.visibility)
if lm.presence is not None: vis = min(vis, float(lm.presence))
out[i] = (lm.x*w, lm.y*h, vis)
```
Normalised landmarks are converted to pixels (x by width, y by height, which keeps angles correct on non-square frames), and visibility is combined with presence. A missing model file raises `FileNotFoundError` with an instruction. It returns `None` when no person is found. It supports `with` and `close()`.

## C11. `coaching/overlay.py` (`OverlayRenderer`)

Drawing only; it never mutates analysis state. `render(frame, result, analyzer)` draws:
- **Skeleton:** `BODY_CONNECTIONS` between landmarks with visibility ≥ 0.5.
- **Reference lines:** a dashed vertical at mid-foot (the bar-path reference) and the floor line.
- **Spine:** the interpolated curve, red when `LUMBAR_ROUNDING` is active, with dots for all nine labels and text for C1/C7/T8/L3/S.
- **Bar:** a circle, amber while `BAR_DRIFT` is active.
- **Angle tags** at the hip and knee.
- **HUD** (phase badge, rep count, hip/knee, trunk/tibia/trunk–tibia with bias, spine deltas, bar drift/height) and a last-rep panel (pull/lock/down seconds, tempo, peak hip, bias, flags).
- **Cue banner:** active flags in severity order (danger first) using the static `CUE_TEXT`. A fresh event's numeric message stays for 3 s and replaces the static text of the same flag (a duplicate-text bug fixed after looking at the rendered frames).

Text is ASCII only (Hershey fonts cannot draw "°") and scales with `h/720`. The shadow thickness was reduced from `thick+2` to `thick+1` after the first render showed dark smudges.

## C12. `scripts/demo_deadlift.py`

Thin CLI. Arguments: `--source` (path or integer webcam index) **or** `--synthetic {good,lumbar_round,butt_wink,stripper,hyperextension,bar_drift}` (mutually exclusive, required), plus `--reps`, `--noise-px`, `--model`, `--out-dir` (default `<repo>/output`, gitignored), `--name`, `--side`, `--no-display`, `--no-video`, `--max-seconds`, `--config` (JSON overriding `DeadliftRuleConfig` fields).

- **Timing:** live uses the wall clock; a file uses `frame_index / fps` (constant-frame-rate assumption, as in the tracker).
- **Synthetic mode:** draws on a 1280×720 canvas with `Body().scaled(0.62, …)` and bypasses MediaPipe.
- **Output:** per-event `[FORM]` lines and per-rep `[REP ]` lines on the console, an annotated `mp4v` video, and `<name>_summary.json`.
- **Writer failure** degrades to "no video" with a warning.

## C13. `scripts/cv/deadlift/deadlift_tracker.py` (670 → 335 lines)

Rewritten as a camera loop over the package, with a compatibility layer so the 18 legacy tests pass **unchanged** (they assert, among other things, `P.ISOMETRIC_HOLD.value == "Isometric Hold"` and `not hasattr(P, "LOCKOUT")`):
- Imports `joint_angle as calculate_angle`, all zone/dwell constants and the filter classes from the package, so `dt.calculate_angle`, `dt.HIP_SETUP_EXIT_DEG` etc. still resolve.
- Defines a legacy `Phase` enum (`CONCENTRIC`, `ISOMETRIC_HOLD`, `ECCENTRIC`) and `_TO_LEGACY` mapping (`REP_COMPLETE → SETUP`).
- `DeadliftStateMachine` is **composition**, not inheritance, because it must expose a legacy-typed `phase`/`transitions`: it wraps `DeadliftRepSegmenter(emit_rep_complete=False)` and maps phases on the way out. The `[PHASE]` print is supplied through `on_transition`.
- The MediaPipe-object helpers (`pixel_coords`, `landmark_reliable`, `extract_joint_angles`) stay in the script because they take MediaPipe objects, which `biomechanics/` must never import.

## C14. `tests/test_deadlift_biomechanics.py` (95 tests)

| Area | What is asserted |
|---|---|
| Geometry/angles | conventions for both facings; hip 85°, knee 120°, trunk 50°, tibia 15°, trunk–tibia 35° recovered exactly; standing tall is 180°/0°; signed hip 194° at −14° lean while interior reads 166°; bar drift ±0.5 for both facings; detector bar overrides wrist; unreliable/NaN landmarks → `None`; side selection; frontal tilt/valgus/varus |
| Smoothing | ≥2.5× jitter reduction at rest; lag <150 ms while moving; passthrough for non-monotonic time and low-visibility landmarks; reset after a 4 s gap |
| Spine proxy | arc fractions equal the vertebral-count values; straight spine → all zeros and evenly spaced points; forward head → cervical > thoracic > lumbar > 0 with the 0.555/0.278 shares; mirrored facings agree; degenerate or folded anchors → `None`; baseline needs samples and is relative |
| Segmenter | full phase order incl. `REP_COMPLETE` (0.35–0.5 s) and the legacy flag; 3 reps counted; lift-off within [truth−0.1, truth+0.7] s; failed pull not counted; hyperextended lockout still reaches LOCKOUT with peak hip >185°; bar detector emits valley then peak and ignores sub-prominence jitter; normalisation is exactly [−180, 180] |
| Bias | boundaries at ±10° (10.0 balanced, 10.01 hip, −10.01 knee) |
| Rules | good rep: no flags; each scenario flags exactly its own rule; rounded set-up is **not** flagged (documented limitation); lumbar threshold boundary (6° and 10° no, 20° and 30° yes); config override suppresses a flag; once-per-rep; `_Latch` hysteresis/dwell; standalone `DeadliftFormAnalyzer` API |
| Robustness | every scenario × {15,30,60} fps × both facings × two noise/seed pairs; 12 seeds at 3 px with no false positives; short dropout survived; long dropout → STANDBY and attempt discarded; standing still never counts; all-`None` input never crashes |
| Output | `json.dumps(allow_nan=False)` succeeds; schema and disclaimers present |
| Demo CLI | synthetic run writes a readable video whose frame count equals `frames_total`, plus JSON; `--config` override works |

---

# Part G: Gap-register entries added in session 2 (`docs/GAP_ANALYSIS.md`)

| ID | Entry |
|---|---|
| G-23 | Deadlift rule set is not in P1–P8. [X1] names deadlift errors (non-neutral spine, bar position, locking knees too early) with no thresholds. Table of each rule's evidence |
| G-24 | Spine-proxy construction: anchors, Hermite curves, vertebral-count fractions (our derivation), baseline-relative measures, hard limitation |
| G-25 | [P8]'s ±10° (a squat regression, evaluated at peak knee flexion) reused as a descriptive deadlift label at lift-off |
| G-26 | Segmentation: legacy machine kept, `REP_COMPLETE` added, bar half-cycles as a second opinion, compatibility shim |
| G-27 | Signed hip angle so hyperextension isn't read as a descent |
| G-17 (extended) | Notes the new `KeypointSmoother` and the measured parameters |

---

# Part H: Bugs found and fixed during development

Each came from running the pipeline and looking at the results, then fixing the code rather than loosening a test.

**1. Lift-off time was wrong (6.47 s for a rep starting at 1.2 s).** `_track_bottom` also runs during the descent, overwriting `_liftoff_t` with the end-of-rep time before the rep was recorded. Fix: snapshot into `_attempt_liftoff_t` in `_start_attempt`, use it in `_finish_rep`, and make the `liftoff_t` property phase-aware. After the fix the estimate is 1.57 s vs truth 1.2 s (≈0.35 s filter lag, now documented).

**2. Lumbar rounding was never flagged.** The generator rounded the spine during the set-up hold (the pose function treated `s=0` while ascending as "rounded"), so the baseline absorbed it. This was a generator bug, but it exposed the real baseline limitation, now a test. Fix: `_bump` ramps in over the first 8% of travel and is zero at the floor.

**3. Hyperextension produced a false descent.** The interior hip angle falls when the trunk leans back, which looked like hip flexion, so LOCKOUT was never entered. Fix: `hip_signed_deg` (G-27), fed to the segmenter and used by the lockout rule. Peak hip now reads 194° for a −14° lean.

**4. Butt-wink events were attributed to the next rep** (even a phantom "rep 3" in a 2-rep clip). Fix: deferred finalisation until `REP_COMPLETE` ends (C8). Each rep now carries exactly its own verdict.

**5. Keypoint smoother only halved jitter.** Re-tuned from (1.5, 8.0) to (1.0, 3.0) by measurement (C3).

**6. Discontinuity in the synthetic descent rounding** (pose jumped 28° in one frame at the floor). The symmetric `_bump` removed it and `butt_wink` became its own scenario.

**Smaller:** a dead no-op block in `BarHalfCycleDetector.update` (removed before tests); the duplicate "Dual-tier" section in `CLAUDE.md` (B2); my latch test used a 0.1 s "continuous" run against a 0.2 s dwell, so I rewrote the test data, not the latch; overlay text shadow smudges and a duplicated cue line (C11); unused import, ambiguous `l` variable names and ~20 long lines (flake8 clean afterwards).

---

# Part I: Verification

- **Tests:** 113 passed (18 legacy untouched + 95 new), about 13 s.
- **Stress matrix** (ad hoc script, not a committed test): 6 scenarios × fps {15,30,60} × noise {1,2,3,4} px × 4 seeds × both facings = 576 runs, all with the expected rep count and flag set; zero false positives on `good`. Caveat: synthetic noise is independent per frame, while real tracker error is correlated and systematic, so this is not evidence of real-world sensitivity.
- **Mutation checks:** four deliberate bugs were injected into a copy and each failed a specific test, after which the source was restored and diffed identical: feeding the segmenter the interior hip angle, never collecting the spine baseline, inverting the hip/knee bias return, and removing the lift-off snapshot.
- **Real MediaPipe path:** `--source` on an annotated synthetic mp4 loaded the model and looped without crashing, with 0/121 frames tracked. That proves loading, the file loop and the no-person path, **not** detection accuracy on real lifters.
- **Visual check:** two frames of the rendered output were opened and inspected.
- **Legacy verbose path:** `DeadliftStateMachine(verbose=True)` was exercised directly; the `[PHASE]` lines print with legacy names.

---

# Part J: Known limitations

1. **The spine proxy cannot see true lumbar curvature** (C4, G-24). It reacts to head/upper-back drop relative to the trunk, and a lifter who sets up already rounded is invisible to it. All its cues are labelled "(proxy)". Replace it with SpinePose (G-01) before drawing any real conclusion.
2. **Thresholds are provisional.** Only ±10° is from a paper, and it is a squat value. Stripper, hyperextension and bar-drift rules are coaching practice. Tune them on labelled footage.
3. **No real lifting footage was available.** All evidence is synthetic.
4. **Phase times are detection times** and lag physical events by a few hundred ms. Compare them with hand-labelled ground truth against the SRS ±150 ms criterion before quoting them.
5. **The wrist stands in for the bar** unless a YOLO box is supplied; the YOLO detector isn't wired in yet.
6. **Single-side, side-view only** for the deadlift pipeline. Frontal angles exist but aren't used by the demo.
7. **CI** only lints `scripts/` and doesn't run pytest; `ROADMAP.md` Phase 5 still says "Hybrid Model (Hardware-Calibrated)", which should be reworded per D-01.

---

# Addendum (2026-11-18): uploaded-video input

**Correction to part B2.** "100% camera-only" was the wrong wording: it implied live-camera-only. The released app is *sensor-free* (no sEMG or body sensor) and accepts **live camera and uploaded video**, as Context.md §1 already says. Corrected in `CLAUDE.md` (rule 4, tier table, commands), `docs/GAP_ANALYSIS.md` (D-01), `docs/ARCHITECTURE_AND_PIPELINES.md` (ARCH-4), `docs/METHODOLOGY_SPEC.md` and `README.md`.

| File | Change |
|---|---|
| `src/myomotion/vision/video_input.py` (new) | `validate_video(path, limits)` returns `VideoValidation(ok, errors, warnings, info)`: checks existence, empty file, extension, size, that a frame decodes, fps, duration, resolution. Never raises for a bad file. `iter_video_frames(path, max_seconds)` yields `(t_s, frame)` using `CAP_PROP_POS_MSEC` (correct for variable-frame-rate phone video), falling back to `index/fps`; times are strictly increasing; rotation metadata requested via `CAP_PROP_ORIENTATION_AUTO`. `assess_tracking(tracked, total, reps)` returns plain-language warnings (no person, tracked <60% of frames, no complete rep). Limits are in the frozen `VideoLimits` dataclass |
| `scripts/demo_deadlift.py` | `--source` with a file path now validates first and exits with code 2 plus reasons if rejected. New `--validate-only`. `camera_frames` was split into `webcam_frames` (wall clock) and `uploaded_frames` (container timestamps). The JSON gains `meta.input` (`uploaded_video`/`live_camera`/`synthetic`), `meta.video` (validation) and `meta.quality_warnings` |
| `tests/test_video_input.py` (new, 20 tests) | formats decode (mp4/mov with mp4v and H.264, m4v, avi, mkv), rejection reasons, soft vs hard limits, timestamps, fallback timing via a fake capture, CLI exit codes, and an end-to-end upload through the real MediaPipe model (skipped if `models/` has no model) |
| `docs/VIDEO_INPUT_SPEC.md` (new) | the explicit accepted format and filming requirements |
| `docs/GAP_ANALYSIS.md` | new G-28 (upload limits are provisional product choices) |

Verified: the listed containers and H.264/MPEG-4 decode with exact timestamps. **Not verified** (no ffmpeg to build test files): HEVC and rotation-metadata portrait footage.

Test status: 171 pass; 5 tests in `tests/test_record_session_pty.py` (Aayan's sEMG recorder) fail with a pyserial pty `ioctl` error. They fail identically on committed `HEAD` with these changes stashed, so they are pre-existing and unrelated.

---

# Addendum 2 (2026-11-18): first live-webcam test, no skeleton and no reps

**Reproduced before fixing** (synthetic lifter, ad hoc script): hiding ear, or heel+toe, or ankle+feet gave `frames_with_skeleton=0, reps=0`; and at <= 6 fps `reps=0` even with the full body visible.

| Cause | Fix |
|---|---|
| `extract_side_pose` required ear, heel and toe; one missing joint discarded the frame | `landmarks.py`: `CORE_SAGITTAL = (shoulder, hip, knee, ankle)` required; `OPTIONAL_SAGITTAL = (ear, heel, foot, wrist)` each disable only their own feature. `extract_side_pose(..., require=, facing=)` includes only reliable optional joints; `SidePose.has(name)` |
| Facing came only from heel/toe | `estimate_facing` returns `(direction, weight)`: toe-heel 3, nose-ear 2, knee-ankle 1. `DeadliftAnalyzer._vote_facing` accumulates weighted votes and locks at +/-12 |
| Bar drift needs mid-foot | `compute_deadlift_angles`: without heel+toe `feet_visible=False`, `bar_drift_foot=None` (never guessed), floor estimated as ankle.y + 0.12 x torso |
| Spine proxy needed ear | `build_spine_proxy` returns None when ear/shoulder/hip are absent; the pipeline builds a `partial_spine` whenever those three are visible even if the legs are not |
| Nothing was drawn and nothing explained | New `FramingReport` / `framing_report` (missing core joint, facing the camera if both shoulders are > 0.55 torso apart, body < 30% of frame height). `FrameResult` gains `framing`, `partial_spine`, and `landmarks` is always set when a person is detected |
| Velocity estimator stalled below ~8 fps | `SlidingSlope(window_s, min_samples, max_window_s)`: if the 0.4 s window has < 4 samples it stretches back over the last 4 as long as they span <= 1.0 s. The segmenter uses `max_window_s=1.0`. Identical to the old behaviour at >= 10 fps |
| Overlay drew the full skeleton only when everything was valid | `coaching/overlay.py` rewritten: draws only the tracked near-side chain (heel/toe-ankle-knee-hip-shoulder-ear, shoulder-wrist), green dot = seen, red ring = missing core joint; proxy spine whenever possible; live checklist of all 8 joints plus advice lines; a "next step" hint (`next_step_hint`) stating what the counter is waiting for, using the real thresholds; fps readout and a low-fps warning |
| Console | `demo_deadlift.py` passes `image_wh` and prints `[FRAME]` lines when the framing problem changes |

New docs: `docs/DEADLIFT_TRACKING_REQUIREMENTS.md`. New gap entry: G-29. New tests: `tests/test_tracking_robustness.py` (21 tests). Result: 192 pass (excluding the 6 pre-existing pty tests).

---

# Addendum 3 (2026-11-18): both sides, any heading, real spine

**Reported:** only the right side tracked; nothing when facing the camera; a curved line from head to C1 for the left side or front view; spine points not lying on a curved spine.

**Reproduced / diagnosed first.** (a) Side lock: after the lifter turned, the locked side stayed "right" and 0 of 189 frames got angles. (b) The user's own recorded run (`output/webcam0_*`) showed the proxy bowing like a banana on an *upright standing* back and flagging "rounding" on every rep; the heel-to-toe direction measured from MediaPipe's 3-D landmarks showed the lifter was ~38 degrees off side-on. (c) MediaPipe world landmarks verified on those frames: y is down; 3-D trunk angle within ~3-4 degrees of the 2-D one.

| Area | Change |
|---|---|
| Side | `DeadliftAnalyzer._update_side`: re-evaluated every frame, switch only when the other side is > 0.15 more visible for 0.4 s; `--side` forces. Switch resets facing and spine baseline (`DeadliftFormAnalyzer.reset_baseline`) |
| Facing | `_vote_facing`: decaying weighted vote (replaces the hard lock at +/-12); uses the 3-D forward vector in angled views |
| 3-D angles | new `biomechanics/pose3d.py`: `compute_angles_3d`, `estimate_forward`, `heading_span_equivalent`. `compute_deadlift_angles(..., world=, view=)` uses them when the person is not side-on |
| View | `landmarks.py`: `shoulder_span_ratio`, `view_name`, `side_scores`, thresholds `ANGLED_ENTER/EXIT_SPAN_RATIO` (0.30 / 0.22). The 3-D hip axis decides when world data exist (a 45 degree turn was misclassified as "side" with the 2-D shoulder cue because the far shoulder is hidden) |
| Framing | `FramingReport.view`; "facing the camera" is now a note, not a blocking problem |
| Synthetic | `synthetic.py`: `build_landmarks_3d(yaw_deg)`, `generate_clip_3d`: any heading, both sides, MediaPipe-style world coordinates. `--yaw` in the demo |
| Spine (non-side) | `build_spine_proxy(straight=True)` on mid-ear/mid-shoulder/mid-hip: a straight midline, `curvature_visible=False`; no rounding rules; bar drift off |
| Spine (real) | `build_spine_from_points` (angles from detected points, Catmull-Rom through every point, rejects unordered/low-confidence points); `vision/spine_model.py` (`SpinePoseTracker`, `person_bbox`, `bridge_error`, `spinepose_available`); `DeadliftRuleConfig.lumbar_enter_deg_model/exit` (12 / 8 deg); the baseline restarts when the source changes |
| Proxy cues | `DeadliftRuleConfig.proxy_spine_cues`: library default True (logic tested), **demo default off**, `--proxy-spine-cues` to enable |
| Overlay | both sides drawn (near side bold + labelled, far side thin grey), shoulder and hip lines, spine curve with a source tag, "View: side / angled" and "Spine (proxy: display only)" rows |
| Demo | `--spine auto|spinepose|proxy`, `--spine-size`, `--spine-every`, `--yaw`, `--proxy-spine-cues`; `PoseTracker.detect_full` returns world landmarks |
| Real footage | the new pipeline on the user's recorded run: same 4 reps as the old code; 73 frames used the 3-D path; the spurious `lockout_hyperextension` (a 2-D foreshortening artefact) disappeared |

**Blocked, not done:** installing SpinePose (`onnxruntime`, `tqdm`, `OneEuroFilter`, `spinepose --no-deps`) into the project venv was denied by the permission classifier because it adds third-party code and model downloads to the user's environment. The integration is written and unit-tested with a stand-in model; the real network has **not** been run.

Tests: 233 pass (excluding the 6 pre-existing pty tests): `tests/test_views_and_spine.py` (new, 40), updates to `test_tracking_robustness.py` and the demo CLI tests.

---

# Addendum 4 (2026-11-18): a spine estimate without SpinePose (mobile-friendly)

**Request.** Do something without SpinePose (heavy for phones; the app is meant for phones), using the papers' methodology; say so if SpinePose + MediaPipe is better.

**What the papers allow (read from the extracted text of `docs/resarchpapers/`).** Only P1, P8, P5 (and X1, X2) mention the spine; P2, P3, P4, P6, P7 do not. None offers a spine estimator without a trained model: OpenCap (X2, line 595) says hip and shoulder keypoints are "insufficient for robustly determining sagittal-plane hip, pelvis, and lumbar kinematics" and uses an LSTM marker augmenter trained on mocap data; Fitness-AQA (P5 Table 5) finds image features better than 2-D pose features for lumbar errors (F1 0.61-0.63 vs 0.54). So image evidence is needed; the cheapest phone-friendly source is MediaPipe's own person mask. This is an external method (logged as G-32), not from P1-P8.

| Area | Change |
|---|---|
| New module | `biomechanics/spine_contour.py`: `extract_back_contour` (ray-cast from the hip-shoulder axis to the back edge of the mask, 28 stations, small holes tolerated, robust quartic fit), `spine_from_contour` (spine = back surface moved forward by 0.065 torso, S1 at t=0.08 to C7 at t=1.06 along the chord, nine labels at the vertebral-count fractions, cervical points continue to the ear), `estimate_spine_from_mask`, `synthetic_mask` |
| Shared builder | `build_spine_from_points(..., source=)`: "spinepose" or "contour" |
| Pipeline | `DeadliftAnalyzer.process(..., mask=)`; `_spine_input`: detected points win, else the mask when side-on, else the proxy; front/angled views never use the silhouette for curvature |
| Rules | `lumbar_enter_deg_contour` / `_exit` = 15 / 10 deg (conservative), a third lumbar latch, message "possible back rounding (back contour)", event confidence "contour" |
| Vision | `PoseTracker(masks=True)`, `PoseDetection.mask`. **Bug fixed:** the mask was a view into MediaPipe's own buffer (reused for the next frame); now copied |
| Demo | `--spine auto|spinepose|contour|proxy`; auto = SpinePose if installed else contour; the synthetic lifter gets silhouettes via `generate_clip_3d_masked` / `back_profile_for` |
| Overlay | contour source tag, colour and HUD row |
| Docs | G-32 (with the SpinePose comparison and a roadmap), tracking requirements, README, CLAUDE.md; stale `docs/Research Papers/` paths changed to `docs/resarchpapers/` |

**Evidence on the user's real recording (586 frames, 1080p, loose clothing).** Contour lies on the back (the old proxy ran through the front of the torso and bowed on an upright back); lumbar change relative to set-up goes toward extension at lockout (-12 to -22 deg) and back to ~0 on descent; quartic fit chosen over cubic/quintic by measurement; no false rounding warnings in 4 reps (the proxy raised 5). Cost: +2.1 ms (mask) +0.71 ms (contour) per frame on a laptop CPU.

**Not validated:** accuracy against markers or SpinePose; clothing and lower-back (gluteal) limits; a genuinely rounded back was never in the footage, so cue sensitivity is unknown.

Tests: 251 pass (excluding the 6 pre-existing pty tests): `tests/test_spine_contour.py` (new, 18).
