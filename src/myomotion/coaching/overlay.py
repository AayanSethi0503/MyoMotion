"""OpenCV overlay for the deadlift demo: tracked joints, proxy spine, live angles, phase/rep HUD, framing guidance.

Drawing only: it reads a ``FrameResult`` and the analyzer, and never changes analysis state. Cue arbitration
(priority, cooldown, speech) is the job of the future ``coaching/cue_engine.py`` (G-04); here the banner simply
shows active flags in severity order. OpenCV's Hershey fonts are ASCII only ("deg", never a degree sign).

What is drawn (only the joints the analysis uses, not the full 33-point skeleton):
  * BOTH sides' chains heel/toe - ankle - knee - hip - shoulder - ear, plus the arm to the wrist (the bar proxy),
    with the shoulder line and hip line when both sides are seen. The side facing the camera is drawn bold and labelled;
    a joint is a filled green dot when seen and a hollow red ring when MediaPipe is not confident about it;
  * the spine curve (C1 ... sacrum): through the points detected by SpinePose when available (the curve passes
    exactly through them), else the geometric proxy; red while a rounding flag is active. From the front or back the
    curve is a straight midline, labelled as such, because curvature cannot be seen from that angle;
  * a checklist of which tracked joints are visible, the framing problems, and a hint about what the rep counter
    is waiting for next, so a missing rep always comes with a reason.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from myomotion.biomechanics.deadlift_pipeline import DeadliftAnalyzer, FrameResult
from myomotion.biomechanics.deadlift_rules import BIAS_LABEL, FLAG_SEVERITY, FormFlag, Severity
from myomotion.biomechanics.landmarks import SIDE_INDICES, TRACKED_SAGITTAL
from myomotion.biomechanics.rep_segmenter import (HIP_LOCKOUT_ENTER_DEG, HIP_SETUP_ENTER_DEG, KNEE_LOCKOUT_DEG,
                                                  Phase)
from myomotion.biomechanics.spine_proxy import SPINE_LABELS

# BGR
PHASE_COLOURS: Dict[Phase, Tuple[int, int, int]] = {
    Phase.STANDBY: (120, 120, 120),
    Phase.SETUP: (200, 200, 200),
    Phase.CONCENTRIC_PULL: (0, 200, 0),
    Phase.LOCKOUT: (0, 180, 255),
    Phase.ECCENTRIC_DESCENT: (0, 100, 255),
    Phase.REP_COMPLETE: (255, 200, 0),
}
GREEN, AMBER, RED = (0, 200, 0), (0, 170, 255), (40, 40, 230)
WHITE, GREY, CYAN = (240, 240, 240), (170, 170, 170), (255, 255, 0)
BONE = (235, 235, 235)

CUE_TEXT = {
    FormFlag.LUMBAR_ROUNDING: "CHEST UP - possible back rounding (proxy)",
    FormFlag.HIPS_RISE_EARLY: "HIPS RISING TOO FAST - push the floor away",
    FormFlag.LOCKOUT_HYPEREXTENSION: "DON'T LEAN BACK AT LOCKOUT - finish tall",
    FormFlag.BAR_DRIFT: "KEEP THE BAR OVER MID-FOOT",
}

# The tracked chain, ordered from the floor up. A bone is drawn only when both ends are visible.
CHAIN_BONES = [("heel", "foot"), ("heel", "ankle"), ("foot", "ankle"), ("ankle", "knee"), ("knee", "hip"),
               ("hip", "shoulder"), ("shoulder", "ear"), ("shoulder", "wrist")]
JOINT_LABELS = {"ear": "ear", "shoulder": "shoulder", "hip": "hip", "knee": "knee", "ankle": "ankle",
                "heel": "heel", "foot": "toe", "wrist": "wrist (bar)"}
CORE = ("shoulder", "hip", "knee", "ankle")
SPINE_DOT_LABELS = {"C1": "C1", "C7": "C7", "T8": "T8", "L3": "L3", "SACRUM": "S"}
EVENT_DISPLAY_S = 3.0           # how long the numeric message of a fresh event stays on screen
LOW_FPS = 8.0                   # below this the rep counter becomes unreliable (see filters.SlidingSlope)


def _text(img, txt: str, org: Tuple[int, int], scale: float, colour, thick: int = 1, shadow: bool = True) -> None:
    if shadow:
        cv2.putText(img, txt, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1,
                    cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _panel(img, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.55) -> None:
    sub = img[max(y0, 0):y1, max(x0, 0):x1]
    if sub.size:
        img[max(y0, 0):y1, max(x0, 0):x1] = (sub * (1 - alpha)).astype(img.dtype)


def next_step_hint(res: FrameResult) -> Tuple[str, Tuple[int, int, int]]:
    """What the rep counter is waiting for right now, in plain words."""
    a = res.angles
    if a is None:
        return "Not tracking yet: fix the items listed on the left", AMBER
    hip = a.hip_signed_deg
    if res.phase == Phase.STANDBY:
        if hip > HIP_SETUP_ENTER_DEG:
            return (f"START: hinge forward into your set-up (hip now {hip:.0f} deg, "
                    f"needs <= {HIP_SETUP_ENTER_DEG:.0f}) and hold still", AMBER)
        return "START: hold still for a moment to register your set-up", AMBER
    if res.phase in (Phase.SETUP, Phase.REP_COMPLETE):
        msg = "Rep counted. " if res.phase == Phase.REP_COMPLETE else "Set-up registered. "
        return (msg + f"Now pull to a full lockout (hip >= {HIP_LOCKOUT_ENTER_DEG:.0f}, "
                f"knees >= {KNEE_LOCKOUT_DEG:.0f} deg)", GREEN)
    if res.phase == Phase.CONCENTRIC_PULL:
        return (f"Stand fully tall to count the rep (hip {hip:.0f} / knee {a.knee_deg:.0f} deg; "
                f"needs >= {HIP_LOCKOUT_ENTER_DEG:.0f} / {KNEE_LOCKOUT_DEG:.0f})", GREEN)
    if res.phase == Phase.LOCKOUT:
        return "Lockout reached. Lower under control back to the start position", GREEN
    return f"Lower to the start position (hip <= {HIP_SETUP_ENTER_DEG:.0f} deg, now {hip:.0f}) and pause", GREEN


class OverlayRenderer:
    """Stateful only for display timing (fresh-event message, frame-rate estimate)."""

    def __init__(self):
        self._event_msg: Optional[Tuple[float, str, Severity, FormFlag]] = None   # (t_expire, text, severity, flag)
        self._last_t: Optional[float] = None
        self._fps: Optional[float] = None

    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray, res: FrameResult, an: DeadliftAnalyzer) -> np.ndarray:
        k = max(0.5, frame.shape[0] / 720.0)
        for ev in res.new_events:
            self._event_msg = (res.t_s + EVENT_DISPLAY_S, ev.message, ev.severity, ev.flag)
        if self._last_t is not None and res.t_s > self._last_t:
            inst = 1.0 / (res.t_s - self._last_t)
            self._fps = inst if self._fps is None else 0.9 * self._fps + 0.1 * inst
        self._last_t = res.t_s

        side = res.angles.side if res.angles is not None else (res.framing.side if res.framing else None)
        if res.landmarks is not None and side is not None:
            self._draw_chain(frame, res.landmarks, "left" if side == "right" else "right", k, near=False)
            self._draw_chain(frame, res.landmarks, side, k, near=True)
            self._draw_body_lines(frame, res.landmarks, k)
            self._draw_spine(frame, res, k)
        if res.angles is not None:
            self._draw_reference_lines(frame, res, k)
            self._draw_bar(frame, res, k)
            self._draw_angle_tags(frame, res, k)
        self._draw_hud(frame, res, an, k)
        self._draw_checklist(frame, res, k)
        self._draw_cues(frame, res, an, k)
        return frame

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_chain(frame, lm: np.ndarray, side: str, k: float, near: bool = True) -> None:
        """One side's tracked chain. The near side (facing the camera) is bold and labelled; the other side is thin,
        grey and unlabelled, and only drawn where MediaPipe is confident about both ends of a bone."""
        idx = SIDE_INDICES[side]
        pts = {n: (int(lm[idx[n], 0]), int(lm[idx[n], 1])) for n in TRACKED_SAGITTAL
               if np.all(np.isfinite(lm[idx[n], :2]))}
        seen = {n: bool(lm[idx[n], 2] >= 0.5) for n in pts}
        bone_colour = BONE if near else (150, 150, 150)
        for a, b in CHAIN_BONES:
            if a in pts and b in pts and seen[a] and seen[b]:
                cv2.line(frame, pts[a], pts[b], bone_colour, max(2, int((3 if near else 2) * k)), cv2.LINE_AA)
        for n, p in pts.items():
            r = max(4, int((6 if near else 4) * k))
            if seen[n]:
                cv2.circle(frame, p, r, (60, 220, 90) if near else (110, 170, 120), -1, cv2.LINE_AA)
                cv2.circle(frame, p, r, (0, 0, 0), 1, cv2.LINE_AA)
            elif near and n in CORE:                          # a missing core joint: show where it is guessed to be
                cv2.circle(frame, p, r + 2, RED, 2, cv2.LINE_AA)
            else:
                continue
            if near:
                _text(frame, JOINT_LABELS[n], (p[0] + int(10 * k), p[1] - int(8 * k)), 0.42 * k,
                      WHITE if seen[n] else RED)

    @staticmethod
    def _draw_body_lines(frame, lm: np.ndarray, k: float) -> None:
        """Shoulder line and hip line when both sides are seen: shows the turn of the body at a glance."""
        for a, b in ((11, 12), (23, 24)):
            if lm[a, 2] >= 0.5 and lm[b, 2] >= 0.5:
                cv2.line(frame, (int(lm[a, 0]), int(lm[a, 1])), (int(lm[b, 0]), int(lm[b, 1])), (200, 170, 90),
                         max(1, int(2 * k)), cv2.LINE_AA)

    @staticmethod
    def _draw_reference_lines(frame, res: FrameResult, k: float) -> None:
        """Dashed vertical through mid-foot (the bar-path reference) and the floor line. Needs both feet joints."""
        a = res.angles
        if not a.feet_visible:
            return
        heel, toe = a.pose["heel"], a.pose["foot"]
        x = int(0.5 * (heel[0] + toe[0]))
        floor_y = int(max(heel[1], toe[1]))
        top = int(max(0, floor_y - 3.2 * a.torso_len_px))
        for y in range(top, floor_y, int(14 * k)):
            cv2.line(frame, (x, y), (x, min(y + int(7 * k), floor_y)), (255, 160, 60), max(1, int(1.5 * k)),
                     cv2.LINE_AA)
        cv2.line(frame, (int(heel[0] - 40 * k), floor_y), (int(toe[0] + 40 * k), floor_y), (120, 120, 120), 1,
                 cv2.LINE_AA)

    @staticmethod
    def _draw_spine(frame, res: FrameResult, k: float) -> None:
        sp = res.angles.spine if res.angles is not None else res.partial_spine
        if sp is None:
            return
        model = sp.source in ("spinepose", "contour")
        if not sp.curvature_visible:
            colour = (200, 150, 200)                          # muted: a midline, not a measured curve
        elif FormFlag.LUMBAR_ROUNDING in res.active_flags:
            colour = RED
        else:
            colour = (255, 200, 60) if model else (255, 120, 255)
        pts = sp.curve.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, colour, max(2, int(4 * k)), cv2.LINE_AA)
        for label in SPINE_LABELS:
            x, y = (int(v) for v in sp.points[label])
            cv2.circle(frame, (x, y), max(3, int(5 * k)), colour, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(3, int(5 * k)), (0, 0, 0), 1, cv2.LINE_AA)
            if label in SPINE_DOT_LABELS:
                _text(frame, SPINE_DOT_LABELS[label], (x - int(30 * k), y + int(4 * k)), 0.4 * k, colour)
        top = sp.points["C1"]
        name = {"spinepose": "SpinePose", "contour": "back contour"}.get(sp.source, "spine proxy")
        tag = name + ("" if sp.curvature_visible else " (curve not visible here)")
        _text(frame, tag, (int(top[0]) + int(14 * k), int(top[1]) - int(14 * k)), 0.42 * k, colour)

    @staticmethod
    def _draw_bar(frame, res: FrameResult, k: float) -> None:
        a = res.angles
        if a.bar_xy is None:
            return
        c = (int(a.bar_xy[0]), int(a.bar_xy[1]))
        colour = AMBER if FormFlag.BAR_DRIFT in res.active_flags else CYAN
        cv2.circle(frame, c, max(5, int(9 * k)), colour, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_angle_tags(frame, res: FrameResult, k: float) -> None:
        a = res.angles
        hip, knee = a.pose["hip"], a.pose["knee"]
        off = int(70 * k * a.facing)                          # on the side the person faces, clear of the body
        _text(frame, f"{a.hip_deg:.0f}", (int(hip[0] + off), int(hip[1] - 10 * k)), 0.6 * k, (0, 255, 0), 2)
        _text(frame, f"{a.knee_deg:.0f}", (int(knee[0] + off), int(knee[1] - 10 * k)), 0.6 * k, (0, 255, 0), 2)

    # ------------------------------------------------------------------
    def _draw_hud(self, frame, res: FrameResult, an: DeadliftAnalyzer, k: float) -> None:
        _panel(frame, 0, 0, int(400 * k), int(330 * k))
        x, y, dy = int(14 * k), int(40 * k), int(26 * k)
        a = res.angles
        if a is None:
            _text(frame, "NOT TRACKING", (x, y), 0.95 * k, AMBER, 2)
        else:
            _text(frame, res.phase.value.upper(), (x, y), 0.95 * k, PHASE_COLOURS[res.phase], 2)
        _text(frame, f"Reps: {res.rep_count}", (x, y + dy), 0.8 * k, WHITE, 2)
        fps_txt = f"{self._fps:.0f} fps" if self._fps else "- fps"
        _text(frame, fps_txt, (int(300 * k), y + dy), 0.55 * k, AMBER if (self._fps or 99) < LOW_FPS else GREY)
        if a is not None:
            rows = [
                f"Hip {a.hip_signed_deg:5.1f}   Knee {a.knee_deg:5.1f} deg",
                f"Trunk {a.trunk_incl_deg:+5.1f}   Tibia {a.tibia_incl_deg:+5.1f} deg",
                f"Trunk-Tibia {a.trunk_tibia_deg:+5.1f}  {BIAS_LABEL[res.bias].split(' (')[0] if res.bias else ''}",
            ]
            rows.insert(0, f"View: {'side (exact 2D)' if res.view == 'side' else 'angled (3D estimate)'}"
                           + ("" if (res.view == 'side' or res.angle_source == 'world') else "  NO 3D DATA"))
            sd = res.spine_delta
            if a.spine is not None and not a.spine.curvature_visible:
                rows.append("Spine: curve not visible from this angle")
            elif a.spine is not None:
                rows.append({"spinepose": "Spine (SpinePose), rel. to set-up:",
                             "contour": "Spine (back contour), rel. to set-up:"}.get(
                    a.spine.source, "Spine (proxy: display only), rel. to set-up:"))
                rows.append(f"  lumbar {sd.lumbar_deg:+4.1f}  thoracic {sd.thoracic_deg:+4.1f}"
                            f"  cerv {sd.cervical_deg:+4.1f}" if sd is not None else "  baseline: waiting for set-up")
            else:
                rows.append("Spine: off (ear not visible)")
            if a.bar_drift_foot is not None:
                rows.append(f"Bar drift {a.bar_drift_foot:+.2f} ft-len   height {a.bar_height_torso:.2f} torso")
            for i, r in enumerate(rows):
                dim = r.startswith("  ") or r.startswith("Spine") or r.startswith("View")
                _text(frame, r, (x, y + dy * (i + 2)), 0.5 * k, GREY if dim else WHITE)
        self._draw_last_rep(frame, an, k)

    @staticmethod
    def _draw_last_rep(frame, an: DeadliftAnalyzer, k: float) -> None:
        if not an.reps:
            return
        w = frame.shape[1]
        r = an.reps[-1]
        f = r["form"]
        lines = [f"Rep {r['rep_index']}: pull {r['concentric_s']:.1f}s  lock {r['lockout_s']:.1f}s"
                 f"  down {r['eccentric_s']:.1f}s",
                 f"tempo ecc/conc {r['tempo_ratio']:.2f}   peak hip {r['peak_hip_deg']:.0f} deg",
                 f"{f['bias'].replace('_', ' ')}  ({f['trunk_tibia_at_liftoff_deg']:+.0f} deg)   flags: "
                 f"{', '.join(f['flags']) if f['flags'] else 'none'}"]
        bw = int(560 * k)
        _panel(frame, w - bw, 0, w, int(95 * k))
        for i, ln in enumerate(lines):
            _text(frame, ln, (w - bw + int(10 * k), int((26 + 26 * i) * k)), 0.5 * k,
                  AMBER if f["flags"] else GREEN)

    @staticmethod
    def _draw_checklist(frame, res: FrameResult, k: float) -> None:
        """Which tracked joints MediaPipe sees, and what to fix. Shown under the HUD on the left."""
        fr = res.framing
        top = int(340 * k)
        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        if fr is None:
            lines.append(("No person detected", RED))
            lines.append(("Stand in view, side-on, whole body in frame", AMBER))
        else:
            for name in TRACKED_SAGITTAL:
                core = name in CORE
                ok = fr.visible[name]
                tag = "ok " if ok else ("MISSING" if core else "off")
                colour = GREEN if ok else (RED if core else GREY)
                lines.append((f"{tag:7s} {JOINT_LABELS[name]}{'' if core else '  (optional)'}", colour))
            lines += [(p, AMBER) for p in fr.problems]
            lines += [(n, GREY) for n in fr.notes]
        lh = int(22 * k)
        _panel(frame, 0, top - int(8 * k), int(440 * k), top + lh * len(lines) + int(6 * k))
        for i, (txt, colour) in enumerate(lines):
            _text(frame, txt, (int(12 * k), top + lh * (i + 1) - int(6 * k)), 0.41 * k, colour)

    def _draw_cues(self, frame, res: FrameResult, an: DeadliftAnalyzer, k: float) -> None:
        h, w = frame.shape[:2]
        active = sorted(res.active_flags, key=lambda f: (FLAG_SEVERITY[f] != Severity.DANGER, f.value))
        fresh = self._event_msg if (self._event_msg and res.t_s <= self._event_msg[0]) else None
        lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (CUE_TEXT[f], RED if FLAG_SEVERITY[f] == Severity.DANGER else AMBER)
            for f in active if not (fresh and fresh[3] == f)]   # a fresh event's numeric text replaces the static one
        if fresh:
            lines.append((fresh[1], RED if fresh[2] == Severity.DANGER else AMBER))
        if self._fps is not None and self._fps < LOW_FPS:
            lines.append((f"LOW FRAME RATE ({self._fps:.0f} fps): rep counting needs ~{LOW_FPS:.0f}+ fps. "
                          "Close other apps or use a lighter model", RED))
        lines.append(next_step_hint(res))                       # always last: the line nearest the bottom edge
        lh = int(36 * k)
        top = h - lh * len(lines) - int(14 * k)
        _panel(frame, 0, top - int(8 * k), w, h, alpha=0.6)
        for i, (txt, colour) in enumerate(lines):
            _text(frame, txt, (int(16 * k), top + lh * (i + 1) - int(10 * k)), 0.7 * k, colour, 2)
