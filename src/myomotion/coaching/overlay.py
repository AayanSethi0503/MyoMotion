"""OpenCV overlay for the deadlift demo: skeleton, proxy spine curve, live angles, phase/rep HUD, form cues.

Drawing only: it reads a ``FrameResult`` and the analyzer, and never changes analysis state. Cue arbitration
(priority, cooldown, speech) is the job of the future ``coaching/cue_engine.py`` (G-04); here the banner simply
shows active flags in severity order. OpenCV's Hershey fonts are ASCII only ("deg", never a degree sign).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from myomotion.biomechanics.deadlift_pipeline import DeadliftAnalyzer, FrameResult
from myomotion.biomechanics.deadlift_rules import BIAS_LABEL, FLAG_SEVERITY, FormFlag, Severity
from myomotion.biomechanics.rep_segmenter import Phase
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

CUE_TEXT = {
    FormFlag.LUMBAR_ROUNDING: "CHEST UP - possible back rounding (proxy)",
    FormFlag.HIPS_RISE_EARLY: "HIPS RISING TOO FAST - push the floor away",
    FormFlag.LOCKOUT_HYPEREXTENSION: "DON'T LEAN BACK AT LOCKOUT - finish tall",
    FormFlag.BAR_DRIFT: "KEEP THE BAR OVER MID-FOOT",
}

# MediaPipe body connections (near/far limbs, trunk, feet)
BODY_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25),
                    (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32), (7, 11),
                    (8, 12)]
SPINE_DOT_LABELS = {"C1": "C1", "C7": "C7", "T8": "T8", "L3": "L3", "SACRUM": "S"}
EVENT_DISPLAY_S = 3.0           # how long the numeric message of a fresh event stays on screen


def _text(img, txt: str, org: Tuple[int, int], scale: float, colour, thick: int = 1, shadow: bool = True) -> None:
    if shadow:
        cv2.putText(img, txt, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1,
                    cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _panel(img, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.55) -> None:
    sub = img[y0:y1, x0:x1]
    if sub.size:
        img[y0:y1, x0:x1] = (sub * (1 - alpha)).astype(img.dtype)


class OverlayRenderer:
    """Stateful only for display timing (how long a fresh event message stays visible)."""

    def __init__(self):
        self._event_msg: Optional[Tuple[float, str, Severity, FormFlag]] = None   # (t_expire, text, severity, flag)

    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray, res: FrameResult, an: DeadliftAnalyzer) -> np.ndarray:
        h, w = frame.shape[:2]
        k = max(0.5, h / 720.0)
        for ev in res.new_events:
            self._event_msg = (res.t_s + EVENT_DISPLAY_S, ev.message, ev.severity, ev.flag)

        if res.landmarks is not None and res.angles is not None:
            self._draw_skeleton(frame, res.landmarks, k)
            self._draw_reference_lines(frame, res, k)
            self._draw_spine(frame, res, k)
            self._draw_bar(frame, res, k)
            self._draw_angle_tags(frame, res, k)
        self._draw_hud(frame, res, an, k)
        self._draw_cues(frame, res, an, k)
        return frame

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_skeleton(frame, lm: np.ndarray, k: float) -> None:
        for a, b in BODY_CONNECTIONS:
            if lm[a, 2] >= 0.5 and lm[b, 2] >= 0.5:
                cv2.line(frame, tuple(int(v) for v in lm[a, :2]), tuple(int(v) for v in lm[b, :2]),
                         (230, 230, 230), max(1, int(2 * k)), cv2.LINE_AA)
        for i in range(11, 33):
            if lm[i, 2] >= 0.5:
                cv2.circle(frame, (int(lm[i, 0]), int(lm[i, 1])), max(2, int(4 * k)), (60, 220, 90), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_reference_lines(frame, res: FrameResult, k: float) -> None:
        """Dashed vertical through mid-foot (the bar-path reference) and the floor line."""
        a = res.angles
        heel, toe = a.pose["heel"], a.pose["foot"]
        x = int(0.5 * (heel[0] + toe[0]))
        floor_y = int(max(heel[1], toe[1]))
        top = int(max(0, floor_y - 3.2 * a.torso_len_px))
        for y in range(top, floor_y, int(14 * k)):
            cv2.line(frame, (x, y), (x, min(y + int(7 * k), floor_y)), (255, 160, 60),
                     max(1, int(1.5 * k)), cv2.LINE_AA)
        cv2.line(frame, (int(heel[0] - 40 * k), floor_y), (int(toe[0] + 40 * k), floor_y),
                 (120, 120, 120), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_spine(frame, res: FrameResult, k: float) -> None:
        sp = res.angles.spine
        if sp is None:
            return
        colour = RED if FormFlag.LUMBAR_ROUNDING in res.active_flags else (255, 120, 255)
        pts = sp.curve.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, colour, max(2, int(4 * k)), cv2.LINE_AA)
        for label in SPINE_LABELS:
            x, y = (int(v) for v in sp.points[label])
            cv2.circle(frame, (x, y), max(3, int(5 * k)), colour, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(3, int(5 * k)), (0, 0, 0), 1, cv2.LINE_AA)
            if label in SPINE_DOT_LABELS:
                _text(frame, SPINE_DOT_LABELS[label], (x + int(8 * k), y + int(4 * k)), 0.4 * k, colour)

    @staticmethod
    def _draw_bar(frame, res: FrameResult, k: float) -> None:
        a = res.angles
        if a.bar_xy is None:
            return
        c = (int(a.bar_xy[0]), int(a.bar_xy[1]))
        colour = AMBER if FormFlag.BAR_DRIFT in res.active_flags else CYAN
        cv2.circle(frame, c, max(5, int(9 * k)), colour, 2, cv2.LINE_AA)
        label = "bar" if a.bar_source == "bar" else "bar (wrist)"
        _text(frame, label, (c[0] + int(12 * k), c[1] + int(4 * k)), 0.42 * k, colour)

    @staticmethod
    def _draw_angle_tags(frame, res: FrameResult, k: float) -> None:
        a = res.angles
        hip, knee = a.pose["hip"], a.pose["knee"]
        off = int(-70 * k * a.facing)
        _text(frame, f"{a.hip_deg:.0f}", (int(hip[0] + off), int(hip[1] - 10 * k)), 0.6 * k, (0, 255, 0), 2)
        _text(frame, f"{a.knee_deg:.0f}", (int(knee[0] + off), int(knee[1] - 10 * k)), 0.6 * k, (0, 255, 0), 2)

    # ------------------------------------------------------------------
    def _draw_hud(self, frame, res: FrameResult, an: DeadliftAnalyzer, k: float) -> None:
        h, w = frame.shape[:2]
        pw = int(400 * k)
        _panel(frame, 0, 0, pw, int(330 * k))
        x, y, dy = int(14 * k), int(40 * k), int(26 * k)
        _text(frame, res.phase.value.upper(), (x, y), 0.95 * k, PHASE_COLOURS[res.phase], 2)
        y += dy
        _text(frame, f"Reps: {res.rep_count}", (x, y), 0.8 * k, WHITE, 2)
        a = res.angles
        if a is None:
            _text(frame, "NO POSE - step into frame (side view)" if not res.tracking_ok else "tracking...",
                  (x, y + dy), 0.55 * k, AMBER)
            self._draw_last_rep(frame, an, k)
            return
        rows = [
            f"Hip {a.hip_deg:5.1f}   Knee {a.knee_deg:5.1f} deg",
            f"Trunk {a.trunk_incl_deg:+5.1f}   Tibia {a.tibia_incl_deg:+5.1f} deg",
            f"Trunk-Tibia {a.trunk_tibia_deg:+5.1f}  {BIAS_LABEL[res.bias].split(' (')[0] if res.bias else ''}",
        ]
        sd = res.spine_delta
        if a.spine is not None:
            rows.append("Spine proxy (rel. to set-up):")
            if sd is not None:
                rows.append(f"  lumbar {sd.lumbar_deg:+4.1f}  thoracic {sd.thoracic_deg:+4.1f}"
                            f"  cerv {sd.cervical_deg:+4.1f}")
            else:
                rows.append("  baseline: waiting for set-up")
        if a.bar_drift_foot is not None:
            rows.append(f"Bar drift {a.bar_drift_foot:+.2f} ft-len   height {a.bar_height_torso:.2f} torso")
        for i, r in enumerate(rows):
            dim = r.startswith("  ") or r.startswith("Spine")
            _text(frame, r, (x, y + dy * (i + 1)), 0.5 * k, GREY if dim else WHITE)
        self._draw_last_rep(frame, an, k)

    @staticmethod
    def _draw_last_rep(frame, an: DeadliftAnalyzer, k: float) -> None:
        if not an.reps:
            return
        h, w = frame.shape[:2]
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

    def _draw_cues(self, frame, res: FrameResult, an: DeadliftAnalyzer, k: float) -> None:
        h, w = frame.shape[:2]
        active = sorted(res.active_flags, key=lambda f: (FLAG_SEVERITY[f] != Severity.DANGER, f.value))
        fresh = self._event_msg if (self._event_msg and res.t_s <= self._event_msg[0]) else None
        lines: List[Tuple[str, Tuple[int, int, int]]] = [
            (CUE_TEXT[f], RED if FLAG_SEVERITY[f] == Severity.DANGER else AMBER)
            for f in active if not (fresh and fresh[3] == f)]   # a fresh event's numeric text replaces the static one
        if fresh:
            lines.append((fresh[1], RED if fresh[2] == Severity.DANGER else AMBER))
        if not lines:
            if res.phase in (Phase.CONCENTRIC_PULL, Phase.LOCKOUT, Phase.ECCENTRIC_DESCENT) and res.angles is not None:
                lines = [("Form OK", GREEN)]
            else:
                return
        lh = int(36 * k)
        top = h - lh * len(lines) - int(14 * k)
        _panel(frame, 0, top - int(8 * k), w, h, alpha=0.6)
        for i, (txt, colour) in enumerate(lines):
            _text(frame, txt, (int(16 * k), top + lh * (i + 1) - int(10 * k)), 0.8 * k, colour, 2)
