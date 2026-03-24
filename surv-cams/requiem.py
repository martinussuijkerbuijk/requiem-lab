#!/home/martinus/Projects/requiem/surv-cams/requiem/bin/python3
"""
requiem.py
==========
REQUIEM — Surveillance Art System

State machine
-------------
  VIGIL  →  HUNT  →  CENTER  →  ZOOM  →  ANALYZE  →  BLESS
               ↑___________________________________|  (loop)

CLI commands (type while running)
----------------------------------
  start       begin the automated sequence (HUNT → CENTER → ZOOM → ANALYZE → BLESS loop)
  stop        pause and return to vigil (idle)
  analyze     force analysis on current target
  bless       force holy filter
  home        move camera to preset 1
  reset       zoom out and return to vigil
  status      print current state info
  help        list all commands
  quit / q    exit
"""

import argparse
import math
import os
import random
import sys
import threading
import time
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from camera_controller import CameraController
from run_scan import draw_halo, draw_mask
from run_analysis import draw_targeting_brackets, draw_forensic_hud, get_distance


# ──────────────────────────────────────────────────────────────────────────────
# States
# ──────────────────────────────────────────────────────────────────────────────

class State(Enum):
    VIGIL   = "VIGIL"    # idle — waiting for start command
    HUNT    = "HUNT"     # random pan + face detection boxes only
    CENTER  = "CENTER"   # pan to align crosshair on face; hold; then zoom
    ZOOM    = "ZOOM"     # hardware + digital zoom only — no panning
    ANALYZE = "ANALYZE"  # zoom stopped, full analysis overlay + countdown
    BLESS   = "BLESS"    # holy filter


# ──────────────────────────────────────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────────────────────────────────────

CENTER_TIMEOUT      = 15.0   # give up centering and return to HUNT after this
CENTER_HOLD_S       =  1.5   # seconds face must stay in DEAD_ZONE before zoom starts
CENTER_LOST_S       =  3.0   # seconds without any detection → back to HUNT
CLOSE_ZONE_PX       = 200    # "good enough" radius — triggers zoom if held long enough
CLOSE_HOLD_S        =  4.0   # seconds within CLOSE_ZONE_PX before zoom triggers anyway
CLOSE_HYSTERESIS_S  =  1.5   # must be outside zone this long before close timer resets
TARGET_MAX_JUMP_PX  = 0.35   # max target switch as fraction of frame width (35%)

ZOOM_TIMEOUT      = 20.0   # give up zooming after this many seconds
ZOOM_LOST_S       =  4.0   # seconds without any detection before re-hunt
ANALYZE_DURATION  = 7.0    # seconds to hold analysis overlay after zoom completes
BLESS_DURATION    = 5.0    # seconds of holy filter
ZOOM_OUT_DURATION = 3.0    # seconds to zoom back out before re-hunting

DETECT_EVERY_N = 4          # run YOLO every N frames for CPU budget

# Digital zoom — software crop+resize; works regardless of camera zoom support
ZOOM_STEP        = 0.010   # factor added per frame (~30 fps → reaches 2.5× in ~5 s)
MAX_DIGITAL_ZOOM = 2.5     # transition to ANALYZE when this is reached


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette per state (BGR)
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    State.VIGIL:   (0,  180,  80),
    State.HUNT:    (0,  220, 120),
    State.CENTER:  (0,   50, 255),   # red — aggressive target-lock feel
    State.ZOOM:    (0,  200, 255),
    State.ANALYZE: (255, 255,  0),
    State.BLESS:   (30, 200, 255),
}


# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────

class RequiemApp:

    def __init__(self, cam: CameraController, args: argparse.Namespace):
        self.cam  = cam
        self.args = args

        self.state             = State.VIGIL
        self._state_entered_at = time.time()

        # Models
        self.yolo      : YOLO | None = None
        self.face_mesh               = None
        self._mp_face_mesh           = None

        # Detection / tracking
        self._detected  : list[tuple] = []
        self._target    : tuple | None = None   # (x1, y1, x2, y2)
        self._last_seen_at : float     = 0.0    # time.time() of last successful detection
        self._frame_count              = 0
        self._centering_status         = 'adjusting'
        self._last_psych               = "COMPLIANT"
        self._zoom_started             = False   # prevents re-issuing zoom every frame
        self._zoom_factor              = 1.0     # digital zoom level (1.0 = no zoom)
        self._zoom_cx                  = 0       # digital zoom centre x
        self._zoom_cy                  = 0       # digital zoom centre y
        self._centered_since  : float | None = None  # when face first hit dead zone
        self._close_since     : float | None = None  # when face first entered close zone
        self._close_exited_at : float | None = None  # when face last left close zone

        # Patrol sweep timing
        self._sweep_stop        : float = 0
        self._sweep_pause_until : float = 0
        self._sweeping          : bool  = False

        # Holy mask
        self._mask_img: np.ndarray | None = None

        # CLI
        self._cmd_queue : list[str] = []
        self._cmd_lock               = threading.Lock()

        # Session ledger
        self._blessed_log: list[dict] = []

    # ─── Model loading ───────────────────────────────────────────────────────

    def load_models(self):
        print("[Requiem] Loading YOLO face detection model …")
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt",
        )
        self.yolo = YOLO(model_path)
        print(f"[Requiem] YOLO loaded: {model_path}")

        print("[Requiem] Loading MediaPipe Face Mesh …")
        self._mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[Requiem] MediaPipe ready.")

        mask_path = "input/Test_mask_v1.png"
        if os.path.exists(mask_path):
            self._mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            print("[Requiem] Holy mask loaded.")
        else:
            print(f"[Requiem] Mask not found at {mask_path} — halo-only mode.")

    # ─── State helpers ───────────────────────────────────────────────────────

    def _transition(self, new_state: State):
        print(f"[Requiem] {self.state.value} → {new_state.value}")
        self.state             = new_state
        self._state_entered_at = time.time()

    def _elapsed(self) -> float:
        return time.time() - self._state_entered_at

    def _start_hunt(self):
        """Common helper: reset tracking state and enter HUNT."""
        self.cam.stop_zoom()
        self.cam.stop_movement()
        self._target            = None
        self._detected          = []
        self._sweeping          = False
        self._sweep_pause_until = 0
        self._zoom_started      = False
        self._zoom_factor       = 1.0
        self._zoom_cx           = 0
        self._zoom_cy           = 0
        self._centered_since    = None
        self._close_since       = None
        self._close_exited_at   = None
        self._last_seen_at      = time.time()
        self._transition(State.HUNT)

    # ─── CLI ─────────────────────────────────────────────────────────────────

    def start_cli(self):
        t = threading.Thread(target=self._cli_loop, daemon=True, name="cli")
        t.start()

    def _cli_loop(self):
        print("\n[REQUIEM CLI]  Commands: start | stop | analyze | bless | home | reset | status | help | quit\n")
        while True:
            try:
                cmd = input("> ").strip().lower()
                if cmd:
                    with self._cmd_lock:
                        self._cmd_queue.append(cmd)
            except EOFError:
                break

    def _flush_commands(self):
        with self._cmd_lock:
            cmds, self._cmd_queue = self._cmd_queue[:], []
        for cmd in cmds:
            self._handle(cmd)

    def _handle(self, cmd: str):
        verb = cmd.split()[0] if cmd else ''

        if verb in ('start', 'hunt'):
            self._start_hunt()

        elif verb == 'stop':
            self.cam.stop_movement()
            self.cam.stop_zoom()
            self._target = None
            self._transition(State.VIGIL)

        elif verb == 'analyze':
            if self._target:
                self.cam.stop_zoom()
                self.cam.stop_movement()
                self._transition(State.ANALYZE)
            else:
                print("[CLI] No target — run 'start' first.")

        elif verb == 'bless':
            self._transition(State.BLESS)

        elif verb == 'home':
            self.cam.stop_movement()
            self.cam.go_to_preset(index=1)
            print("[CLI] Moving to home preset.")

        elif verb == 'reset':
            self.cam.stop_movement()
            self.cam.zoom_to_minimum()
            self._target = None
            self._transition(State.VIGIL)
            print("[CLI] Reset.")

        elif verb == 'status':
            print(f"  State   : {self.state.value}")
            print(f"  Faces   : {len(self._detected)}")
            print(f"  Target  : {self._target}")
            print(f"  Blessed : {len(self._blessed_log)}")
            for entry in self._blessed_log[-3:]:
                print(f"    {entry}")

        elif verb in ('quit', 'exit', 'q'):
            print("[Requiem] Shutting down.")
            self.cam.stop()
            cv2.destroyAllWindows()
            sys.exit(0)

        elif verb == 'help':
            print(
                "  start   – begin automated sequence (HUNT → ZOOM → ANALYZE → BLESS)\n"
                "  stop    – pause, return to vigil\n"
                "  analyze – force analysis of current target\n"
                "  bless   – force apply holy filter\n"
                "  home    – move camera to preset 1\n"
                "  reset   – zoom out + return to vigil\n"
                "  status  – show current status\n"
                "  quit    – exit"
            )
        else:
            print(f"[CLI] Unknown: '{cmd}'.  Type 'help'.")

    # ─── Face detection ──────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> list[tuple]:
        results = self.yolo(frame, verbose=False, conf=0.45, stream=True)
        boxes = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append(tuple(coords))
        return boxes

    def _nearest_to_target(self, faces: list[tuple], target: tuple) -> tuple:
        tx = (target[0] + target[2]) / 2
        ty = (target[1] + target[3]) / 2
        return min(faces, key=lambda f: abs((f[0]+f[2])/2 - tx) + abs((f[1]+f[3])/2 - ty))

    def _pick_target(self, faces: list[tuple]) -> tuple:
        """Weighted random selection — larger (closer) faces more likely."""
        sizes = [(f[2]-f[0]) * (f[3]-f[1]) for f in faces]
        total = sum(sizes)
        if total == 0:
            return random.choice(faces)
        r = random.uniform(0, total)
        cumulative = 0
        for face, size in zip(faces, sizes):
            cumulative += size
            if r <= cumulative:
                return face
        return faces[-1]

    # ─── Digital zoom ────────────────────────────────────────────────────────

    @staticmethod
    def _apply_digital_zoom(
        frame: np.ndarray,
        factor: float,
        cx: int, cy: int,
    ) -> np.ndarray:
        """
        Crop the frame around (cx, cy) by 1/factor and scale back to original size.
        factor=1.0 → no change; factor=2.0 → 2× magnification.
        """
        if factor <= 1.0:
            return frame
        h, w   = frame.shape[:2]
        crop_w = max(1, int(w / factor))
        crop_h = max(1, int(h / factor))
        x1     = max(0, min(cx - crop_w // 2, w - crop_w))
        y1     = max(0, min(cy - crop_h // 2, h - crop_h))
        crop   = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    # ─── Renderers ───────────────────────────────────────────────────────────

    def _render_vigil(self, frame: np.ndarray) -> np.ndarray:
        color = PALETTE[State.VIGIL]
        cv2.putText(frame, "REQUIEM  //  STANDBY", (20, 36),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, color, 2)
        cv2.putText(frame, "type 'start' to begin", (20, 62),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, color, 1)
        return frame

    def _render_hunt(self, frame: np.ndarray) -> np.ndarray:
        color  = PALETTE[State.HUNT]
        white  = (255, 255, 255)
        for (x1, y1, x2, y2) in self._detected:
            # Solid rectangle so it's always visible against any background
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Sci-fi corner brackets on top
            draw_targeting_brackets(frame, x1, y1, x2, y2, white, thickness=2, length=20)
        cv2.putText(frame, f"HUNT  //  FACES: {len(self._detected)}", (20, 36),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, color, 2)
        return frame

    def _render_center(self, frame: np.ndarray, status: str) -> np.ndarray:
        """Targeting display during the CENTER state."""
        h, w   = frame.shape[:2]
        color  = PALETTE[State.CENTER]
        cx_f   = w // 2
        cy_f   = h // 2

        if self._target:
            x1, y1, x2, y2 = self._target
            elapsed = self._elapsed()
            pulse   = abs(math.sin(elapsed * 7))
            llen    = int(22 + 12 * pulse)
            thick   = 2 + int(pulse * 2)

            # Pulsing targeting brackets on the face
            draw_targeting_brackets(frame, x1, y1, x2, y2, color,
                                    thickness=thick, length=llen)

            # Line from face centre to frame crosshair
            face_cx = (x1 + x2) // 2
            face_cy = (y1 + y2) // 2
            cv2.line(frame, (face_cx, face_cy), (cx_f, cy_f),
                     color, 1, cv2.LINE_AA)

        # Fixed crosshair at frame centre
        cv2.drawMarker(frame, (cx_f, cy_f), (255, 255, 255),
                       cv2.MARKER_CROSS, 50, 2, cv2.LINE_AA)

        # Hold-bar: fills as the face stays centred
        if self._centered_since is not None:
            held    = time.time() - self._centered_since
            progress = min(held / CENTER_HOLD_S, 1.0)
            bar_w   = int(w * progress)
            cv2.rectangle(frame, (0, h - 10), (bar_w, h), color, -1)
            label = f"LOCKED  {int(progress * 100)}%"
        else:
            label = "CENTERING"

        cv2.putText(frame, f"CENTER  //  {label}", (20, 36),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, color, 2)
        return frame

    # Resolution the forensic HUD was designed for
    _PROC_W = 1280
    _PROC_H = 720

    def _render_analysis_overlay(
        self,
        frame: np.ndarray,
        label: str,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Full forensic overlay from run_analysis.py.
        Used by both ZOOM (show_progress=False) and ANALYZE (show_progress=True).
        Frame is normalised to 1280×720 so the HUD elements always fill the screen.
        """
        # Normalise to the HUD design resolution
        frame = cv2.resize(frame, (self._PROC_W, self._PROC_H))
        h, w  = self._PROC_H, self._PROC_W
        current_time = time.time()
        elapsed      = self._elapsed()

        # ── Blue-tinted monochrome base ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        base = cv2.multiply(base, np.array([1.6, 1.0, 0.6]))
        base = np.clip(base, 0, 255).astype(np.uint8)

        # ── MediaPipe face mesh ──
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = self.face_mesh.process(rgb)

        LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        LIPS      = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                     291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        neon_pink = (255, 50, 255)
        neon_cyan = (255, 255, 0)

        if results_mp.multi_face_landmarks:
            for face_lm in results_mp.multi_face_landmarks:
                lm = [(int(pt.x * w), int(pt.y * h)) for pt in face_lm.landmark]

                pad = 40
                xs  = [p[0] for p in lm]
                ys  = [p[1] for p in lm]
                x1  = max(0, min(xs) - pad)
                y1  = max(0, min(ys) - pad)
                x2  = min(w, max(xs) + pad)
                y2  = min(h, max(ys) + pad)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Psychedelic HSV shift on face region
                roi = frame[y1:y2, x1:x2]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] + int((current_time * 100) % 180)) % 180
                hsv[:, :, 1] = 255
                base[y1:y2, x1:x2] = cv2.addWeighted(
                    base[y1:y2, x1:x2], 0.2,
                    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0.8, 0)

                # Eye and lip mesh lines
                for i in range(len(LEFT_EYE)):
                    cv2.line(base, lm[LEFT_EYE[i]],  lm[LEFT_EYE[(i+1)  % len(LEFT_EYE)]],  neon_cyan, 2)
                for i in range(len(RIGHT_EYE)):
                    cv2.line(base, lm[RIGHT_EYE[i]], lm[RIGHT_EYE[(i+1) % len(RIGHT_EYE)]], neon_cyan, 2)
                for i in range(len(LIPS)):
                    cv2.line(base, lm[LIPS[i]],      lm[LIPS[(i+1)      % len(LIPS)]],      neon_pink, 2)

                # Iris crosshairs
                if len(lm) > 473:
                    cv2.drawMarker(base, lm[473], (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
                    cv2.drawMarker(base, lm[468], (0, 0, 255), cv2.MARKER_CROSS, 20, 1)

                # Pseudo-emotion classification
                eye_h  = get_distance(lm[159], lm[145])
                eye_w  = get_distance(lm[33],  lm[133])
                ear    = eye_h / eye_w if eye_w > 0 else 0
                mouth_h = get_distance(lm[13], lm[14])

                psych, tension, ui_color = "COMPLIANT", "NOMINAL", (0, 255, 0)
                if mouth_h > 20:
                    psych, tension, ui_color = "SHOCKED / VOCALIZING", "ELEVATED", (0, 165, 255)
                elif ear < 0.2:
                    psych, tension, ui_color = "DEFENSIVE / SKEPTICAL", "HIGH",     neon_pink
                elif ear > 0.35:
                    psych, tension, ui_color = "FEAR / ALERTNESS",      "CRITICAL", (0, 0, 255)

                self._last_psych = psych

                draw_targeting_brackets(base, x1, y1, x2, y2, ui_color, thickness=3, length=40)

                font = cv2.FONT_HERSHEY_PLAIN
                hex_id = f"ID:{random.randint(0x1000, 0xFFFF):04X}"
                cv2.putText(base, hex_id,                   (x1,      y1 - 35), font, 1.2, ui_color,  2)
                cv2.putText(base, f"TENSION: {tension}",    (x1,      y1 - 15), font, 1.2, ui_color,  2)
                cv2.putText(base, "PSYCH_EVAL:",             (x2 + 10, y1 + 20), font, 1.2, neon_cyan, 2)
                cv2.putText(base, psych,                     (x2 + 10, y1 + 40), font, 1.2, ui_color,  2)
                cv2.putText(base, f"PUPIL: {ear:.2f}mm",    (x2 + 10, y1 + 60), font, 1.0, neon_cyan, 2)
                cv2.putText(base, f"VOCAL: {mouth_h:.1f}",  (x2 + 10, y1 + 80), font, 1.0, neon_cyan, 2)

        # Drone HUD
        draw_forensic_hud(base, w, h, elapsed)

        # Moving scanline
        cv2.line(base,
                 (0, int((current_time * 180) % h)),
                 (w, int((current_time * 180) % h)),
                 (0, 255, 0), 2)

        # Noise grain
        noise = np.random.randint(0, 25, base.shape, dtype='uint8')
        base  = cv2.add(base, noise)

        # Status label + optional progress bar
        if show_progress:
            progress = min(elapsed / ANALYZE_DURATION, 1.0)
            cv2.rectangle(base, (0, h - 8), (int(w * progress), h), (0, 255, 0), -1)
            cv2.putText(base, f"{label}  //  {int(progress * 100):3d}%", (20, 36),
                        cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 255, 0), 2)
        else:
            cv2.putText(base, label, (20, 36),
                        cv2.FONT_HERSHEY_PLAIN, 1.6, PALETTE[State.ZOOM], 2)

        return base

    def _render_bless(self, frame: np.ndarray) -> np.ndarray:
        """Holy filter — halo + mask from run_scan.py."""
        h, w    = frame.shape[:2]
        elapsed = self._elapsed()

        # Re-detect faces on detection frames so halos stay on
        if self._frame_count % DETECT_EVERY_N == 0:
            self._detected = self._detect(frame)

        # Warm golden grade, fade in over 1.2 s
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        warm = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        warm = cv2.multiply(warm, np.array([0.55, 0.80, 1.15]))
        warm = np.clip(warm, 0, 255).astype(np.uint8)
        fade  = min(elapsed / 1.2, 1.0)
        frame = cv2.addWeighted(frame, 1.0 - fade, warm, fade, 0)

        for (x1, y1, x2, y2) in self._detected:
            draw_halo(frame, x1, y1, x2, y2)
            if self._mask_img is not None:
                draw_mask(frame, x1, y1, x2, y2, self._mask_img)

        # Pulsing "BLESSED" title
        pulse = 0.7 + 0.3 * abs(math.sin(elapsed * 3))
        color = (int(30 * pulse), int(180 * pulse), int(255 * pulse))
        label = "BLESSED"
        ts    = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
        cv2.putText(frame, label, ((w - ts[0]) // 2, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)

        remaining = max(0.0, BLESS_DURATION - elapsed)
        cv2.putText(frame, f"returning in {remaining:.1f}s", (20, h - 16),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 180, 200), 1)

        return frame

    # ─── Main loop ───────────────────────────────────────────────────────────

    def run(self):
        print("[Requiem] Main loop started.")

        while True:
            self._flush_commands()
            self._frame_count += 1

            frame = self.cam.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            h, w       = frame.shape[:2]
            run_detect = (self._frame_count % DETECT_EVERY_N == 0)

            # ── VIGIL ────────────────────────────────────────────────────────
            if self.state == State.VIGIL:
                frame = self._render_vigil(frame)

            # ── HUNT ─────────────────────────────────────────────────────────
            elif self.state == State.HUNT:
                if run_detect:
                    self._detected = self._detect(frame)

                # Manage patrol sweep
                now = time.time()
                if self._sweeping:
                    if now >= self._sweep_stop:
                        self.cam.stop_movement()
                        self._sweeping = False
                elif now >= self._sweep_pause_until:
                    self._sweep_stop, self._sweep_pause_until = self.cam.begin_sweep()
                    self._sweeping = True

                frame = self._render_hunt(frame)

                # Probabilistically pick a target (~2.5% chance per detection frame)
                if self._detected and run_detect and random.random() < 0.065:
                    self.cam.stop_movement()
                    self._sweeping       = False
                    self._target         = self._pick_target(self._detected)
                    self._last_seen_at    = time.time()
                    self._centered_since  = None
                    self._close_since     = None
                    self._close_exited_at = None
                    self._transition(State.CENTER)

            # ── CENTER ───────────────────────────────────────────────────────
            elif self.state == State.CENTER:
                if run_detect:
                    faces = self._detect(frame)
                    if faces:
                        self._last_seen_at = time.time()
                        nearest = self._nearest_to_target(faces, self._target) \
                            if self._target else faces[0]
                        # Sticky target: only update if nearest face is plausibly
                        # the same person (within TARGET_MAX_JUMP_PX of frame width).
                        if self._target:
                            tx = (self._target[0] + self._target[2]) / 2
                            ty = (self._target[1] + self._target[3]) / 2
                            nx = (nearest[0] + nearest[2]) / 2
                            ny = (nearest[1] + nearest[3]) / 2
                            if math.hypot(nx - tx, ny - ty) < w * TARGET_MAX_JUMP_PX:
                                self._target = nearest
                            # else: keep old target — nearby face is a different person
                        else:
                            self._target = nearest

                face_lost = (time.time() - self._last_seen_at) > CENTER_LOST_S
                if face_lost or self._elapsed() > CENTER_TIMEOUT:
                    print("[Requiem] Centering failed — back to HUNT.")
                    self._start_hunt()
                    continue

                if self._target:
                    x1, y1, x2, y2 = self._target
                    status = self.cam.center_on_face(x1, y1, x2, y2, w, h)

                    cx_face = (x1 + x2) / 2
                    cy_face = (y1 + y2) / 2
                    dist = math.hypot(cx_face - w / 2, cy_face - h / 2)
                    now  = time.time()

                    # Tier 1: precise lock within dead zone
                    if status == 'centered':
                        if self._centered_since is None:
                            self._centered_since = now
                    else:
                        self._centered_since = None

                    # Tier 2: "close enough" fallback with hysteresis
                    # Timer only resets after being outside the zone for CLOSE_HYSTERESIS_S,
                    # so brief overshoots from camera bursts don't kill the lock.
                    if dist < CLOSE_ZONE_PX:
                        self._close_exited_at = None
                        if self._close_since is None:
                            self._close_since = now
                            print(f"[Center] close-zone entered  dist={dist:.0f}px")
                    else:
                        if self._close_exited_at is None:
                            self._close_exited_at = now
                        elif now - self._close_exited_at > CLOSE_HYSTERESIS_S:
                            if self._close_since is not None:
                                print(f"[Center] close-zone lost  dist={dist:.0f}px")
                            self._close_since     = None
                            self._close_exited_at = None

                    # Progress print (every ~1 s)
                    if self._close_since is not None and int(now) != int(self._close_since):
                        held = now - self._close_since
                        print(f"[Center] close-lock {held:.1f}/{CLOSE_HOLD_S:.0f}s  "
                              f"dist={dist:.0f}px", end="\r")

                    # Trigger zoom on either tier
                    precise_lock = (self._centered_since is not None and
                                    now - self._centered_since >= CENTER_HOLD_S)
                    close_lock   = (self._close_since is not None and
                                    now - self._close_since >= CLOSE_HOLD_S)

                    if precise_lock or close_lock:
                        reason = "precise lock" if precise_lock else "close-enough lock"
                        print(f"\n[Requiem] {reason} (dist={dist:.0f}px) → ZOOM")
                        self.cam.stop_movement()
                        self._zoom_cx = (x1 + x2) // 2
                        self._zoom_cy = (y1 + y2) // 2
                        self._transition(State.ZOOM)

                    frame = self._render_center(frame, status)

            # ── ZOOM ─────────────────────────────────────────────────────────
            elif self.state == State.ZOOM:
                # Face is already centred (done in CENTER state).
                # Start hardware zoom once and leave it running — no panning here.
                if not self._zoom_started:
                    self.cam.zoom_in_slow()
                    self._zoom_started = True

                # Update face tracking (detection only, no pan corrections)
                if run_detect:
                    faces = self._detect(frame)
                    if faces:
                        self._last_seen_at = time.time()
                        self._target = self._nearest_to_target(faces, self._target) \
                            if self._target else faces[0]
                        # Keep digital zoom centre on the face
                        x1, y1, x2, y2 = self._target
                        self._zoom_cx = (x1 + x2) // 2
                        self._zoom_cy = (y1 + y2) // 2

                if (time.time() - self._last_seen_at) > ZOOM_LOST_S:
                    print("[Requiem] Target lost during zoom — back to HUNT.")
                    self.cam.stop_zoom()
                    self._start_hunt()
                    continue

                if self._elapsed() > ZOOM_TIMEOUT:
                    print("[Requiem] Zoom timeout — back to HUNT.")
                    self.cam.stop_zoom()
                    self._start_hunt()
                    continue

                # Advance digital zoom
                self._zoom_factor = min(self._zoom_factor + ZOOM_STEP, MAX_DIGITAL_ZOOM)
                pct    = int((self._zoom_factor - 1.0) / (MAX_DIGITAL_ZOOM - 1.0) * 100)
                zoomed = self._apply_digital_zoom(
                    frame, self._zoom_factor, self._zoom_cx, self._zoom_cy)
                frame  = self._render_analysis_overlay(zoomed, f"ZOOM  //  {pct}%")

                if self._zoom_factor >= MAX_DIGITAL_ZOOM:
                    self.cam.stop_zoom()
                    self._zoom_started = False
                    self._transition(State.ANALYZE)

                cv2.imshow("REQUIEM", frame)
                cv2.waitKey(1)
                continue

            # ── ANALYZE ──────────────────────────────────────────────────────
            elif self.state == State.ANALYZE:
                zoomed = self._apply_digital_zoom(
                    frame, MAX_DIGITAL_ZOOM, self._zoom_cx, self._zoom_cy)
                frame = self._render_analysis_overlay(
                    zoomed, "ANALYZE", show_progress=True)

                if self._elapsed() >= ANALYZE_DURATION:
                    self._transition(State.BLESS)

            # ── BLESS ────────────────────────────────────────────────────────
            elif self.state == State.BLESS:
                zoomed = self._apply_digital_zoom(
                    frame, MAX_DIGITAL_ZOOM, self._zoom_cx, self._zoom_cy)
                frame = self._render_bless(zoomed)

                if self._elapsed() >= BLESS_DURATION:
                    entry = {
                        "time":  time.strftime("%H:%M:%S"),
                        "psych": self._last_psych,
                        "bbox":  self._target,
                    }
                    self._blessed_log.append(entry)
                    print(f"[Requiem] ✦ BLESSED #{len(self._blessed_log):03d}  "
                          f"{entry['time']}  {entry['psych']}")

                    # Zoom out then loop back to hunting
                    self.cam.zoom_to_minimum()
                    time.sleep(ZOOM_OUT_DURATION)
                    self.cam.stop_zoom()
                    self._start_hunt()

            cv2.imshow("REQUIEM", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.stop()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="REQUIEM — Surveillance Art System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ip",       default="192.168.1.126", help="Camera IP address")
    p.add_argument("--user",     default="admin",         help="Camera username")
    p.add_argument("--password", default="requiem-2026",  help="Camera password")
    p.add_argument("--profile",  default="main",          choices=["main", "sub"],
                   help="Stream profile (main=high quality, sub=low latency)")
    p.add_argument("--start",     action="store_true",
                   help="Begin the automated sequence immediately on launch")
    p.add_argument("--pan-range",   type=float, default=120.0,
                   help="Total horizontal sweep range in degrees (camera stays within ±half this)")
    p.add_argument("--tilt-up",     type=float, default=45.0,
                   help="Max upward tilt in degrees from start position")
    p.add_argument("--tilt-down",   type=float, default=30.0,
                   help="Max downward tilt in degrees from start position")
    p.add_argument("--home-preset", type=int, default=0,
                   help="Move camera to this preset index on startup (0 = skip)")
    p.add_argument("--width",    type=int, default=1280,
                   help="Display window width  (max 1920)")
    p.add_argument("--height",   type=int, default=720,
                   help="Display window height (max 1080)")
    return p.parse_args()


def main():
    args = parse_args()

    cam = CameraController(
        ip=args.ip,
        username=args.user,
        password=args.password,
        profile=args.profile,
        pan_limit=args.pan_range / 2,
        tilt_up_limit=args.tilt_up,
        tilt_down_limit=args.tilt_down,
    )
    cam.connect()
    if args.home_preset > 0:
        print(f"[Requiem] Moving to home preset {args.home_preset} …")
        cam.go_to_preset(index=args.home_preset)
        time.sleep(3.0)   # allow camera to reach position before streaming
    cam.start_stream()

    print("[Requiem] Waiting for first frame …")
    timeout = time.time() + 15
    while cam.get_frame() is None:
        if time.time() > timeout:
            print("[Requiem] ERROR: no frame received after 15 s. Check connection.")
            cam.stop()
            sys.exit(1)
        time.sleep(0.1)
    print("[Requiem] Stream live.")

    disp_w = min(max(args.width,  320), 1920)
    disp_h = min(max(args.height, 240), 1080)
    cv2.namedWindow("REQUIEM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("REQUIEM", disp_w, disp_h)

    app = RequiemApp(cam, args)
    app.load_models()
    app.start_cli()

    if args.start:
        app._start_hunt()

    app.run()


if __name__ == "__main__":
    main()
