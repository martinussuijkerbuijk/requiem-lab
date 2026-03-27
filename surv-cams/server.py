#!/home/martinus/Projects/requiem/surv-cams/requiem/bin/python3
"""
REQUIEM // server.py
====================
Web backend for the REQUIEM surveillance art system.
Inference + state machine run in a background thread.
The browser handles all rendering.

Endpoints
---------
  GET  /            browser UI (templates/index.html)
  GET  /video_feed  MJPEG stream  — <img> handles it natively
  GET  /ws          WebSocket    — pushes detection JSON at ~30 fps
  POST /ptz         PTZ stub     — { "cmd": "left"|"right"|..., "speed": n }
  POST /command     State machine — { "cmd": "start"|"stop"|"bless"|... }

Config
------
  --ip         Reolink camera IP  (leave empty for webcam)
  --source     Video source if no IP: 0 = webcam, or rtsp://...
  --host/port  Server bind address (default 0.0.0.0:5000)
  --start      Begin HUNT immediately on launch
"""

from __future__ import annotations

import argparse
import collections
import json
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
from flask import Flask, Response, render_template, request, jsonify
from flask_sock import Sock
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

try:
    from camera_controller import CameraController
    _HAS_CAM = True
except Exception as _cam_import_err:
    _HAS_CAM = False
    print(f"[Requiem] WARNING: camera_controller import failed: {_cam_import_err}")


# ── Timing (mirrors requiem.py) ───────────────────────────────────────────────

CENTER_TIMEOUT     = 15.0
CENTER_LOST_S      = 5.0   # raised from 3.0 — camera needs ~1.5s to settle after sweep
DEAD_ZONE_PX       = 70    # must match camera_controller.DEAD_ZONE_PX
CENTER_HOLD_N      = 3     # consecutive centered reads required before committing to ZOOM
ZOOM_TIMEOUT       = 20.0
ZOOM_LOST_S        = 4.0
ANALYZE_DURATION   = 15.0   # raised — audience needs time to read the analysis
BLESS_DURATION     = 10.0   # raised — give the blessing ritual space to breathe
ZOOM_OUT_DURATION  = 3.0
DETECT_EVERY_N     = 4
ZOOM_STEP          = 0.010
MAX_DIGITAL_ZOOM   = 2.5

# Processing resolution — frames are downscaled to this immediately after
# acquisition.  YOLO, MediaPipe, digital zoom, and MJPEG encoding all run
# on this size.  Keeps CPU/GPU load proportional to a 1280-wide frame
# regardless of whether the camera outputs 1080p or 4K.
PROC_W = 1280
PROC_H = 720

# HUNT systematic sweep (2 full 60° sweeps before face locking)
# At HUNT_SWEEP_SPEED PTZ units and DEG_PER_SPEED_PER_SEC=1.0 → speed=deg/s
HUNT_SWEEP_SPEED  = 8     # PTZ speed passed to move_left/move_right
HUNT_SWEEP_HALF_T = 3.5    # seconds per 30° half-arc (30 / 20 deg·s⁻¹)
HUNT_DETECT_T     = 5.0    # seconds to wait for a face after sweep before re-sweeping

# Legs: (direction, duration_s)  —  two full sweeps + return to centre
_HUNT_LEGS = [
    ('right', HUNT_SWEEP_HALF_T),        # 0: home → right edge
    ('left',  HUNT_SWEEP_HALF_T * 2),    # 1: right → left  (sweep 1 done)
    ('right', HUNT_SWEEP_HALF_T * 2),    # 2: left  → right (sweep 2 done)
    ('left',  HUNT_SWEEP_HALF_T),        # 3: right → home  (return)
]

# MediaPipe landmark indices
_L_EYE = [362, 385, 387, 263, 373, 380]
_R_EYE = [33,  160, 158, 133, 153, 144]
_LIPS  = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
          291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


# ── Shared state (detection thread → Flask handlers) ─────────────────────────

_lock  = threading.Lock()
_frame: bytes | None = None
_data: dict = {
    "state": "VIGIL", "faces": [], "mp_faces": [], "target": None,
    "zoom": 1.0, "frame_w": 1280, "frame_h": 720, "t": 0.0, "elapsed": 0.0,
    "analyze_p": 0.0, "bless_p": 0.0, "center_p": 0.0,
    "blessed_n": 0, "last_psych": "COMPLIANT",
}
_cmd_deque: collections.deque = collections.deque()


# ── State machine ─────────────────────────────────────────────────────────────

class State(Enum):
    VIGIL   = "VIGIL"
    HUNT    = "HUNT"
    CENTER  = "CENTER"
    ZOOM    = "ZOOM"
    ANALYZE = "ANALYZE"
    BLESS   = "BLESS"


def _dist(p1, p2) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _digital_zoom(frame: np.ndarray, factor: float, cx: int, cy: int) -> np.ndarray:
    if factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    cw   = max(1, int(w / factor))
    ch   = max(1, int(h / factor))
    x1   = max(0, min(cx - cw // 2, w - cw))
    y1   = max(0, min(cy - ch // 2, h - ch))
    return cv2.resize(frame[y1:y1 + ch, x1:x1 + cw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


class RequiemEngine:
    """
    Inference + state machine. Zero rendering — outputs JPEG bytes + JSON.
    Runs entirely in a background thread.
    """

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._cam  = None
        self._cap  = None

        self.yolo      = None
        self.face_mesh = None

        self.state             = State.VIGIL
        self._state_entered_at = time.time()
        self._frame_count      = 0

        self._detected   : list[tuple] = []
        self._target     : tuple | None = None
        self._last_seen  : float = 0.0
        self._last_psych : str   = "COMPLIANT"

        self._zoom_factor  : float = 1.0
        self._zoom_cx      : int   = 0
        self._zoom_cy      : int   = 0
        self._zoom_started : bool  = False

        # Stable target tracking for CENTER state.
        # _smooth_cx/cy: EMA-smoothed centroid, only updated when camera is NOT moving.
        #   Reduces YOLO jitter so DEAD_ZONE check is stable.
        # _centered_count: consecutive frames where center_on_face returned 'centered'.
        #   Requires CENTER_HOLD_N before committing to ZOOM.
        # _target_centroid: search anchor for the lock-radius filter.
        #   Only updated when camera is settled, preventing neighbor drift during bursts.
        # _LOCK_RADIUS_F: max fraction of frame width a detection may stray from anchor.
        self._smooth_cx       : float | None = None
        self._smooth_cy       : float | None = None
        self._centered_count  : int   = 0
        self._target_centroid : tuple[int, int] | None = None
        self._LOCK_RADIUS_F   : float = 0.20   # ~192px at 960w; tracks through a burst

        # HUNT systematic sweep state
        # _hunt_leg: -1=not started, 0-3=sweep leg index, 4=detect phase (fallback)
        self._hunt_leg        : int        = -1
        self._hunt_leg_end    : float      = 0.0
        self._hunt_detect_end : float      = 0.0
        self._hunt_seen_faces : list[tuple] = []  # faces collected during sweep

        self._blessed_log : list[dict] = []

    # ── Startup ───────────────────────────────────────────────────────────────

    def load_models(self):
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"[Requiem] CUDA available: {cuda_ok}"
              + (f"  device: {torch.cuda.get_device_name(0)}" if cuda_ok else
                 "  ← WARNING: running on CPU, expect ~600 ms/frame"))

        print("[Requiem] Loading YOLO …")
        path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        self.yolo = YOLO(path)
        if cuda_ok:
            self.yolo.to('cuda')
            # Warmup at the actual processing resolution so CUDA kernels for
            # that input size are compiled before the first real frame.
            dummy = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)
            self.yolo(dummy, verbose=False)
            print("[Requiem] YOLO warmed up on CUDA.")
        print("[Requiem] Loading MediaPipe …")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=10, refine_landmarks=True,
            min_detection_confidence=0.3, min_tracking_confidence=0.3)
        # CLAHE for grayscale/IR camera frames — MediaPipe was trained on
        # colour images; enhancing local contrast helps the face detector fire
        # on monochrome surveillance footage.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print("[Requiem] Models ready.")

    def connect_camera(self):
        args = self._args
        if args.ip and not _HAS_CAM:
            raise RuntimeError(
                f"--ip was given but camera_controller could not be imported. "
                f"Check the warning above for the root cause.")
        if args.ip and _HAS_CAM:
            self._cam = CameraController(
                ip=args.ip, username=args.user, password=args.password,
                profile=args.profile,
                pan_limit=args.pan_range / 2,
                tilt_up_limit=args.tilt_up,
                tilt_down_limit=args.tilt_down,
            )
            self._cam.connect()
            self._cam.start_stream()
            deadline = time.time() + 15
            while self._cam.get_frame() is None:
                if time.time() > deadline:
                    print("[Requiem] ERROR: no frame received after 15 s.")
                    sys.exit(1)
                time.sleep(0.1)
            # Move to home preset AFTER stream is live — some firmware ignores
            # PTZ commands issued before the first frame has been delivered.
            if args.home_preset >= 0:
                print(f"[Requiem] Moving to preset {args.home_preset} …")
                self._cam.go_to_preset(index=args.home_preset)
                time.sleep(3.0)
        else:
            src = getattr(args, 'source', 0)
            self._cap = cv2.VideoCapture(src)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[Requiem] Camera live.")
        if self._cam:
            self._cam.test_ptz()
            # Always kill both lights at startup — sent as one atomic request so
            # the firmware cannot auto-switch to white LED when IR is disabled.
            self._cam.all_lights_off()

    # ── Frame source ──────────────────────────────────────────────────────────

    def _get_frame(self) -> np.ndarray | None:
        if self._cam:
            return self._cam.get_frame()
        ret, frame = self._cap.read()
        return frame if ret else None

    # ── Detection ─────────────────────────────────────────────────────────────

    _detect_count = 0
    _detect_total_ms = 0.0

    # Minimum face dimension in pixels — rejects tiny false positives on
    # headrests/backgrounds that are too small to be a real face at this range.
    _MIN_FACE_PX = 45

    def _detect(self, frame: np.ndarray, conf: float = 0.45) -> list[tuple]:
        t0 = time.time()
        boxes = []
        for r in self.yolo(frame, verbose=False, conf=conf, stream=True, half=True):
            for box in r.boxes:
                b = tuple(box.xyxy[0].cpu().numpy().astype(int))
                if (b[2] - b[0]) >= self._MIN_FACE_PX and (b[3] - b[1]) >= self._MIN_FACE_PX:
                    boxes.append(b)
        ms = (time.time() - t0) * 1000
        RequiemEngine._detect_count += 1
        RequiemEngine._detect_total_ms += ms
        if RequiemEngine._detect_count % 30 == 0:
            avg = RequiemEngine._detect_total_ms / RequiemEngine._detect_count
            print(f"[YOLO] last={ms:.0f}ms  avg={avg:.0f}ms  calls={RequiemEngine._detect_count}")
        return boxes

    # Maximum pixel width fed to MediaPipe — checked against the actual input
    # frame dimensions (not the output coordinate space w, h).
    _MP_MAX_W = 960

    def _run_mediapipe(self, frame: np.ndarray, w: int, h: int) -> list[dict]:
        # w, h are the OUTPUT coordinate space (zoomed frame dimensions).
        # Resize based on the ACTUAL input pixel width, not w.
        fw = frame.shape[1]
        if fw > self._MP_MAX_W:
            scale = self._MP_MAX_W / fw
            fh    = frame.shape[0]
            small = cv2.resize(frame, (self._MP_MAX_W, int(fh * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame

        # IR/B&W cameras produce grayscale frames (R≈G≈B).  MediaPipe was
        # trained on colour images — BlazeFace fires poorly on flat monochrome.
        # Detect IR mode via --ir flag OR automatically by checking colour
        # variance.  Fix: apply CLAHE to boost local contrast so the face
        # detector can find edges and landmarks.
        force_ir = getattr(self._args, 'ir', False)
        if force_ir:
            is_ir = True
        else:
            b_ch, _, r_ch = cv2.split(small)
            is_ir = float(np.std(r_ch.astype(np.int16) - b_ch.astype(np.int16))) < 8.0
        if is_ir:
            gray     = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            enhanced = self._clahe.apply(gray)
            small    = cv2.merge([enhanced, enhanced, enhanced])

        rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        out = []
        if not results.multi_face_landmarks:
            return out
        for face_lm in results.multi_face_landmarks:
            lm = [(int(p.x * w), int(p.y * h)) for p in face_lm.landmark]

            ear     = _dist(lm[159], lm[145]) / max(_dist(lm[33], lm[133]), 1e-6)
            mouth_h = _dist(lm[13], lm[14])

            if mouth_h > 20:
                psych, tension = "SHOCKED / VOCALIZING", "ELEVATED"
            elif ear < 0.2:
                psych, tension = "DEFENSIVE / SKEPTICAL", "HIGH"
            elif ear > 0.35:
                psych, tension = "FEAR / ALERTNESS",     "CRITICAL"
            else:
                psych, tension = "COMPLIANT",            "NOMINAL"

            out.append({
                "id":            f"{random.randint(0x1000, 0xFFFF):04X}",
                "lm_left_eye":   [lm[i] for i in _L_EYE],
                "lm_right_eye":  [lm[i] for i in _R_EYE],
                "lm_lips":       [lm[i] for i in _LIPS],
                "lm_left_iris":  list(lm[473]) if len(lm) > 473 else None,
                "lm_right_iris": list(lm[468]) if len(lm) > 468 else None,
                "psych":   psych,
                "tension": tension,
                "ear":     round(ear, 3),
                "mouth_h": round(mouth_h, 1),
            })
        if out:
            self._last_psych = out[0]["psych"]
        return out

    # ── State helpers ─────────────────────────────────────────────────────────

    def _elapsed(self) -> float:
        return time.time() - self._state_entered_at

    def _transition(self, s: State):
        print(f"[Requiem] {self.state.value} → {s.value}")
        self.state = s
        self._state_entered_at = time.time()
        if self._cam and getattr(self._args, 'spotlight', False):
            if s in (State.ANALYZE, State.BLESS):
                print(f"[Lights] {s.value} → spotlight ON  (IR off, white 100%)")
                self._cam.set_ir_lights("Off")
                self._cam.set_white_led(True, bright=100)
            elif s in (State.HUNT, State.VIGIL):
                print(f"[Lights] {s.value} → all off")
                self._cam.all_lights_off()

    def _start_hunt(self):
        if self._cam:
            self._cam.stop_zoom()
            self._cam.stop_movement()
        self._target = None
        self._detected = []
        self._hunt_leg        = -1
        self._hunt_leg_end    = 0.0
        self._hunt_detect_end = 0.0
        self._hunt_seen_faces = []
        self._zoom_started = False
        self._zoom_factor  = 1.0
        self._zoom_cx = self._zoom_cy = 0
        self._smooth_cx = self._smooth_cy = None
        self._centered_count  = 0
        self._target_centroid = None
        self._last_seen = time.time()
        self._transition(State.HUNT)

    # ── Commands ──────────────────────────────────────────────────────────────

    def _flush_commands(self):
        while _cmd_deque:
            self._handle(_cmd_deque.popleft())

    def _handle(self, cmd: str):
        v = cmd.split()[0] if cmd else ''
        if v in ('start', 'hunt'):
            self._start_hunt()
        elif v == 'stop':
            if self._cam:
                self._cam.stop_movement()
                self._cam.stop_zoom()
            self._target = None
            self._transition(State.VIGIL)
        elif v == 'bless':
            self._transition(State.BLESS)
        elif v == 'analyze' and self._target:
            if self._cam:
                self._cam.stop_zoom()
                self._cam.stop_movement()
            self._transition(State.ANALYZE)
        elif v == 'home' and self._cam:
            self._cam.stop_movement()
            self._cam.go_to_preset(index=1)
        elif v == 'reset':
            if self._cam:
                self._cam.stop_movement()
                self._cam.zoom_to_minimum()
            self._target = None
            self._transition(State.VIGIL)

    # ── Target helpers ────────────────────────────────────────────────────────

    def _nearest(self, faces: list[tuple], target: tuple) -> tuple:
        tx, ty = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
        return min(faces,
                   key=lambda f: abs((f[0]+f[2])/2 - tx) + abs((f[1]+f[3])/2 - ty))

    def _lock_target(self, face: tuple, now: float):
        """Set the tracking target from a chosen face bbox and reset CENTER state."""
        self._target = face
        self._last_seen = now
        cx0, cy0 = (face[0] + face[2]) // 2, (face[1] + face[3]) // 2
        self._target_centroid = (cx0, cy0)
        self._smooth_cx       = float(cx0)
        self._smooth_cy       = float(cy0)
        self._centered_count  = 0

    def _pick(self, faces: list[tuple]) -> tuple:
        sizes = [(f[2]-f[0]) * (f[3]-f[1]) for f in faces]
        total = sum(sizes) or 1
        r, cum = random.uniform(0, total), 0.0
        for face, s in zip(faces, sizes):
            cum += s
            if r <= cum:
                return face
        return faces[-1]

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        global _frame, _data

        while True:
            self._flush_commands()
            self._frame_count += 1
            now        = time.time()
            elapsed    = self._elapsed()
            run_detect = (self._frame_count % DETECT_EVERY_N == 0)

            frame = self._get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            # Downscale to processing resolution immediately — YOLO, MediaPipe,
            # digital zoom, and MJPEG encoding all work on this smaller frame.
            fh, fw = frame.shape[:2]
            if fw != PROC_W or fh != PROC_H:
                frame = cv2.resize(frame, (PROC_W, PROC_H),
                                   interpolation=cv2.INTER_LINEAR)

            h, w = frame.shape[:2]   # now always PROC_H × PROC_W

            # Keep the unzoomed frame for MediaPipe — digital zoom enlarges the
            # face too much and BlazeFace misses oversized faces.  We'll extract
            # just the crop region that the zoom would show and run mesh on that.
            pre_zoom_frame = frame

            # ── VIGIL ─────────────────────────────────────────────────────────
            if self.state == State.VIGIL:
                # Keep detection running so boxes appear immediately when
                # the audience is watching before a hunt cycle starts.
                if run_detect:
                    self._detected = self._detect(frame)

            # ── HUNT ──────────────────────────────────────────────────────────
            elif self.state == State.HUNT:
                if run_detect:
                    # Lower confidence during sweep — motion blur reduces YOLO certainty
                    self._detected = self._detect(frame, conf=0.30)

                if self._cam:
                    # ── Systematic 2-sweep patrol ─────────────────────────────
                    # During sweep legs: detect runs but we only collect faces,
                    # never lock.  After sweep, pick a random face immediately.
                    # Detect phase is a fallback for when sweep found nothing.

                    if self._hunt_leg == -1:
                        # Kick off the first sweep leg
                        self._hunt_seen_faces = []
                        direction, duration = _HUNT_LEGS[0]
                        self._cam.start_pan(direction, HUNT_SWEEP_SPEED)
                        self._hunt_leg     = 0
                        self._hunt_leg_end = now + duration
                        print(f"[Hunt] leg 0: pan {direction} for {duration:.1f}s")

                    elif self._hunt_leg < len(_HUNT_LEGS):
                        # Accumulate faces seen during this sweep
                        if run_detect and self._detected:
                            self._hunt_seen_faces.extend(self._detected)

                        if now >= self._hunt_leg_end:
                            self._cam.stop_movement_async()   # non-blocking
                            next_leg = self._hunt_leg + 1
                            if next_leg < len(_HUNT_LEGS):
                                direction, duration = _HUNT_LEGS[next_leg]
                                self._cam.start_pan(direction, HUNT_SWEEP_SPEED)
                                self._hunt_leg     = next_leg
                                self._hunt_leg_end = now + duration
                                print(f"[Hunt] leg {next_leg}: pan {direction} for {duration:.1f}s")
                            else:
                                # All 4 legs done
                                if self._hunt_seen_faces:
                                    # Faces were spotted during sweep.
                                    # Don't use their coords — the camera has moved since
                                    # they were captured.  Enter CENTER with no target;
                                    # the first live detection will pick a fresh face.
                                    n = len(self._hunt_seen_faces)
                                    print(f"[Hunt] sweep done — {n} detections during sweep, entering CENTER for fresh lock")
                                    self._target          = None
                                    self._target_centroid = None
                                    self._smooth_cx       = None
                                    self._smooth_cy       = None
                                    self._centered_count  = 0
                                    self._last_seen       = now   # prevent immediate timeout
                                    self._transition(State.CENTER)
                                else:
                                    # Nothing seen — enter short detect phase
                                    self._hunt_leg        = len(_HUNT_LEGS)   # 4
                                    self._hunt_detect_end = now + HUNT_DETECT_T
                                    print("[Hunt] sweep done — no faces seen, entering detect phase")

                    else:
                        # Detect-phase fallback: camera stopped, lock on first face
                        if self._detected and run_detect:
                            print("[Hunt] face found in detect phase — locking")
                            self._lock_target(self._pick(self._detected), now)
                            self._transition(State.CENTER)
                        elif now >= self._hunt_detect_end:
                            # Still nothing — restart sweep
                            self._hunt_leg = -1
                            print("[Hunt] detect phase timed out — restarting sweep")
                else:
                    # No PTZ camera — lock on any face with low probability
                    if self._detected and run_detect and random.random() < 0.065:
                        self._lock_target(self._pick(self._detected), now)
                        self._transition(State.CENTER)

            # ── CENTER ────────────────────────────────────────────────────────
            elif self.state == State.CENTER:
                if run_detect:
                    faces = self._detect(frame)
                    self._detected = faces   # keep browser display current

                    if self._target is None and not faces:
                        # Haven't locked anything yet — don't start the lost-timer
                        # until we've had at least one successful detection.
                        self._last_seen = now

                    if faces:
                        if self._target is None:
                            # No target yet (e.g. first frame after sweep) — pick
                            # a face from the current live frame so coordinates are fresh.
                            self._lock_target(self._pick(faces), now)
                            print(f"[Center] locked fresh target  faces_in_frame={len(faces)}")
                        else:
                            anchor_cx = float(self._target_centroid[0]) if self._target_centroid \
                                        else (self._target[0] + self._target[2]) / 2.0
                            anchor_cy = float(self._target_centroid[1]) if self._target_centroid \
                                        else (self._target[1] + self._target[3]) / 2.0

                            # ① Loose presence: any face within 35 % of frame width
                            #    counts as "still here" — keeps _last_seen alive even
                            #    when the camera has panned and the face shifted in frame.
                            presence_r = w * 0.35
                            if any(math.hypot((f[0]+f[2])/2 - anchor_cx,
                                              (f[1]+f[3])/2 - anchor_cy) < presence_r
                                   for f in faces):
                                self._last_seen = now

                            # ② Tight centroid refinement — only when camera is settled.
                            #    During a burst all faces shift; updating the anchor then
                            #    would let a neighbour drift into the lock radius.
                            cam_settled = (self._cam is None or not self._cam.is_moving())
                            if cam_settled:
                                lock_r = w * 0.18
                                tight = [f for f in faces
                                         if math.hypot((f[0]+f[2])/2 - anchor_cx,
                                                       (f[1]+f[3])/2 - anchor_cy) < lock_r]
                                if tight:
                                    best = min(tight,
                                               key=lambda f: math.hypot((f[0]+f[2])/2 - anchor_cx,
                                                                         (f[1]+f[3])/2 - anchor_cy))
                                    bw = best[2] - best[0]
                                    bh = best[3] - best[1]
                                    new_cx = (best[0] + best[2]) / 2.0
                                    new_cy = (best[1] + best[3]) / 2.0

                                    a = 0.40
                                    self._smooth_cx = self._smooth_cx*(1-a) + new_cx*a \
                                                      if self._smooth_cx is not None else new_cx
                                    self._smooth_cy = self._smooth_cy*(1-a) + new_cy*a \
                                                      if self._smooth_cy is not None else new_cy

                                    self._target_centroid = (int(self._smooth_cx),
                                                             int(self._smooth_cy))
                                    self._target = (
                                        int(self._smooth_cx - bw/2),
                                        int(self._smooth_cy - bh/2),
                                        int(self._smooth_cx + bw/2),
                                        int(self._smooth_cy + bh/2),
                                    )

                if (now - self._last_seen) > CENTER_LOST_S or elapsed > CENTER_TIMEOUT:
                    reason = "timeout" if elapsed > CENTER_TIMEOUT else "lost"
                    print(f"[Center] → HUNT ({reason})  "
                          f"last_seen_age={now-self._last_seen:.1f}s  "
                          f"elapsed={elapsed:.1f}s  "
                          f"target={'set' if self._target else 'None'}")
                    self._start_hunt()
                    continue

                if self._target:
                    x1, y1, x2, y2 = self._target

                    if self._cam:
                        status   = self._cam.center_on_face(x1, y1, x2, y2, w, h)
                        centered = (status == 'centered')
                    else:
                        cx_f = (x1 + x2) / 2
                        cy_f = (y1 + y2) / 2
                        centered = math.hypot(cx_f - w/2, cy_f - h/2) < DEAD_ZONE_PX

                    # Require CENTER_HOLD_N consecutive centered readings before ZOOM.
                    # A single detection can be a jitter fluke; N in a row is not.
                    if centered:
                        self._centered_count += 1
                        if self._centered_count >= CENTER_HOLD_N:
                            if self._cam:
                                self._cam.stop_movement()
                            self._zoom_cx = (x1 + x2) // 2
                            self._zoom_cy = (y1 + y2) // 2
                            self._transition(State.ZOOM)
                    else:
                        self._centered_count = 0

            # ── ZOOM ──────────────────────────────────────────────────────────
            elif self.state == State.ZOOM:
                if not self._zoom_started:
                    if self._cam:
                        self._cam.zoom_in_slow()
                    self._zoom_started = True

                # Keep the zoom center locked on the chosen face.
                # Do NOT re-run detection here — in a crowded scene the detector
                # will jump to any nearby face.  Only check for presence to
                # maintain the lost-target timeout, using a tight proximity window.
                if run_detect:
                    faces = self._detect(frame)
                    if faces and self._zoom_cx and self._zoom_cy:
                        lock_r = w * self._LOCK_RADIUS_F
                        nearby = [f for f in faces
                                  if math.hypot((f[0]+f[2])/2 - self._zoom_cx,
                                                (f[1]+f[3])/2 - self._zoom_cy) < lock_r]
                        if nearby:
                            self._last_seen = now
                            # Gently nudge zoom center toward the face (≤10% per frame)
                            best = min(nearby,
                                       key=lambda f: math.hypot((f[0]+f[2])/2 - self._zoom_cx,
                                                                 (f[1]+f[3])/2 - self._zoom_cy))
                            self._zoom_cx = int(self._zoom_cx * 0.9 + (best[0]+best[2])/2 * 0.1)
                            self._zoom_cy = int(self._zoom_cy * 0.9 + (best[1]+best[3])/2 * 0.1)
                            # Keep _target in sync with zoom anchor so that _ibox
                            # transforms it to the correct position in ANALYZE/BLESS.
                            bw = best[2] - best[0]
                            bh = best[3] - best[1]
                            self._target = (
                                int(self._zoom_cx - bw / 2),
                                int(self._zoom_cy - bh / 2),
                                int(self._zoom_cx + bw / 2),
                                int(self._zoom_cy + bh / 2),
                            )

                if (now - self._last_seen) > ZOOM_LOST_S:
                    if self._cam:
                        self._cam.stop_zoom()
                    self._start_hunt()
                    continue
                if elapsed > ZOOM_TIMEOUT:
                    if self._cam:
                        self._cam.stop_zoom()
                    self._start_hunt()
                    continue

                self._zoom_factor = min(self._zoom_factor + ZOOM_STEP, MAX_DIGITAL_ZOOM)
                frame = _digital_zoom(frame, self._zoom_factor, self._zoom_cx, self._zoom_cy)

                if self._zoom_factor >= MAX_DIGITAL_ZOOM:
                    if self._cam:
                        self._cam.stop_zoom()
                    self._zoom_started = False
                    self._transition(State.ANALYZE)

            # ── ANALYZE ───────────────────────────────────────────────────────
            elif self.state == State.ANALYZE:
                frame = _digital_zoom(frame, MAX_DIGITAL_ZOOM, self._zoom_cx, self._zoom_cy)
                if elapsed >= ANALYZE_DURATION:
                    self._transition(State.BLESS)

            # ── BLESS ─────────────────────────────────────────────────────────
            elif self.state == State.BLESS:
                if run_detect:
                    # Detect on the pre-zoom frame so YOLO returns pre-zoom
                    # coordinates — _ibox will then map them to display space.
                    # (Running detect on the already-zoomed frame would produce
                    # coords that _ibox double-transforms to wrong positions.)
                    self._detected = self._detect(pre_zoom_frame)
                frame = _digital_zoom(frame, MAX_DIGITAL_ZOOM, self._zoom_cx, self._zoom_cy)

                if elapsed >= BLESS_DURATION:
                    entry = {
                        "time":  time.strftime("%H:%M:%S"),
                        "psych": self._last_psych,
                        "bbox":  list(self._target) if self._target else None,
                    }
                    self._blessed_log.append(entry)
                    print(f"[Requiem] ✦ BLESSED #{len(self._blessed_log):03d}  "
                          f"{entry['time']}  {entry['psych']}")
                    if self._cam:
                        self._cam.zoom_to_minimum()
                        time.sleep(ZOOM_OUT_DURATION)
                        self._cam.stop_zoom()
                    self._start_hunt()

            # ── Build snapshot ────────────────────────────────────────────────
            needs_mesh = self.state in (State.ZOOM, State.ANALYZE, State.BLESS)
            mp_faces   = []
            if needs_mesh:
                if self._zoom_factor > 1.0 and self._zoom_cx and self._zoom_cy:
                    # Extract the same crop region that _digital_zoom shows.
                    # Running MediaPipe on this crop avoids the "face too large"
                    # failure that occurs on the zoomed full frame.
                    # Landmarks (p.x * w, p.y * h) land in zoomed-frame space
                    # because the crop IS exactly what the zoomed frame displays.
                    _zf  = self._zoom_factor
                    _cw  = max(1, int(w / _zf))
                    _ch  = max(1, int(h / _zf))
                    _x1  = max(0, min(self._zoom_cx - _cw // 2, w - _cw))
                    _y1  = max(0, min(self._zoom_cy - _ch // 2, h - _ch))
                    # np array slices are non-contiguous views; MediaPipe
                    # needs a contiguous buffer.
                    crop = np.ascontiguousarray(
                        pre_zoom_frame[_y1:_y1 + _ch, _x1:_x1 + _cw])
                    mp_faces = self._run_mediapipe(crop, w, h)

                    # Debug: save crop when at full zoom so we see what
                    # MediaPipe actually receives (not the near-full-frame
                    # at low zoom_factor which fires first otherwise).
                    if (not mp_faces and self._detected and _zf >= 2.0
                            and not getattr(self, '_mp_debug_saved', False)):
                        cv2.imwrite('/tmp/mp_debug_crop.jpg', crop)
                        print(f"[Mesh] DEBUG: mp_faces=0  YOLO found face  "
                              f"crop_shape={crop.shape}  "
                              f"x1={_x1} y1={_y1} cw={_cw} ch={_ch}  "
                              f"zoom_factor={_zf:.2f}  "
                              f"saved /tmp/mp_debug_crop.jpg")
                        self._mp_debug_saved = True
                else:
                    mp_faces = self._run_mediapipe(frame, w, h)

                if self._frame_count % 30 == 0:
                    print(f"[Mesh] state={self.state.value}  mp_faces={len(mp_faces)}"
                          f"  zoom_cx={self._zoom_cx}  zoom_cy={self._zoom_cy}"
                          f"  zoom_factor={self._zoom_factor:.2f}")

            cp = 0.0  # no hold phase — center_p is unused now but kept for API compat

            # ── Coordinate transform: pre-zoom → zoomed display space ──────────
            # YOLO detections and the target bbox are in the pre-zoom 1280×720
            # frame.  In ZOOM/ANALYZE/BLESS the MJPEG shows the zoomed crop, so
            # we must map those coords before sending them to the browser.
            # MediaPipe landmarks use the crop as input and p.x*w as output, so
            # they are already in zoomed display space — no transform needed.
            in_zoom = (self.state in (State.ZOOM, State.ANALYZE, State.BLESS)
                       and self._zoom_factor > 1.0
                       and self._zoom_cx and self._zoom_cy)

            if in_zoom:
                _zf   = self._zoom_factor
                _cw_z = max(1, int(w / _zf))
                _ch_z = max(1, int(h / _zf))
                _xo   = max(0, min(self._zoom_cx - _cw_z // 2, w - _cw_z))
                _yo   = max(0, min(self._zoom_cy - _ch_z // 2, h - _ch_z))

                def _ibox(b):
                    return [int((b[0] - _xo) * _zf), int((b[1] - _yo) * _zf),
                            int((b[2] - _xo) * _zf), int((b[3] - _yo) * _zf)]

                # Always keep only the MediaPipe face closest to display centre.
                # This handles both multi-face false detections AND single
                # detections on wrong objects (headrests, background figures).
                # Reject outright if the closest face is >40% of frame width
                # away from centre — it's not the zoomed subject.
                if mp_faces:
                    def _mp_dist(f):
                        if f.get('lm_left_eye'):
                            lx, ly = f['lm_left_eye'][0]
                            return math.hypot(lx - w / 2, ly - h / 2)
                        return float('inf')
                    closest = min(mp_faces, key=_mp_dist)
                    mp_faces = [closest] if _mp_dist(closest) < w * 0.40 else []
            else:
                def _ibox(b):
                    return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]

            tbox = _ibox(self._target) if self._target is not None else None

            snap = {
                "state":     self.state.value,
                "faces":     [
                    {"bbox": _ibox(b), "is_target": (_ibox(b) == tbox)}
                    for b in self._detected
                ],
                "mp_faces":  mp_faces,
                "target":    tbox,
                "zoom":      round(float(self._zoom_factor), 3),
                "frame_w":   int(w),
                "frame_h":   int(h),
                "t":         round(now, 3),
                "elapsed":   round(elapsed, 3),
                "analyze_p": round(min(elapsed / ANALYZE_DURATION, 1.0), 3)
                             if self.state == State.ANALYZE else 0.0,
                "bless_p":   round(min(elapsed / BLESS_DURATION, 1.0), 3)
                             if self.state == State.BLESS else 0.0,
                "center_p":  round(cp, 3),
                "blessed_n": len(self._blessed_log),
                "last_psych": self._last_psych,
            }

            sw, sh = self._args.stream_width, self._args.stream_height
            stream_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
            _, buf = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpg_bytes = buf.tobytes()

            # Confirm JSON serialises cleanly before storing
            try:
                json.dumps(snap)
            except (TypeError, ValueError) as je:
                print(f"[Engine] JSON ERROR: {je}")
                snap = _data   # fall back to previous good snapshot

            with _lock:
                _frame = jpg_bytes
                _data  = snap

            if self._frame_count % 90 == 0:   # ~every 3 s
                print(f"[Engine] frame={self._frame_count}  "
                      f"state={self.state.value}  "
                      f"jpg={len(jpg_bytes)}B  "
                      f"faces={len(self._detected)}")


# ── Flask app ─────────────────────────────────────────────────────────────────

app  = Flask(__name__)
sock = Sock(app)


def _mjpeg_stream():
    client_id = id(threading.current_thread())
    print(f"[MJPEG] client {client_id} connected")
    frames_sent = 0
    try:
        while True:
            with _lock:
                jpg = _frame
            if jpg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n'
                       b'\r\n' + jpg + b'\r\n')
                frames_sent += 1
                if frames_sent % 90 == 1:
                    print(f"[MJPEG] client {client_id} sent {frames_sent} frames")
            time.sleep(0.033)
    except GeneratorExit:
        print(f"[MJPEG] client {client_id} disconnected after {frames_sent} frames")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    with _lock:
        has_frame = _frame is not None
        frame_size = len(_frame) if _frame else 0
        state_val  = _data.get('state', '?')
    return jsonify({
        "ok": True,
        "has_frame": has_frame,
        "frame_bytes": frame_size,
        "state": state_val,
    })


@app.route('/test.jpg')
def test_jpg():
    """Single JPEG snapshot — use this to confirm image delivery works."""
    with _lock:
        jpg = _frame
    if jpg is None:
        return "No frame yet", 503
    return Response(jpg, mimetype='image/jpeg')


@app.route('/video_feed')
def video_feed():
    return Response(
        _mjpeg_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
    )


@sock.route('/ws')
def ws_handler(ws):
    client_id = id(threading.current_thread())
    print(f"[WS] client {client_id} connected")
    msgs_sent = 0
    try:
        while True:
            with _lock:
                data = json.dumps(_data)
            ws.send(data)
            msgs_sent += 1
            if msgs_sent % 90 == 1:
                print(f"[WS] client {client_id} sent {msgs_sent} msgs")
            time.sleep(0.033)
    except Exception as e:
        print(f"[WS] client {client_id} closed after {msgs_sent} msgs: {e}")


@app.route('/ptz', methods=['POST'])
def ptz():
    data = request.get_json(force=True, silent=True) or {}
    print(f"[PTZ] {data}")
    return jsonify({"ok": True})


@app.route('/command', methods=['POST'])
def command():
    data = request.get_json(force=True, silent=True) or {}
    cmd  = data.get('cmd', '').strip()
    if cmd:
        _cmd_deque.append(cmd)
    return jsonify({"ok": True, "cmd": cmd})


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="REQUIEM — Web Surveillance Art Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ip',          default='',    help='Reolink IP (empty = webcam)')
    p.add_argument('--user',        default='admin')
    p.add_argument('--password',    default='requiem-2026')
    p.add_argument('--profile',     default='main', choices=['main', 'sub'])
    p.add_argument('--source',      default=0,
                   type=lambda v: int(v) if str(v).isdigit() else v,
                   help='Webcam index or rtsp:// URL (used when --ip is empty)')
    p.add_argument('--host',        default='0.0.0.0')
    p.add_argument('--port',        default=5000, type=int)
    p.add_argument('--pan-range',   default=120.0, type=float)
    p.add_argument('--tilt-up',     default=45.0,  type=float)
    p.add_argument('--tilt-down',   default=30.0,  type=float)
    p.add_argument('--home-preset', default=0,     type=int,
                   help='PTZ preset index to go to on startup (-1 = skip)')
    p.add_argument('--stream-width',  default=960, type=int,
                   help='Width of MJPEG stream sent to browser')
    p.add_argument('--stream-height', default=480, type=int,
                   help='Height of MJPEG stream sent to browser')
    p.add_argument('--start',       action='store_true', help='Begin HUNT on launch')
    p.add_argument('--ir',          action='store_true',
                   help='Force CLAHE enhancement for IR/B&W camera mode. '
                        'Auto-detected when omitted (uses colour-variance check).')
    p.add_argument('--spotlight',   action='store_true',
                   help='Enable white spotlight during ANALYZE/BLESS. '
                        'Turns IR off and white LED on when the subject is analysed, '
                        'restores IR Auto when returning to HUNT/VIGIL. '
                        'Requires a camera with a physical white LED.')
    return p.parse_args()


def main():
    args   = parse_args()
    engine = RequiemEngine(args)

    print("[Requiem] Loading models …")
    engine.load_models()
    engine.connect_camera()

    if args.start:
        engine._start_hunt()

    def _run_engine():
        try:
            engine.run()
        except Exception:
            import traceback
            traceback.print_exc()
            print("[Engine] CRASHED — see traceback above")

    threading.Thread(target=_run_engine, daemon=True, name='engine').start()

    print(f"[Requiem] ── http://{args.host}:{args.port}/ ──")
    print(f"[Requiem] Test single frame: http://{args.host}:{args.port}/test.jpg")
    print(f"[Requiem] Health check:      http://{args.host}:{args.port}/health")
    app.run(host=args.host, port=args.port,
            debug=False, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
