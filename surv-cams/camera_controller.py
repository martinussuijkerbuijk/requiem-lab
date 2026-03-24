#!/home/martinus/Projects/Requiem/requiem/bin/python3
"""
camera_controller.py
====================
Reolink camera interface module for the REQUIEM application.
Handles: connection, RTSP streaming, PTZ movement, zoom control, face centering.

Pan/tilt range limits are enforced via a time-based position estimator since
the Reolink API provides no absolute position feedback.
"""

import threading
import time
import math
import random

import cv2
import numpy as np

from reolinkapi import Camera


# ──────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ──────────────────────────────────────────────────────────────────────────────

# PTZ centering
DEAD_ZONE_PX   = 45    # pixels from frame centre before issuing correction
PAN_SPEED_MIN  = 8
PAN_SPEED_MAX  = 30
BURST_S        = 0.15  # how long each correction movement lasts before stopping
INTER_BURST_S  = 0.35  # pause between bursts (camera settles, frame updates)

# Zoom
ZOOM_SPEED_TRACK = 22  # slow creep during ZOOM state
ZOOM_SPEED_RESET = 60  # fast zoom-out on release

# Patrol sweep
SWEEP_SPEEDS    = [12, 18, 25, 30]
SWEEP_DURATIONS = (0.4, 1.8)   # seconds of movement per burst
SWEEP_PAUSES    = (0.5, 2.0)   # seconds of stillness between bursts

# Position estimator
# Rough calibration: at speed=25, camera moves ~1 degree per second.
# Adjust DEG_PER_SPEED_PER_SEC until the limits feel right for your camera.
DEG_PER_SPEED_PER_SEC = 1.0

# Default operating range limits (degrees from starting centre position)
# x: total 120° → ±60° from centre
# y: up 45° / down 30° from centre (asymmetric — avoids floor / low audience angle)
DEFAULT_PAN_LIMIT       = 60.0
DEFAULT_TILT_UP_LIMIT   = 45.0
DEFAULT_TILT_DOWN_LIMIT = 30.0


# ──────────────────────────────────────────────────────────────────────────────

class CameraController:
    """
    Wraps the Reolink Camera API for REQUIEM.

    The camera's current position is estimated via dead-reckoning
    (speed × duration) since the API provides no position feedback.
    """

    def __init__(
        self,
        ip: str,
        username: str,
        password: str,
        profile: str = 'main',
        pan_limit: float = DEFAULT_PAN_LIMIT,
        tilt_up_limit: float = DEFAULT_TILT_UP_LIMIT,
        tilt_down_limit: float = DEFAULT_TILT_DOWN_LIMIT,
    ):
        self.ip        = ip
        self.username  = username
        self.password  = password
        self.profile   = profile

        self._pan_limit       = pan_limit
        self._tilt_up_limit   = tilt_up_limit    # degrees above start position
        self._tilt_down_limit = tilt_down_limit  # degrees below start position

        self._camera: Camera | None = None

        # Thread-safe frame buffer
        self._frame: np.ndarray | None = None
        self._frame_lock  = threading.Lock()
        self._stream_active = False
        self._stream_thread: threading.Thread | None = None

        # Dead-reckoning position estimator (degrees from start)
        self._pan_pos  = 0.0
        self._tilt_pos = 0.0

        # Centering burst-and-stop state
        self._burst_active  = False   # True while a correction burst is in progress
        self._burst_end     = 0.0     # time.time() when to stop the current burst
        self._next_burst_at = 0.0     # earliest time the next burst may start
        self._burst_axis    = 'h'     # alternates 'h' (pan) / 'v' (tilt)

    # ─── Connection ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        print(f"[Camera] Connecting to {self.ip} ...")
        self._camera = Camera(self.ip, self.username, self.password, profile=self.profile)
        print("[Camera] Connected.")
        try:
            self._camera.set_osd(
                osd_channel_enabled=False,
                osd_channel_name="",
                osd_time_enabled=False,
                osd_watermark_enabled=False,
            )
            print("[Camera] OSD cleared (channel name, time, watermark off).")
        except Exception as exc:
            print(f"[Camera] Could not clear OSD: {exc}")
        return True

    # ─── Streaming ───────────────────────────────────────────────────────────

    def start_stream(self):
        if self._stream_active:
            return
        self._stream_active = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="cam-stream")
        self._stream_thread.start()
        print("[Camera] Stream thread started.")

    def _stream_loop(self):
        try:
            stream = self._camera.open_video_stream()
            for frame in stream:
                if not self._stream_active:
                    break
                with self._frame_lock:
                    self._frame = frame
        except Exception as exc:
            print(f"[Camera] Stream error: {exc}")
            self._stream_active = False

    def get_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stream_active = False
        if self._camera:
            try:
                self._camera.stop_ptz()
                self._camera.stop_zooming()
            except Exception:
                pass
        print("[Camera] Stopped.")

    # ─── PTZ – patrol sweep ──────────────────────────────────────────────────

    def begin_sweep(self) -> tuple[float, float]:
        """
        Issue one bounded random pan burst.

        Direction is biased away from the position limit so the camera
        stays within the configured pan/tilt range.

        Returns (stop_time, pause_until).
        """
        pan_ratio  = self._pan_pos  / self._pan_limit   if self._pan_limit  > 0 else 0
        tilt_ratio = self._tilt_pos / self._tilt_down_limit  if self._tilt_down_limit > 0 else 0

        # Build the candidate direction list weighted to stay within bounds.
        # pan_ratio > 0 → camera is to the right → prefer Left
        # pan_ratio < 0 → camera is to the left  → prefer Right
        def _h_weight(dir_name: str) -> float:
            """Higher weight when direction moves away from the limit."""
            if 'Right' in dir_name and 'Left' not in dir_name:
                return max(0.1, 1.0 - pan_ratio)   # weak when already right
            if 'Left' in dir_name and 'Right' not in dir_name:
                return max(0.1, 1.0 + pan_ratio)   # weak when already left
            return 1.0

        candidates = ['Left', 'Right', 'LeftUp', 'RightUp', 'LeftDown', 'RightDown']
        weights    = [_h_weight(d) * 3 if d in ('Left', 'Right') else _h_weight(d)
                      for d in candidates]

        direction = random.choices(candidates, weights=weights, k=1)[0]
        speed     = random.choice(SWEEP_SPEEDS)
        duration  = random.uniform(*SWEEP_DURATIONS)
        pause     = random.uniform(*SWEEP_PAUSES)

        # Update estimated position
        delta = speed * duration * DEG_PER_SPEED_PER_SEC
        if 'Right' in direction:
            self._pan_pos = min(self._pan_pos + delta, self._pan_limit)
        elif 'Left' in direction:
            self._pan_pos = max(self._pan_pos - delta, -self._pan_limit)

        if 'Up' in direction:
            self._tilt_pos = max(self._tilt_pos - delta * 0.5, -self._tilt_up_limit)
        elif 'Down' in direction:
            self._tilt_pos = min(self._tilt_pos + delta * 0.5,  self._tilt_down_limit)

        op_map = {
            'Left':      self._camera.move_left,
            'Right':     self._camera.move_right,
            'LeftUp':    self._camera.move_left_up,
            'RightUp':   self._camera.move_right_up,
            'LeftDown':  self._camera.move_left_down,
            'RightDown': self._camera.move_right_down,
        }
        try:
            op_map[direction](speed=speed)
        except Exception as exc:
            print(f"[Camera] Sweep command failed: {exc}")

        now = time.time()
        return now + duration, now + duration + pause

    def stop_movement(self):
        try:
            self._camera.stop_ptz()
        except Exception:
            pass
        self._burst_active  = False
        self._next_burst_at = 0.0
        self._burst_axis    = 'h'

    # ─── PTZ – face centering ────────────────────────────────────────────────

    def center_on_face(
        self,
        x1: int, y1: int, x2: int, y2: int,
        frame_w: int, frame_h: int,
    ) -> str:
        """
        Incrementally centre the face using alternating single-axis bursts.

        Pattern: issue a short move burst (BURST_S) → stop → pause (INTER_BURST_S)
        → repeat on the other axis.  Alternating h/v avoids diagonal drift and
        lets the frame settle between corrections.

        Returns 'centered' if within dead zone, 'adjusting' otherwise.
        """
        cx_face = (x1 + x2) / 2
        cy_face = (y1 + y2) / 2
        dx = cx_face - frame_w / 2
        dy = cy_face - frame_h / 2
        dist = math.hypot(dx, dy)

        if dist < DEAD_ZONE_PX:
            # Make sure any in-progress burst is stopped when we're centred
            if self._burst_active:
                try:
                    self._camera.stop_ptz()
                except Exception:
                    pass
                self._burst_active = False
            return 'centered'

        now = time.time()

        # ── If a burst is in progress, check whether it's time to stop it ────
        if self._burst_active:
            if now >= self._burst_end:
                try:
                    self._camera.stop_ptz()
                except Exception:
                    pass
                self._burst_active  = False
                self._next_burst_at = now + INTER_BURST_S
                # Flip axis for next burst
                self._burst_axis = 'v' if self._burst_axis == 'h' else 'h'
            return 'adjusting'

        # ── Inter-burst pause: camera settling / frame updating ───────────────
        if now < self._next_burst_at:
            return 'adjusting'

        # ── Start the next burst on the current axis ──────────────────────────
        norm  = min(dist / (frame_w / 2), 1.0)
        speed = int(PAN_SPEED_MIN + norm * (PAN_SPEED_MAX - PAN_SPEED_MIN))

        # Headroom is direction-sensitive: only block movement that would push
        # FURTHER past the limit, never block movement back toward centre.
        pan_ok_right  = self._pan_pos  < self._pan_limit
        pan_ok_left   = self._pan_pos  > -self._pan_limit
        tilt_ok_down  = self._tilt_pos < self._tilt_down_limit
        tilt_ok_up    = self._tilt_pos > -self._tilt_up_limit

        moved = False
        cmd   = None
        try:
            if self._burst_axis == 'h' and abs(dx) > DEAD_ZONE_PX:
                if dx > 0 and pan_ok_right:
                    self._camera.move_right(speed=speed)
                    cmd = f"move_right  speed={speed}"
                    delta = speed * BURST_S * DEG_PER_SPEED_PER_SEC
                    self._pan_pos = min(self._pan_pos + delta,  self._pan_limit)
                    moved = True
                elif dx < 0 and pan_ok_left:
                    self._camera.move_left(speed=speed)
                    cmd = f"move_left   speed={speed}"
                    delta = speed * BURST_S * DEG_PER_SPEED_PER_SEC
                    self._pan_pos = max(self._pan_pos - delta, -self._pan_limit)
                    moved = True

            elif self._burst_axis == 'v' and abs(dy) > DEAD_ZONE_PX:
                if dy > 0 and tilt_ok_down:
                    self._camera.move_down(speed=speed)
                    cmd = f"move_down   speed={speed}"
                    delta = speed * BURST_S * DEG_PER_SPEED_PER_SEC
                    self._tilt_pos = min(self._tilt_pos + delta,  self._tilt_down_limit)
                    moved = True
                elif dy < 0 and tilt_ok_up:
                    self._camera.move_up(speed=speed)
                    cmd = f"move_up     speed={speed}"
                    delta = speed * BURST_S * DEG_PER_SPEED_PER_SEC
                    self._tilt_pos = max(self._tilt_pos - delta, -self._tilt_up_limit)
                    moved = True
        except Exception as exc:
            print(f"[Camera] PTZ error: {exc}")

        print(f"[Center] axis={self._burst_axis}  dx={dx:+.0f}  dy={dy:+.0f}  "
              f"pan_pos={self._pan_pos:+.1f}  tilt_pos={self._tilt_pos:+.1f}  "
              f"cmd={cmd or 'BLOCKED'}")

        if moved:
            self._burst_active = True
            self._burst_end    = now + BURST_S
        else:
            # Current axis already in dead zone — skip to other axis immediately
            self._burst_axis    = 'v' if self._burst_axis == 'h' else 'h'
            self._next_burst_at = now + INTER_BURST_S

        return 'adjusting'

    # ─── Zoom ─────────────────────────────────────────────────────────────────

    def zoom_in_slow(self):
        try:
            self._camera.start_zooming_in(speed=ZOOM_SPEED_TRACK)
        except Exception:
            pass

    def zoom_out_full(self):
        try:
            self._camera.start_zooming_out(speed=ZOOM_SPEED_RESET)
        except Exception:
            pass

    def zoom_to_minimum(self):
        """Absolute zoom-out via StartZoomFocus; falls back to continuous out."""
        try:
            self._camera.start_zoom_pos(0)
        except Exception:
            self.zoom_out_full()

    def stop_zoom(self):
        try:
            self._camera.stop_zooming()
        except Exception:
            pass

    def go_to_preset(self, index: int = 1, speed: int = 60):
        try:
            self._camera.go_to_preset(speed=speed, index=index)
        except Exception:
            pass

    # ─── Utility ─────────────────────────────────────────────────────────────

    @staticmethod
    def face_is_large_enough(
        x1: int, y1: int, x2: int, y2: int,
        frame_w: int, frame_h: int,
        target_ratio: float = 0.20,
    ) -> bool:
        """Return True once face area ≥ target_ratio of frame area (default 20%)."""
        return ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h) >= target_ratio
