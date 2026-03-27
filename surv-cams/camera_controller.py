#!/home/martinus/Projects/Requiem/requiem/bin/python3
"""
camera_controller.py
====================
Reolink camera interface module for the REQUIEM application.
Handles: connection, RTSP streaming, PTZ movement, zoom control, face centering.

Pan/tilt range limits are enforced via a time-based position estimator since
the Reolink API provides no absolute position feedback.
"""

from __future__ import annotations

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
DEAD_ZONE_PX   = 70    # pixels from frame centre — wide enough to absorb YOLO jitter
PAN_SPEED_MIN  = 8
PAN_SPEED_MAX  = 30
BURST_S        = 0.25  # LAN HTTP round-trip ~50 ms; need ≥ 0.20 s for visible movement
INTER_BURST_S  = 0.60  # longer settle → camera stops + new detection frame arrives

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
        self._panning       = False   # True while a hunt sweep pan is active

        # PTZ serialisation lock — prevents concurrent HTTP calls racing on the
        # camera connection or triggering simultaneous re-login attempts.
        # All _ptz_bg() calls acquire this before touching the camera API.
        self._ptz_lock = threading.Lock()

    # ─── Connection ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        print(f"[Camera] Connecting to {self.ip} ...")
        self._camera = Camera(self.ip, self.username, self.password, profile=self.profile)
        # Explicit login — some reolinkapi versions only log in lazily on first call,
        # which can race with model loading and leave the token stale.
        try:
            self._camera.login()
            time.sleep(0.5)   # give the camera a moment to activate the new token
            print("[Camera] Connected and authenticated.")
        except Exception as exc:
            print(f"[Camera] Login error: {exc}")
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

    def _relogin(self) -> bool:
        """Re-authenticate. Called automatically when rspCode -6 is detected."""
        try:
            self._camera.login()
            print("[Camera] Re-login OK.")
            return True
        except Exception as exc:
            print(f"[Camera] Re-login FAILED: {exc}")
            return False

    def _ptz_call_locked(self, fn, **kwargs):
        """
        Core PTZ HTTP call with auth-retry.  Caller must hold self._ptz_lock.
        """
        try:
            rv = fn(**kwargs)
        except Exception as exc:
            print(f"[Camera] PTZ call error: {exc}")
            return None

        if isinstance(rv, list) and rv:
            err = rv[0].get('error', {})
            if err.get('rspCode') == -6:
                print("[Camera] Session expired — re-logging in and retrying …")
                if self._relogin():
                    try:
                        rv = fn(**kwargs)
                    except Exception as exc:
                        print(f"[Camera] PTZ retry error: {exc}")
                        return None
        return rv

    def _ptz_call(self, fn, **kwargs):
        """Synchronous PTZ call — blocks the caller until the HTTP round-trip completes.
        Use for startup / one-shot operations (preset, zoom, test).
        Do NOT call from the main engine loop."""
        with self._ptz_lock:
            return self._ptz_call_locked(fn, **kwargs)

    def _ptz_bg(self, fn, **kwargs):
        """Non-blocking PTZ call — returns immediately, executes in a daemon thread.
        The _ptz_lock serialises concurrent calls so commands reach the camera in order
        and re-login races are impossible.  Use for all move/stop commands in the
        engine loop so the main thread never stalls on a camera HTTP round-trip."""
        def _run():
            with self._ptz_lock:
                self._ptz_call_locked(fn, **kwargs)
        threading.Thread(target=_run, daemon=True, name='ptz-bg').start()

    def test_ptz(self) -> bool:
        """
        Send a small left-right nudge and confirm the camera responds.
        Prints a clear PASS / FAIL message.

        Returns True if the API accepted the command (code 0 in response).
        Note: even PASS only means the API accepted the command — if the camera
        has AI Tracking, Guard Position, or PTZ Patrol enabled it may still
        override the movement.  Disable those features in the camera web UI.
        """
        import time as _time
        print("[Camera] PTZ self-test: nudge left …")
        try:
            rv = self._ptz_call(self._camera.move_left, speed=20)
            _time.sleep(0.30)
            self._ptz_call(self._camera.stop_ptz)
            # reolinkapi returns a list of response dicts; code 0 = success
            ok = False
            if isinstance(rv, list) and rv:
                ok = rv[0].get('code', -1) == 0
            elif isinstance(rv, bool):
                ok = rv
            elif rv is None:
                ok = True   # older lib versions return None on success
            if ok:
                print("[Camera] PTZ self-test PASS — API accepted the command.")
                print("[Camera] If the camera did NOT physically move, check:")
                print("         • AI Smart Tracking → OFF")
                print("         • Guard Position / Guard Tour → OFF")
                print("         • PTZ Patrol → OFF")
                print("         (All found in the camera's web UI under PTZ or AI settings)")
            else:
                print(f"[Camera] PTZ self-test FAIL — camera returned: {rv}")
                print("[Camera] Possible causes: wrong password, locked PTZ, unsupported firmware.")
            return ok
        except Exception as exc:
            print(f"[Camera] PTZ self-test ERROR: {exc}")
            return False

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
            self._ptz_call(self._camera.stop_ptz)
            self._ptz_call(self._camera.stop_zooming)
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
        self._ptz_call(op_map[direction], speed=speed)

        now = time.time()
        return now + duration, now + duration + pause

    def is_moving(self) -> bool:
        """True while a centering burst OR a hunt sweep pan is in progress."""
        return self._burst_active or self._panning

    def stop_movement(self):
        """Synchronous stop — use at state transitions where the camera must be
        stationary before proceeding (e.g. entering CENTER after HUNT)."""
        self._ptz_call(self._camera.stop_ptz)
        self._burst_active  = False
        self._next_burst_at = 0.0
        self._burst_axis    = 'h'
        self._panning       = False

    def stop_movement_async(self):
        """Non-blocking stop — use inside the main engine loop (e.g. sweep leg ends)
        so the engine thread does not stall on the HTTP round-trip."""
        self._ptz_bg(self._camera.stop_ptz)
        self._burst_active  = False
        self._next_burst_at = 0.0
        self._burst_axis    = 'h'
        self._panning       = False

    def start_pan(self, direction: str, speed: int = 20):
        """Start continuous pan in 'left' or 'right'. Non-blocking."""
        fn = self._camera.move_left if direction == 'left' else self._camera.move_right
        self._panning = True
        self._ptz_bg(fn, speed=speed)

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
            if self._burst_active:
                self._ptz_bg(self._camera.stop_ptz)   # non-blocking
                self._burst_active = False
            return 'centered'

        now = time.time()

        # ── If a burst is in progress, check whether it's time to stop it ────
        if self._burst_active:
            if now >= self._burst_end:
                self._ptz_bg(self._camera.stop_ptz)    # non-blocking
                self._burst_active  = False
                self._next_burst_at = now + INTER_BURST_S
            return 'adjusting'

        # ── Inter-burst pause: camera settling / frame updating ───────────────
        if now < self._next_burst_at:
            return 'adjusting'

        # ── Pick axis: always correct the larger error; skip centred axes ─────
        # This prevents wasting bursts on an axis that is already in dead zone.
        h_needed = abs(dx) > DEAD_ZONE_PX
        v_needed = abs(dy) > DEAD_ZONE_PX

        if h_needed and v_needed:
            self._burst_axis = 'h' if abs(dx) >= abs(dy) else 'v'
        elif h_needed:
            self._burst_axis = 'h'
        elif v_needed:
            self._burst_axis = 'v'
        # else: both in dead zone — already handled by dist < DEAD_ZONE_PX above

        norm  = min(dist / (frame_w / 2), 1.0)
        speed = int(PAN_SPEED_MIN + norm * (PAN_SPEED_MAX - PAN_SPEED_MIN))

        pan_ok_right  = self._pan_pos  < self._pan_limit
        pan_ok_left   = self._pan_pos  > -self._pan_limit
        tilt_ok_down  = self._tilt_pos < self._tilt_down_limit
        tilt_ok_up    = self._tilt_pos > -self._tilt_up_limit

        moved = False
        cmd   = None
        if self._burst_axis == 'h':
            if dx > 0 and pan_ok_right:
                self._ptz_bg(self._camera.move_right, speed=speed)
                cmd = f"move_right  speed={speed}"
                self._pan_pos = min(
                    self._pan_pos + speed * BURST_S * DEG_PER_SPEED_PER_SEC,
                    self._pan_limit)
                moved = True
            elif dx < 0 and pan_ok_left:
                self._ptz_bg(self._camera.move_left, speed=speed)
                cmd = f"move_left   speed={speed}"
                self._pan_pos = max(
                    self._pan_pos - speed * BURST_S * DEG_PER_SPEED_PER_SEC,
                    -self._pan_limit)
                moved = True
        else:  # 'v'
            if dy > 0 and tilt_ok_down:
                self._ptz_bg(self._camera.move_down, speed=speed)
                cmd = f"move_down   speed={speed}"
                self._tilt_pos = min(
                    self._tilt_pos + speed * BURST_S * DEG_PER_SPEED_PER_SEC,
                    self._tilt_down_limit)
                moved = True
            elif dy < 0 and tilt_ok_up:
                self._ptz_bg(self._camera.move_up, speed=speed)
                cmd = f"move_up     speed={speed}"
                self._tilt_pos = max(
                    self._tilt_pos - speed * BURST_S * DEG_PER_SPEED_PER_SEC,
                    -self._tilt_up_limit)
                moved = True

        print(f"[Center] axis={self._burst_axis}  dx={dx:+.0f}  dy={dy:+.0f}  "
              f"pan_pos={self._pan_pos:+.1f}  tilt_pos={self._tilt_pos:+.1f}  "
              f"cmd={cmd or 'BLOCKED'}")

        if moved:
            self._burst_active = True
            self._burst_end    = now + BURST_S
        else:
            # Movement blocked by position limit — try other axis, short wait
            self._burst_axis    = 'v' if self._burst_axis == 'h' else 'h'
            self._next_burst_at = now + INTER_BURST_S

        return 'adjusting'

    # ─── Zoom ─────────────────────────────────────────────────────────────────

    def zoom_in_slow(self):
        self._ptz_call(self._camera.start_zooming_in, speed=ZOOM_SPEED_TRACK)

    def zoom_out_full(self):
        self._ptz_call(self._camera.start_zooming_out, speed=ZOOM_SPEED_RESET)

    def zoom_to_minimum(self):
        """Absolute zoom-out via StartZoomFocus; falls back to continuous out."""
        fn = getattr(self._camera, 'start_zoom_pos', None)
        if fn is None or self._ptz_call(fn, 0) is None:
            self.zoom_out_full()

    def stop_zoom(self):
        self._ptz_call(self._camera.stop_zooming)

    def go_to_preset(self, index: int = 1, speed: int = 60):
        print(f"[Camera] go_to_preset index={index} speed={speed}")
        rv = self._ptz_call(self._camera.go_to_preset, speed=speed, index=index)
        print(f"[Camera] go_to_preset response: {rv}")

    # ─── Lights ───────────────────────────────────────────────────────────────

    @staticmethod
    def _light_status(rv) -> str:
        """Extract a short OK/ERR summary from a camera light API response."""
        try:
            code = rv[0].get("code", "?") if isinstance(rv, list) else "?"
            return "OK" if str(code) == "0" else f"ERR code={code}"
        except Exception:
            return f"ERR {rv}"

    def set_ir_lights(self, state: str = "Auto"):
        """Control IR LEDs.  state: 'Auto' | 'Off'"""
        body = [{"cmd": "SetIrLights", "action": 0,
                 "param": {"IrLights": {"channel": 0, "state": state}}}]
        try:
            rv = self._camera._execute_command("SetIrLights", body)
            print(f"[Lights] IR={state}  {self._light_status(rv)}")
        except Exception as e:
            print(f"[Lights] IR={state}  ERR {e}")

    def set_white_led(self, on: bool, bright: int = 100):
        """Control white spotlight.  on=True turns it on at given brightness (0-100)."""
        # mode 0 = off/disabled, mode 1 = always-on when state=1.
        # When turning off, force mode=0 so the firmware doesn't auto-reactivate.
        body = [{"cmd": "SetWhiteLed", "action": 0,
                 "param": {"WhiteLed": {"channel": 0, "state": int(on),
                                        "bright": bright if on else 0,
                                        "mode": 1 if on else 0}}}]
        try:
            rv = self._camera._execute_command("SetWhiteLed", body)
            label = f"ON bright={bright}%" if on else "OFF"
            print(f"[Lights] white={label}  {self._light_status(rv)}")
        except Exception as e:
            print(f"[Lights] white={'ON' if on else 'OFF'}  ERR {e}")

    def all_lights_off(self):
        """Turn off both IR and white LED in a single request so the firmware
        cannot auto-switch between them when IR is disabled."""
        body = [
            {"cmd": "SetIrLights", "action": 0,
             "param": {"IrLights": {"channel": 0, "state": "Off"}}},
            {"cmd": "SetWhiteLed", "action": 0,
             "param": {"WhiteLed": {"channel": 0, "state": 0,
                                    "bright": 0, "mode": 0}}},
        ]
        try:
            rv = self._camera._execute_command("SetIrLights", body, multi=True)
            print(f"[Lights] IR=Off white=OFF  {self._light_status(rv)}")
        except Exception as e:
            print(f"[Lights] IR=Off white=OFF  ERR {e}")

    # ─── Utility ─────────────────────────────────────────────────────────────

    @staticmethod
    def face_is_large_enough(
        x1: int, y1: int, x2: int, y2: int,
        frame_w: int, frame_h: int,
        target_ratio: float = 0.20,
    ) -> bool:
        """Return True once face area ≥ target_ratio of frame area (default 20%)."""
        return ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h) >= target_ratio
