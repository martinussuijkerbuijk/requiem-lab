#!/home/martinus/Projects/Requiem/requiem/bin/python3
"""
calibrate.py
============
Interactive camera calibration for REQUIEM.

Point the camera at your audience centre using the arrow keys,
then press S to save the position as preset 1.
The main script (requiem.py) loads this preset on startup via --home-preset 1.

Controls
--------
  Arrow keys   Pan / tilt (hold for continuous movement)
  W A X D      Same as arrows (reliable fallback: W=up, A=left, X=down, D=right)
  +  /  -      Increase / decrease movement speed
  S            Save current position as preset 1  (audience centre)
  R            Recall preset 1  (jump back to saved centre)
  Q / Esc      Quit
"""

import argparse
import sys
import time

import cv2
import numpy as np
from reolinkapi import Camera


# ──────────────────────────────────────────────────────────────────────────────
# Key codes — cv2.waitKeyEx returns different values depending on OS / X11
# Both known Linux variants are listed for each arrow key.
# WASD also work as fallback (W=up, A=left, D=right, X=down).
# ──────────────────────────────────────────────────────────────────────────────

KEYS_UP    = {2490368, 65362}   # X11 variant A / variant B
KEYS_DOWN  = {2621440, 65364}
KEYS_LEFT  = {2424832, 65361}
KEYS_RIGHT = {2555904, 65363}

KEY_ESC   = 27
KEY_S     = ord('s')
KEY_R     = ord('r')
KEY_Q     = ord('q')
KEY_PLUS  = ord('+')
KEY_MINUS = ord('-')

# WASD fallback (ASCII, always reliable)
KEY_W = ord('w')   # up
KEY_A = ord('a')   # left
KEY_X = ord('x')   # down
KEY_D = ord('d')   # right

# Movement burst: camera moves for this long after each keypress (seconds).
# Arrow keys repeat at ~30 Hz when held, so movement is continuous while held.
BURST_S = 0.15

SPEEDS = [8, 15, 25, 35, 50]
DEFAULT_SPEED_IDX = 2   # 25


# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="REQUIEM camera calibration — position camera, press S to save.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ip",       default="192.168.1.126", help="Camera IP address")
    p.add_argument("--user",     default="admin",         help="Camera username")
    p.add_argument("--password", default="requiem-2026",  help="Camera password")
    p.add_argument("--profile",  default="sub", choices=["main", "sub"],
                   help="Stream profile (sub recommended for low-latency preview)")
    p.add_argument("--preset",   default=1, type=int,
                   help="Preset index to save/recall (default: 1)")
    p.add_argument("--width",    default=960, type=int, help="Preview window width")
    p.add_argument("--height",   default=540, type=int, help="Preview window height")
    return p.parse_args()


def draw_overlay(frame: np.ndarray, speed: int, speed_idx: int,
                 saved: bool, status_msg: str) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Centre crosshair
    cv2.line(frame, (cx - 40, cy), (cx + 40, cy), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 40), (cx, cy + 40), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 1, cv2.LINE_AA)

    # Help text (bottom-left)
    lines = [
        "ARROWS / W A X D  pan / tilt",
        "+ / -             speed",
        "S                 save as centre",
        "R                 recall centre",
        "Q / ESC           quit",
    ]
    font = cv2.FONT_HERSHEY_PLAIN
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, h - 12 - i * 18), font, 1.0, (0, 200, 80), 1)

    # Speed indicator (top-left)
    speed_bar = " ".join(
        f"[{s}]" if j == speed_idx else f" {s} "
        for j, s in enumerate(SPEEDS)
    )
    cv2.putText(frame, f"SPEED  {speed_bar}", (12, 28), font, 1.0, (0, 200, 80), 1)

    # Status message (top-right area)
    color = (0, 255, 180) if saved else (0, 180, 255)
    cv2.putText(frame, status_msg, (12, 56), font, 1.2, color, 2)

    return frame


def main():
    args = parse_args()

    print(f"[Calibrate] Connecting to {args.ip} …")
    cam = Camera(args.ip, args.user, args.password, profile=args.profile)
    print("[Calibrate] Connected.")

    stream = cam.open_video_stream()

    cv2.namedWindow("REQUIEM — Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("REQUIEM — Calibration", args.width, args.height)

    speed_idx  = DEFAULT_SPEED_IDX
    speed      = SPEEDS[speed_idx]
    saved      = False
    status_msg = "Position camera, then press S to save"
    moving     = False        # True while a movement command is active
    stop_at    = 0.0          # time.time() when to issue stop

    print("\n  Arrows / W A X D to move   |   + / -  speed   |   S save   |   R recall   |   Q quit")
    print("  (raw key codes printed to terminal — if arrows don't respond, use W A X D)\n")

    for raw_frame in stream:
        # ── auto-stop after burst duration ────────────────────────────────────
        now = time.time()
        if moving and now >= stop_at:
            try:
                cam.stop_ptz()
            except Exception:
                pass
            moving = False

        # ── draw overlay ──────────────────────────────────────────────────────
        frame = draw_overlay(raw_frame.copy(), speed, speed_idx, saved, status_msg)
        cv2.imshow("REQUIEM — Calibration", frame)

        # ── key handling ──────────────────────────────────────────────────────
        key = cv2.waitKeyEx(1)
        if key == -1:
            continue

        masked = key & 0xFF   # for regular ASCII keys

        # Debug: print raw code for any unrecognised key so codes can be tuned
        if key != -1:
            print(f"[key] raw={key}  masked={masked}", end="\r")

        # Movement keys — restart burst timer on every press (held = continuous)
        move_cmd = None
        if   key in KEYS_UP    or masked == KEY_W: move_cmd = cam.move_up
        elif key in KEYS_DOWN  or masked == KEY_X: move_cmd = cam.move_down
        elif key in KEYS_LEFT  or masked == KEY_A: move_cmd = cam.move_left
        elif key in KEYS_RIGHT or masked == KEY_D: move_cmd = cam.move_right

        if move_cmd is not None:
            try:
                move_cmd(speed=speed)
            except Exception as exc:
                print(f"[Calibrate] Move error: {exc}")
            moving  = True
            stop_at = time.time() + BURST_S
            continue

        # Speed adjust
        if masked == KEY_PLUS:
            speed_idx = min(speed_idx + 1, len(SPEEDS) - 1)
            speed = SPEEDS[speed_idx]
            print(f"[Calibrate] Speed → {speed}")
            continue

        if masked == KEY_MINUS:
            speed_idx = max(speed_idx - 1, 0)
            speed = SPEEDS[speed_idx]
            print(f"[Calibrate] Speed → {speed}")
            continue

        # Save preset
        if masked == KEY_S:
            try:
                cam.stop_ptz()
                time.sleep(0.2)   # let camera settle before saving
                cam.add_preset(preset=args.preset, name="requiem-centre")
                saved      = True
                status_msg = f"SAVED  →  preset {args.preset}  (requiem-centre)"
                print(f"[Calibrate] Position saved as preset {args.preset}.")
                print(f"[Calibrate] Launch requiem.py with:  --home-preset {args.preset}")
            except Exception as exc:
                status_msg = f"SAVE FAILED: {exc}"
                print(f"[Calibrate] Save failed: {exc}")
            continue

        # Recall preset
        if masked == KEY_R:
            try:
                cam.go_to_preset(speed=40, index=args.preset)
                status_msg = f"Recalled preset {args.preset}"
                print(f"[Calibrate] Moved to preset {args.preset}.")
            except Exception as exc:
                status_msg = f"RECALL FAILED: {exc}"
                print(f"[Calibrate] Recall failed: {exc}")
            continue

        # Quit
        if masked in (KEY_Q, KEY_ESC):
            break

    try:
        cam.stop_ptz()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("[Calibrate] Done.")


if __name__ == "__main__":
    main()
