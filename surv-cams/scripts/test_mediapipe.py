#!/usr/bin/env python3
"""
Quick MediaPipe FaceMesh test on webcam or a static image.

Usage:
    python scripts/test_mediapipe.py           # webcam (device 0)
    python scripts/test_mediapipe.py --src 1   # alternate webcam
    python scripts/test_mediapipe.py --image /path/to/face.jpg

Press Q to quit.
"""

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src',       type=int,   default=0,    help='webcam device index')
    ap.add_argument('--image',     type=str,   default=None, help='test on a static image instead')
    ap.add_argument('--conf',      type=float, default=0.4,  help='min_detection_confidence')
    ap.add_argument('--width',     type=int,   default=1280)
    ap.add_argument('--height',    type=int,   default=720)
    ap.add_argument('--grayscale', action='store_true',
                    help='convert to grayscale before detection (simulates IR camera)')
    args = ap.parse_args()

    mp_fm   = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_sty  = mp.solutions.drawing_styles

    face_mesh = mp_fm.FaceMesh(
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=args.conf,
        min_tracking_confidence=0.4,
    )

    print(f"[Test] MediaPipe version: {mp.__version__}")
    print(f"[Test] min_detection_confidence={args.conf}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def prepare(img):
        """Apply grayscale simulation + CLAHE if requested or detected."""
        if args.grayscale:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            e = clahe.apply(g)
            return cv2.merge([e, e, e])
        # Auto-detect grayscale (IR camera): R≈G≈B
        b, g, r = cv2.split(img)
        if float(np.std(r.astype(np.int16) - b.astype(np.int16))) < 8.0:
            print("[Test] Grayscale/IR image detected — applying CLAHE")
            g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            e  = clahe.apply(g2)
            return cv2.merge([e, e, e])
        return img

    # ── Static image mode ────────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[Test] ERROR: could not read {args.image}")
            return
        h, w = frame.shape[:2]
        # Test both raw and enhanced
        for label, img in [('RAW', frame), ('ENHANCED', prepare(frame))]:
            rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            n = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
            print(f"[Test] {label}  faces={n}  shape={img.shape}")
            if results.multi_face_landmarks and label == 'ENHANCED':
                for lm in results.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        img, lm,
                        mp_fm.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_sty.get_default_face_mesh_tesselation_style())
                cv2.imshow(f'MediaPipe {label}', img)
        cv2.waitKey(0)
        return

    # ── Webcam mode ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Test] Camera opened  requested={args.width}×{args.height}  "
          f"actual={actual_w}×{actual_h}")

    fps_t  = time.time()
    fps_n  = 0
    fps    = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Test] Frame read failed — stopping.")
            break

        fps_n += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps    = fps_n / (now - fps_t)
            fps_n  = 0
            fps_t  = now

        h, w = frame.shape[:2]

        t0      = time.time()
        mp_in   = prepare(frame)
        rgb     = cv2.cvtColor(mp_in, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        mp_ms   = (time.time() - t0) * 1000

        n = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0

        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame, lm,
                    mp_fm.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_sty.get_default_face_mesh_tesselation_style())
                mp_draw.draw_landmarks(
                    frame, lm,
                    mp_fm.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_sty.get_default_face_mesh_contours_style())

        # ── Overlay ──────────────────────────────────────────────────────────
        def txt(s, y, color=(0, 255, 80)):
            cv2.putText(frame, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)

        status_col = (0, 220, 50) if n > 0 else (0, 80, 255)
        txt(f"faces={n}   mp={mp_ms:.0f}ms   fps={fps:.1f}", 28, status_col)
        txt(f"resolution={w}x{h}   conf={args.conf}", 52)

        if n == 0:
            txt("NO FACE DETECTED", h // 2, (0, 60, 255))

        cv2.imshow('MediaPipe FaceMesh test  (Q=quit)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        # Also print to terminal every 60 frames
        if fps_n == 1:
            print(f"[Test] faces={n}  mp={mp_ms:.0f}ms  fps={fps:.1f}  "
                  f"res={w}x{h}")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == '__main__':
    main()
