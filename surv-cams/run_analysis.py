import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random

def draw_targeting_brackets(frame, x1, y1, x2, y2, color, thickness=1, length=20):
    """Draws sci-fi style corner brackets instead of a full rectangle."""
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

def get_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def draw_hud_scale(frame, x, y, label, current_val, val_list, is_left=True, highlight_color=(0,0,255)):
    """Helper to draw the vertical forensic telemetry ladders."""
    green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_PLAIN
    
    # Label
    cv2.putText(frame, label, (x - 20 if is_left else x, y - 15), font, 1.0, green, 1)
    
    # Vertical Line (thickened)
    h = len(val_list) * 20
    cv2.line(frame, (x, y), (x, y + h), green, 1)
    
    for i, val in enumerate(val_list):
        ty = y + (i * 20)
        # Tick marks (thickened)
        tick_len = 8 if i % 2 != 0 else 15
        tx_end = x + tick_len if is_left else x - tick_len
        cv2.line(frame, (x, ty), (tx_end, ty), green, 1)
        
        # Text placement
        text_x = x + 18 if is_left else x - 40
        
        # Highlight current value box
        if val == current_val:
            # Box background
            cv2.rectangle(frame, (text_x - 5, ty - 12), (text_x + 35, ty + 5), highlight_color, -1)
            cv2.putText(frame, str(val), (text_x, ty + 2), font, 1.0, (0, 0, 0), 1)
        else:
            cv2.putText(frame, str(val), (text_x, ty + 2), font, 0.9, green, 1)


def draw_forensic_hud(frame, w, h, elapsed):
    """Draws the global drone/search-and-detect interface."""
    green = (0, 255, 0)
    red = (0, 0, 255)
    
    # --- TOP ROLL ARC ---
    cx, cy = w // 2, h // 2
    arc_radius = 200
    for angle in range(-45, 46, 5):
        rad = math.radians(angle - 90)
        r1 = arc_radius if angle % 15 == 0 else arc_radius + 10
        r2 = arc_radius + 20
        start_pt = (int(cx + r1 * math.cos(rad)), int(150 + r1 * math.sin(rad)))
        end_pt = (int(cx + r2 * math.cos(rad)), int(150 + r2 * math.sin(rad)))
        # Thickened arc lines
        cv2.line(frame, start_pt, end_pt, green, 3 if angle % 15 == 0 else 2)
    
    # Roll indicator triangle (thickened)
    cv2.drawMarker(frame, (cx, 150 - arc_radius + 5), green, cv2.MARKER_TRIANGLE_DOWN, 20, 1)
    cv2.putText(frame, "ROLL", (cx - 25, 150 - arc_radius + 35), cv2.FONT_HERSHEY_PLAIN, 1.2, green, 1)

    # --- CENTER RETICLE ---
    gap = 20
    length = 60
    # Dashed horizon line (thickened)
    for dx in range(-200, -50, 30):
        cv2.line(frame, (cx + dx, cy), (cx + dx + 15, cy), green, 1)
    for dx in range(50, 200, 30):
        cv2.line(frame, (cx + dx, cy), (cx + dx + 15, cy), green, 1)
        
    cv2.line(frame, (cx, cy - gap), (cx, cy - gap - length), green, 1)
    cv2.line(frame, (cx, cy + gap), (cx, cy + gap + length), green, 1)
    
    # Pitch ladder center bars (thickened)
    for offset in [-60, -30, 30, 60]:
        bar_len = 20 if abs(offset) == 30 else 40
        cv2.line(frame, (cx - bar_len, cy + offset), (cx + bar_len, cy + offset), green, 1)
        cv2.line(frame, (cx - bar_len, cy + offset), (cx - bar_len, cy + offset + (6 if offset > 0 else -6)), green, 1)
        cv2.line(frame, (cx + bar_len, cy + offset), (cx + bar_len, cy + offset + (6 if offset > 0 else -6)), green, 1)

    # --- SIDE SCALES ---
    # Simulate slight telemetry fluctuations based on time
    wobble = int(math.sin(elapsed * 2) * 2)
    
    # Left Scales: Speed & Pitch
    draw_hud_scale(frame, 50, cy - 100, "SPEED m/s", 39, [43, 42, 41, 40, 39, 38, 37, 36, 35, 34], highlight_color=red)
    draw_hud_scale(frame, 180, cy - 100, "PITCH", 4, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], highlight_color=green)
    
    # Right Scales: Altitude & Climb
    draw_hud_scale(frame, w - 180, cy - 100, "AGL ALT m", 87, [110, 105, 100, 95, 90, 87, 80, 75, 70, 65], is_left=False, highlight_color=green)
    draw_hud_scale(frame, w - 50, cy - 100, "CLI m/s", 0.1, [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4], is_left=False, highlight_color=green)


def main():
    print("--- INITIATING FORENSIC CACOPHONY PROTOCOL ---")

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=30, # Increased to track up to 30 faces in the audience
        refine_landmarks=True, 
        min_detection_confidence=0.5, # Lowered slightly to better catch faces further back in the crowd
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_width, frame_height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    start_time = time.time()
    scanline_y = 0

    print("System Online. Press 'q' to sever connection.")

    while True:
        ret, original_frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed = current_time - start_time
        
        # 1. COLD OMNIPOTENT BASELINE (Blueish Aesthetic Reverted)
        gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        base_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Deep blue/cyan tint for the background
        base_frame = cv2.multiply(base_frame, np.array([1.6, 1.0, 0.6])) 
        base_frame = np.clip(base_frame, 0, 255).astype(np.uint8)

        # 2. MEDIAPIPE INFERENCE
        rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Landmark indices
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = original_frame.shape
                
                # Extract pixel coordinates
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
                
                # Calculate Bounding Box for the HSV Anomaly effect
                x_coords = [p[0] for p in landmarks]
                y_coords = [p[1] for p in landmarks]
                
                # Add padding to the face bounding box
                pad = 40
                x1, y1 = max(0, min(x_coords) - pad), max(0, min(y_coords) - pad)
                x2, y2 = min(w, max(x_coords) + pad), min(h, max(y_coords) + pad)

                if x2 <= x1 or y2 <= y1: continue

                # 3. HSV ANOMALY EXTRACTION (The Funky Meat-Scan on top of Blue)
                roi = original_frame[y1:y2, x1:x2]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Hyper-fast pulsing Hue, Maxed Saturation
                hue_shift = int((current_time * 100) % 180)
                hsv_roi[:, :, 0] = (hsv_roi[:, :, 0] + hue_shift) % 180
                hsv_roi[:, :, 1] = 255 
                
                processed_roi = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)
                base_frame[y1:y2, x1:x2] = cv2.addWeighted(base_frame[y1:y2, x1:x2], 0.2, processed_roi, 0.8, 0)

                # 4. DRAW INVASIVE STALKER MESH (Neon Funky Colors - Thickened)
                neon_pink = (255, 50, 255)
                neon_cyan = (255, 255, 0)
                
                # Draw Eyes (thickened to 3)
                for i in range(len(LEFT_EYE)):
                    cv2.line(base_frame, landmarks[LEFT_EYE[i]], landmarks[LEFT_EYE[(i+1)%len(LEFT_EYE)]], neon_cyan, 2)
                for i in range(len(RIGHT_EYE)):
                    cv2.line(base_frame, landmarks[RIGHT_EYE[i]], landmarks[RIGHT_EYE[(i+1)%len(RIGHT_EYE)]], neon_cyan, 2)
                
                # Draw Lips (thickened to 3)
                for i in range(len(LIPS)):
                    cv2.line(base_frame, landmarks[LIPS[i]], landmarks[LIPS[(i+1)%len(LIPS)]], neon_pink, 2)

                # Crosshairs on Irises (thickened and larger)
                cv2.drawMarker(base_frame, landmarks[473], (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.drawMarker(base_frame, landmarks[468], (0, 0, 255), cv2.MARKER_CROSS, 20, 1)

                # 5. PSEUDO-EMOTION CALCULATIONS
                eye_h = get_distance(landmarks[159], landmarks[145])
                eye_w = get_distance(landmarks[33], landmarks[133])
                ear = eye_h / eye_w if eye_w > 0 else 0
                mouth_h = get_distance(landmarks[13], landmarks[14])
                
                psych_state = "COMPLIANT"
                tension_lvl = "NOMINAL"
                ui_color = (0, 255, 0) # Green default
                
                if mouth_h > 20: 
                    psych_state = "SHOCKED / VOCALIZING"
                    tension_lvl = "ELEVATED"
                    ui_color = (0, 165, 255) # Orange
                elif ear < 0.2: 
                    psych_state = "DEFENSIVE / SKEPTICAL"
                    tension_lvl = "HIGH"
                    ui_color = neon_pink
                elif ear > 0.35: 
                    psych_state = "FEAR / ALERTNESS"
                    tension_lvl = "CRITICAL"
                    ui_color = (0, 0, 255) # Red

                # 6. AGGRESSIVE UI OVERLAYS (Targeting - Thickened)
                draw_targeting_brackets(base_frame, x1, y1, x2, y2, ui_color, thickness=3, length=40)
                
                font = cv2.FONT_HERSHEY_PLAIN
                hex_id = f"ID:{random.randint(0x1000, 0xFFFF):04X}"
                
                # Left side: Identity & Threat
                cv2.putText(base_frame, hex_id, (x1, y1 - 35), font, 1.2, ui_color, 2)
                cv2.putText(base_frame, f"TENSION: {tension_lvl}", (x1, y1 - 15), font, 1.2, ui_color, 2)
                
                # Right side: Psych State scrolling text
                cv2.putText(base_frame, f"PSYCH_EVAL:", (x2 + 10, y1 + 20), font, 1.2, neon_cyan, 2)
                cv2.putText(base_frame, psych_state, (x2 + 10, y1 + 40), font, 1.2, ui_color, 2)
                cv2.putText(base_frame, f"PUPIL: {ear:.2f}mm", (x2 + 10, y1 + 60), font, 1.0, neon_cyan, 2)
                cv2.putText(base_frame, f"VOCAL: {mouth_h:.1f}", (x2 + 10, y1 + 80), font, 1.0, neon_cyan, 2)

        # 7. DRAW GLOBAL DRONE HUD
        draw_forensic_hud(base_frame, frame_width, frame_height, elapsed)

        # 8. GLOBAL EFFECTS (Scanline & Glitch Noise)
        scanline_y += 8
        if scanline_y > frame_height:
            scanline_y = 0
        # Thickened scanline
        cv2.line(base_frame, (0, scanline_y), (frame_width, scanline_y), (0, 255, 0), 2)

        # Apply noise to make it feel like a raw, gritty camera feed
        noise = np.random.randint(0, 30, base_frame.shape, dtype='uint8')
        base_frame = cv2.add(base_frame, noise)
        
        cv2.imshow("Omnipotent Funky Vision", base_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()