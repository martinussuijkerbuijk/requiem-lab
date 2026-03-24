import cv2
import numpy as np
import time
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def draw_halo(frame, x1, y1, x2, y2):
    """
    Draws a glowing halo effect above the detected face bounding box.
    """
    # Calculate dimensions based on face width
    face_width = x2 - x1

    base_thickness = max(1, int(face_width * 0.06))
    
    halo_width = int(face_width * 1.2)
    halo_height = int(face_width * 0.3)
    
    # Position the halo slightly above the top of the bounding box
    center_x = int((x1 + x2) / 2)
    center_y = y1 - int(face_width * 0.2)
    
    # We create a separate overlay to handle transparency (alpha blending)
    overlay = frame.copy()
    
    # Colors for the glow (Golden/Yellow)
    # BGR format: (Blue, Green, Red)
    outer_color = (0, 200, 255) # Lighter gold
    inner_color = (0, 255, 255) # Bright yellow
    
    # ERROR FIX: Ensure axes are never negative to prevent OpenCV assertion errors
    # This happens if the face is too small or near the edge.
    axis_major = max(1, halo_width // 2)
    axis_minor = max(1, halo_height // 2)
    
    # Draw multiple ellipses to create a 'glow' gradient effect
    # --- DYNAMIC DRAWING ---
    # Outer glow: Uses the full base_thickness
    cv2.ellipse(overlay, (center_x, center_y), (axis_major, axis_minor), 
                0, 0, 360, outer_color, base_thickness, cv2.LINE_AA)
    
    # Middle ring: Usually half as thick as the outer glow
    mid_thickness = max(1, base_thickness // 2)
    cv2.ellipse(overlay, (center_x, center_y), (axis_major, axis_minor), 
                0, 0, 360, inner_color, mid_thickness, cv2.LINE_AA)
    
    # Inner bright core: Always thin (1-2px) to keep the "edge" sharp
    core_thickness = max(1, mid_thickness // 2)
    core_major = max(1, axis_major - core_thickness)
    core_minor = max(1, axis_minor - core_thickness)
    cv2.ellipse(overlay, (center_x, center_y), (core_major, core_minor), 
                0, 0, 360, (255, 255, 255), core_thickness, cv2.LINE_AA)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# MONKEY PATCH for NumPy 1.24+
if not hasattr(np, 'bool'):
    np.bool = bool

def overlay_transparent(background, overlay, x, y, size=None):
    """
    Overlays a transparent PNG onto a background image at (x, y).
    """
    background_width, background_height = background.shape[1], background.shape[0]
    
    if size is not None:
        overlay = cv2.resize(overlay, (size, size), interpolation=cv2.INTER_AREA)
    
    h, w = overlay.shape[:2]

    # Boundary check to prevent crashing if face is near edge
    if x + w > background_width: w = background_width - x
    if y + h > background_height: h = background_height - y
    if x < 0: x = 0
    if y < 0: y = 0
    if w <= 0 or h <= 0: return background

    overlay_image = overlay[:h, :w, 0:3]
    mask = overlay[:h, :w, 3] / 255.0

    # Apply alpha blending
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (1.0 - mask) * background[y:y+h, x:x+w, c] + mask * overlay_image[:, :, c]

    return background


def draw_mask(frame, x1, y1, x2, y2, mask_img):
    """
    Scales and places the mask image onto the face area.
    """
    face_width = x2 - x1
    
    # Scale the mask to be slightly wider than the face
    mask_size = int(face_width * 1.3)
    
    # Calculate top-left corner to center the mask over the face
    # We shift it up slightly so it aligns with eyes/forehead
    pos_x = int((x1 + x2) / 2 - mask_size / 2)
    pos_y = int(y1 - (mask_size * 0.1)) 
    
    overlay_transparent(frame, mask_img, pos_x, pos_y, size=mask_size)


def main():
    print("--- JETSON HALO/MASK FACE DETECTOR ---")

    # 1. Load Mask Image (Ensure you have a 'mask.png' in your folder)
    # Using a 4-channel read (IMREAD_UNCHANGED) is CRITICAL
    mask_path = "input/Test_mask_v1.png" 
    if not os.path.exists(mask_path):
        print(f"Error: {mask_path} not found. Please place a transparent PNG named mask.png in the directory.")
        # Create a dummy colored square if mask is missing so the script doesn't crash
        mask_img = np.zeros((256, 256, 4), dtype=np.uint8)
        mask_img[:,:,1] = 255; mask_img[:,:,3] = 128 # Semi-transparent green square
    else:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    
    # Download model if not present (using your specific face detector)
    try:
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        print(f"Model loaded from: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return

    # Load Model
    # On Jetson, 'device=0' utilizes the integrated Maxwell/Pascal/Xavier GPU
    model = YOLO(model_path)

    # Camera Setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Set resolution (Lower resolution = Higher FPS on Jetson Nano)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_time = 0

    print("Starting Inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
   
        # Inference
        # verbose=False keeps the console clean
        # stream=True is more memory efficient for video
        results = model(frame, verbose=False, conf=0.5, device=0, stream=True)
        
        for result in results:
            for box in result.boxes:
                # Extract coordinates
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                
                # Apply the Halo Effect
                draw_halo(frame, x1, y1, x2, y2)
                draw_mask(frame, x1, y1, x2, y2, mask_img)
                
                # Optional: Subtle bounding box
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Calculate and Display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # UI Overlay
        cv2.rectangle(frame, (0, 0), (150, 40), (0, 0, 0), -1) # Background for FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Halo Filter - Jetson", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()