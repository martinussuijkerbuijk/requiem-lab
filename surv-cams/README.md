# REQUIEM — Surveillance Art System

A PTZ camera-based art installation that autonomously hunts, selects, and ritually processes faces in a crowd. The system pairs a Reolink IP camera with computer-vision overlays to perform a repeating five-stage ceremony on anyone it detects.

---

## Requirements

- Python 3.11+ (virtual environment at `requiem/bin/python3`)
- Reolink PTZ IP camera accessible on the local network
- A camera preset saved in the Reolink app pointing at your audience (optional but recommended)
- `input/Test_mask_v1.png` — transparent PNG used in the BLESS stage (optional; halo-only if missing)

### Python dependencies

```
opencv-python
mediapipe
numpy
ultralytics
huggingface_hub
reolinkapi
```

Install into the venv:

```bash
uv pip install --python requiem/bin/python3 \
    opencv-python mediapipe numpy ultralytics huggingface_hub reolinkapi
```

The YOLOv8 face-detection model (`arnabdhar/YOLOv8-Face-Detection`) is downloaded automatically from HuggingFace on first run.

---

## Files

| File | Role |
|---|---|
| `requiem.py` | Main application — state machine, rendering, CLI |
| `camera_controller.py` | Reolink camera wrapper — streaming, PTZ, zoom, position estimator |
| `run_scan.py` | Source of `draw_halo`, `draw_mask` used in BLESS |
| `run_analysis.py` | Source of `draw_targeting_brackets`, `draw_forensic_hud`, `get_distance` used in ZOOM/ANALYZE |
| `input/Test_mask_v1.png` | Transparent PNG mask overlaid on face during BLESS |

---

## Usage

```bash
./requiem/bin/python3 requiem.py [OPTIONS]
```

### Launch options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--ip` | string | `192.168.1.126` | Camera IP address |
| `--user` | string | `admin` | Camera username |
| `--password` | string | `requiem-2026` | Camera password |
| `--profile` | `main` \| `sub` | `main` | Stream profile. `main` = high quality, `sub` = lower resolution / lower latency |
| `--start` | flag | off | Begin the automated sequence immediately on launch without waiting for a CLI command |
| `--pan-range` | degrees | `120` | Total horizontal sweep range. Camera stays within ±half this value from its starting position (e.g. 120° → ±60°) |
| `--tilt-range` | degrees | `90` | Total vertical sweep range. Camera stays within ±half this value (e.g. 90° → ±45°) |
| `--home-preset` | integer | `0` | Move camera to this saved preset index before starting the stream. `0` = skip. Use to ensure the camera always faces the audience. |
| `--width` | pixels | `1280` | Display window width (clamped 320–1920) |
| `--height` | pixels | `720` | Display window height (clamped 240–1080) |

### Examples

Minimal launch — connect and wait for a CLI command:
```bash
./requiem/bin/python3 requiem.py
```

Start the sequence automatically, camera faces audience via preset 1:
```bash
./requiem/bin/python3 requiem.py --home-preset 1 --start
```

Custom display size and narrower sweep range:
```bash
./requiem/bin/python3 requiem.py --width 960 --height 540 --pan-range 80
```

Low-resolution stream for faster frame delivery:
```bash
./requiem/bin/python3 requiem.py --profile sub --start
```

---

## State machine

The system moves through six states in a continuous loop:

```
VIGIL → HUNT → CENTER → ZOOM → ANALYZE → BLESS
                 ↑_________________________________|
```

### VIGIL
Idle. The camera does not move. The display shows `REQUIEM // STANDBY`. No detection runs. Waits for the `start` CLI command.

### HUNT
The camera performs randomised patrol sweeps across the audience. YOLOv8 detects faces every 4 frames. All detected faces are outlined with green rectangles and sci-fi corner brackets. The sweep direction is weighted to keep the camera within the configured pan/tilt range.

When at least one face is detected, the system selects a target using weighted random selection (larger, closer faces are more likely) and transitions to CENTER.

### CENTER
The camera pans to align the selected face with the frame crosshair. A pulsing targeting bracket tracks the face. A line connects the face centre to the frame crosshair.

- If the face leaves the frame for more than 25 consecutive frames, or centering takes longer than 10 seconds, the system returns to HUNT.
- Once the face is within the dead zone (45 px from centre), a hold bar fills along the bottom of the frame. After 1.5 seconds of stable lock, the camera stops panning and the system transitions to ZOOM.

### ZOOM
Hardware zoom starts once (Reolink `start_zooming_in`, speed 22). Simultaneously, digital zoom (software crop + resize) increases from 1.0× to 2.5× over approximately 5 seconds. The forensic analysis overlay from `run_analysis.py` is rendered live during this phase, including the drone HUD and face mesh. A progress label (`ZOOM // N%`) tracks the zoom level.

No pan commands are issued during ZOOM — pan and zoom share the same Reolink API channel and cannot run simultaneously.

When digital zoom reaches 2.5×, hardware zoom stops and the system transitions to ANALYZE.

### ANALYZE
The frame is held at maximum digital zoom (2.5×) centred on the locked face. The full forensic overlay renders:
- MediaPipe 468-point face mesh (eyes, lips, iris crosshairs)
- Psychedelic HSV face ROI (hue-shifted, fully saturated)
- Drone HUD (roll arc, centre reticle, pitch ladder, four telemetry scales)
- Pseudo-emotion classification: COMPLIANT / SHOCKED / DEFENSIVE / FEAR
- Per-face metrics: pupil aperture ratio (EAR), mouth opening distance

A progress bar counts down 7 seconds, then transitions to BLESS.

### BLESS
The frame is held at maximum digital zoom. A warm golden tone is applied to the image. A glowing gold/yellow elliptical halo is drawn above the head. If `input/Test_mask_v1.png` is present it is overlaid on the face with alpha blending.

After 5 seconds, the camera zooms back out to minimum, a log entry is written to the session ledger, and the system returns to HUNT to select the next subject.

---

## CLI commands

Type commands at any time while the application is running. Commands are processed between frames.

| Command | Effect |
|---|---|
| `start` | Enter HUNT and begin the automated sequence |
| `hunt` | Alias for `start` |
| `stop` | Stop all movement and zoom, clear target, return to VIGIL |
| `analyze` | Force-transition to ANALYZE immediately (requires an active target) |
| `bless` | Force-transition to BLESS immediately |
| `home` | Move camera to saved preset 1 (useful to re-centre between runs) |
| `reset` | Zoom out to minimum, stop movement, clear target, return to VIGIL |
| `status` | Print current state, number of detected faces, active target bounding box, and the last 3 blessed log entries |
| `help` | List all commands |
| `quit` / `q` | Shut down cleanly — stops camera, closes window, exits |

Press `q` in the OpenCV display window as an alternative to the `quit` command.

---

## Camera position management

The Reolink API provides no absolute position feedback. Position is estimated via dead-reckoning: `speed × duration × DEG_PER_SPEED_PER_SEC` (default calibration: speed 25 ≈ 1°/s).

The `--pan-range` and `--tilt-range` flags set the total sweep window. All patrol sweeps and face-centering moves respect these limits. When the estimated position approaches a limit, sweep directions are weighted away from it.

To set the audience-facing centre position:
1. Aim the camera at the audience centre using the Reolink app.
2. Save it as a preset (e.g. preset 1) in the app.
3. Launch with `--home-preset 1`. The camera will move to that preset before streaming begins, making it the dead-reckoning origin.

---

## Tuning constants (in source files)

### `camera_controller.py`

| Constant | Default | Meaning |
|---|---|---|
| `DEAD_ZONE_PX` | `45` | Pixels from frame centre before a PTZ correction is issued |
| `PAN_SPEED_MIN` | `10` | Minimum pan speed for face centering |
| `PAN_SPEED_MAX` | `50` | Maximum pan speed for face centering |
| `PTZ_THROTTLE_S` | `0.5` | Minimum seconds between consecutive PTZ correction commands |
| `ZOOM_SPEED_TRACK` | `22` | Hardware zoom speed during ZOOM state |
| `ZOOM_SPEED_RESET` | `60` | Hardware zoom-out speed after BLESS |
| `SWEEP_SPEEDS` | `[12,18,25,30]` | Patrol sweep speed candidates |
| `SWEEP_DURATIONS` | `(0.4, 1.8) s` | Per-burst movement duration range |
| `SWEEP_PAUSES` | `(0.5, 2.0) s` | Stillness duration between sweep bursts |
| `DEG_PER_SPEED_PER_SEC` | `1.0` | Calibration factor for the dead-reckoning estimator |
| `DEFAULT_PAN_LIMIT` | `60°` | Half of the default `--pan-range` |
| `DEFAULT_TILT_LIMIT` | `45°` | Half of the default `--tilt-range` |

### `requiem.py`

| Constant | Default | Meaning |
|---|---|---|
| `CENTER_TIMEOUT` | `10.0 s` | Abandon centering after this long |
| `CENTER_HOLD_S` | `1.5 s` | Face must stay in dead zone this long before zoom begins |
| `CENTER_LOST_LIMIT` | `25 frames` | Consecutive missed detections before returning to HUNT |
| `ZOOM_TIMEOUT` | `20.0 s` | Abandon zoom after this long |
| `ZOOM_LOST_LIMIT` | `40 frames` | Missed detections before re-hunting during zoom |
| `ANALYZE_DURATION` | `7.0 s` | Duration of the ANALYZE overlay |
| `BLESS_DURATION` | `5.0 s` | Duration of the BLESS holy filter |
| `ZOOM_OUT_DURATION` | `3.0 s` | Wait for camera to zoom out before re-hunting |
| `DETECT_EVERY_N` | `4` | Run YOLO every N frames (CPU budget control) |
| `ZOOM_STEP` | `0.010` | Digital zoom factor added per frame |
| `MAX_DIGITAL_ZOOM` | `2.5×` | Digital zoom level that triggers transition to ANALYZE |
