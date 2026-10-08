# UAV Autonomous Tracking

A modular C++ autonomous perception and flight control pipeline for onboard UAV applications. The system uses YOLOv8n for real-time object detection, a custom ByteTrack implementation with Kalman filtering for multi-object tracking, and MAVSDK for PX4 flight control.

---

## System Architecture

```
Mac / Windows (Development)
├── C++ UAV App
│   ├── Capture      — webcam / RTSP / video file / image sequence feed
│   ├── Detector     — YOLOv8n ONNX inference via OpenCV DNN
│   ├── ByteTracker  — multi-object tracking with Kalman filter
│   └── Commander    — MAVSDK flight control (arm, offboard, follow, land)
└── QGroundControl ← udp://:14550

Mac Terminal / WSL2 (Ubuntu 22.04)
└── PX4 SITL v1.18.0-alpha1 + Gazebo Harmonic
    └── MAVLink → 127.0.0.1

Jetson Orin (Deployment)
├── C++ UAV App (same codebase, CUDA backend)
└── PX4 on real hardware via serial/UDP
```

---

## Repositories

| Repo | Contents |
|---|---|
| [uav-autonomous-tracking](https://github.com/Anvi2801-dot/uav-autonomous-tracking) | C++ UAV app |
| [px4_custom_files](https://github.com/Anvi2801-dot/px4_custom_files) | Custom airframe, Gazebo world, PX4 config |

---

## Project Structure

```
uav/
├── src/
│   ├── main.cpp
│   ├── Capture.cpp/.hpp        — camera acquisition
│   ├── Detector.cpp/.hpp       — YOLO inference (dynamic class count)
│   ├── Commander.cpp/.hpp      — flight state machine
│   └── tracking/
│       ├── Tracker.hpp         — abstract interface (swap ByteTrack ↔ BotSort here)
│       ├── Track.hpp
│       ├── KalmanFilter.hpp/.cpp
│       └── ByteTracker.hpp/.cpp
├── include/
├── models/                     — gitignored, download from Releases
│   ├── best.onnx               — Run 2 finetuned (person + car)
│   ├── best.pt                 — Run 2 finetuned weights
│   ├── best.names              — 2-class names: person, car
│   ├── yolov8n-visdrone.onnx   — HuggingFace pretrained (10-class, remapped to 2)
│   ├── yolov8n-visdrone.pt     — HuggingFace pretrained weights
│   ├── visdrone.names          — display names for pretrained model (person, car)
│   ├── yolov8n.onnx            — baseline COCO pretrained
│   └── coco.names              — 80 COCO class labels
├── scripts/
│   ├── visdrone_to_yolo.py     — converts VisDrone VID + DET to YOLO format
│   ├── uavdt_to_yolo.py        — converts UAVDT to YOLO format
│   ├── train_visdrone.py       — YOLOv8n finetuning pipeline
│   ├── eval_visdrone.py        — VisDrone evaluation (precision/recall/F1)
│   ├── infer_video.py          — raw video inference + analytics
│   ├── bytetrack_sweep.py      — ByteTrack parameter sweep analysis
│   └── requirements.txt
├── analysis.md                 — Run 1 finetuning analysis
├── analysis2.md                — Run 2 finetuning analysis
├── CMakeLists.txt
├── DEPENDENCIES.md
└── README.md
```

---

## Models

Download model files from the [GitHub Releases](https://github.com/Anvi2801-dot/uav-autonomous-tracking/releases) page and place them in `uav/models/`.

| File | Description | Classes |
|---|---|---|
| `best.pt` | Run 2 finetuned on VisDrone VID + DET | person, car |
| `best.onnx` | ONNX export of best.pt for C++ deployment | person, car |
| `yolov8n-visdrone.pt` | HuggingFace pretrained ([mshamrai/yolov8n-visdrone](https://huggingface.co/mshamrai/yolov8n-visdrone)) | 10-class VisDrone |
| `yolov8n-visdrone.onnx` | ONNX export of pretrained model | 10-class → remapped to person, car |
| `yolov8n.onnx` | Baseline COCO pretrained (download from Ultralytics) | 80 COCO classes |

### Switching Models

In `main.cpp`, change the Detector line and rebuild:

```cpp
// Run 2 finetuned (person + car):
Detector detector("../models/best.onnx", "../models/best.names");

// HuggingFace pretrained (10-class remapped to person + car):
// Detector detector("../models/yolov8n-visdrone.onnx", "../models/visdrone.names", 10);
```

---

## Platform Stack

| Component | Mac (dev) | Windows (dev) | Jetson (deploy) |
|---|---|---|---|
| Compiler | Clang (Homebrew) | MSVC (VS 2022) | GCC (apt) |
| MAVSDK | 3.17.1 Homebrew | 3.17.1 pre-built MSVC | 3.17.1 from source |
| OpenCV | 4.13.0 Homebrew | 4.12.0 pre-built MSVC | Built from source + CUDA |
| YOLO backend | DNN_BACKEND_OPENCV | DNN_BACKEND_OPENCV | DNN_BACKEND_CUDA |
| YOLO target | DNN_TARGET_CPU | DNN_TARGET_CPU | DNN_TARGET_CUDA |
| PX4 | Native SITL | WSL2 SITL | Real hardware |
| Gazebo | Native Harmonic | WSL2 Harmonic | N/A |

---

## Performance

| Metric | Mac (M3 Pro) | Windows | Jetson |
|---|---|---|---|
| Inference | ~37–55ms | ~350ms | ~20–40ms |
| FPS | 17–28 | 2 | 15–25 |
| Backend | OpenCV CPU | OpenCV CPU | OpenCV CUDA |

---

## Prerequisites

### C++ (all platforms)
- CMake ≥ 3.16
- OpenCV 4.x
- MAVSDK 3.17.1

### Mac (Homebrew)
```bash
brew install cmake opencv mavsdk
```

### Jetson
- JetPack 6.x (includes CUDA, cuDNN, TensorRT)
- OpenCV built from source with CUDA
- MAVSDK built from source

### Python scripts (training/eval pipeline)
```bash
pip3 install -r scripts/requirements.txt
```

---

## Build

### macOS
```bash
cd ~/uav_workspace/uav
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.logicalcpu)
```

### Windows (x64 Native Tools Command Prompt for VS 2022)
```bash
cd C:\Users\anvi.s\uav_workspace\uav\build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Jetson
```bash
cd ~/uav_workspace/uav
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## Running the Project

### Step 1 — Launch PX4 SITL

**Mac:**
```bash
cd ~/uav_workspace/PX4-Autopilot
make px4_sitl gz_x500_lidar_custom
```

**WSL2 (Windows):**
```bash
cd /home/anvis/PX4-Autopilot
make px4_sitl gz_x500_lidar_custom
```

Wait for `Ready for takeoff!` and `pxh>` prompt. Then wait an additional **15 seconds** before running the UAV app.

### Step 2 — Set failsafe params (first run only)
```
pxh> param set COM_RCL_EXCEPT 4
pxh> param set NAV_DLL_ACT 0
pxh> param set NAV_RCL_ACT 0
pxh> param save
```

### Step 3 — Broadcast MAVLink (Windows/WSL only)
```
pxh> mavlink start -u 14540 -t 127.0.0.1 -r 4000000 -m onboard
```
Mac uses localhost by default — skip this step.

### Step 4 — Launch QGroundControl (Windows)
Open QGC → UDP Comm Link on port 14550. Disable UDP AutoConnect in General settings.

### Step 5 — Run UAV App

**Mac / Jetson:**
```bash
cd ~/uav_workspace/uav/build && ./uav
```

**Windows:**
```bash
cd C:\Users\anvi.s\uav_workspace\uav\build && .\uav.exe
```

### Step 6 — Startup prompts

**Source selection:**
```
[1] Launch with Live Webcam Feed
[2] Run via Pre-recorded Reference Video File
[3] Run via Image Sequence Folder (e.g. VisDrone)
Select input option (1, 2 or 3):
```

**Mode selection:**
```
[1] Detection      - bounding boxes only, no drone
[2] Classification - bounding boxes + labels, no drone
[3] Tracking       - boxes + labels + drone follows target
Enter mode (1, 2 or 3):
```

**Detection config:**
```
Enter classes to detect (comma separated, or 'all'): person, car
Enter class to TRACK/FOLLOW (drone follows this): person   ← mode 3 only
```

### Runtime Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `l` | Manual land |
| — | Auto-lands after 20s with no target |

---

## Vision Pipeline

```
Camera frame (640x480)
  → YOLOv8n ONNX inference (scoreThreshold=0.25, nmsThreshold=0.45)
  → NMS on raw detections
  → filter by user-selected classes
  → ByteTracker update
  → grace period (30 frames) before switching target
  → select lowest-ID confirmed track (hit_streak ≥ 3)
  → lock onto target ID
  → Commander.processTarget()
```

### HUD Color Coding
- 🔵 Blue — person (all modes, always)
- 🟢 Green — non-person follow target (mode 3)
- 🟠 Orange — other detected classes (mode 3)
- 🔴 Red dot — center of follow target (mode 3)
- Centre guide lines shown in mode 3 only

---

## YOLOv8n Finetuning on VisDrone

### Run 1 — 5 Classes, VID Only (baseline finetuning)

| Model | Precision | Recall | F1 | mAP50 |
|---|---|---|---|---|
| Baseline yolov8n.pt (COCO) | 0.679 | 0.237 | 0.351 | — |
| Run 1 best.pt (conf=0.25, iou=0.3) | 0.643 | 0.456 | **0.533** | 0.312 |

Best epoch: 7 — overfit due to insufficient data (56 sequences).

### Run 2 — 2 Classes, VID + DET (current model)

| Model | mAP50 | Best Epoch | Overfit |
|---|---|---|---|
| Baseline yolov8n.pt | — | — | — |
| Run 1 (5 classes, VID only) | 0.312 | 7 | Yes |
| **Run 2 (2 classes, VID + DET)** | **0.579** | **38** | **No** |

Dataset: 30,672 train images (VisDrone VID train + DET train), 3,394 val images. Trained on Google Colab T4 GPU (~15 min/epoch).

### HuggingFace Pretrained Comparison

| Model | F1 (VisDrone test-dev) | Classes |
|---|---|---|
| Run 2 finetuned | — | 2 (person, car) |
| HuggingFace pretrained | 0.2 | 10 (full VisDrone) |

Run 2 finetuned outperforms the HuggingFace pretrained model on VisDrone evaluation despite having fewer classes, due to focused 2-class training on a larger dataset.

### Finetuning Pipeline

```bash
# Convert VisDrone VID + DET to YOLO format
python3 scripts/visdrone_to_yolo.py \
    --dataset /path/to/VisDrone2019-VID-train \
    --output  /path/to/visdrone_yolo --split train

# Train
python3 scripts/train_visdrone.py \
    --data /path/to/visdrone_yolo/visdrone.yaml \
    --model yolov8n.pt \
    --epochs 50 --batch 16 --imgsz 640 --patience 15
```

---

## ByteTrack Configuration

```cpp
std::unique_ptr<Tracker> tracker = std::make_unique<ByteTracker>(
    0.4f,   // high confidence threshold
    0.15f,  // low confidence threshold
    0.5f,   // IoU match threshold
    80      // max lost frames (~4-5s at 17fps)
);
```

**Target switching logic:** locked target is held for 120 frames after loss before switching to the next lowest-ID confirmed track.

---

## Jetson Deployment

```bash
# Transfer best.pt to Jetson via scp over Tailscale
scp best.pt nvidia@<jetson-tailscale-ip>:~/uav_workspace/uav/models/

# Export to TensorRT on Jetson (run once)
python3 -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='engine', device=0, half=True)
"
# Produces best.engine — update Detector.cpp to load this instead of best.onnx
```

### OpenCV from source (CUDA backend)
```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_CUDA=ON \
      -D WITH_FFMPEG=OFF \
      -D WITH_GSTREAMER=ON \
      -D CUDA_ARCH_BIN="<jetson compute capability>" \
      ..
make -j$(nproc)
sudo make install && sudo ldconfig
```

### MAVSDK from source
```bash
git clone https://github.com/mavlink/MAVSDK.git
cd MAVSDK && git checkout v3.17.1
git submodule update --init --recursive
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j4
sudo cmake --install build
```

### RTSP Camera (SIYI via WiFi hotspot)
```
rtsp://192.168.144.25:8554/main.264
```

---

## WSL2 Setup (Windows)

**`~/.bashrc`:**
```bash
export GZ_IP=127.0.0.1
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
```

**`C:\Users\anvi.s\.wslconfig`:**
```
[wsl2]
memory=4GB
swap=8GB
processors=2
networkingMode=mirrored
```

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| OOM during PX4 build | `JOBS=1`, `swap=8GB` in `.wslconfig`, close all apps |
| Gazebo blank white window | `LIBGL_ALWAYS_SOFTWARE=1` in `~/.bashrc` |
| Gazebo window not opening | `pkill -f "gz sim" && pkill -f px4`, relaunch |
| Airframe not found after build | Copy airframe into build rootfs manually |
| Command 520 unsupported | Harmless — PX4 v1.18/MAVSDK v3.17 mismatch, drone still arms |
| ODOMETRY warning spam | `make px4_sitl ... 2>&1 \| grep -v "ODOMETRY"` |
| Drone disarms pre-offboard | Increase sleep after armable to 10s |
| Takeoff timeout | Increase to 60s, climb speed -1.5 m/s |
| Sensor missing warnings | Wait 15s after `Ready for takeoff` before running app |
| MAVSDK not discovering (SITL) | Use `udpin://0.0.0.0:14540`, wait 15s after PX4 ready |
| MAVSDK not discovering (real HW) | Use `udpout://192.168.110.141:14580` |
| ByteTrack ID jumping | `lockedTargetId` + grace period (30 frames) + `hit_streak≥3` |
| Target snapping to wrong object | Grace period of 30 frames before switching |
| Auto-land too aggressive | `NO_TARGET_TIMEOUT_SEC=20` |
| Duplicate boxes | NMS on raw detections before tracker |
| Ghost boxes after object leaves | Skip drawing Lost state tracks |
| Non-person tracking unstable | YOLOv8n limitation — use larger model |
| OpenCV FFmpeg build error on Jetson | `-D WITH_FFMPEG=OFF -D WITH_GSTREAMER=ON` |
| `mavsdk.dll` not found (Windows) | Add `C:\mavsdk\release\bin` to system PATH |
| LiDAR reads ~29m on ground | Expected — ground plane in `default.sdf` is ~29m below spawn |
| Pretrained model class mismatch | Pass `numModelClasses=10` to Detector constructor |
| PX4 instance already running | `pkill -f px4 && pkill -f "gz sim"` before relaunching |

---
