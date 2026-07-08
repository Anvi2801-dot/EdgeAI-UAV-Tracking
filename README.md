# UAV Autonomous Tracking

A modular C++ autonomous perception and flight control pipeline for onboard UAV applications. The system uses YOLOv8n for real-time object detection, a custom ByteTrack implementation with Kalman filtering for multi-object tracking, and MAVSDK for PX4 flight control.

---

## System Architecture

```
Mac / Windows (Development)
├── C++ UAV App
│   ├── Capture      — webcam / RTSP camera feed
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
│   ├── Detector.cpp/.hpp       — YOLO inference
│   ├── Commander.cpp/.hpp      — flight state machine
│   └── tracking/
│       ├── Tracker.hpp         
│       ├── Track.hpp
│       ├── KalmanFilter.hpp/.cpp
│       └── ByteTracker.hpp/.cpp
├── include/
├── models/
│   ├── yolov8n.onnx           
│   └── coco.names              
├── scripts/                    
│   ├── convert_visdrone_to_yolo.py
│   ├── train_visdrone.py
│   ├── eval_visdrone.py
│   └── requirements.txt
├── CMakeLists.txt
├── DEPENDENCIES.md
├── requirements.txt
└── README.md
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

| Metric | Mac (M3 Pro) | Windows | Jetson (expected) |
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
cd scripts/
pip3 install -r requirements.txt
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

Wait for `Ready for takeoff!` and `pxh>` prompt.

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
```
Select input option (1 or 2): 1
Enter classes to DETECT: person, car     ← or 'all'
Enter class to TRACK/FOLLOW: person
```

### Runtime Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `l` | Manual land |
| — | Auto-lands after 10s with no target |

---

## Vision Pipeline

```
Camera frame (640x480)
  → YOLOv8n ONNX inference
  → NMS on raw detections
  → filter by user-selected classes
  → ByteTracker update
  → select lowest-ID confirmed track
  → lock onto target ID
  → Commander.processTarget()
```

### HUD Color Coding
- 🟢 Green — follow target
- 🟠 Orange — same class as target, not followed
- 🔵 Blue — other detected classes
- 🔴 Red dot — center of follow target

---

## Fine-tuning YOLOv8n on VisDrone

The baseline `yolov8n.pt` (COCO-pretrained) has limited recall on small aerial objects (Precision=0.679, Recall=0.237, F1=0.351 on VisDrone-VID test-dev). A fine-tuning pipeline is included in `scripts/`.

---

## Jetson Setup

### OpenCV from source (with CUDA)
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

Verify:
```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda
# Should show: CUDA: YES
```

### MAVSDK from source (no ARM64 pre-built)
```bash
git clone https://github.com/mavlink/MAVSDK.git
cd MAVSDK && git checkout v3.17.1
git submodule update --init --recursive
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build -j4
sudo cmake --install build
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
| Airframe not found after build | Copy airframe into build rootfs manually |
| Command 520 unsupported | Harmless — PX4 v1.18/MAVSDK v3.17 mismatch, drone still arms |
| ODOMETRY warning spam | `make px4_sitl ... 2>&1 \| grep -v "ODOMETRY"` |
| Drone disarms pre-offboard | Increase sleep after armable to 10s |
| Takeoff timeout | Increase to 60s, climb speed -1.5 m/s |
| Sensor missing warnings | Wait 15s after `Ready for takeoff` before running app |
| ByteTrack ID jumping | `lockedTargetId` + `hit_streak≥3` + NMS on detections |
| Duplicate boxes | NMS on raw detections before tracker |
| Ghost boxes after object leaves | Skip drawing Lost state tracks |
| Non-person tracking unstable | YOLOv8n limitation — use larger model |
| OpenCV FFmpeg build error on Jetson | `-D WITH_FFMPEG=OFF -D WITH_GSTREAMER=ON` |
| `mavsdk.dll` not found (Windows) | Add `C:\mavsdk\release\bin` to system PATH |
| LiDAR reads ~29m on ground | Expected — ground plane in `default.sdf` is ~29m below spawn |
