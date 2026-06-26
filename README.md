# UAV Autonomous Tracking

This repository contains a modular C++ autonomous perception pipeline designed for onboard UAV applications. The system utilizes YOLOv8 for real-time target detection and a state-machine-based commander to generate flight setpoints.

## Key Features
- **Real-Time Perception:** YOLOv8n inference via OpenCV DNN.
- **Multi-Axis Tracking:** Coordinated logic for Yaw (centering) and Pitch (distance maintenance).
- **Control Stability:** 10-frame moving average filter to eliminate command flickering.
- **Modular Architecture:** Distinct separation between Camera Acquisition, AI Detection, and Flight Logic.

## Performance (MacBook M3 Pro)
- **Inference Latency:** ~37ms 
- **Control Frequency:** ~28 FPS
- **Architecture:** ARM64 (Optimized for transition to NVIDIA Jetson)

## Project Structure
- `src/Capture.cpp`: Handles camera acquisition and frame mirroring.
- `src/Detector.cpp`: Manages YOLO inference and target centroid calculation.
- `src/Commander.cpp`: Navigation state machine and smoothing filters.
- `src/main.cpp`: Orchestrates the real-time perception-control loop.

## Tech Stack
- **Language:** C++17
- **Library:** OpenCV 4.x
- **Model:** YOLOv8n (ONNX)
- **Build System:** CMake

## How to Build
Ensure OpenCV 4.x is installed on your system.

```bash
mkdir build && cd build
cmake ..
make
./uav
