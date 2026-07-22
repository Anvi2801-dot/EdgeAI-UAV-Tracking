# UAV Video Inference Analysis — REC_0009.mp4

## Test Configuration

| Parameter | Value |
|---|---|
| Video | REC_0009.mp4 |
| Duration | ~1:49 mins |
| Total frames | 2,709 |
| Video FPS | 25.0 |
| Model | YOLOv8n finetuned — 2 classes |
| Confidence threshold | 0.25 |
| Hardware | Mac M3 Pro (CPU inference) |

---

## Inference Performance

| Metric | Classification | Tracking |
|---|---|---|
| Avg inference | 23.5ms | 23.8ms |
| Avg FPS | 42.5 | 42.1 |
| Min inference | 20.0ms | 20.4ms |
| Max inference | 775.5ms | 2653.1ms |

---

## Detection Counts

| Class | Classification | Tracking |
|---|---|---|
| Person — total detections | 3,865 | 1,781 |
| Person — avg confidence | 0.439 | 0.546 |
| Person — per frame | 1.43 | 0.66 |
| Car — total detections | 12,369 | 10,024 |
| Car — avg confidence | 0.595 | 0.649 |
| Car — per frame | 4.57 | 3.70 |

---

## Tracking Analytics 

| Metric | Value |
|---|---|
| Unique track IDs assigned | 530 |
| Person tracks | 160 |
| Car tracks | 370 |
| Avg track lifetime | 22.2 frames (0.9s) |
| Max track lifetime | 618 frames (24.7s) |
| Target switches | 823 |

---
