# Model Comparison Analysis — YOLOv8n-VisDrone (Pretrained) vs Run 2 Finetuned

## Overview

Comparison of two models on the same aerial video (REC_0009.mp4) under identical conditions.

| Parameter | Value |
|---|---|
| Video | REC_0009.mp4 (~1:49 mins, 2,709 frames @ 25fps) |
| Conf threshold | 0.25 |
| Mode | Classification (Mode 2) |
| Hardware | Mac M3 Pro (CPU inference) |

---

## Models

| Model | Classes | Training Data | mAP50 |
|---|---|---|---|
| Run 2 Finetuned (best.pt) | 2 (person, car) | VisDrone VID + DET, 30,672 images | 0.579 |
| HuggingFace Pretrained (yolov8n-visdrone.pt) | 10 (full VisDrone) | VisDrone (unknown split) | 0.341 |

---

## Inference Performance

| Metric | Run 2 Finetuned | HuggingFace Pretrained |
|---|---|---|
| Avg inference | 23.5ms | 22.5ms |
| Avg FPS | 42.5 | 44.4 |
| Min inference | 20.0ms | 19.9ms |
| Max inference | 775.5ms | 980.0ms |

Both models run at comparable speed (~22-23ms avg). The pretrained model is marginally faster (~2 FPS) likely due to fewer post-processing steps with 10 classes vs the same architecture.

---

## Detection Counts Comparison

### Person Detection

| Metric | Run 2 Finetuned | HuggingFace Pretrained |
|---|---|---|
| Person total | 3,865 | 2,453 (pedestrian) + 1,593 (people) = **4,046** |
| Person avg conf | 0.439 | 0.395 (pedestrian) / 0.369 (people) |
| Person per frame | 1.43 | 0.91 + 0.59 = **1.50** |

The pretrained model detects slightly more people per frame (1.50 vs 1.43) by splitting them into `pedestrian` and `people` categories. However average confidence is lower (0.395/0.369 vs 0.439), suggesting more borderline detections.

### Car Detection

| Metric | Run 2 Finetuned | HuggingFace Pretrained |
|---|---|---|
| Car total | 12,369 | 13,155 |
| Car avg conf | 0.595 | 0.585 |
| Car per frame | 4.57 | 4.86 |

Car detection is comparable — the pretrained model detects slightly more cars per frame (4.86 vs 4.57) at similar confidence.

### Additional Classes (Pretrained Only)

| Class | Total | Avg Conf | Per Frame |
|---|---|---|---|
| van | 2,211 | 0.467 | 0.82 |
| truck | 983 | 0.519 | 0.36 |
| motor | 530 | 0.401 | 0.20 |
| bus | 301 | 0.438 | 0.11 |
| bicycle | 111 | 0.370 | 0.04 |
| tricycle | 59 | 0.384 | 0.02 |
| awning-tricycle | 17 | 0.322 | 0.01 |

The pretrained model detects a full range of vehicle and person types that the finetuned model completely ignores. Van detection (0.82/frame) is notable — these were being classified as cars in the finetuned model, inflating its car count.

---

## Key Observations

**1. Person detection — comparable with different granularity**
The pretrained model splits people into `pedestrian` (individual, clearly visible) and `people` (groups, partially occluded) — 4,046 total vs 3,865 from the finetuned model. Slightly more detections but lower confidence per detection.

**2. Car detection — pretrained slightly better**
13,155 vs 12,369 total car detections. The pretrained model also separately detects vans (2,211), trucks (983), and buses (301) which were being lumped into the car class by the finetuned model. This means the finetuned model's car count was inflated by vans and trucks.

**3. Confidence — finetuned model wins**
Run 2 finetuned has higher avg confidence for both person (0.439 vs 0.395) and car (0.595 vs 0.585). This is expected — a model trained specifically on 2 classes becomes more decisive on those classes.

**4. mAP paradox**
Despite the pretrained model having lower mAP50 (0.341 vs 0.579), it detects more objects per frame. This is because mAP measures precision-recall tradeoff across all classes — with 10 classes including rare ones (tricycle, awning-tricycle), overall mAP is pulled down even if common class detection is strong.

---

## Summary

| Aspect | Winner | Reason |
|---|---|---|
| Inference speed | Pretrained (44.4 FPS) | Marginally faster |
| Person detection count | Pretrained (4,046 vs 3,865) | Detects more people |
| Car detection count | Pretrained (13,155 vs 12,369) | More car detections |
| Detection confidence | Finetuned (0.439/0.595) | More decisive on 2 classes |
| Class diversity | Pretrained | 10 classes vs 2 |
| mAP50 | Finetuned (0.579 vs 0.341) | Focused 2-class training |

---

## Recommendation

For the UAV tracking use case (person + car following):

- **Use the pretrained model** if the goal is broader scene understanding — it detects more object types and slightly more people and cars per frame
- **Use the finetuned Run 2 model** if the goal is high-confidence person and car detection specifically — it is more decisive and has higher mAP on those two classes

For deployment on Jetson with TensorRT, both can be exported to `.engine` format. The pretrained model's additional classes add minimal overhead.