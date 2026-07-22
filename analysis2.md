# YOLOv8n VisDrone Finetuning — Run 2 Analysis

## Overview

Second finetuning run addressing the limitations identified in Run 1. Key changes:
- Reduced from 5 classes → **2 classes (person + car)**
- Added **VisDrone-DET** dataset (+6,471 images) alongside VID
- Trained on **Google Colab T4 GPU**
- Early stopping patience set to **15 epochs**

---

## Dataset

| Split | Images | Notes |
|---|---|---|
| Train | 30,672 | VID train frames + DET train images |
| Val | 3,394 | VID val frames + DET val images |

Combined dataset is ~5x larger than Run 1 (56 VID sequences only).

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | yolov8n.pt (COCO pretrained) |
| Classes | 2 (person, car) |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 |
| Learning rate | 0.01 (auto-adjusted by Ultralytics) |
| Patience | 15 |
| Hardware | Tesla T4 GPU (Google Colab) |
| ~Time per epoch | ~15 minutes |

---

## Results

### mAP50 per Epoch

| Epoch | mAP50 | Epoch | mAP50 |
|---|---|---|---|
| 1 | 0.505 | 26 | 0.573 |
| 2 | 0.518 | 27 | 0.574 |
| 3 | 0.533 | 28 | 0.575 |
| 4 | 0.538 | 29 | 0.575 |
| 5 | 0.547 | 30 | 0.577 |
| 6 | 0.550 | 31 | 0.577 |
| 7 | 0.559 | 32 | 0.577 |
| 8 | 0.566 | 33 | 0.578 |
| 9 | 0.563 | 34 | 0.578 |
| 10 | 0.564 | 35 | 0.578 |
| 11 | 0.568 | 36 | 0.578 |
| 12 | 0.566 | 37 | 0.578 |
| 13 | 0.574 | 38 | 0.579 |
| 14 | 0.569 | 39 | 0.578 |
| 15 | 0.570 | 40 | 0.578 |
| 16 | 0.569 | 41 | 0.578 |
| 17 | 0.571 | 42 | 0.578 |
| 18 | 0.572 | 43 | 0.578 |
| 19 | 0.574 | 44 | 0.578 |
| 20 | 0.573 | 45 | 0.578 |
| 21 | 0.578 | 46 | 0.578 |
| 22 | 0.578 | 47 | 0.578 |
| 23 | 0.576 | 48 | 0.578 |
| 24 | 0.574 | 49 | 0.578 |
| 25 | 0.573 | 50 | 0.578 |

### Key Metrics

| Metric | Run 1 (5 classes) | Run 2 (2 classes) |
|---|---|---|
| Best mAP50 | 0.312 | **0.579** |
| Best epoch | 7 | 38 |
| Overfitting | Yes (val loss diverged at epoch 7) | No |
| Completed all 50 epochs | No (early stopped) | Yes |

---

## Analysis

### Overfitting — Resolved

Run 1 overfit severely — best epoch was epoch 7 out of 50, after which val loss diverged. Run 2 shows no such behaviour. mAP improves consistently from epoch 1 through epoch 38, then plateaus gracefully around 0.578–0.579 without declining. The patience counter never triggered because mAP kept making small incremental gains throughout.

Root cause fix was correct: dropping 3 underperforming classes (motorcycle, bicycle, truck/bus) and adding the DET dataset gave the model sufficient data diversity to generalise properly.

### mAP Improvement

Best mAP50 improved from **0.312 → 0.579**, an increase of **+85%** over Run 1. This is a significant jump attributable to both the class reduction and the larger dataset.

### Training Curve Shape

- **Epochs 1–8:** Steep improvement (+0.061 mAP)
- **Epochs 8–22:** Slower but consistent gains (+0.013 mAP)
- **Epochs 22–38:** Gradual plateau (+0.001 mAP)
- **Epochs 38–50:** Fully plateaued — model has converged

The plateau from epoch 38 onwards suggests the model has extracted most of what it can from this dataset. Further gains would require more data (e.g. real SIYI flight footage) or a larger model (yolov8s/m).

### Best Epoch

Best checkpoint saved at **epoch 38** (mAP50 = 0.579). This is located at:
```
runs/visdrone/yolov8n_ft/weights/best.pt
```

---

## Comparison with Baseline

| Model | mAP50 | Notes |
|---|---|---|
| Baseline yolov8n.pt (COCO) | — | Ground-level training data, poor on aerial |
| Run 1 — finetuned (5 classes, conf=0.25, iou=0.3) | F1=0.533 | Overfit at epoch 7, limited data |
| Run 2 — finetuned (2 classes, conf=0.25) | **mAP50=0.579** | No overfitting, full 50 epochs |

---
