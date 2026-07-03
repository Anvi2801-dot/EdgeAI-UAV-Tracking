"""
Evaluate YOLOv8n on VisDrone-VID sequences.
Computes precision/recall/IoU-based matching against ground truth annotations,
with VisDrone -> COCO class mapping.

Usage:
    python3 eval_visdrone.py --dataset /home/nvidia/VisDrone2019-VID-test-dev \
                              --model /path/to/yolov8n.pt \
                              --sequences uav0000009_03358_v uav0000073_00600_v
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO

# ── VisDrone class ID -> COCO class ID mapping ──
# VisDrone classes: 0=ignored,1=pedestrian,2=people,3=bicycle,4=car,5=van,
#                    6=truck,7=tricycle,8=awning-tricycle,9=bus,10=motor
VISDRONE_TO_COCO = {
    1: 0,   # pedestrian -> person
    2: 0,   # people -> person
    3: 1,   # bicycle -> bicycle
    4: 2,   # car -> car
    5: 2,   # van -> car
    6: 7,   # truck -> truck
    9: 5,   # bus -> bus
    10: 3,  # motor -> motorcycle
    # 0, 7, 8 have no COCO equivalent -> skipped
}

IOU_THRESHOLD = 0.5


def load_visdrone_annotations(txt_path):
    """
    Parses VisDrone-VID annotation file.
    Format per row: frame_id,target_id,x,y,w,h,score,class,truncation,occlusion
    Returns: dict {frame_id: [(coco_class_id, x, y, w, h), ...]}
    """
    frame_gt = defaultdict(list)
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            score = int(parts[6])
            vis_class = int(parts[7])

            if score == 0:  # ignored region per VisDrone spec
                continue
            if vis_class not in VISDRONE_TO_COCO:
                continue  # skip classes with no COCO equivalent

            coco_class = VISDRONE_TO_COCO[vis_class]
            frame_gt[frame_id].append((coco_class, x, y, w, h))
    return frame_gt


def iou(boxA, boxB):
    """boxes as (x, y, w, h)"""
    ax1, ay1, aw, ah = boxA
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = boxB
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = aw * ah
    areaB = bw * bh
    union = areaA + areaB - inter_area

    return inter_area / union if union > 0 else 0


def evaluate_sequence(model, dataset_path, seq_name):
    seq_dir = Path(dataset_path) / "sequences" / seq_name
    ann_path = Path(dataset_path) / "annotations" / f"{seq_name}.txt"

    if not seq_dir.exists() or not ann_path.exists():
        print(f"[Skip] Missing data for sequence: {seq_name}")
        return None

    frame_gt = load_visdrone_annotations(ann_path)
    frame_files = sorted(seq_dir.glob("*.jpg"))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for frame_path in frame_files:
        frame_id = int(frame_path.stem)  # e.g. "0000001" -> 1
        gt_boxes = frame_gt.get(frame_id, [])

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        results = model.predict(frame, verbose=False, conf=0.25)[0]

        pred_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_boxes.append((cls_id, x1, y1, x2 - x1, y2 - y1))

        matched_gt = set()
        for pred_cls, px, py, pw, ph in pred_boxes:
            best_iou = 0
            best_idx = -1
            for idx, (gt_cls, gx, gy, gw, gh) in enumerate(gt_boxes):
                if idx in matched_gt or gt_cls != pred_cls:
                    continue
                score = iou((px, py, pw, ph), (gx, gy, gw, gh))
                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            if best_iou >= IOU_THRESHOLD:
                total_tp += 1
                matched_gt.add(best_idx)
            else:
                total_fp += 1

        total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n[{seq_name}] frames={len(frame_files)} TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"[{seq_name}] Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

    return {"tp": total_tp, "fp": total_fp, "fn": total_fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to VisDrone2019-VID-test-dev folder")
    parser.add_argument("--model", required=True, help="Path to yolov8n.pt")
    parser.add_argument("--sequences", nargs="+", required=False, default=None, help="Sequence names to evaluate (omit to run all)")
    args = parser.parse_args()

    model = YOLO(args.model)

    overall = {"tp": 0, "fp": 0, "fn": 0}
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = sorted(os.listdir(os.path.join(args.dataset, "sequences")))
        print(f"[Info] Running on all {len(sequences)} sequences...")

    for seq in sequences:
        result = evaluate_sequence(model, args.dataset, seq)
        if result:
            overall["tp"] += result["tp"]
            overall["fp"] += result["fp"]
            overall["fn"] += result["fn"]

    precision = overall["tp"] / (overall["tp"] + overall["fp"]) if (overall["tp"] + overall["fp"]) > 0 else 0
    recall = overall["tp"] / (overall["tp"] + overall["fn"]) if (overall["tp"] + overall["fn"]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n===== OVERALL =====")
    print(f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']}")
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
