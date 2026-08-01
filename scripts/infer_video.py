"""
UAV Video Inference & Analytics Script
Runs YOLOv8 inference on a raw video file with optional tracking.
Saves annotated video + per-frame CSV + summary analytics.

Usage:
    python3 infer_video.py \
        --video  /path/to/video.mp4 \
        --model  /path/to/best.pt \
        --names  /path/to/best.names \
        --mode   2 \
        --conf   0.25 \
        --output /path/to/output_folder
"""

import argparse
import csv
import os
import time
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO

# ── ByteTrack-style simple tracker using YOLO built-in tracker ──
TRACKER_CONFIG = "bytetrack.yaml"  # built into ultralytics


def load_names(names_path):
    with open(names_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def draw_box(frame, box, label, color):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True,         help="Path to input video")
    parser.add_argument("--model",  required=True,         help="Path to best.pt")
    parser.add_argument("--names",  required=True,         help="Path to best.names")
    parser.add_argument("--mode",   type=int, default=2,   help="1=Detection, 2=Classification, 3=Tracking")
    parser.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", default="./infer_output", help="Output folder")
    args = parser.parse_args()

    CLASS_NAMES = load_names(args.names)
    model = YOLO(args.model)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {args.video}")
        return

    fps_in     = cap.get(cv2.CAP_PROP_FPS)
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mode_label = {1: "detection", 2: "classification", 3: "tracking"}[args.mode]
    video_out_path = output_path / f"annotated_{mode_label}.mp4"
    csv_out_path   = output_path / f"frame_stats_{mode_label}.csv"
    summary_path   = output_path / f"summary_{mode_label}.txt"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_out_path), fourcc, fps_in, (640, 480))

    print(f"[Info] Video:   {args.video}")
    print(f"[Info] Model:   {args.model}")
    print(f"[Info] Mode:    {args.mode} ({mode_label})")
    print(f"[Info] Conf:    {args.conf}")
    print(f"[Info] Output:  {output_path}")
    print(f"[Info] Frames:  {total_frames} @ {fps_in:.1f}fps")
    print(f"[Info] Processing...")

    # ── Analytics accumulators ──
    frame_records = []
    inference_times = []
    class_detection_counts = defaultdict(int)
    class_confidence_sums  = defaultdict(float)

    # Tracking analytics
    track_lifetimes    = defaultdict(int)   # track_id -> frame count
    track_classes      = {}                  # track_id -> class_id
    target_switches    = 0
    last_target_id     = -1

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_idx += 1

        t0 = time.perf_counter()

        if args.mode == 3:
            results = model.track(frame, persist=True, verbose=False,
                                  conf=args.conf, tracker=TRACKER_CONFIG)[0]
        else:
            results = model.predict(frame, verbose=False, conf=args.conf)[0]

        t1 = time.perf_counter()
        inf_ms = (t1 - t0) * 1000
        inference_times.append(inf_ms)
        fps_live = 1000.0 / inf_ms

        frame_det_counts = defaultdict(int)
        frame_det_confs  = defaultdict(list)

        for box in results.boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            xyxy    = box.xyxy[0].tolist()
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "obj"

            # Color: person=blue, car=green
            color = (255, 0, 0) if cls_id == 0 else (0, 255, 0)

            if args.mode == 1:
                cv2.rectangle(frame,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])), color, 2)

            elif args.mode == 2:
                label = f"{cls_name} {int(conf * 100)}%"
                draw_box(frame, xyxy, label, color)

            elif args.mode == 3:
                track_id = int(box.id[0]) if box.id is not None else -1
                label = f"{cls_name} ID:{track_id}"
                draw_box(frame, xyxy, label, color)

                if track_id != -1:
                    track_lifetimes[track_id] += 1
                    track_classes[track_id] = cls_id

                    # Detect target switches (lowest ID person)
                    if cls_id == 0:
                        if last_target_id == -1:
                            last_target_id = track_id
                        elif track_id != last_target_id:
                            target_switches += 1
                            last_target_id = track_id

            class_detection_counts[cls_name] += 1
            class_confidence_sums[cls_name]  += conf
            frame_det_counts[cls_name] += 1
            frame_det_confs[cls_name].append(conf)

        # HUD
        hud = (f"Inference: {int(inf_ms)}ms | FPS: {int(fps_live)}"
               f" | Detections: {len(results.boxes)} | Mode: {mode_label}")
        cv2.putText(frame, hud, (10, 465),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)

        writer.write(frame)

        # Per-frame record
        record = {"frame": frame_idx, "inference_ms": round(inf_ms, 2), "fps": round(fps_live, 2)}
        for cn in CLASS_NAMES:
            record[f"{cn}_count"]      = frame_det_counts[cn]
            record[f"{cn}_avg_conf"]   = round(sum(frame_det_confs[cn]) / len(frame_det_confs[cn]), 3) \
                                         if frame_det_confs[cn] else 0.0
        frame_records.append(record)

        if frame_idx % 100 == 0:
            print(f"  [{frame_idx}/{total_frames}] {inf_ms:.1f}ms/frame")

    cap.release()
    writer.release()

    # ── Write CSV ──
    if frame_records:
        with open(csv_out_path, 'w', newline='') as f:
            writer_csv = csv.DictWriter(f, fieldnames=frame_records[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(frame_records)

    # ── Compute summary ──
    avg_inf   = sum(inference_times) / len(inference_times) if inference_times else 0
    avg_fps   = 1000.0 / avg_inf if avg_inf > 0 else 0
    min_inf   = min(inference_times) if inference_times else 0
    max_inf   = max(inference_times) if inference_times else 0

    lines = []
    lines.append("===== UAV VIDEO INFERENCE SUMMARY =====")
    lines.append(f"Video:         {args.video}")
    lines.append(f"Model:         {args.model}")
    lines.append(f"Mode:          {args.mode} ({mode_label})")
    lines.append(f"Conf threshold:{args.conf}")
    lines.append(f"Total frames:  {frame_idx}")
    lines.append(f"Video FPS:     {fps_in:.1f}")
    lines.append("")
    lines.append("── Inference Performance ──")
    lines.append(f"Avg inference: {avg_inf:.1f}ms")
    lines.append(f"Avg FPS:       {avg_fps:.1f}")
    lines.append(f"Min inference: {min_inf:.1f}ms")
    lines.append(f"Max inference: {max_inf:.1f}ms")
    lines.append("")
    lines.append("── Detection Counts ──")
    for cls_name in CLASS_NAMES:
        total = class_detection_counts[cls_name]
        avg_conf = class_confidence_sums[cls_name] / total if total > 0 else 0
        lines.append(f"{cls_name:12s}  total={total:6d}  avg_conf={avg_conf:.3f}  per_frame={total/frame_idx:.2f}")

    if args.mode == 3:
        lines.append("")
        lines.append("── Tracking Analytics ──")
        lines.append(f"Unique track IDs:   {len(track_lifetimes)}")
        lines.append(f"Target switches:    {target_switches}")
        if track_lifetimes:
            avg_life = sum(track_lifetimes.values()) / len(track_lifetimes)
            max_life = max(track_lifetimes.values())
            lines.append(f"Avg track lifetime: {avg_life:.1f} frames ({avg_life/fps_in:.1f}s)")
            lines.append(f"Max track lifetime: {max_life} frames ({max_life/fps_in:.1f}s)")
            # Per-class track breakdown
            for cls_id, cls_name in enumerate(CLASS_NAMES):
                ids = [tid for tid, cid in track_classes.items() if cid == cls_id]
                lines.append(f"{cls_name:12s}  unique tracks={len(ids)}")

    lines.append("")
    lines.append("── Output Files ──")
    lines.append(f"Annotated video: {video_out_path}")
    lines.append(f"Per-frame CSV:   {csv_out_path}")
    lines.append(f"This summary:    {summary_path}")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(summary_path, 'w') as f:
        f.write(summary_text)

    print(f"\n[Done] All outputs saved to: {output_path}")


if __name__ == "__main__":
    main()