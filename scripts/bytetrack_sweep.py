"""
ByteTrack Parameter Sweep Analysis
Runs ByteTrack on a video with varying parameters and logs analytics for each combination.

Usage:
    python3 bytetrack_sweep.py \
        --video  /path/to/video.mp4 \
        --model  /path/to/best.pt \
        --names  /path/to/best.names \
        --conf   0.25 \
        --output /path/to/output_folder
"""

import argparse
import csv
import os
import time
from pathlib import Path
from collections import defaultdict
from itertools import product

import cv2
from ultralytics import YOLO

# ── Parameter grid to sweep ──
PARAM_GRID = {
    "high_conf":     [0.3, 0.4, 0.5],
    "low_conf":      [0.1, 0.15, 0.2],
    "iou_threshold": [0.4, 0.5, 0.6],
    "max_lost":      [30, 60, 80],
}


def load_names(names_path):
    with open(names_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def run_sweep(model, video_path, CLASS_NAMES, conf_threshold, params, run_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps_in       = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write custom bytetrack yaml for this run
    tracker_cfg_path = f"/tmp/bytetrack_sweep_{run_id}.yaml"
    with open(tracker_cfg_path, 'w') as f:
        f.write(f"""tracker_type: bytetrack
track_high_thresh: {params['high_conf']}
track_low_thresh: {params['low_conf']}
new_track_thresh: {params['high_conf']}
track_buffer: {params['max_lost']}
match_thresh: {params['iou_threshold']}
fuse_score: True
""")

    inference_times = []
    track_lifetimes = defaultdict(int)
    track_classes   = {}
    target_switches = 0
    last_target_id  = -1
    class_counts    = defaultdict(int)
    class_conf_sums = defaultdict(float)
    frame_idx       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_idx += 1

        t0 = time.perf_counter()
        results = model.track(frame, persist=True, verbose=False,
                              conf=conf_threshold, tracker=tracker_cfg_path)[0]
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "obj"

            class_counts[cls_name]    += 1
            class_conf_sums[cls_name] += conf

            if track_id != -1:
                track_lifetimes[track_id] += 1
                track_classes[track_id] = cls_id

                if cls_id == 0:
                    if last_target_id == -1:
                        last_target_id = track_id
                    elif track_id != last_target_id:
                        target_switches += 1
                        last_target_id = track_id

    cap.release()

    avg_inf    = sum(inference_times) / len(inference_times) if inference_times else 0
    avg_fps    = 1000.0 / avg_inf if avg_inf > 0 else 0
    unique_ids = len(track_lifetimes)
    avg_life   = sum(track_lifetimes.values()) / unique_ids if unique_ids > 0 else 0
    max_life   = max(track_lifetimes.values()) if track_lifetimes else 0

    person_tracks = sum(1 for tid, cid in track_classes.items() if cid == 0)
    car_tracks    = sum(1 for tid, cid in track_classes.items() if cid == 1)

    result = {
        "run_id":                run_id,
        "high_conf":             params["high_conf"],
        "low_conf":              params["low_conf"],
        "iou_threshold":         params["iou_threshold"],
        "max_lost":              params["max_lost"],
        "avg_inf_ms":            round(avg_inf, 2),
        "avg_fps":               round(avg_fps, 2),
        "total_frames":          frame_idx,
        "unique_track_ids":      unique_ids,
        "person_tracks":         person_tracks,
        "car_tracks":            car_tracks,
        "avg_track_life_frames": round(avg_life, 1),
        "avg_track_life_sec":    round(avg_life / fps_in, 2) if fps_in > 0 else 0,
        "max_track_life_frames": max_life,
        "max_track_life_sec":    round(max_life / fps_in, 2) if fps_in > 0 else 0,
        "target_switches":       target_switches,
    }

    for cls_name in CLASS_NAMES:
        total    = class_counts[cls_name]
        avg_conf = class_conf_sums[cls_name] / total if total > 0 else 0
        result[f"{cls_name}_detections"] = total
        result[f"{cls_name}_avg_conf"]   = round(avg_conf, 3)
        result[f"{cls_name}_per_frame"]  = round(total / frame_idx, 2) if frame_idx > 0 else 0

    os.remove(tracker_cfg_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True,            help="Path to input video")
    parser.add_argument("--model",  required=True,            help="Path to best.pt")
    parser.add_argument("--names",  required=True,            help="Path to best.names")
    parser.add_argument("--conf",   type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--output", default="./sweep_output", help="Output folder")
    args = parser.parse_args()

    CLASS_NAMES = load_names(args.names)
    model = YOLO(args.model)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path     = output_path / "bytetrack_sweep_results.csv"
    summary_path = output_path / "bytetrack_sweep_summary.txt"

    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(product(*values))

    total_runs = len(combos)
    print(f"[Info] Running {total_runs} parameter combinations...")
    print(f"[Info] Video:  {args.video}")
    print(f"[Info] Model:  {args.model}")
    print(f"[Info] Output: {output_path}\n")

    all_results = []

    for run_id, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"[Run {run_id}/{total_runs}] high_conf={params['high_conf']} "
              f"low_conf={params['low_conf']} iou={params['iou_threshold']} "
              f"max_lost={params['max_lost']}")

        result = run_sweep(model, args.video, CLASS_NAMES, args.conf, params, run_id)
        if result:
            all_results.append(result)
            print(f"  → unique_ids={result['unique_track_ids']} "
                  f"avg_life={result['avg_track_life_sec']}s "
                  f"target_switches={result['target_switches']} "
                  f"avg_fps={result['avg_fps']}")

    # Write CSV
    if all_results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # Find best configs
    best_switches = min(all_results, key=lambda x: x["target_switches"])
    best_life     = max(all_results, key=lambda x: x["avg_track_life_sec"])
    best_ids      = min(all_results, key=lambda x: x["unique_track_ids"])

    lines = []
    lines.append("===== BYTETRACK PARAMETER SWEEP SUMMARY =====")
    lines.append(f"Video:       {args.video}")
    lines.append(f"Model:       {args.model}")
    lines.append(f"Conf thresh: {args.conf}")
    lines.append(f"Total runs:  {total_runs}")
    lines.append("")

    lines.append("── Best Config: Fewest Target Switches ──")
    lines.append(f"  high_conf={best_switches['high_conf']}  low_conf={best_switches['low_conf']}  "
                 f"iou={best_switches['iou_threshold']}  max_lost={best_switches['max_lost']}")
    lines.append(f"  target_switches={best_switches['target_switches']}  "
                 f"avg_life={best_switches['avg_track_life_sec']}s  "
                 f"unique_ids={best_switches['unique_track_ids']}")

    lines.append("")
    lines.append("── Best Config: Longest Avg Track Lifetime ──")
    lines.append(f"  high_conf={best_life['high_conf']}  low_conf={best_life['low_conf']}  "
                 f"iou={best_life['iou_threshold']}  max_lost={best_life['max_lost']}")
    lines.append(f"  avg_life={best_life['avg_track_life_sec']}s  "
                 f"target_switches={best_life['target_switches']}  "
                 f"unique_ids={best_life['unique_track_ids']}")

    lines.append("")
    lines.append("── Best Config: Fewest Unique Track IDs (least fragmentation) ──")
    lines.append(f"  high_conf={best_ids['high_conf']}  low_conf={best_ids['low_conf']}  "
                 f"iou={best_ids['iou_threshold']}  max_lost={best_ids['max_lost']}")
    lines.append(f"  unique_ids={best_ids['unique_track_ids']}  "
                 f"avg_life={best_ids['avg_track_life_sec']}s  "
                 f"target_switches={best_ids['target_switches']}")

    lines.append("")
    lines.append("── All Results (sorted by target_switches asc) ──")
    lines.append(f"{'Run':>4} {'high_conf':>10} {'low_conf':>9} {'iou':>5} {'max_lost':>9} "
                 f"{'unique_ids':>11} {'avg_life(s)':>12} {'switches':>9} {'avg_fps':>8}")
    lines.append("-" * 85)
    for r in sorted(all_results, key=lambda x: x["target_switches"]):
        lines.append(f"{r['run_id']:>4} {r['high_conf']:>10} {r['low_conf']:>9} "
                     f"{r['iou_threshold']:>5} {r['max_lost']:>9} "
                     f"{r['unique_track_ids']:>11} {r['avg_track_life_sec']:>12} "
                     f"{r['target_switches']:>9} {r['avg_fps']:>8}")

    lines.append("")
    lines.append(f"Full results CSV: {csv_path}")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(summary_path, 'w') as f:
        f.write(summary_text)

    print(f"\n[Done] Results saved to: {output_path}")


if __name__ == "__main__":
    main()