"""
Finetune YOLOv8n on VisDrone dataset (converted to YOLO format).

Usage:
    python3 train_visdrone.py \
        --data /path/to/visdrone_yolo/visdrone.yaml \
        --model yolov8n.pt \
        --epochs 50 \
        --batch 16 \
        --imgsz 640 \
        --project runs/visdrone \
        --name yolov8n_ft
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True,              help="Path to visdrone.yaml")
    parser.add_argument("--model",   default="yolov8n.pt",       help="Base model (default: yolov8n.pt)")
    parser.add_argument("--epochs",  type=int,   default=50,     help="Number of epochs (default: 50)")
    parser.add_argument("--batch",   type=int,   default=16,     help="Batch size (default: 16)")
    parser.add_argument("--imgsz",   type=int,   default=640,    help="Input image size (default: 640)")
    parser.add_argument("--lr",      type=float, default=0.01,   help="Initial learning rate (default: 0.01)")
    parser.add_argument("--project", default="runs/visdrone",    help="Output project folder")
    parser.add_argument("--name",    default="yolov8n_ft",       help="Run name")
    parser.add_argument("--patience",type=int,   default=15,     help="To prevent overfitting")
    args = parser.parse_args()

    model = YOLO(args.model)

    print(f"[Info] Starting finetuning:")
    print(f"       model  = {args.model}")
    print(f"       data   = {args.data}")
    print(f"       epochs = {args.epochs}")
    print(f"       batch  = {args.batch}")
    print(f"       imgsz  = {args.imgsz}")
    print(f"       lr      = {args.lr}")
    print(f"       patience= {args.patience}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        project=args.project,
        name=args.name,
        device="mps",        # Apple Silicon GPU — change to 0 for CUDA on Jetson
        workers=4,
        patience=args.patience,
        save=True,
        save_period=10,      # save checkpoint every 10 epochs
        val=True,
        verbose=True,
    )

    print(f"\n[Done] Best weights saved to: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()