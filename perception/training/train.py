"""
Train YOLO26n on the bell-detection dataset.

Lazy-prepares the dataset (calls prepare() if dataset.yaml is missing), then
fine-tunes from the pretrained yolo26n.pt and runs a final test-set eval.

Usage:
    python -m perception.training.train
    python -m perception.training.train --epochs 50 --batch 16 --name run2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from perception.training.prepare_dataset import prepare

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_DATASET = REPO_ROOT / "perception" / "dataset"
DEFAULT_WEIGHTS = HERE / "weights" / "yolo26n.pt"
DEFAULT_PROJECT = HERE / "runs"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train YOLO26n bell detector."
    )
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64,
                    help="total batch across all GPUs (must be multiple of "
                         "device count; auto-batch -1 is single-GPU only)")
    ap.add_argument("--device", default="0,1",
                    help="cuda device id, 'cpu', or comma list e.g. '0,1' "
                         "(DDP multi-GPU when >1)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="bell_yolo26n")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--no-test", action="store_true",
                    help="skip post-training test-set eval")
    return ap.parse_args()


def main():
    args = parse_args()

    yaml_path = prepare(DEFAULT_DATASET, HERE, rebuild=False, seed=args.seed)

    if not args.weights.is_file():
        raise FileNotFoundError(
            f"pretrained weights not found at {args.weights}; "
            "ultralytics will auto-download yolo26n.pt if you run "
            f"`python -c 'from ultralytics import YOLO; YOLO(\"yolo26n.pt\")'` "
            f"and then move the file there."
        )

    model = YOLO(str(args.weights))

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=str(DEFAULT_PROJECT),
        name=args.name,
        single_cls=True,
        cos_lr=True,
        amp=True,
        close_mosaic=10,
        exist_ok=False,
    )

    if args.no_test:
        return

    run_dir = DEFAULT_PROJECT / args.name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        print(f"[train] best.pt not found at {best_pt}, skipping test eval")
        return

    print(f"\n[train] evaluating {best_pt} on test split ...")
    best = YOLO(str(best_pt))
    val_device = args.device.split(",")[0] if isinstance(args.device, str) else args.device
    metrics = best.val(
        data=str(yaml_path),
        split="test",
        imgsz=args.imgsz,
        device=val_device,
        project=str(DEFAULT_PROJECT),
        name=f"{args.name}_test",
    )
    print(f"[train] test mAP@0.5      = {metrics.box.map50:.4f}")
    print(f"[train] test mAP@0.5:0.95 = {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
