"""
Interactive viewer for trained-model predictions on the test split.

Loads a trained YOLO weights file (default: latest `runs/*/weights/best.pt`),
runs inference once over `data/images/test/`, and shows each image with
ground-truth bboxes (green) and predicted bboxes (red, with confidence).

Keys (cv2 window in focus):
    →, n, space   next
    ←, p          previous
    s             save current frame as pred_<stem>.jpg in CWD
    q, Esc        quit

Usage:
    python -m perception.training.visualize_predictions
    python -m perception.training.visualize_predictions --conf 0.5
    python -m perception.training.visualize_predictions --weights path/to/best.pt
    python -m perception.training.visualize_predictions --split val
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from perception.training.visualize_dataset import (
    draw_bbox_on_image,
    iter_pairs_for_split,
    read_yolo_bboxes,
)

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE / "runs"

# X11/GTK keycodes returned by cv2.waitKeyEx() on Linux for arrow keys.
KEY_LEFT = 65361
KEY_RIGHT = 65363


def find_latest_best(runs_root: Path) -> Path:
    """Return the newest `runs/*/weights/best.pt` by mtime, or raise."""
    candidates = list(runs_root.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no best.pt found under {runs_root}/*/weights/. "
            "Train first or pass --weights."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_predictions(
    model: YOLO,
    pairs: List[Tuple[Path, Path]],
    imgsz: int,
    conf: float,
    device: str,
) -> Dict[str, List[Tuple[float, float, float, float, float]]]:
    """Run inference once over all images, return {img_name: [(cx,cy,w,h,conf), ...]}.

    Bboxes are normalised (YOLO format) so they can be drawn with the same
    helper used for ground-truth labels.
    """
    out: dict[str, list[tuple[float, float, float, float, float]]] = {}
    img_paths = [str(img) for img, _ in pairs]
    results = model.predict(
        source=img_paths,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        save=False,
        stream=False,
    )
    for path_str, res in zip(img_paths, results):
        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xywhn = res.boxes.xywhn.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (cx, cy, w, h), c in zip(xywhn, confs):
                boxes.append((float(cx), float(cy), float(w), float(h), float(c)))
        out[Path(path_str).name] = boxes
    return out


def draw_predictions(
    image: np.ndarray,
    preds: List[Tuple[float, float, float, float, float]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Overlay predicted bboxes (red by default) with confidence label above each."""
    out = image.copy()
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cx, cy, bw, bh, c in preds:
        x1 = int(round((cx - bw / 2) * w))
        y1 = int(round((cy - bh / 2) * h))
        x2 = int(round((cx + bw / 2) * w))
        y2 = int(round((cy + bh / 2) * h))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{c:.2f}"
        ty = max(y1 - 5, 12)
        cv2.putText(out, label, (x1, ty), font, 0.5, color, 2)
    return out


def annotate(
    image: np.ndarray,
    gt: List[Tuple[float, float, float, float]],
    preds: List[Tuple[float, float, float, float, float]],
    filename: str,
    index: int,
    total: int,
) -> np.ndarray:
    """Draw GT (green) + preds (red) + HUD (filename, index, counts)."""
    annotated = draw_bbox_on_image(image, gt, color=(0, 255, 0), thickness=2)
    annotated = draw_predictions(annotated, preds)
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, filename, (10, 25), font, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated, f"{index + 1}/{total}",
                (w - 110, 25), font, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated, f"GT={len(gt)}  pred={len(preds)}",
                (10, h - 15), font, 0.6, (0, 255, 255), 2)
    return annotated


def run_viewer(
    pairs: List[Tuple[Path, Path]],
    preds_by_name: Dict[str, List[Tuple[float, float, float, float, float]]],
) -> None:
    if not pairs:
        print("[pred-viz] no images to show")
        return

    n = len(pairs)
    print(f"[pred-viz] {n} images. "
          "Keys: →/n/space=next, ←/p=prev, s=save, q/Esc=quit")
    cv2.namedWindow("predictions", cv2.WINDOW_NORMAL)
    i = 0
    while True:
        img_path, lab_path = pairs[i]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[pred-viz] could not read {img_path}, skipping")
            i = (i + 1) % n
            continue
        gt = read_yolo_bboxes(lab_path)
        preds = preds_by_name.get(img_path.name, [])
        annotated = annotate(image, gt, preds, img_path.name, i, n)
        cv2.imshow("predictions", annotated)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):  # 27 = Esc
            break
        if key in (ord("n"), ord(" "), KEY_RIGHT):
            i = (i + 1) % n
        elif key in (ord("p"), KEY_LEFT):
            i = (i - 1) % n
        elif key == ord("s"):
            out = Path.cwd() / f"pred_{img_path.stem}.jpg"
            cv2.imwrite(str(out), annotated)
            print(f"[pred-viz] saved {out}")
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="Interactive viewer for trained-model predictions on test split.",
    )
    ap.add_argument("--training-root", type=Path, default=HERE,
                    help=f"training root containing data/ (default: {HERE})")
    ap.add_argument("--weights", type=Path, default=None,
                    help="path to best.pt (default: newest runs/*/weights/best.pt)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold for predictions (default 0.25)")
    ap.add_argument("--device", default="0",
                    help="cuda device id or 'cpu' (default '0')")
    args = ap.parse_args()

    weights = args.weights or find_latest_best(DEFAULT_RUNS)
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    print(f"[pred-viz] weights: {weights}")

    pairs = iter_pairs_for_split(
        args.training_root, args.split, include_originals=True,
    )
    if not pairs:
        print(f"[pred-viz] no pairs found for split={args.split}")
        return

    print(f"[pred-viz] running inference on {len(pairs)} {args.split} images ...")
    model = YOLO(str(weights))
    preds_by_name = run_predictions(
        model, pairs, args.imgsz, args.conf, args.device,
    )

    run_viewer(pairs, preds_by_name)


if __name__ == "__main__":
    main()
