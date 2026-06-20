"""
Real-time YOLO inference on the RealSense D435i color stream.

Loads a trained YOLO weights file (default: latest
`perception/training/runs/*/weights/best.pt`), pulls aligned color+depth from
the camera, draws each detection's bbox with confidence and the depth (m) at
the bbox-center pixel, and overlays current FPS.

Keys (cv2 window in focus):
    q, Esc        quit

Usage:
    python -m perception.detection.realtime_infer
    python -m perception.detection.realtime_infer --conf 0.5
    python -m perception.detection.realtime_infer --weights path/to/best.pt
    python -m perception.detection.realtime_infer --coco                 # COCO yolov8n.pt, sports-ball only
    python -m perception.detection.realtime_infer --coco --coco-classes 32 38
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from perception.common.realsense_wrapper import RealSenseCamera, apply_depth_colormap
from perception.config import CAMERA

HERE = Path(__file__).resolve().parent
TRAINING_RUNS = HERE.parent / "training" / "runs"

IMGSZ = 640
FPS_EMA_ALPHA = 0.1

COCO_DEFAULT_WEIGHTS = "yolov8n.pt"
COCO_SPORTS_BALL_CLS = 32  # COCO class id for "sports ball" (includes tennis balls)


def find_latest_best(runs_root: Path) -> Path:
    """Return the newest `runs/*/weights/best.pt` by mtime, or raise."""
    candidates = list(runs_root.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no best.pt found under {runs_root}/*/weights/. "
            "Train first or pass --weights."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def bbox_center_depth(depth_frame, x1: int, y1: int, x2: int, y2: int,
                      width: int, height: int) -> Optional[float]:
    """Depth (m) at the bbox-center pixel; None if 0 / out of range."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    if not (0 <= cx < width and 0 <= cy < height):
        return None
    d = depth_frame.get_distance(cx, cy)
    if d <= 0.0:
        return None
    return d


def draw_detections(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    cls_ids: np.ndarray,
    names: dict,
    depth_frame,
) -> np.ndarray:
    """Overlay bbox + 'name conf  d.dd m' label for each detection."""
    out = img
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)
    for (x1f, y1f, x2f, y2f), c, k in zip(boxes_xyxy, confs, cls_ids):
        x1, y1, x2, y2 = int(round(x1f)), int(round(y1f)), int(round(x2f)), int(round(y2f))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        d = bbox_center_depth(depth_frame, x1, y1, x2, y2, w, h)
        d_str = f"{d:.2f}m" if d is not None else "n/a"
        name = names.get(int(k), str(int(k))) if isinstance(names, dict) else str(int(k))
        label = f"{name} {float(c):.2f}  {d_str}"

        ty = max(y1 - 5, 12)
        cv2.putText(out, label, (x1, ty), font, 0.5, color, 2)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(out, (cx, cy), 3, color, -1)
    return out


def draw_fps(img: np.ndarray, fps: Optional[float]) -> None:
    if fps is None:
        return
    cv2.putText(
        img, f"{fps:5.1f} FPS", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Real-time YOLO inference on RealSense D435i color stream.",
    )
    ap.add_argument("--weights", type=Path, default=None,
                    help="path to best.pt (default: newest training/runs/*/weights/best.pt; "
                         "ignored default when --coco is set)")
    ap.add_argument("--conf", type=float, default=0.2,
                    help="confidence threshold (default 0.25)")
    ap.add_argument("--device", type=str, default="0",
                    help="inference device: '0' (CUDA), 'cpu', 'mps' on macOS (default '0')")
    ap.add_argument("--coco", action="store_true",
                    help="use a COCO-pretrained YOLO model (default yolov8n.pt) and "
                         "filter to 'sports ball' (class 32) — covers tennis balls")
    ap.add_argument("--coco-classes", type=int, nargs="+", default=None,
                    help="override the COCO class id filter (default: [32]); only used with --coco")
    args = ap.parse_args()

    if args.coco:
        weights_arg = args.weights if args.weights is not None else Path(COCO_DEFAULT_WEIGHTS)
        class_filter: Optional[List[int]] = args.coco_classes or [COCO_SPORTS_BALL_CLS]
    else:
        weights_arg = args.weights or find_latest_best(TRAINING_RUNS)
        class_filter = None

    # ultralytics resolves bare names like "yolov8n.pt" itself (auto-download);
    # only verify existence for explicit on-disk paths.
    weights_str = str(weights_arg)
    is_bare_name = weights_arg.parent == Path(".") and weights_arg.suffix == ".pt"
    if not is_bare_name and not weights_arg.is_file():
        raise FileNotFoundError(f"weights not found: {weights_arg}")

    print(f"[realtime] weights: {weights_str}")
    print(f"[realtime] conf:    {args.conf}")
    if class_filter is not None:
        print(f"[realtime] coco classes filter: {class_filter}")

    model = YOLO(weights_str)
    names = getattr(model, "names", {})

    fps: Optional[float] = None
    last_t: Optional[float] = None

    cv2.namedWindow("realtime", cv2.WINDOW_NORMAL)
    with RealSenseCamera(CAMERA) as camera:
        camera.warmup(num_frames=30)
        while True:
            color, depth, depth_frame = camera.get_frames()
            if color is None or depth_frame is None:
                continue

            predict_kwargs = dict(
                source=color,
                imgsz=IMGSZ,
                conf=args.conf,
                device=args.device,
                verbose=False,
                save=False,
                stream=False,
            )
            if class_filter is not None:
                predict_kwargs["classes"] = class_filter
            results = model.predict(**predict_kwargs)
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy()
                color = draw_detections(color, xyxy, confs, cls_ids, names, depth_frame)

            now = time.perf_counter()
            if last_t is not None:
                inst = 1.0 / max(now - last_t, 1e-6)
                fps = inst if fps is None else (1 - FPS_EMA_ALPHA) * fps + FPS_EMA_ALPHA * inst
            last_t = now
            draw_fps(color, fps)

            depth_vis = apply_depth_colormap(depth, depth_frame, camera.colorizer)
            if depth_vis.shape[:2] != color.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (color.shape[1], color.shape[0]))
            panel = np.hstack([color, depth_vis])
            cv2.imshow("realtime", panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
