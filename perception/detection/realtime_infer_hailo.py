"""
Real-time YOLO inference on the RealSense D435i color stream — Hailo-8 NPU backend.

Mirrors the logic of `realtime_infer.py` (per-bbox depth at center pixel, FPS
overlay, q/Esc to quit) but replaces the ultralytics PyTorch model with a
HailoRT pipeline running on the AI HAT+ NPU.

Pi5 venv:
    ~/CapstoneDesign2026/.venv311_hailo (Python 3.11, hailo_platform 4.20)

Keys (cv2 window in focus):
    q, Esc        quit

Usage:
    source ~/CapstoneDesign2026/.venv311_hailo/bin/activate
    cd ~/CapstoneDesign2026
    python -m perception.detection.realtime_infer_hailo
    python -m perception.detection.realtime_infer_hailo --conf 0.5
    python -m perception.detection.realtime_infer_hailo --hef hailo_models/yolov11n.hef
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    InferVStreams,
)

from perception.common.realsense_wrapper import RealSenseCamera
from perception.config import CAMERA

HERE = Path(__file__).resolve().parent
HEF_DIR = HERE.parent.parent / "hailo_models"

FPS_EMA_ALPHA = 0.1


def find_latest_hef(hef_dir: Path) -> Path:
    """Return newest `<hef_dir>/*.hef` by mtime, or raise."""
    candidates = list(hef_dir.glob("*.hef"))
    if not candidates:
        raise FileNotFoundError(
            f"no *.hef found under {hef_dir}/. "
            "Compile a model on desktop (Hailo DFC) or pass --hef."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def letterbox_to_rgb(
    img_bgr: np.ndarray,
    target_hw: Tuple[int, int],
    pad_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize-with-aspect-pad to (th, tw); BGR→RGB. Returns (rgb_canvas, ratio, (pad_x, pad_y))."""
    th, tw = target_hw
    h0, w0 = img_bgr.shape[:2]
    r = min(tw / w0, th / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    px = (tw - new_w) // 2
    py = (th - new_h) // 2
    canvas[py:py + new_h, px:px + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb, r, (px, py)


def parse_hailo_nms_by_class(
    raw_out,
    *,
    conf_thr: float,
    orig_w: int,
    orig_h: int,
    model_w: int,
    model_h: int,
    ratio: float,
    pad_xy: Tuple[int, int],
    classes_filter: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HailoRT 'HAILO NMS BY CLASS' output → (xyxy, confs, class_ids) in original-image pixels.

    Output shape:
        list (batch). Each batch entry is a list of length num_classes; each
        per-class entry is an (n_dets, 5) float array with rows
        [y_min, x_min, y_max, x_max, score], coords normalized to model-input space.
    """
    # batch dim may or may not be present; unwrap if needed
    if isinstance(raw_out, list) and len(raw_out) > 0 and isinstance(raw_out[0], list):
        per_class = raw_out[0]
    else:
        per_class = raw_out

    px, py = pad_xy
    boxes_xyxy: List[List[float]] = []
    confs: List[float] = []
    cls_ids: List[int] = []

    for cls_id, dets in enumerate(per_class):
        if classes_filter is not None and cls_id not in classes_filter:
            continue
        if dets is None or len(dets) == 0:
            continue
        for det in dets:
            ymin, xmin, ymax, xmax, score = (float(v) for v in det[:5])
            if score < conf_thr:
                continue
            # normalized → model-input pixels
            x1m, y1m, x2m, y2m = xmin * model_w, ymin * model_h, xmax * model_w, ymax * model_h
            # undo letterbox pad → scale to original
            x1 = (x1m - px) / ratio
            y1 = (y1m - py) / ratio
            x2 = (x2m - px) / ratio
            y2 = (y2m - py) / ratio
            x1 = max(0.0, min(orig_w - 1, x1))
            y1 = max(0.0, min(orig_h - 1, y1))
            x2 = max(0.0, min(orig_w - 1, x2))
            y2 = max(0.0, min(orig_h - 1, y2))
            boxes_xyxy.append([x1, y1, x2, y2])
            confs.append(score)
            cls_ids.append(cls_id)

    if boxes_xyxy:
        return (
            np.asarray(boxes_xyxy, dtype=np.float32),
            np.asarray(confs, dtype=np.float32),
            np.asarray(cls_ids, dtype=np.int32),
        )
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
    )


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
    depth_frame,
) -> np.ndarray:
    """Overlay bbox + 'conf  d.dd m' label for each detection."""
    out = img
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)
    for (x1f, y1f, x2f, y2f), c in zip(boxes_xyxy, confs):
        x1, y1, x2, y2 = int(round(x1f)), int(round(y1f)), int(round(x2f)), int(round(y2f))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        d = bbox_center_depth(depth_frame, x1, y1, x2, y2, w, h)
        d_str = f"{d:.2f}m" if d is not None else "n/a"
        label = f"{float(c):.2f}  {d_str}"

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
        description="Real-time YOLO inference on RealSense D435i — Hailo-8 backend.",
    )
    ap.add_argument("--hef", type=Path, default=None,
                    help="path to .hef (default: newest hailo_models/*.hef)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold (default 0.25)")
    args = ap.parse_args()

    hef_path = args.hef or find_latest_hef(HEF_DIR)
    if not hef_path.is_file():
        raise FileNotFoundError(f"hef not found: {hef_path}")
    print(f"[realtime] hef:  {hef_path}")
    print(f"[realtime] conf: {args.conf}")

    hef = HEF(str(hef_path))
    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]
    model_h, model_w = int(in_info.shape[0]), int(in_info.shape[1])
    print(f"[realtime] model input:  {in_info.name}  {model_h}x{model_w}")
    print(f"[realtime] model output: {out_info.name}")

    fps: Optional[float] = None
    last_t: Optional[float] = None

    cv2.namedWindow("realtime_hailo", cv2.WINDOW_NORMAL)

    with VDevice() as target, RealSenseCamera(CAMERA) as camera:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef, cfg)[0]
        ng_params = ng.create_params()
        in_p = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_p = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

        camera.warmup(num_frames=30)

        with InferVStreams(ng, in_p, out_p) as pipeline:
            with ng.activate(ng_params):
                while True:
                    color, _depth, depth_frame = camera.get_frames()
                    if color is None or depth_frame is None:
                        continue
                    h0, w0 = color.shape[:2]

                    inp_rgb, ratio, pad_xy = letterbox_to_rgb(color, (model_h, model_w))
                    inp = inp_rgb[None, ...]  # add batch dim → (1, H, W, 3)

                    out_dict = pipeline.infer({in_info.name: inp})
                    raw = out_dict[out_info.name]

                    xyxy, confs, _cls = parse_hailo_nms_by_class(
                        raw,
                        conf_thr=args.conf,
                        orig_w=w0, orig_h=h0,
                        model_w=model_w, model_h=model_h,
                        ratio=ratio, pad_xy=pad_xy,
                    )
                    if len(xyxy) > 0:
                        color = draw_detections(color, xyxy, confs, depth_frame)

                    now = time.perf_counter()
                    if last_t is not None:
                        inst = 1.0 / max(now - last_t, 1e-6)
                        fps = inst if fps is None else (1 - FPS_EMA_ALPHA) * fps + FPS_EMA_ALPHA * inst
                    last_t = now
                    draw_fps(color, fps)

                    cv2.imshow("realtime_hailo", color)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
