#!/usr/bin/env python3
"""Detect target on RealSense color stream (Hailo) → print center depth.

Standalone verification tool: each frame runs the Hailo YOLO26 detector, takes the
highest-conf bbox, and prints the depth at the bbox center (both the raw single-pixel
depth and the robust ROI-median used by the real pipeline).

All knobs live in config.yaml (probe.*, detector.hef). CLI only accepts
--config / --hw-reset (override probe.hw_reset toggle).

Run with the Hailo venv:
    .venv311_hailo/bin/python run_center_depth_probe.py
    .venv311_hailo/bin/python run_center_depth_probe.py --hw-reset
"""
from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path

from config_loader import load_args
from perception.common.realsense_wrapper import RealSenseCamera
from perception.config import CAMERA
from perception.detection.hailo_yolo26 import HailoYolo26Detector
from perception.detection.visual_servo_target import compute_target_depth

HERE = Path(__file__).resolve().parent
DEFAULT_HEF = HERE / "perception" / "detection" / "outdoor_v2.hef"


def main() -> None:
    args = load_args(
        prog="probe",
        allow_overrides=("hw_reset",),
    )
    print(f"[probe] config    : {args.config_path}")

    # detector.hef (yaml) takes precedence; fall back to probe's own default.
    hef = args.hef if args.hef is not None else DEFAULT_HEF
    if not Path(hef).is_file():
        raise SystemExit(f"HEF not found: {hef}")

    print(f"[probe] hef       : {hef}")
    print(f"[probe] conf      : {args.probe_conf}")
    print(f"[probe] roi_frac  : {args.probe_roi_frac}")
    print("[probe] Ctrl-C to stop\n")

    with ExitStack() as stack:
        det = stack.enter_context(
            HailoYolo26Detector(Path(hef), conf=args.probe_conf)
        )
        cam = stack.enter_context(
            RealSenseCamera(CAMERA, hardware_reset_on_start=args.probe_hw_reset)
        )
        cam.warmup(num_frames=30)

        n = 0
        t_fps = time.time()
        fps_count = 0
        fps = 0.0
        try:
            while args.probe_max_frames == 0 or n < args.probe_max_frames:
                color, depth, depth_frame = cam.get_frames()
                if color is None or depth_frame is None:
                    continue
                n += 1
                fps_count += 1
                if time.time() - t_fps >= 1.0:
                    fps = fps_count / (time.time() - t_fps)
                    fps_count = 0
                    t_fps = time.time()

                result = det.predict(color)
                if result is None:
                    print(f"[{n:5d}] {fps:4.1f}fps  no target")
                    continue

                bbox, conf = result
                x1, y1, x2, y2 = bbox
                cx = int(round((x1 + x2) / 2.0))
                cy = int(round((y1 + y2) / 2.0))

                # raw single-pixel depth at the bbox center (meters, 0 = hole)
                px_depth = depth_frame.get_distance(cx, cy)
                # robust ROI-median depth (same as Phase1/Phase2 pipeline)
                roi_depth = compute_target_depth(
                    depth, bbox,
                    roi_frac=args.probe_roi_frac,
                    min_valid_pixels=args.probe_min_valid,
                    depth_scale_m=0.001,
                )
                roi_str = f"{roi_depth:.3f} m" if roi_depth is not None else "  n/a "
                print(
                    f"[{n:5d}] {fps:4.1f}fps  conf={conf:.2f}  "
                    f"center=({cx:3d},{cy:3d})  "
                    f"px_depth={px_depth:.3f} m  roi_depth={roi_str}"
                )
        except KeyboardInterrupt:
            print("\n[probe] stopped")


if __name__ == "__main__":
    main()
