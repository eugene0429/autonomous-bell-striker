#!/usr/bin/env python3
"""Phase 1 visual-servo runner — real RealSense + YOLO + OpenRB.

Drives the rover under the bell using YOLO bbox + depth + active tilt
servoing (no SLAM). Mirrors:
  - perception/detection/realtime_infer.py    (YOLO loading + RealSense)
  - perception/detection/visual_servo_target.py (depth ROI median)
  - Driving/visual_servo_driver.py            (FSM-driven control loop)

Phase 2 is NOT included — this is Phase 1 only ("drive to directly under the bell").

Backends:
  --backend ultralytics  : PyTorch YOLO .pt (CUDA/CPU/MPS). default.
  --backend hailo        : Hailo-8/8L NPU with .hef (Pi5 AI HAT+).
                           uses perception.detection.hailo_yolo26 decoder.

Usage:
    All parameters are loaded from config.yaml. The CLI only accepts the
    --config / --dry-run / --debug-detect toggles (run_phase1_visual_servo.py
    shares the same yaml as pipeline.py even when run standalone).

    python3 run_phase1_visual_servo.py
    python3 run_phase1_visual_servo.py --config configs/sweep.yaml
    python3 run_phase1_visual_servo.py --dry-run        # serial off (no rover)
"""
from __future__ import annotations

import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np

from config_loader import load_args, visual_servo_config_from_args
from Driving.visual_servo_controller import VisualServoController
from Driving.visual_servo_driver import VisualServoPhase1Driver
from Driving.wheel_motor import WheelMotorClient, WheelMotorConfig
from LevelingPlatform.tilt_motor import (
    TiltAsyncClient,
    TiltClient,
    TiltMotorConfig,
)
from perception.common.realsense_wrapper import RealSenseCamera
from perception.config import CAMERA
from perception.detection.visual_servo_target import compute_target_depth

HERE = Path(__file__).resolve().parent
TRAINING_RUNS = HERE / "perception" / "training" / "runs"
INDOOR_PT_FALLBACK = HERE / "perception" / "detection" / "indoor.pt"
INDOOR_HEF_FALLBACK = HERE / "perception" / "detection" / "indoor.hef"


def find_latest_best(runs_root: Path) -> Path:
    """Newest perception/training/runs/*/weights/best.pt by mtime.

    Mirrors perception/detection/realtime_infer.py:find_latest_best, inlined
    here so we don't import that module (it pulls in cv2 at load-time, which
    is unnecessary for this script).
    """
    candidates = list(runs_root.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no best.pt under {runs_root}/*/weights/. "
            "Train first or pass --weights."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)

IMGSZ = 640
WARMUP_FRAMES = 30

BBox = Tuple[int, int, int, int]


# ───────────────────────────── detector backends ────────────────────────────
class Detector(Protocol):
    """Backend-agnostic single-best-detection interface used by RealRobot."""

    def predict(self, color_bgr: np.ndarray) -> Optional[Tuple[BBox, float]]:
        """Return ((x1,y1,x2,y2), conf) for the highest-conf detection, or None."""
        ...


class UltralyticsDetector:
    """PyTorch YOLO backend (ultralytics). Stateless wrt frames."""

    def __init__(self, weights: Path, conf: float, device: str,
                 class_filter: Optional[List[int]]):
        from ultralytics import YOLO   # lazy: avoid torch import for hailo backend
        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device
        self.class_filter = class_filter

    def predict(self, color_bgr: np.ndarray) -> Optional[Tuple[BBox, float]]:
        kwargs = dict(
            source=color_bgr, imgsz=IMGSZ, conf=self.conf,
            device=self.device, verbose=False, save=False, stream=False,
        )
        if self.class_filter is not None:
            kwargs["classes"] = self.class_filter

        res = self.model.predict(**kwargs)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        confs = res.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        x1, y1, x2, y2 = (int(round(v)) for v in res.boxes.xyxy[idx].cpu().numpy())
        return (x1, y1, x2, y2), float(confs[idx])


# ─────────────────────────────── robot adapter ──────────────────────────────
class RealRobot:
    """Adapter wiring RealSense + Detector + serial into the Robot interface
    expected by VisualServoPhase1Driver.

    Class name is exactly "RealRobot" — the driver inspects
    `type(robot).__name__` to decide whether to apply hardware-paced timing
    (settle_s between sweep steps, dt sleep in the main loop)."""

    def __init__(
        self,
        camera: RealSenseCamera,
        detector: Detector,
        wheel: WheelMotorClient,
        tilt_async: TiltAsyncClient,
        tilt_sync: TiltClient,
        roi_frac: float = 0.4,
        min_valid_pixels: int = 10,
        debug_detect: bool = False,
    ):
        self.camera = camera
        self.detector = detector
        self.wheel = wheel
        self.tilt_async = tilt_async
        self.tilt_sync = tilt_sync
        self.roi_frac = roi_frac
        self.min_valid_pixels = min_valid_pixels
        self.debug_detect = debug_detect
        self._tilt_deg: float = 0.0

    # ── visual-servo detection ──
    def get_visual_servo_detection(self):
        color, depth, depth_frame = self.camera.get_frames()
        if color is None or depth_frame is None:
            if self.debug_detect:
                print("  [detect] no frame")
            return None

        # spec §9: single-object assumption — backend returns highest-conf bbox
        pred = self.detector.predict(color)
        if pred is None:
            if self.debug_detect:
                print("  [detect] no YOLO box")
            return None
        bbox, conf_val = pred

        # depth_m may be None: D435i depth is only intermittently valid for a
        # bell (curved/metal) at long range (~2.5m) because the IR pattern
        # doesn't return. Treating a depth failure as "target lost" would drop
        # the steering lock and rotate blindly into SEARCH → falling into a
        # limit cycle. Depth is only needed for forward speed, so if a bbox
        # exists, pass depth_m=None through unchanged so the controller keeps
        # steering/tilting and only stops driving forward.
        depth_m = compute_target_depth(
            depth, bbox,
            roi_frac=self.roi_frac,
            min_valid_pixels=self.min_valid_pixels,
            depth_scale_m=0.001,
        )
        if self.debug_detect:
            x1, y1, x2, y2 = bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            dstr = f"{depth_m:.2f}m" if depth_m is not None else "None"
            print(f"  [detect] conf={conf_val:.2f} cx={cx:6.1f} cy={cy:6.1f} "
                  f"bbox=({x1},{y1},{x2},{y2}) wh=({x2-x1}x{y2-y1}) depth={dstr}")
        return {"bbox": bbox, "conf": conf_val, "depth_m": depth_m}

    # ── tilt ──
    def get_tilt_deg(self) -> float:
        return float(self._tilt_deg)

    def send_tilt_async(self, step: int) -> None:
        self.tilt_async.send(step)
        self._tilt_deg = float(step) / self.tilt_async.cfg.steps_per_deg

    def tilt_camera(self, deg: float) -> None:
        """Sync TILT (motion-complete) — used by the tilt-sweep bootstrap."""
        step = self.tilt_sync.step_from_deg(deg)
        self.tilt_sync.tilt(step)
        self._tilt_deg = float(deg)

    # ── wheel ──
    def send_wheel_omegas(self, omega_left: float, omega_right: float,
                          dt: float) -> None:
        self.wheel.drive(omega_left, omega_right)


# ─────────────────────────────── CLI ────────────────────────────────────────
# All knobs live in config.yaml; this runner only accepts --config / --dry-run
# / --debug-detect from CLI (see config_loader.load_args).


def resolve_weights(arg: Optional[Path]) -> Path:
    if arg is not None:
        if not arg.is_file():
            raise FileNotFoundError(f"weights not found: {arg}")
        return arg
    try:
        return find_latest_best(TRAINING_RUNS)
    except FileNotFoundError:
        if INDOOR_PT_FALLBACK.is_file():
            return INDOOR_PT_FALLBACK
        raise FileNotFoundError(
            "no weights found under perception/training/runs/*/weights/best.pt "
            f"and no fallback at {INDOOR_PT_FALLBACK}. "
            "Train first or pass --weights."
        )


def resolve_hef(arg: Optional[Path]) -> Path:
    if arg is not None:
        if not arg.is_file():
            raise FileNotFoundError(f"hef not found: {arg}")
        return arg
    if INDOOR_HEF_FALLBACK.is_file():
        return INDOOR_HEF_FALLBACK
    raise FileNotFoundError(
        f"no .hef at {INDOOR_HEF_FALLBACK}. "
        "Compile via perception/detection/HAILO_HEF_CONVERT.md or pass --hef."
    )


def build_detector_ctx(args):
    """Return a context manager yielding a Detector instance for the chosen backend.

    ultralytics: weights loaded eagerly; returned via nullcontext (no cleanup).
    hailo     : HailoYolo26Detector owns VDevice + InferVStreams lifecycle.
    """
    if args.backend == "ultralytics":
        weights = resolve_weights(args.weights)
        print(f"[phase1] backend : ultralytics ({weights})")
        print(f"[phase1] device  : {args.device}")
        return nullcontext(
            UltralyticsDetector(weights, args.conf, args.device, args.classes)
        )

    from perception.detection.hailo_yolo26 import HailoYolo26Detector
    hef = resolve_hef(args.hef)
    print(f"[phase1] backend : hailo ({hef})")
    return HailoYolo26Detector(hef, args.conf)


# ─────────────────────────────── main ───────────────────────────────────────
def main():
    args = load_args(
        prog="phase1",
        allow_overrides=("dry_run", "debug_detect"),
    )
    print(f"[phase1] config  : {args.config_path}")

    detector_ctx = build_detector_ctx(args)

    print(f"[phase1] conf    : {args.conf}")
    print(f"[phase1] port    : {args.port}{' (dry-run)' if args.dry_run else ''}")
    print(f"[phase1] dt      : {args.dt:.3f}s ({1.0 / args.dt:.1f}Hz)")
    print(f"[phase1] v_max   : {args.v_max:.2f} m/s   ω_max: {args.omega_max:.2f} rad/s")
    print(f"[phase1] coast   : {args.coast_frames} frames × {args.coast_scale:.2f}")
    print(f"[phase1] creep   : {args.creep_v:.2f} m/s × {args.creep_s:.1f}s "
          f"(retries: {args.creep_retries})")
    print(f"[phase1] timeout : {args.phase1_timeout:.1f}s")

    wheel_cfg = WheelMotorConfig(
        port=args.port, baud=args.baud, dry_run=args.dry_run,
        # Since tilt_sync._ser = wheel._ser is shared, set the wheel-side
        # timeout to 5s too so the sync TILT motion-complete (waitMotion up to
        # 4s) isn't cut off. The wheel's sync commands (PING/STOP) respond
        # immediately, so there's no side effect.
        sync_read_timeout_sec=5.0,
    )
    tilt_cfg = TiltMotorConfig(
        port=args.port, baud=args.baud, dry_run=args.dry_run,
    )

    # WheelMotorClient owns the single OpenRB serial; the tilt clients piggy-back
    # on its open file descriptor (OpenRB has one USB-CDC, spec §6.2).
    # hardware_reset_on_start=True: recover from USB suspend
    # ('failed to set power state') after a dirty prior session. ~5s startup delay.
    with ExitStack() as stack:
        detector = stack.enter_context(detector_ctx)
        camera = stack.enter_context(
            RealSenseCamera(CAMERA, hardware_reset_on_start=True)
        )
        wheel = stack.enter_context(WheelMotorClient(wheel_cfg))
        camera.warmup(num_frames=WARMUP_FRAMES)

        tilt_async = TiltAsyncClient(tilt_cfg)
        tilt_sync = TiltClient(tilt_cfg)
        if not args.dry_run:
            tilt_async._ser = wheel._ser   # shared FD; single-threaded loop
            tilt_sync._ser = wheel._ser

        robot = RealRobot(
            camera=camera, detector=detector, wheel=wheel,
            tilt_async=tilt_async, tilt_sync=tilt_sync,
            roi_frac=args.depth_roi_frac,
            min_valid_pixels=args.depth_min_valid,
            debug_detect=args.debug_detect,
        )

        # Full yaml-derived VisualServoConfig (kp_v / d_stop_m / debounce / …
        # — same values pipeline.py builds via config_loader).
        ctrl = VisualServoController(visual_servo_config_from_args(args))
        driver = VisualServoPhase1Driver(
            robot=robot,
            target_provider=None,    # real path doesn't use the sim bypass
            ctrl=ctrl,
            dt=args.dt,
            timeout_s=args.phase1_timeout,
            log_every_s=args.log_every,
            bootstrap_creep_v=args.creep_v,
            bootstrap_creep_s=args.creep_s,
            bootstrap_creep_retries=args.creep_retries,
        )

        ok = driver.run()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
