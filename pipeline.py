"""
Capstone 2026 — Full Pipeline Orchestrator
==========================================

Phase 1 (driving) → Phase 2 (aiming & strike ×N) integrated runner.

This file runs the behaviors of [run_phase1_visual_servo.py](run_phase1_visual_servo.py)
and [run_phase2_aiming.py](run_phase2_aiming.py) **in a single process**,
back to back. Sharing a single OpenRB serial + a single RealSense device
across the two phases is the core responsibility of this orchestrator.

Run modes
--------
--mode sim   : Pure Python simulation without camera/motors (runs anywhere)
--mode real  : RealSense + YOLO + OpenRB real hardware (Pi5 environment)

Phase transition
---------
[Phase 1: Driving]   YOLO bbox + depth + active tilt servoing
                      (same as run_phase1_visual_servo.py)

[Phase 2: Aiming & Strike ×N]
    camera 90° tilt → 1s measurement window (per-axis median plate-frame target)
    → launcher offset/tilt correction → LevelingIK → leveling motors AIM →
    STRIKE (flywheel + loader single command)
    Re-estimate the bell position on every strike (or reuse the first aim
    with --static).

CLI
---
All parameters are loaded from config.yaml. CLI arguments are only the three
most frequently toggled (--mode / --dry-run / --debug-detect) plus --config
for selecting the yaml path.

    python3 pipeline.py                                # config.yaml default
    python3 pipeline.py --config configs/sim.yaml      # different yaml
    python3 pipeline.py --mode sim                     # override yaml.mode
    python3 pipeline.py --dry-run --debug-detect       # toggle override
"""
from __future__ import annotations

import sys
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

# ── Add package paths (integrated script run from root) ──
ROOT = Path(__file__).resolve().parent
for sub in ("Driving", "LevelingPlatform", "perception"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from controller import ControllerConfig, DrivingController          # noqa: E402
from leveling_ik import LevelingConfig, LevelingIK                  # noqa: E402
from detection.dummy_detector import (                              # noqa: E402
    DummyTargetConfig, DummyTargetProvider,
)
from Driving.visual_servo_controller import VisualServoConfig       # noqa: E402
from config_loader import (                                          # noqa: E402
    load_args, visual_servo_config_from_args,
)

# ── Same default paths/constants as the run_phase~ scripts ──
TRAINING_RUNS = ROOT / "perception" / "training" / "runs"
INDOOR_PT_FALLBACK = ROOT / "perception" / "detection" / "indoor.pt"
INDOOR_HEF_FALLBACK = ROOT / "perception" / "detection" / "outdoor_v2.hef"

IMGSZ = 640
WARMUP_FRAMES = 30

BBox = Tuple[int, int, int, int]


# ─────────────────────────────────────────────────────────────────────
# Detector backends (same implementation as the run_phase~ scripts; kept in manual sync)
# ─────────────────────────────────────────────────────────────────────
class _Detector(Protocol):
    """Backend-agnostic single-best-detection. Mirrors phase1/phase2 runner."""

    def predict(self, color_bgr: np.ndarray) -> Optional[Tuple[BBox, float]]: ...


def _find_latest_best(runs_root: Path) -> Path:
    candidates = list(runs_root.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no best.pt under {runs_root}/*/weights/. "
            "Train first or pass --weights."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


class _UltralyticsDetector:
    """PyTorch YOLO backend (ultralytics). Single-best-conf detection."""

    def __init__(self, weights: Path, conf: float, device: str,
                 class_filter: Optional[List[int]]):
        from ultralytics import YOLO   # lazy: hailo path doesn't need torch
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


class _PredictToDetectAdapter:
    """Phase 2 estimator expects detector.detect(color) → list[{bbox, conf}].
    Wrap phase1-style .predict() → list of 0 or 1."""

    def __init__(self, detector: _Detector):
        self._d = detector

    def detect(self, color):
        pred = self._d.predict(color)
        if pred is None:
            return []
        bbox, conf = pred
        return [{"bbox": bbox, "conf": conf}]


def _resolve_weights(arg: Optional[Path]) -> Path:
    if arg is not None:
        if not arg.is_file():
            raise FileNotFoundError(f"weights not found: {arg}")
        return arg
    try:
        return _find_latest_best(TRAINING_RUNS)
    except FileNotFoundError:
        if INDOOR_PT_FALLBACK.is_file():
            return INDOOR_PT_FALLBACK
        raise FileNotFoundError(
            "no weights found under perception/training/runs/*/weights/best.pt "
            f"and no fallback at {INDOOR_PT_FALLBACK}. "
            "Train first or pass --weights."
        )


def _resolve_hef(arg: Optional[Path]) -> Path:
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


def _build_detector_ctx(args):
    """Context-manager yielding a Detector instance for the chosen backend."""
    if args.backend == "ultralytics":
        weights = _resolve_weights(args.weights)
        print(f"[pipeline] backend : ultralytics ({weights})")
        print(f"[pipeline] device  : {args.device}")
        return nullcontext(
            _UltralyticsDetector(weights, args.conf, args.device, args.classes)
        )
    from perception.detection.hailo_yolo26 import HailoYolo26Detector
    hef = _resolve_hef(args.hef)
    print(f"[pipeline] backend : hailo ({hef})")
    return HailoYolo26Detector(hef, args.conf)


# ─────────────────────────────────────────────────────────────────────
# Robot adapters — Phase 1 visual servo detection bridge + motor sinks.
# ─────────────────────────────────────────────────────────────────────
class SimulatedRobot:
    """Pure Python simulation. Differential-drive forward kinematics + dummy detection."""

    def __init__(
        self,
        start_xy: Tuple[float, float] = (0.0, 0.0),
        start_theta: float = 0.0,
        wheel_diameter: float = 0.10,
        wheel_base: float = 0.30,
    ):
        self.x, self.y = start_xy
        self.theta = start_theta
        self.wheel_diameter = wheel_diameter
        self.wheel_base = wheel_base
        self._tilt_deg = 0.0
        self._fired = 0

    # ── lifecycle ──
    def start(self) -> None:
        print(f"[SIM] robot ready @ ({self.x:.2f}, {self.y:.2f}, "
              f"{np.degrees(self.theta):.1f}°)")

    def stop(self) -> None:
        print(f"[SIM] robot shutdown (fired {self._fired} times)")

    # ── motor sinks ──
    def send_wheel_omegas(self, omega_left: float, omega_right: float, dt: float) -> None:
        r = self.wheel_diameter / 2.0
        v_L = omega_left * r
        v_R = omega_right * r
        v   = 0.5 * (v_L + v_R)
        w   = (v_R - v_L) / self.wheel_base
        self.x    += v * np.cos(self.theta) * dt
        self.y    += v * np.sin(self.theta) * dt
        self.theta = self._wrap_angle(self.theta + w * dt)

    # ── tilt ──
    def tilt_camera(self, deg: float) -> None:
        self._tilt_deg = deg
        print(f"[SIM] camera tilt → {deg:+.1f}°  (sync)")

    def send_tilt_async(self, step: int) -> None:
        self._tilt_deg = step * (90.0 / 1024.0)

    def get_tilt_deg(self) -> float:
        return float(self._tilt_deg)

    # ── visual-servo detection bypass (sim only) ──
    def set_visual_servo_target_provider(self, provider) -> None:
        """Driver injects the DummyTargetProvider so robot can synthesize bbox."""
        self._vs_provider = provider

    def get_visual_servo_detection(self):
        if not hasattr(self, "_vs_provider") or self._vs_provider is None:
            return None
        return self._vs_provider.get_visual_servo_detection(
            robot_x=self.x, robot_y=self.y, robot_theta=self.theta,
            tilt_deg=self._tilt_deg,
        )

    def send_leveling_angles(self, angles_deg, encoder_steps) -> None:
        print(f"[SIM] leveling motors ← deg={[f'{a:+.2f}' for a in angles_deg]}  "
              f"steps={encoder_steps}")

    def fire(self) -> None:
        self._fired += 1
        print(f"[SIM] *** FIRE #{self._fired} ***")

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class RealRobot:
    """Real hardware adapter — RealSense + YOLO + OpenRB.

    The hardware clients (camera, detector, wheel, tilt, leveling) are
    lifecycle-managed by an ExitStack in main() and injected here. RealRobot
    does not construct the clients itself — an external owner is needed to
    share the single OpenRB serial FD across phases
    ([COMMUNICATION_PROTOCOL.md](docs/COMMUNICATION_PROTOCOL.md) §6.2).
    """

    # The class name "RealRobot" is inspected by VisualServoPhase1Driver via
    # `type(robot).__name__` (sim/real pacing difference) — do not rename it.

    def __init__(
        self,
        camera,
        detector: _Detector,
        wheel,
        tilt_async,
        tilt_sync,
        leveling,
        fire_rpm: int = 8000,
        fire_hold_ms: int = 1000,
        roi_frac: float = 0.4,
        min_valid_pixels: int = 10,
        debug_detect: bool = False,
    ):
        self.camera = camera
        self.detector = detector
        self.wheel = wheel
        self.tilt_async = tilt_async
        self.tilt_sync = tilt_sync
        self.leveling = leveling
        self.fire_rpm = fire_rpm
        self.fire_hold_ms = fire_hold_ms
        self.roi_frac = roi_frac
        self.min_valid_pixels = min_valid_pixels
        self.debug_detect = debug_detect
        self._tilt_deg: float = 0.0
        self._fired: int = 0

    # ── lifecycle ──
    def start(self) -> None:
        # The camera / serial / detector are already entered by the ExitStack,
        # and warmup is already completed in main().
        pass

    def stop(self) -> None:
        print(f"[REAL] shutdown (fired {self._fired} times)")

    # ── Phase 1 visual servo: bbox + depth from RealSense + YOLO ──
    def get_visual_servo_detection(self):
        # Mirrors run_phase1_visual_servo.RealRobot.get_visual_servo_detection.
        from perception.detection.visual_servo_target import compute_target_depth

        color, depth, depth_frame = self.camera.get_frames()
        if color is None or depth_frame is None:
            if self.debug_detect:
                print("  [detect] no frame")
            return None

        pred = self.detector.predict(color)
        if pred is None:
            if self.debug_detect:
                print("  [detect] no YOLO box")
            return None
        bbox, conf_val = pred

        # depth_m may be None: D435i depth is only intermittently valid for a
        # bell (curved/metal) at long range (~2.5m) because the IR pattern
        # doesn't return. Treating a depth failure as "target lost" would drop
        # the steering lock and rotate into SEARCH → limit cycle. If a bbox
        # exists, pass depth_m=None so the controller keeps steering/tilting
        # and only stops driving forward.
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

    # ── wheel ──
    def send_wheel_omegas(self, omega_left: float, omega_right: float,
                          dt: float) -> None:
        self.wheel.drive(omega_left, omega_right)

    # ── tilt ──
    def get_tilt_deg(self) -> float:
        return float(self._tilt_deg)

    def send_tilt_async(self, step: int) -> None:
        self.tilt_async.send(step)
        self._tilt_deg = float(step) / self.tilt_async.cfg.steps_per_deg

    def tilt_camera(self, deg: float) -> None:
        """Sync TILT (motion-complete)."""
        step = self.tilt_sync.step_from_deg(deg)
        self.tilt_sync.tilt(step)
        self._tilt_deg = float(deg)
        print(f"[REAL] camera tilt → {deg:+.1f}° (sync, step={step})")

    # ── leveling ──
    def send_leveling_angles(self, angles_deg, encoder_steps) -> None:
        self.leveling.aim({"angles_steps": encoder_steps})
        print(f"[REAL] leveling AIM steps={encoder_steps}")

    # ── flywheel + loader ──
    def fire(self) -> None:
        self._fired += 1
        # COMMUNICATION_PROTOCOL.md §4: STRIKE <rpm> <hold_ms> is a single sync
        # command that handles spin-up, load (=actual shot), and spin-down.
        cmd = f"STRIKE {self.fire_rpm} {self.fire_hold_ms}"
        self.leveling._command(cmd)
        print(f"[REAL] *** FIRE #{self._fired} *** ({cmd})")

    # ── split SPIN / LOAD (lead-aim, center-aim modes) ──
    # Decouples flywheel spin-up latency from per-shot trigger so the aim
    # modes can pre-spin once and LOAD-instantly at the right moment. Same
    # primitives RealPhase2Robot exposes in run_phase2_aiming.py.
    def spin_up(self, rpm: int) -> None:
        self.leveling._command(f"SPIN {rpm} {rpm}")
        print(f"[REAL] SPIN {rpm} {rpm}")

    def spin_down(self) -> None:
        self.leveling._command("SPIN 0 0")
        print("[REAL] SPIN 0 0")

    def load(self) -> None:
        self._fired += 1
        self.leveling._command("LOAD")
        print(f"[REAL] *** LOAD #{self._fired} ***")


# ─────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────
class CapstonePipeline:
    """Phase 1 (Driving) → Phase 2 (Aiming & Strike ×N) integrated runner."""

    def __init__(
        self,
        robot,
        target_provider,                       # Phase 1 dummy provider (sim path)
        ctrl: DrivingController,
        ik: LevelingIK,
        dt: float = 0.067,                     # 15 Hz
        phase1_timeout_sec: float = 60.0,
        num_strikes: int = 2,
        strike_interval_sec: float = 1.0,
        # ── Phase 1 visual servo ──
        # Controller knobs flow through `vs_cfg` (built from yaml via
        # config_loader.visual_servo_config_from_args). Driver-level knobs
        # (log cadence, bootstrap creep) stay as discrete kwargs because
        # they're not part of VisualServoConfig.
        vs_cfg: Optional[VisualServoConfig] = None,
        vs_log_every_s: float = 0.5,
        vs_creep_v: float = 0.2,
        vs_creep_s: float = 3.0,
        vs_creep_retries: int = 3,
        # ── Phase 2 ──
        phase2_target_provider=None,
        tilt_settle_sec: float = 0.5,
        plate_settle_sec: float = 0.3,
        tilt_deg: float = 90.0,
        static_aim: bool = False,
        launcher_offset: Tuple[float, float, float] = (-0.01, 0.0, 0.0),
        launcher_tilt_deg: Tuple[float, float] = (0.0, -5.0),
        # ── Phase 2 mode dispatch (lead-aim / center-aim) ──
        # aim_mode != "static" delegates phase 2 to the run_phase2_aiming
        # helpers which need direct camera + per-frame estimator access (the
        # 1 s median provider is bypassed). For "static" these can be None.
        aim_mode: str = "static",         # "static" | "lead" | "center"
        camera=None,
        estimator=None,
        aim_args=None,                     # argparse.Namespace, passed to
                                           # run_phase2_lead_aim /
                                           # run_phase2_center_aim verbatim
    ):
        self.robot = robot
        self.target_provider = target_provider
        self.ctrl = ctrl
        self.ik = ik
        self.dt = dt
        self.phase1_timeout_sec = phase1_timeout_sec
        self.num_strikes = num_strikes
        self.strike_interval_sec = strike_interval_sec

        # vs_cfg is built externally so the same yaml-derived
        # VisualServoConfig is used by pipeline + standalone phase1 runner.
        if vs_cfg is None:
            vs_cfg = VisualServoConfig(
                wheel_diameter=ctrl.cfg.wheel_diameter,
                wheel_base=ctrl.cfg.wheel_base,
                dt=dt,
            )
        self.vs_cfg = vs_cfg
        self.vs_log_every_s = vs_log_every_s
        self.vs_creep_v = vs_creep_v
        self.vs_creep_s = vs_creep_s
        self.vs_creep_retries = vs_creep_retries

        self.phase2_target_provider = phase2_target_provider or target_provider
        self.tilt_settle_sec = tilt_settle_sec
        self.plate_settle_sec = plate_settle_sec
        self.tilt_deg = tilt_deg
        self.static_aim = static_aim
        self.launcher_offset = np.asarray(launcher_offset, dtype=float)
        self.launcher_tilt_deg = launcher_tilt_deg

        # Phase 2 mode dispatch (see __init__ docstring for valid modes).
        if aim_mode not in ("static", "lead", "center"):
            raise ValueError(
                f"aim_mode must be 'static' | 'lead' | 'center', got {aim_mode!r}"
            )
        self.aim_mode = aim_mode
        self.camera = camera
        self.estimator = estimator
        self.aim_args = aim_args

    def run(self) -> bool:
        print("=" * 70)
        print("  Capstone 2026 Pipeline START")
        print("=" * 70)

        # Explicit leveling home — so it doesn't start from the residual pose of a prior run.
        print("[PIPELINE] leveling home → aim 0 0 0")
        self.robot.send_leveling_angles([0.0, 0.0, 0.0], [0, 0, 0])

        ok = self.phase1_driving()
        if not ok:
            print("[PIPELINE] Phase 1 failed → abort")
            return False

        print()
        ok = self.phase2_aiming()
        print()
        print("=" * 70)
        print(f"  Capstone Pipeline {'COMPLETE' if ok else 'FAILED in Phase 2'}")
        print("=" * 70)
        return ok

    # ── Phase 1 (visual servo) ──
    def phase1_driving(self) -> bool:
        from Driving.visual_servo_controller import VisualServoController
        from Driving.visual_servo_driver import VisualServoPhase1Driver

        vs_ctrl = VisualServoController(self.vs_cfg)
        driver = VisualServoPhase1Driver(
            self.robot, self.target_provider, vs_ctrl,
            dt=self.dt,
            timeout_s=self.phase1_timeout_sec,
            log_every_s=self.vs_log_every_s,
            bootstrap_creep_v=self.vs_creep_v,
            bootstrap_creep_s=self.vs_creep_s,
            bootstrap_creep_retries=self.vs_creep_retries,
        )
        return driver.run()

    # ── Phase 2 ──  (mirrors run_phase2_aiming.run_phase2)
    def phase2_aiming(self) -> bool:
        # Dispatch by mode. Lead / center modes delegate to the canonical
        # implementations in run_phase2_aiming so behavior matches the
        # standalone runner exactly (no manual sync needed).
        if self.aim_mode in ("lead", "center"):
            if self.camera is None or self.estimator is None or self.aim_args is None:
                raise RuntimeError(
                    f"aim_mode={self.aim_mode!r} requires camera, estimator, "
                    "and aim_args to be passed to CapstonePipeline "
                    "(real mode only — sim path is static aim)"
                )
            # Lazy import: avoid run_phase2_aiming → torch/hailo deps when
            # static-aim or sim mode is used.
            from run_phase2_aiming import (
                RealPhase2Robot, run_phase2_center_aim, run_phase2_lead_aim,
            )
            # Wrap the pipeline RealRobot in the RealPhase2Robot interface
            # only if the underlying robot doesn't already provide the
            # required spin_up/spin_down/load methods. Pipeline.RealRobot
            # now exposes them directly (mirror of RealPhase2Robot), so
            # we pass `self.robot` through.
            run_fn = (run_phase2_center_aim if self.aim_mode == "center"
                      else run_phase2_lead_aim)
            return run_fn(self.robot, self.camera, self.estimator,
                          self.ik, self.aim_args)

        from detection.phase2_target import Phase2MeasurementError   # lazy

        print(f"── PHASE 2: AIMING & STRIKE x{self.num_strikes} ──")

        self.robot.tilt_camera(self.tilt_deg)
        # Explicit leveling home — so the plate doesn't start from the residual
        # pose of a prior run / Phase 1 (same as run_phase2_aiming.run_phase2).
        self.robot.send_leveling_angles([0.0, 0.0, 0.0], [0, 0, 0])
        time.sleep(self.tilt_settle_sec)

        successful = 0
        cached_aim = None
        for shot in range(1, self.num_strikes + 1):
            print(f"\n  ── shot {shot}/{self.num_strikes} ──")

            if self.static_aim and cached_aim is not None:
                out = cached_aim
                print(f"  [static] reusing prior aim → "
                      f"angles_steps={out['angles_steps']}")
            else:
                try:
                    target_xyz = self.phase2_target_provider.get_phase2_target()
                except Phase2MeasurementError as e:
                    print(f"  ✗ measurement failed: {e} — skip shot")
                    continue

                print(f"  target (plate frame): ({target_xyz[0]:+.3f}, "
                      f"{target_xyz[1]:+.3f}, {target_xyz[2]:+.3f}) m")

                # Launcher offset: the projectile exits from the (R @ L) position
                # along the plate normal, but for a large range relative to a small
                # L, under the R≈I assumption this is just subtracted from the target.
                aim_xyz = np.asarray(target_xyz, dtype=float) - self.launcher_offset

                # Launcher angular misalignment: if the projectile direction does
                # not exactly match the plate normal, shift the target laterally by d·sin(a).
                a_x = np.deg2rad(self.launcher_tilt_deg[0])
                a_y = np.deg2rad(self.launcher_tilt_deg[1])
                if a_x != 0.0 or a_y != 0.0:
                    plate_center = np.array([0.0, 0.0, self.ik.cfg.H0])
                    d = float(np.linalg.norm(aim_xyz - plate_center))
                    aim_xyz = aim_xyz - d * np.array(
                        [np.sin(a_x), np.sin(a_y), 0.0]
                    )

                out = self.ik.aim_at(aim_xyz)
                if out["angles_deg"] is None:
                    print("  ✗ leg length infeasible — skip")
                    continue

                ball = ", ".join(f"{b:.2f}" for b in out["ball_deg"])
                print(f"  motor angles : {[f'{a:+.3f}' for a in out['angles_deg']]} deg")
                print(f"  encoder steps: {out['angles_steps']}")
                print(f"  ball P deg   : [{ball}] (lim={self.ik.cfg.ball_max_deg})")
                print(f"  feasible     : {out['ok']}")
                if not out["ok"]:
                    print("  ✗ ball joint limit exceeded — skip shot "
                          "(Phase 1 positioning assumption violated)")
                    continue
                cached_aim = out

            self.robot.send_leveling_angles(out["angles_deg"], out["angles_steps"])
            time.sleep(self.plate_settle_sec)
            self.robot.fire()
            successful += 1

            if shot < self.num_strikes:
                time.sleep(self.strike_interval_sec)

        print(f"\n  → {successful}/{self.num_strikes} strikes executed")
        return successful == self.num_strikes


# ─────────────────────────────────────────────────────────────────────
# Hardware setup helpers (real mode)
# ─────────────────────────────────────────────────────────────────────
def _build_real_hardware(args, stack: ExitStack):
    """Construct + lifecycle-manage real hardware clients via ExitStack.

    Returns (camera, detector, wheel, tilt_async, tilt_sync, leveling).

    Single OpenRB serial: wheel is the FD owner. Tilt async/sync and leveling
    piggy-back on the same FD ([COMMUNICATION_PROTOCOL.md](docs/COMMUNICATION_PROTOCOL.md) §6.2).
    """
    from Driving.wheel_motor import WheelMotorClient, WheelMotorConfig
    from LevelingPlatform.leveling_motor import (
        LevelingMotorClient, MotorClientConfig,
    )
    from LevelingPlatform.tilt_motor import (
        TiltAsyncClient, TiltClient, TiltMotorConfig,
    )
    from perception.common.realsense_wrapper import RealSenseCamera
    from perception.config import CAMERA

    # Detector backend
    detector_ctx = _build_detector_ctx(args)
    detector = stack.enter_context(detector_ctx)

    # OpenRB single-serial FD: wheel = owner.
    wheel_cfg = WheelMotorConfig(
        port=args.port, baud=args.baud, dry_run=args.dry_run,
        # Keep the wheel side at 5s too so the tilt sync motion-complete
        # (waitMotion up to 4s) isn't cut off. The wheel sync (PING/STOP)
        # responds immediately, so there's no side effect.
        sync_read_timeout_sec=5.0,
    )
    leveling_cfg = MotorClientConfig(
        port=args.port, baud=args.baud, dry_run=args.dry_run,
    )
    tilt_cfg = TiltMotorConfig(
        port=args.port, baud=args.baud, dry_run=args.dry_run,
    )

    wheel = stack.enter_context(WheelMotorClient(wheel_cfg))

    # Leveling, tilt clients piggy-back on the wheel FD.
    leveling = LevelingMotorClient(leveling_cfg)
    tilt_async = TiltAsyncClient(tilt_cfg)
    tilt_sync = TiltClient(tilt_cfg)
    if not args.dry_run:
        leveling._ser = wheel._ser
        tilt_async._ser = wheel._ser
        tilt_sync._ser = wheel._ser

    # Camera: start + warmup before Phase 1 begins.
    camera = stack.enter_context(
        RealSenseCamera(CAMERA, hardware_reset_on_start=True)
    )
    camera.warmup(num_frames=WARMUP_FRAMES)

    return camera, detector, wheel, tilt_async, tilt_sync, leveling


# ─────────────────────────────────────────────────────────────────────
# Build & main
# ─────────────────────────────────────────────────────────────────────
def _resolve_aim_mode(args) -> str:
    """Pick the phase-2 aim mode from the CLI flags.

    Mutually exclusive: --center-aim > --lead-aim > --static > "static".
    Warns when multiple are set so a typo doesn't silently flip the mode.
    """
    flags = [getattr(args, "center_aim", False),
             getattr(args, "lead_aim", False),
             getattr(args, "static", False)]
    set_count = sum(bool(f) for f in flags)
    if set_count > 1:
        print("[pipeline] WARN multiple aim-mode flags set "
              "(--center-aim > --lead-aim > --static)")
    if getattr(args, "center_aim", False):
        return "center"
    if getattr(args, "lead_aim", False):
        return "lead"
    return "static"


def _build_pipeline_common(args, robot, phase2_target_provider,
                            camera=None, estimator=None) -> CapstonePipeline:
    """Common CapstonePipeline construction (sim + real share this).

    `camera`, `estimator` are only used by lead/center aim modes (real
    mode); pass None for static / sim paths.
    """
    target_cfg = DummyTargetConfig(
        phase1_target=(args.phase1_x, args.phase1_y),
        phase2_target=(args.phase2_x, args.phase2_y, args.phase2_z),
        phase2_jitter=args.phase2_jitter,
        vs_bbox_noise_px=args.vs_bbox_noise,
        vs_depth_noise_m=args.vs_depth_noise,
        vs_dropout_prob=args.vs_dropout,
    )
    target_provider = DummyTargetProvider(target_cfg)
    ctrl = DrivingController(ControllerConfig(
        wheel_diameter=args.wheel_diameter,
        wheel_base=args.wheel_base,
    ))
    ik = LevelingIK(LevelingConfig())

    aim_mode = _resolve_aim_mode(args)

    # All controller knobs (kp_v, d_stop_m, ...) flow through yaml →
    # config_loader.visual_servo_config_from_args (one place to keep
    # phase1 standalone + pipeline in lockstep).
    vs_cfg = visual_servo_config_from_args(
        args,
        override_dt=args.dt,
        override_wheel_diameter=ctrl.cfg.wheel_diameter,
        override_wheel_base=ctrl.cfg.wheel_base,
    )

    return CapstonePipeline(
        robot, target_provider, ctrl, ik,
        dt=args.dt,
        phase1_timeout_sec=args.phase1_timeout,
        num_strikes=args.num_strikes,
        strike_interval_sec=args.strike_interval,
        vs_cfg=vs_cfg,
        vs_log_every_s=args.log_every,
        vs_creep_v=args.creep_v,
        vs_creep_s=args.creep_s,
        vs_creep_retries=args.creep_retries,
        phase2_target_provider=phase2_target_provider,
        tilt_settle_sec=args.tilt_settle_sec,
        plate_settle_sec=args.plate_settle_sec,
        tilt_deg=args.tilt_deg,
        static_aim=args.static,
        launcher_offset=(args.launcher_offset_x,
                         args.launcher_offset_y,
                         args.launcher_offset_z),
        launcher_tilt_deg=(args.launcher_tilt_x_deg,
                           args.launcher_tilt_y_deg),
        # Mode dispatch — pass the full args namespace through so the
        # delegated runners see exactly the same knobs they would when
        # invoked standalone (run_phase2_aiming.py).
        aim_mode=aim_mode,
        camera=camera,
        estimator=estimator,
        aim_args=(args if aim_mode in ("lead", "center") else None),
    )


def main():
    # All parameters live in config.yaml (see config_loader.py). CLI is
    # limited to --config + a few common toggles (--mode / --dry-run /
    # --debug-detect).
    args = load_args(
        prog="pipeline",
        allow_overrides=("mode", "dry_run", "debug_detect"),
    )
    print(f"[pipeline] config  : {args.config_path}")

    # ── SIM path: no hardware, no ExitStack needed ──
    if args.mode == "sim":
        robot = SimulatedRobot(
            start_xy=(args.start_x, args.start_y),
            start_theta=np.deg2rad(args.start_theta_deg),
            wheel_diameter=args.wheel_diameter,
            wheel_base=args.wheel_base,
        )
        # SIM mode only supports a static target — center/lead aim need a
        # live camera, so reject them explicitly when requested in sim (fail
        # early so it's not confusing to the user).
        if args.center_aim or args.lead_aim:
            print("[pipeline] ✗ --center-aim / --lead-aim require real "
                  "hardware (camera). Re-run with --mode real, or omit "
                  "the flag for static aim.")
            sys.exit(2)

        # In SIM, Phase 2 measurement also reuses the dummy provider
        sim_dummy_cfg = DummyTargetConfig(
            phase1_target=(args.phase1_x, args.phase1_y),
            phase2_target=(args.phase2_x, args.phase2_y, args.phase2_z),
            phase2_jitter=args.phase2_jitter,
        )
        phase2_provider = DummyTargetProvider(sim_dummy_cfg)

        pipeline = _build_pipeline_common(args, robot, phase2_provider)
        robot.start()
        try:
            ok = pipeline.run()
        finally:
            robot.stop()
        sys.exit(0 if ok else 1)

    # ── REAL path: manage all hardware lifecycle with an ExitStack ──
    with ExitStack() as stack:
        (camera, detector, wheel, tilt_async, tilt_sync,
         leveling) = _build_real_hardware(args, stack)

        robot = RealRobot(
            camera=camera, detector=detector,
            wheel=wheel, tilt_async=tilt_async, tilt_sync=tilt_sync,
            leveling=leveling,
            fire_rpm=args.fire_rpm, fire_hold_ms=args.fire_hold_ms,
            roi_frac=args.depth_roi_frac,
            min_valid_pixels=args.depth_min_valid,
            debug_detect=args.debug_detect,
        )

        # Phase 2 real provider — both extrinsic + min_conf injected from the CLI.
        from detection.phase2_target import (
            CameraToPlateExtrinsic, Phase2TargetEstimator, RealPhase2TargetProvider,
        )
        extrinsic = CameraToPlateExtrinsic(
            t_x_m=args.camera_offset_x,
            t_y_m=args.camera_offset_y,
            t_z_m=args.camera_offset_z,
        )
        estimator = Phase2TargetEstimator(
            camera=camera,
            detector=_PredictToDetectAdapter(detector),
            extrinsic=extrinsic,
            roi_frac=args.depth_roi_frac,
            min_conf=args.min_conf,
        )
        phase2_provider = RealPhase2TargetProvider(
            camera=camera,
            estimator=estimator,
            measurement_duration_s=args.phase2_meas_sec,
            min_valid_frames=args.phase2_min_frames,
        )

        pipeline = _build_pipeline_common(
            args, robot, phase2_provider,
            camera=camera, estimator=estimator,
        )

        robot.start()
        try:
            ok = pipeline.run()
        finally:
            robot.stop()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
