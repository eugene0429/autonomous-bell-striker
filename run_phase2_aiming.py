#!/usr/bin/env python3
"""Phase 2 aiming & strike runner — real RealSense + YOLO + leveling + flywheel.

Sequence: camera 90° up → 1 s measurement (multi-frame median plate-frame target) →
LevelingIK → leveling motors AIM → STRIKE (flywheel + loader).
Phase 1 is NOT included — assumes the robot is already positioned ~under
the bell. Mirrors:
  - run_phase1_visual_servo.py                       (detector backends, hardware lifecycle)
  - perception/detection/phase2_target.py            (CameraToPlateExtrinsic + 1s median provider)
  - pipeline.py:CapstonePipeline.phase2_aiming       (orchestration — re-implemented here standalone)

Backends:
  --backend ultralytics  : PyTorch YOLO .pt (CUDA/CPU/MPS). default.
  --backend hailo        : Hailo-8/8L NPU with .hef (Pi5 AI HAT+).
                           uses perception.detection.hailo_yolo26 decoder.

Usage:
    All parameters are loaded from config.yaml (shares the same yaml as pipeline.py).
    The CLI only accepts the --config / --dry-run toggles.

    python3 run_phase2_aiming.py
    python3 run_phase2_aiming.py --config configs/lead.yaml
    python3 run_phase2_aiming.py --dry-run              # serial off (no hardware)

    The aim mode is selected via the yaml `phase2.aim_mode` key: static | lead | center.
"""
from __future__ import annotations

import sys
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np

from config_loader import load_args
from LevelingPlatform.leveling_ik import LevelingConfig, LevelingIK
from LevelingPlatform.leveling_motor import LevelingMotorClient, MotorClientConfig
from LevelingPlatform.tilt_motor import TiltClient, TiltMotorConfig
from perception.common.realsense_wrapper import RealSenseCamera
from perception.config import CAMERA
from perception.detection.phase2_target import (
    CameraToPlateExtrinsic,
    Phase2MeasurementError,
    Phase2TargetEstimator,
    RealPhase2TargetProvider,
)

HERE = Path(__file__).resolve().parent
TRAINING_RUNS = HERE / "perception" / "training" / "runs"
INDOOR_PT_FALLBACK = HERE / "perception" / "detection" / "indoor.pt"
INDOOR_HEF_FALLBACK = HERE / "perception" / "detection" / "outdoor_v2.hef"

IMGSZ = 640
WARMUP_FRAMES = 30

BBox = Tuple[int, int, int, int]


# ───────────────────────── detector backends ────────────────────────────
class Detector(Protocol):
    """Backend-agnostic single-best-detection interface (mirrors phase1 runner)."""

    def predict(self, color_bgr: np.ndarray) -> Optional[Tuple[BBox, float]]:
        """Return ((x1,y1,x2,y2), conf) for the highest-conf detection, or None."""
        ...


def find_latest_best(runs_root: Path) -> Path:
    """Newest perception/training/runs/*/weights/best.pt by mtime.

    Inlined (also present in run_phase1_visual_servo.py) so this runner does
    not import the phase1 script (avoids accidental coupling).
    """
    candidates = list(runs_root.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no best.pt under {runs_root}/*/weights/. "
            "Train first or pass --weights."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


class UltralyticsDetector:
    """PyTorch YOLO backend (ultralytics)."""

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


class _PredictToDetectAdapter:
    """Wraps a phase1-style Detector (`.predict() -> Optional[(bbox, conf)]`)
    to match `Phase2TargetEstimator`'s expected interface (`.detect() ->
    list[{bbox, conf}]`). Always returns 0 or 1 detection (single-bell
    assumption — spec §8 'Multi-bell scene' is out-of-scope)."""

    def __init__(self, detector: Detector):
        self._d = detector

    def detect(self, color):
        pred = self._d.predict(color)
        if pred is None:
            return []
        bbox, conf = pred
        return [{"bbox": bbox, "conf": conf}]


# ──────────────────── robot adapter (tilt + leveling + flywheel) ────────────────────
class RealPhase2Robot:
    """Adapter providing tilt_camera + send_leveling_angles + fire — the
    interface CapstonePipeline.phase2_aiming uses.

    Single OpenRB serial connection lifecycle owned by `leveling`
    (LevelingMotorClient). `tilt` and the inline STRIKE command piggy-back on
    that file descriptor (one USB-CDC port on OpenRB-150, per
    COMMUNICATION_PROTOCOL.md §7).
    """

    def __init__(
        self,
        tilt: TiltClient,
        leveling: LevelingMotorClient,
        fire_rpm: int = 8000,
        fire_hold_ms: int = 1000,
    ):
        self.tilt = tilt
        self.leveling = leveling
        self.fire_rpm = fire_rpm
        self.fire_hold_ms = fire_hold_ms
        self._fired = 0

    # ── tilt ──
    def tilt_camera(self, deg: float) -> None:
        step = self.tilt.step_from_deg(deg)
        self.tilt.tilt(step)
        print(f"[phase2] camera tilt → {deg:+.1f}° (step={step})")

    # ── leveling ──
    def send_leveling_angles(self, angles_deg, encoder_steps) -> None:
        # LevelingMotorClient.aim() takes the full IK result dict but only
        # consumes `angles_steps`. Wrap the steps in the minimal shape.
        self.leveling.aim({"angles_steps": encoder_steps})
        print(f"[phase2] leveling AIM steps={encoder_steps}")

    # ── flywheel + loader (STRIKE convenience cmd: SPIN → wait → LOAD → SPIN 0 0) ──
    def fire(self) -> None:
        self._fired += 1
        # COMMUNICATION_PROTOCOL.md §4: STRIKE <rpm> <hold_ms> is a single
        # sync command that handles spin-up, load (=actual shot), and spin-down.
        # Using leveling._command keeps a single serial owner; a future
        # LauncherClient facade (proto §7) can replace this without changing
        # the Phase 2 sequence.
        cmd = f"STRIKE {self.fire_rpm} {self.fire_hold_ms}"
        self.leveling._command(cmd)
        print(f"[phase2] *** FIRE #{self._fired} *** ({cmd})")

    # ── split SPIN/LOAD (lead-aim mode) ──
    # Decouples spin-up latency (~1 s) from per-shot trigger, so lead-aim
    # can pre-spin once and trigger LOAD (instantly) at the right moment.
    # COMMUNICATION_PROTOCOL.md §4: SPIN/LOAD are independent sync commands;
    # STRIKE is just SPIN→sleep→LOAD→SPIN 0 0 wrapped on the OpenRB side.
    def spin_up(self, rpm: int) -> None:
        self.leveling._command(f"SPIN {rpm} {rpm}")
        print(f"[phase2] SPIN {rpm} {rpm}")

    def spin_down(self) -> None:
        self.leveling._command("SPIN 0 0")
        print("[phase2] SPIN 0 0")

    def load(self) -> None:
        self._fired += 1
        self.leveling._command("LOAD")
        print(f"[phase2] *** LOAD #{self._fired} ***")


# ─────────────────── launcher correction (shared) ───────────────────────
def _apply_launcher_correction(target_xyz: np.ndarray, ik: LevelingIK, args) -> np.ndarray:
    """Subtract launcher exit offset + small-angle barrel tilt from a
    plate-frame target so plate normal aims the launcher at the bell
    rather than the plate center. Same first-order approximation (R≈I) used by
    both static and lead-aim modes."""
    launcher_offset = np.array([
        args.launcher_offset_x,
        args.launcher_offset_y,
        args.launcher_offset_z,
    ])
    aim_xyz = np.asarray(target_xyz, dtype=float) - launcher_offset
    a_x = np.deg2rad(args.launcher_tilt_x_deg)
    a_y = np.deg2rad(args.launcher_tilt_y_deg)
    if a_x != 0.0 or a_y != 0.0:
        plate_center = np.array([0.0, 0.0, ik.cfg.H0])
        d = float(np.linalg.norm(aim_xyz - plate_center))
        aim_xyz = aim_xyz - d * np.array(
            [np.sin(a_x), np.sin(a_y), 0.0]
        )
    return aim_xyz


# ─────────────────── phase 2 sequence (standalone) ──────────────────────
def run_phase2(
    robot: RealPhase2Robot,
    provider: RealPhase2TargetProvider,
    ik: LevelingIK,
    args,
) -> bool:
    """Mirrors CapstonePipeline.phase2_aiming with args-style hyperparams,
    independent of pipeline.py orchestrator.

    Kept in-sync manually with pipeline.py:phase2_aiming. If the orchestration
    logic there grows non-trivial, lift it to a shared helper instead.
    """
    print(f"── PHASE 2: AIMING & STRIKE x{args.num_strikes} ──")

    robot.tilt_camera(args.tilt_deg)
    # Explicit leveling home so plate starts from a known neutral pose
    # regardless of what phase 1 (or a prior run) left it in.
    robot.send_leveling_angles([0.0, 0.0, 0.0], [0, 0, 0])
    time.sleep(args.tilt_settle_sec)

    successful = 0
    cached_aim = None  # static mode: first successful aim reused for later shots
    for shot in range(1, args.num_strikes + 1):
        print(f"\n  ── shot {shot}/{args.num_strikes} ──")

        if args.static and cached_aim is not None:
            out = cached_aim
            print(f"  [static] reusing prior aim → "
                  f"angles_steps={out['angles_steps']}")
        else:
            try:
                target_xyz = provider.get_phase2_target()
            except Phase2MeasurementError as e:
                print(f"  ✗ measurement failed: {e} — skip shot")
                continue

            print(f"  target (plate frame): ({target_xyz[0]:+.3f}, "
                  f"{target_xyz[1]:+.3f}, {target_xyz[2]:+.3f}) m")

            aim_xyz = _apply_launcher_correction(np.asarray(target_xyz), ik, args)
            out = ik.aim_at(aim_xyz)
            if out["angles_deg"] is None:
                print("  ✗ leg length infeasible — skip")
                continue

            ball = ", ".join(f"{b:.2f}" for b in out["ball_deg"])
            print(f"  motor angles : {[f'{a:+.3f}' for a in out['angles_deg']]} deg")
            print(f"  encoder steps: {out['angles_steps']}")
            print(f"  ball P deg   : [{ball}] (lim={ik.cfg.ball_max_deg})")
            print(f"  feasible     : {out['ok']}")
            if not out["ok"]:
                print("  ✗ ball joint limit exceeded — skip shot "
                      "(Phase 1 positioning assumption violated)")
                continue
            cached_aim = out

        robot.send_leveling_angles(out["angles_deg"], out["angles_steps"])
        time.sleep(args.plate_settle_sec)
        robot.fire()
        successful += 1

        if shot < args.num_strikes:
            time.sleep(args.strike_interval)

    print(f"\n  → {successful}/{args.num_strikes} strikes executed")
    return successful == args.num_strikes


# ─────────────── center-aim calibration helpers ─────────────────────────
def _center_aim_calibrate(camera, estimator, duration_s: float):
    """Observe the bell for `duration_s` and return
    (z_top, z_bot, x, y, tz_samples).

    Used to bootstrap the center-aim mode without a hardcoded z_center —
    just point the camera up and the bell's own motion reveals its top /
    bottom of swing. Caller must ensure `duration_s` is long enough to
    cover one full period worst-case (≥ 2 · H_max).

    `tz_samples` is a list of (t_monotonic, z_plate) tuples for every
    valid detection during the window, in time order. Caller seeds
    CenterAimTracker with this so endpoint detection has already
    triggered (and v̂ is well-defined) before the first shot is
    considered — avoids the "wait one half-cycle for the first endpoint"
    delay that lead-aim suffers.

    Returns None if zero valid detections in the window.

    Endpoint extraction (z_top / z_bot): 5-tap boxcar smoothing →
    min/max. Smoothing suppresses single-frame depth outliers (~30 mm
    noise spikes in aimlog) that would bias raw min/max by tens of mm.
    """
    print(f"\n  ── calibration: observe bell motion for {duration_s:.1f}s ──")
    tz_samples: List[Tuple[float, float]] = []
    xy_samples: List[Tuple[float, float]] = []
    t_start = time.monotonic()
    t_last_log = t_start
    while time.monotonic() - t_start < duration_s:
        color, depth_image, _ = camera.get_frames()
        if color is None:
            continue
        p_plate = estimator.estimate(color, depth_image)
        t_now = time.monotonic()
        if p_plate is not None:
            tz_samples.append((t_now, float(p_plate[2])))
            xy_samples.append((float(p_plate[0]), float(p_plate[1])))
        if t_now - t_last_log > 2.0:
            if tz_samples:
                zs = [s[1] for s in tz_samples]
                print(f"    [t={t_now-t_start:4.1f}s] samples={len(tz_samples)}, "
                      f"z range=[{min(zs):.3f}, {max(zs):.3f}] m "
                      f"(swing {(max(zs)-min(zs))*100:.1f} cm)")
            else:
                print(f"    [t={t_now-t_start:4.1f}s] no detections yet")
            t_last_log = t_now

    if not tz_samples:
        return None

    z_arr = np.asarray([s[1] for s in tz_samples])
    # 5-tap boxcar smoothing — extends edges so length is preserved.
    if z_arr.size >= 5:
        z_padded = np.pad(z_arr, 2, mode='edge')
        z_smooth = np.convolve(z_padded, np.ones(5) / 5.0, mode='valid')
    else:
        z_smooth = z_arr
    z_top = float(z_smooth.max())
    z_bot = float(z_smooth.min())

    xs = np.array([s[0] for s in xy_samples])
    ys = np.array([s[1] for s in xy_samples])
    x_med, y_med = float(np.median(xs)), float(np.median(ys))

    print(f"  → {len(tz_samples)} samples, "
          f"z_top={z_top:.3f}, z_bot={z_bot:.3f} m, "
          f"swing={(z_top-z_bot)*100:.1f} cm")
    print(f"  → bell lateral (median): "
          f"x={x_med:+.3f} m (σ={xs.std()*1000:.1f} mm), "
          f"y={y_med:+.3f} m (σ={ys.std()*1000:.1f} mm)")
    return z_top, z_bot, x_med, y_med, tz_samples


def _center_aim_measure_xy_only(camera, estimator, duration_s: float):
    """Short-window (x, y) median measurement for manual-endpoint mode.

    Returns (x, y) or None if no detections.
    """
    print(f"\n  ── lateral calibration ({duration_s:.1f}s) ──")
    xy_samples: List[Tuple[float, float]] = []
    t_start = time.monotonic()
    while time.monotonic() - t_start < duration_s:
        color, depth_image, _ = camera.get_frames()
        if color is None:
            continue
        p_plate = estimator.estimate(color, depth_image)
        if p_plate is not None:
            xy_samples.append((float(p_plate[0]), float(p_plate[1])))
    if not xy_samples:
        return None
    xs = np.array([s[0] for s in xy_samples])
    ys = np.array([s[1] for s in xy_samples])
    x_med, y_med = float(np.median(xs)), float(np.median(ys))
    print(f"  bell lateral (median of {len(xy_samples)} samples): "
          f"x={x_med:+.3f} m, y={y_med:+.3f} m  "
          f"(σ_x={xs.std()*1000:.1f} mm, σ_y={ys.std()*1000:.1f} mm)")
    return x_med, y_med


# ─────────────── phase 2 lead-aim sequence (moving target) ──────────────
def run_phase2_lead_aim(
    robot: RealPhase2Robot,
    camera,
    estimator,
    ik: LevelingIK,
    args,
) -> bool:
    """Lead-aim variant for a vertically oscillating bell (triangular wave,
    random half-period in [H_min, H_max]).

    Sequence (per shot):
      1. Pre-SPIN flywheel once at the start (decoupled from per-shot trigger
         via the SPIN/LOAD split — STRIKE convenience cmd is NOT used here).
      2. Stream camera frames → estimator.estimate(color, depth) → feed z to
         BellMotionTracker. tracker bootstraps z_center after one endpoint
         detection.
      3. When `tracker.ready` and `tracker.is_safe_to_fire(Δt)` are both True
         AND the inter-shot lockout has elapsed, compute predicted z* =
         z_now + v·Δt, AIM at (x_now, y_now, z*), wait `plate_settle_sec`,
         then LOAD (instantly).
      4. If not safe (predicted z crosses an endpoint within Δt), keep
         observing — caller's "skip a beat" policy: do nothing this frame
         and re-evaluate next frame. The half-cycle reversal will be picked
         up by endpoint detection within a few frames.
      5. After all shots, SPIN 0 0.

    The (x, y) used per shot is the most recent per-frame estimate (handles
    post-strike lateral sway). The 1-second median provider is bypassed.
    """
    # Lazy import: only needed in lead-aim mode, keeps static mode lean.
    from perception.detection.phase2_lead_aim import (
        BellMotionTracker, LeadAimParams,
    )

    print(f"── PHASE 2 (LEAD AIM): AIMING & STRIKE x{args.num_strikes} ──")
    print(f"[phase2 lead] amplitude={args.lead_amplitude_m*100:.1f} cm, "
          f"H ∈ [{args.lead_half_period_min_s:.1f}, {args.lead_half_period_max_s:.1f}] s, "
          f"Δt={args.lead_total_delay_sec:.2f} s, "
          f"inter_shot={args.lead_inter_shot_sec:.1f} s, "
          f"margin={args.lead_safety_margin_m*100:.1f} cm")

    robot.tilt_camera(args.tilt_deg)
    robot.send_leveling_angles([0.0, 0.0, 0.0], [0, 0, 0])
    time.sleep(args.tilt_settle_sec)

    params = LeadAimParams(
        amplitude_m=args.lead_amplitude_m,
        half_period_min_s=args.lead_half_period_min_s,
        half_period_max_s=args.lead_half_period_max_s,
        safety_margin_m=args.lead_safety_margin_m,
    )
    tracker = BellMotionTracker(params)
    last_xy: Optional[Tuple[float, float]] = None

    # Pre-spin once for all shots. Stays on through inter-shot waits;
    # OpenRB has no watchdog on SPIN (proto §3.2) so this is safe.
    robot.spin_up(args.fire_rpm)
    print(f"[phase2 lead] flywheel warmup {args.lead_spin_warmup_sec:.2f} s")
    t_spin_start = time.monotonic()
    # Use warmup window to start populating the tracker — wastes nothing.
    while time.monotonic() - t_spin_start < args.lead_spin_warmup_sec:
        color, depth_image, _ = camera.get_frames()
        if color is None:
            continue
        p_plate = estimator.estimate(color, depth_image)
        if p_plate is not None:
            last_xy = (float(p_plate[0]), float(p_plate[1]))
            tracker.update(time.monotonic(), float(p_plate[2]))

    successful = 0
    last_shot_t: Optional[float] = None
    delay_s = args.lead_total_delay_sec

    try:
        for shot in range(1, args.num_strikes + 1):
            print(f"\n  ── shot {shot}/{args.num_strikes} (lead) ──")
            t_shot_start = time.monotonic()
            fired = False
            last_log_t = 0.0

            while time.monotonic() - t_shot_start < args.lead_max_wait_sec:
                color, depth_image, _ = camera.get_frames()
                if color is None:
                    continue
                p_plate = estimator.estimate(color, depth_image)
                t_now = time.monotonic()
                if p_plate is not None:
                    last_xy = (float(p_plate[0]), float(p_plate[1]))
                    tracker.update(t_now, float(p_plate[2]))

                # Enforce minimum inter-shot interval.
                if last_shot_t is not None and t_now - last_shot_t < args.lead_inter_shot_sec:
                    continue

                # Periodic status log so the wait isn't silent.
                if t_now - last_log_t > 1.0:
                    if tracker.ready:
                        latest = tracker.latest_sample()
                        z_now = latest[1] if latest is not None else float("nan")
                        print(f"    [wait] z={z_now:+.3f} m, "
                              f"v={tracker.velocity*100:+.1f} cm/s, "
                              f"center={tracker.z_center:+.3f}, "
                              f"endpoints={tracker.endpoints_seen}, "
                              f"safe={tracker.is_safe_to_fire(delay_s)}")
                    else:
                        n = len(tracker._samples)
                        print(f"    [wait] tracker warming up "
                              f"(samples={n}, endpoints={tracker.endpoints_seen})")
                    last_log_t = t_now

                if not tracker.ready or last_xy is None:
                    continue
                if not tracker.is_safe_to_fire(delay_s):
                    continue

                # Safe — predict & fire.
                z_pred = tracker.predict_z(delay_s)
                tau = tracker.time_to_next_endpoint()
                print(f"  v={tracker.velocity*100:+.2f} cm/s, "
                      f"z_now={tracker.latest_sample()[1]:+.3f}, "
                      f"z*={z_pred:+.3f} (+{delay_s:.2f}s), "
                      f"τ_endpoint={tau:.2f} s")

                target_xyz = np.array([last_xy[0], last_xy[1], z_pred])
                aim_xyz = _apply_launcher_correction(target_xyz, ik, args)
                out = ik.aim_at(aim_xyz)
                if out["angles_deg"] is None or not out["ok"]:
                    print("  ✗ IK infeasible at predicted z — keep observing")
                    continue

                robot.send_leveling_angles(out["angles_deg"], out["angles_steps"])
                time.sleep(args.plate_settle_sec)
                robot.load()
                last_shot_t = time.monotonic()
                successful += 1
                fired = True
                break

            if not fired:
                print(f"  ✗ shot {shot}: no safe opportunity in "
                      f"{args.lead_max_wait_sec:.1f} s — skip")

        # Hold 2s after the last LOAD before SPIN 0 0 — prevents an RPM drop while the projectile is in flight.
        if last_shot_t is not None:
            hold = 2.0 - (time.monotonic() - last_shot_t)
            if hold > 0:
                time.sleep(hold)
        print(f"\n  → {successful}/{args.num_strikes} strikes executed")
        return successful == args.num_strikes
    finally:
        robot.spin_down()


# ─────────────── phase 2 center-aim sequence (moving target) ─────────────
def run_phase2_center_aim(
    robot: RealPhase2Robot,
    camera,
    estimator,
    ik: LevelingIK,
    args,
) -> bool:
    """Static-center-aim variant for a vertically oscillating bell.

    Premise (verified on aimlog 2026-05-25): the bell's trajectory endpoints
    z_top, z_bot are known a priori (pre-mission calibration). The aim point
    z_center = (z_top + z_bot) / 2 is fixed. Lateral (x, y) is quasi-static —
    measured once at the start. The plate is AIM'd ONCE at
    (x_bell, y_bell, z_center) and held; per-shot we only LOAD (instantly) when
    the bell's predicted z falls inside (bell_radius − safety_margin) of
    z_center.

    Compared to lead-aim:
      + No per-shot AIM motion → smaller Δt (≈ LOAD ack + ball flight);
        works inside fast half-cycles where lead-aim's safety condition
        2·(A − margin) − 2·|v|·Δt > 0 fails (e.g. H=1.7 s, A=16 cm).
      + Endpoint priors only used to plan ROUGH ranges; the actual
        z_top / z_bot are auto-calibrated from the start-of-phase
        observation window (see below). Lead-aim's online z_center
        bootstrap needs ≥1 endpoint observation mid-mission, which
        delays the first shot; center-aim moves that delay up-front
        into the explicit calibration step.
      + IK + ball-joint feasibility solved once.
      − Hit window is the bell radius (~6 cm), narrower than lead-aim's
        "aim at predicted bell position with same radius". So this mode
        needs the velocity fit to be accurate enough for
        |v̂·Δt + σ_z| < bell_radius − margin (aimlog: σ_pred ≈ 16 mm
        at Δt=0.4 s → well inside 6 cm).

    Sequence:
      1. (one-time) Observe bell motion for `center_calibration_sec`
         seconds. Take 5-tap smoothed min/max of z as z_bot / z_top
         (= trajectory endpoints in plate frame). Take per-axis median
         of (x, y) as bell lateral position. If `center_calibration_sec`
         is 0, fall back to `--center-z-top` / `--center-z-bot` CLI args
         plus a short separate (x, y) window.
      2. (one-time) Solve IK at (x_bell, y_bell, z_center) where
         z_center = (z_top + z_bot) / 2. Reject mission if leg length /
         ball joint limits violated.
      3. (one-time) AIM plate; pre-SPIN flywheel; brief warmup window
         feeds the tracker so v̂ is ready before first shot.
      4. Per frame: tracker.update(t, z). If tracker.ready and
         tracker.should_fire(Δt) and inter-shot cooldown elapsed → LOAD.
      5. After all shots: SPIN 0 0.
    """
    # Lazy import: keeps the static / lead-aim paths lean.
    from perception.detection.phase2_center_aim import (
        CenterAimParams, CenterAimTracker,
    )

    print(f"── PHASE 2 (CENTER AIM): AIMING & STRIKE x{args.num_strikes} ──")

    robot.tilt_camera(args.tilt_deg)
    robot.send_leveling_angles([0.0, 0.0, 0.0], [0, 0, 0])
    time.sleep(args.tilt_settle_sec)

    # ── Calibration: observe bell motion → z_top, z_bot, x, y ──
    # Worst-case full period = 2 · H_max (start at extremum → wait H_max for
    # next, then H_max for the other). Default --center-calibration-sec is
    # tuned for H_max = 6 s. Set to 0 to use the explicit CLI overrides
    # below (debug / re-runs of a known scene).
    seed_samples: List[Tuple[float, float]] = []
    if args.center_calibration_sec > 0:
        cal = _center_aim_calibrate(
            camera, estimator,
            duration_s=args.center_calibration_sec,
        )
        if cal is None:
            print(f"  ✗ calibration produced no detections — abort")
            return False
        z_top, z_bot, x_bell, y_bell, seed_samples = cal
        # Sanity: if observed swing < bell diameter, the calibration window
        # almost certainly missed an extremum — z_center will be biased.
        if (z_top - z_bot) < args.center_bell_diameter:
            print(f"  ✗ observed swing {(z_top-z_bot)*100:.1f} cm < bell Ø "
                  f"{args.center_bell_diameter*100:.1f} cm — calibration "
                  f"window likely too short; increase "
                  f"--center-calibration-sec and retry")
            return False
    else:
        # MANUAL endpoint mode: use CLI args for z, short window for x/y.
        z_top, z_bot = args.center_z_top, args.center_z_bot
        cal = _center_aim_measure_xy_only(
            camera, estimator, duration_s=args.center_xy_meas_sec,
        )
        if cal is None:
            print(f"  ✗ no detections in xy calibration window — abort")
            return False
        x_bell, y_bell = cal
        print(f"  using manual endpoints (--center-calibration-sec=0): "
              f"z_top={z_top:.3f}, z_bot={z_bot:.3f} m")

    params = CenterAimParams(
        z_top_m=z_top,
        z_bot_m=z_bot,
        bell_diameter_m=args.center_bell_diameter,
        safety_margin_m=args.center_safety_margin_m,
        fit_window_samples=args.center_fit_window_samples,
        direction_streak_required=args.center_direction_streak,
        direction_streak_eps_m=args.center_direction_streak_eps,
        endpoint_v_eps_mps=args.center_endpoint_v_eps_mps,
    )
    print(f"[phase2 center] z_top={params.z_top_m:.3f} m, "
          f"z_bot={params.z_bot_m:.3f} m, z_center={params.z_center:.3f} m, "
          f"A={params.amplitude_m*100:.1f} cm")
    print(f"[phase2 center] bell Ø={params.bell_diameter_m*100:.1f} cm, "
          f"margin={params.safety_margin_m*100:.1f} cm, "
          f"fire window=|z*-c|≤{params.fire_window_half_m*100:.1f} cm")
    print(f"[phase2 center] Δt={args.center_total_delay_sec:.2f} s, "
          f"inter_shot={args.center_inter_shot_sec:.1f} s")
    print(f"[phase2 center] bell lateral: x={x_bell:+.3f} m, y={y_bell:+.3f} m")

    # ── one-time IK + AIM ──
    target_xyz = np.array([x_bell, y_bell, params.z_center])
    print(f"  target (plate frame): ({target_xyz[0]:+.3f}, "
          f"{target_xyz[1]:+.3f}, {target_xyz[2]:+.3f}) m")
    aim_xyz = _apply_launcher_correction(target_xyz, ik, args)
    out = ik.aim_at(aim_xyz)
    if out["angles_deg"] is None:
        print("  ✗ leg length infeasible at z_center — abort mission")
        return False
    ball = ", ".join(f"{b:.2f}" for b in out["ball_deg"])
    print(f"  motor angles  : {[f'{a:+.3f}' for a in out['angles_deg']]} deg")
    print(f"  encoder steps : {out['angles_steps']}")
    print(f"  ball P deg    : [{ball}] (lim={ik.cfg.ball_max_deg})")
    print(f"  feasible      : {out['ok']}")
    if not out["ok"]:
        print("  ✗ ball joint limit exceeded at z_center — abort "
              "(Phase 1 positioning assumption violated)")
        return False
    robot.send_leveling_angles(out["angles_deg"], out["angles_steps"])

    # ── Tracker init: seed with calibration samples so endpoint detection
    #    has already fired before the first shot decision (avoids the
    #    "wait one half-cycle for first endpoint" delay) ──
    tracker = CenterAimTracker(params)
    for (t_cal, z_cal) in seed_samples:
        tracker.update(t_cal, z_cal)
    if seed_samples:
        print(f"[phase2 center] tracker seeded with {len(seed_samples)} "
              f"calibration samples → {tracker.endpoints_seen} endpoint(s) detected")

    # ── pre-SPIN flywheel + warmup window (refreshes v̂ on fresh frames) ──
    robot.spin_up(args.fire_rpm)
    settle_total = max(args.center_spin_warmup_sec, args.plate_settle_sec)
    print(f"[phase2 center] flywheel warmup + plate settle = {settle_total:.2f} s")

    t_warm_end = time.monotonic() + settle_total
    while time.monotonic() < t_warm_end:
        color, depth_image, _ = camera.get_frames()
        if color is None:
            continue
        p_plate = estimator.estimate(color, depth_image)
        if p_plate is not None:
            tracker.update(time.monotonic(), float(p_plate[2]))

    # ── per-shot loop: track z, fire LOAD when prediction enters window ──
    successful = 0
    last_shot_t: Optional[float] = None
    delay_s = args.center_total_delay_sec

    try:
        for shot in range(1, args.num_strikes + 1):
            print(f"\n  ── shot {shot}/{args.num_strikes} (center) ──")
            t_shot_start = time.monotonic()
            fired = False
            last_log_t = 0.0

            while time.monotonic() - t_shot_start < args.center_max_wait_sec:
                color, depth_image, _ = camera.get_frames()
                if color is None:
                    continue
                p_plate = estimator.estimate(color, depth_image)
                t_now = time.monotonic()
                if p_plate is not None:
                    tracker.update(t_now, float(p_plate[2]))

                # Enforce minimum inter-shot interval.
                if last_shot_t is not None and t_now - last_shot_t < args.center_inter_shot_sec:
                    continue

                # Periodic status log.
                if t_now - last_log_t > 1.0:
                    if tracker.ready:
                        z_now = tracker.latest_sample()[1]
                        z_pred = tracker.predict_z(delay_s)
                        off_cm = (z_pred - params.z_center) * 100
                        print(f"    [wait] z={z_now:+.3f} m, "
                              f"v={tracker.velocity*100:+.1f} cm/s, "
                              f"z*={z_pred:+.3f} m (offset {off_cm:+.2f} cm, "
                              f"need |offset|≤{params.fire_window_half_m*100:.1f}), "
                              f"endpoints={tracker.endpoints_seen}")
                    else:
                        v_str = (f"{tracker.velocity*100:+.1f} cm/s"
                                 if tracker.velocity is not None else "NA")
                        print(f"    [wait] tracker not ready "
                              f"(samples={len(tracker._samples)}, "
                              f"cycle={len(tracker._cycle)}, "
                              f"v={v_str}, endpoints={tracker.endpoints_seen}, "
                              f"streak={tracker.direction_streak}/"
                              f"{params.direction_streak_required})")
                    last_log_t = t_now

                if not tracker.ready:
                    continue
                if not tracker.should_fire(delay_s):
                    continue

                # Fire — plate already at z_center, just LOAD.
                z_pred = tracker.predict_z(delay_s)
                off_cm = (z_pred - params.z_center) * 100
                print(f"  v={tracker.velocity*100:+.2f} cm/s, "
                      f"z_now={tracker.latest_sample()[1]:+.3f}, "
                      f"z*={z_pred:+.3f} (offset {off_cm:+.2f} cm)")

                robot.load()
                last_shot_t = time.monotonic()
                successful += 1
                fired = True
                break

            if not fired:
                print(f"  ✗ shot {shot}: no firing opportunity in "
                      f"{args.center_max_wait_sec:.1f} s — skip")

        # Hold 2s after the last LOAD before SPIN 0 0 — prevents an RPM drop while the projectile is in flight.
        if last_shot_t is not None:
            hold = 2.0 - (time.monotonic() - last_shot_t)
            if hold > 0:
                time.sleep(hold)
        print(f"\n  → {successful}/{args.num_strikes} strikes executed")
        return successful == args.num_strikes
    finally:
        robot.spin_down()


# ──────────────────────────────── CLI ───────────────────────────────────
# All knobs live in config.yaml (see config_loader.py); CLI only accepts
# --config / --dry-run.


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
    """Return a context manager yielding a phase1-style Detector instance.

    ultralytics: weights loaded eagerly; returned via nullcontext (no cleanup).
    hailo     : HailoYolo26Detector owns VDevice + InferVStreams lifecycle.
    """
    if args.backend == "ultralytics":
        weights = resolve_weights(args.weights)
        print(f"[phase2] backend : ultralytics ({weights})")
        print(f"[phase2] device  : {args.device}")
        return nullcontext(
            UltralyticsDetector(weights, args.conf, args.device, args.classes)
        )

    from perception.detection.hailo_yolo26 import HailoYolo26Detector
    hef = resolve_hef(args.hef)
    print(f"[phase2] backend : hailo ({hef})")
    return HailoYolo26Detector(hef, args.conf)


# ──────────────────────────────── main ──────────────────────────────────
def main():
    args = load_args(
        prog="phase2",
        allow_overrides=("dry_run",),
    )
    print(f"[phase2] config     : {args.config_path}")

    detector_ctx = build_detector_ctx(args)

    # aim_mode is a single string in yaml → loader derives the legacy
    # boolean flags (lead_aim / center_aim) so the rest of this file stays
    # unchanged. Mutual exclusion is impossible by construction.

    if args.center_aim:
        mode_tag = " [CENTER AIM]"
    elif args.lead_aim:
        mode_tag = " [LEAD AIM]"
    elif args.static:
        mode_tag = " [STATIC: reuse 1st aim]"
    else:
        mode_tag = ""

    print(f"[phase2] conf       : {args.conf}")
    print(f"[phase2] port       : {args.port}{' (dry-run)' if args.dry_run else ''}")
    print(f"[phase2] num_strikes: {args.num_strikes}{mode_tag}")
    print(f"[phase2] meas_sec   : {args.phase2_meas_sec:.2f}s "
          f"(min valid frames: {args.phase2_min_frames})")
    print(f"[phase2] tilt_deg   : {args.tilt_deg:+.1f}°")
    print(f"[phase2] launcher Δ : "
          f"({args.launcher_offset_x*1000:+.1f}, "
          f"{args.launcher_offset_y*1000:+.1f}, "
          f"{args.launcher_offset_z*1000:+.1f}) mm  "
          f"@({args.launcher_tilt_x_deg:+.2f}, "
          f"{args.launcher_tilt_y_deg:+.2f})°  (plate frame)")
    print(f"[phase2] camera   Δ : "
          f"({args.camera_offset_x*1000:+.1f}, "
          f"{args.camera_offset_y*1000:+.1f}, "
          f"{args.camera_offset_z*1000:+.1f}) mm  (plate frame)")
    print(f"[phase2] fire       : {args.fire_rpm} rpm × {args.fire_hold_ms} ms")

    leveling_cfg = MotorClientConfig(
        port=args.port, baud=args.baud, dry_run=args.dry_run,
    )
    tilt_cfg = TiltMotorConfig(
        port=args.port, baud=args.baud, dry_run=args.dry_run,
    )

    # LevelingMotorClient owns the single OpenRB serial; TiltClient piggy-backs
    # on its open FD (OpenRB has one USB-CDC, proto §6.2).
    # hardware_reset_on_start=True: stabilize the first connection after
    # recovering from USB suspend ('failed to set power state'). ~5s startup delay.
    with ExitStack() as stack:
        detector = stack.enter_context(detector_ctx)
        camera = stack.enter_context(
            RealSenseCamera(CAMERA, hardware_reset_on_start=True)
        )
        leveling = stack.enter_context(LevelingMotorClient(leveling_cfg))
        camera.warmup(num_frames=WARMUP_FRAMES)

        tilt = TiltClient(tilt_cfg)
        if not args.dry_run:
            tilt._ser = leveling._ser   # shared FD; single-threaded loop

        ik = LevelingIK(LevelingConfig())
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
        robot = RealPhase2Robot(
            tilt=tilt, leveling=leveling,
            fire_rpm=args.fire_rpm, fire_hold_ms=args.fire_hold_ms,
        )

        if args.center_aim:
            # Center-aim also consumes the per-frame estimator directly;
            # the 1 s median provider is skipped.
            ok = run_phase2_center_aim(robot, camera, estimator, ik, args)
        elif args.lead_aim:
            # Lead-aim consumes the estimator directly (per-frame z), so the
            # 1 s median provider is skipped entirely.
            ok = run_phase2_lead_aim(robot, camera, estimator, ik, args)
        else:
            provider = RealPhase2TargetProvider(
                camera=camera,
                estimator=estimator,
                measurement_duration_s=args.phase2_meas_sec,
                min_valid_frames=args.phase2_min_frames,
            )
            ok = run_phase2(robot, provider, ik, args)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
