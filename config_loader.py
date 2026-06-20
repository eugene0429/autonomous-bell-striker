"""Shared YAML config loader for Capstone 2026 runners.

Single source of truth for runtime parameters. All four entry-point scripts
(`pipeline.py`, `run_phase1_visual_servo.py`, `run_phase2_aiming.py`,
`run_center_depth_probe.py`) call `load_args()` here so they share one
`config.yaml`.

Public API:
    load_args(prog, allow_overrides=(...)) -> argparse.Namespace

The returned Namespace is **flat** — attribute names match the original
argparse `dest` names so existing code (e.g. `args.v_max`,
`args.lead_total_delay_sec`) keeps working. The flattening is done via the
explicit `_DEST_MAP` table below — adding a new param means adding one
yaml leaf + one row in the table.

Recognised CLI overrides (subset, gated by `allow_overrides`):
    --config PATH        choose a non-default yaml (always allowed)
    --mode {sim, real}   pipeline.py only
    --dry-run            phase runners + pipeline (toggles serial.dry_run)
    --debug-detect       phase 1 + pipeline (toggles debug.detect)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


# yaml-path → argparse-style dest name
# (yaml path is dot-separated; leaves map 1:1 unless the value is a list/tuple
#  that fans out into multiple dests — see `_LIST_MAP`).
_DEST_MAP: dict[str, str] = {
    # top-level
    "mode": "mode",

    # vehicle
    "vehicle.wheel_diameter": "wheel_diameter",
    "vehicle.wheel_base": "wheel_base",

    # loop
    "loop.dt": "dt",

    # sim start
    "sim_start.x": "start_x",
    "sim_start.y": "start_y",
    "sim_start.theta_deg": "start_theta_deg",

    # phase 1 — driver
    "phase1.timeout_sec": "phase1_timeout",
    "phase1.log_every_s": "log_every",

    # phase 1 — controller (VisualServoConfig 1:1)
    "phase1.controller.img_w": "img_w",
    "phase1.controller.img_h": "img_h",
    "phase1.controller.kp_h": "kp_h",
    "phase1.controller.ki_h": "ki_h",
    "phase1.controller.kd_h": "kd_h",
    "phase1.controller.kp_tilt": "kp_tilt",
    "phase1.controller.ki_tilt": "ki_tilt",
    "phase1.controller.tilt_min_deg": "tilt_min_deg",
    "phase1.controller.tilt_max_deg": "tilt_max_deg",
    "phase1.controller.tilt_max_rate_dps": "tilt_max_rate_dps",
    "phase1.controller.kp_v": "kp_v",
    "phase1.controller.v_max": "v_max",
    "phase1.controller.omega_max": "omega_max",
    "phase1.controller.d_stop_m": "d_stop_m",
    "phase1.controller.tilt_stop_range_deg": "tilt_stop_range_deg",
    "phase1.controller.stop_debounce_frames": "stop_debounce_frames",
    "phase1.controller.tilt_brake_start_deg": "tilt_brake_start_deg",
    "phase1.controller.align_enabled": "align_enabled",
    "phase1.controller.align_v": "align_v",
    "phase1.controller.align_tol_m": "align_tol_m",
    "phase1.controller.align_debounce_frames": "align_debounce_frames",
    "phase1.controller.align_timeout_s": "align_timeout_s",
    "phase1.controller.horiz_dist_lp_alpha": "horiz_dist_lp_alpha",
    "phase1.controller.tilt_err_deadband_px": "tilt_err_deadband_px",
    "phase1.controller.coast_lost_frames": "coast_frames",
    "phase1.controller.hold_lost_frames": "hold_lost_frames",
    "phase1.controller.coast_speed_scale": "coast_scale",
    "phase1.controller.search_creep_v": "search_creep_v",
    "phase1.controller.search_timeout_s": "search_timeout_s",

    # phase 1 — bootstrap
    "phase1.bootstrap.creep_v": "creep_v",
    "phase1.bootstrap.creep_s": "creep_s",
    "phase1.bootstrap.creep_retries": "creep_retries",

    # phase 1 — sim dummy target
    "phase1.sim.x": "phase1_x",
    "phase1.sim.y": "phase1_y",
    "phase1.sim.bbox_noise_px": "vs_bbox_noise",
    "phase1.sim.depth_noise_m": "vs_depth_noise",
    "phase1.sim.dropout_prob": "vs_dropout",

    # phase 2 — sequence
    "phase2.num_strikes": "num_strikes",
    "phase2.strike_interval_sec": "strike_interval",
    "phase2.tilt_settle_sec": "tilt_settle_sec",
    "phase2.plate_settle_sec": "plate_settle_sec",
    "phase2.tilt_deg": "tilt_deg",
    "phase2.aim_mode": "aim_mode",
    "phase2.static_reuse": "static",

    # phase 2 — measurement
    "phase2.measurement.duration_sec": "phase2_meas_sec",
    "phase2.measurement.min_frames": "phase2_min_frames",
    "phase2.measurement.min_conf": "min_conf",
    "phase2.measurement.depth_roi_frac": "depth_roi_frac",
    "phase2.measurement.depth_min_valid": "depth_min_valid",

    # phase 2 — sim dummy target
    "phase2.sim.x": "phase2_x",
    "phase2.sim.y": "phase2_y",
    "phase2.sim.z": "phase2_z",
    "phase2.sim.jitter": "phase2_jitter",

    # phase 2 — lead
    "phase2.lead.amplitude_m": "lead_amplitude_m",
    "phase2.lead.half_period_min_s": "lead_half_period_min_s",
    "phase2.lead.half_period_max_s": "lead_half_period_max_s",
    "phase2.lead.total_delay_sec": "lead_total_delay_sec",
    "phase2.lead.inter_shot_sec": "lead_inter_shot_sec",
    "phase2.lead.spin_warmup_sec": "lead_spin_warmup_sec",
    "phase2.lead.max_wait_sec": "lead_max_wait_sec",
    "phase2.lead.safety_margin_m": "lead_safety_margin_m",

    # phase 2 — center
    "phase2.center.calibration_sec": "center_calibration_sec",
    "phase2.center.z_top": "center_z_top",
    "phase2.center.z_bot": "center_z_bot",
    "phase2.center.bell_diameter": "center_bell_diameter",
    "phase2.center.safety_margin_m": "center_safety_margin_m",
    "phase2.center.fit_window_samples": "center_fit_window_samples",
    "phase2.center.direction_streak": "center_direction_streak",
    "phase2.center.direction_streak_eps": "center_direction_streak_eps",
    "phase2.center.endpoint_v_eps_mps": "center_endpoint_v_eps_mps",
    "phase2.center.total_delay_sec": "center_total_delay_sec",
    "phase2.center.inter_shot_sec": "center_inter_shot_sec",
    "phase2.center.xy_meas_sec": "center_xy_meas_sec",
    "phase2.center.spin_warmup_sec": "center_spin_warmup_sec",
    "phase2.center.max_wait_sec": "center_max_wait_sec",

    # detector
    "detector.backend": "backend",
    "detector.conf": "conf",
    "detector.device": "device",
    "detector.weights": "weights",
    "detector.hef": "hef",
    "detector.classes": "classes",

    # serial
    "serial.port": "port",
    "serial.baud": "baud",
    "serial.dry_run": "dry_run",

    # fire
    "fire.rpm": "fire_rpm",
    "fire.hold_ms": "fire_hold_ms",

    # debug
    "debug.detect": "debug_detect",

    # probe
    "probe.conf": "probe_conf",
    "probe.roi_frac": "probe_roi_frac",
    "probe.min_valid": "probe_min_valid",
    "probe.max_frames": "probe_max_frames",
    "probe.hw_reset": "probe_hw_reset",
}

# yaml-path → tuple of (dest_x, dest_y, [dest_z]) for list-valued leaves.
_LIST_MAP: dict[str, tuple[str, ...]] = {
    "phase2.launcher.offset": ("launcher_offset_x", "launcher_offset_y",
                               "launcher_offset_z"),
    "phase2.launcher.tilt_deg": ("launcher_tilt_x_deg", "launcher_tilt_y_deg"),
    "phase2.camera.offset": ("camera_offset_x", "camera_offset_y",
                             "camera_offset_z"),
}

# aim_mode → derived boolean flags (legacy args.lead_aim / args.center_aim).
# pipeline.py + run_phase2_aiming.py both read these as booleans.
_AIM_MODE_FLAGS = ("lead_aim", "center_aim")


def _dig(d: dict, dotted: str) -> Any:
    """Walk a dotted yaml path, return value or raise KeyError."""
    node: Any = d
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"missing config key: {dotted}")
        node = node[key]
    return node


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"config not found: {path}\n"
            f"hint: pass --config <path>, or create {DEFAULT_CONFIG.name} at "
            f"the project root."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    return data


def _flatten(cfg: dict) -> dict[str, Any]:
    """Apply _DEST_MAP + _LIST_MAP to produce a flat {dest: value} dict.

    Missing keys are *not* errors — the loader is forgiving so a script that
    only needs a subset of the params still works on a partial config. Each
    script supplies its own defaults via `defaults=` to load_args(), and
    those are used when the yaml omits a key.
    """
    flat: dict[str, Any] = {}
    for path, dest in _DEST_MAP.items():
        try:
            flat[dest] = _dig(cfg, path)
        except KeyError:
            pass  # leave unset; caller fills with default

    for path, dests in _LIST_MAP.items():
        try:
            v = _dig(cfg, path)
        except KeyError:
            continue
        if not isinstance(v, (list, tuple)) or len(v) != len(dests):
            raise ValueError(
                f"{path}: expected list of {len(dests)} numbers, got {v!r}"
            )
        for d, x in zip(dests, v):
            flat[d] = x

    # aim_mode → boolean flags
    mode = flat.get("aim_mode", "static")
    flat["lead_aim"] = (mode == "lead")
    flat["center_aim"] = (mode == "center")

    # weights / hef: yaml stores strings, the runners want Path | None
    for k in ("weights", "hef"):
        v = flat.get(k)
        if isinstance(v, str):
            flat[k] = Path(v)

    return flat


def _build_parser(prog: str, allow_overrides: Iterable[str]) -> argparse.ArgumentParser:
    """Build the minimal CLI parser. `--config` is always allowed.

    `allow_overrides` enables script-specific toggles:
      "mode"          : --mode {sim, real}      (pipeline.py)
      "dry_run"       : --dry-run               (phase1/phase2/pipeline)
      "debug_detect"  : --debug-detect          (phase1/pipeline)
      "hw_reset"      : --hw-reset              (probe)
    """
    allow = set(allow_overrides)
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help=f"yaml config path (default: {DEFAULT_CONFIG.name})")
    if "mode" in allow:
        ap.add_argument("--mode", choices=("sim", "real"), default=None,
                        help="override config.mode")
    if "dry_run" in allow:
        ap.add_argument("--dry-run", action="store_true", default=None,
                        help="override config.serial.dry_run = true")
    if "debug_detect" in allow:
        ap.add_argument("--debug-detect", action="store_true", default=None,
                        help="override config.debug.detect = true")
    if "hw_reset" in allow:
        ap.add_argument("--hw-reset", action="store_true", default=None,
                        help="override config.probe.hw_reset = true")
    return ap


def load_args(
    prog: str,
    allow_overrides: Iterable[str] = (),
    defaults: Optional[dict[str, Any]] = None,
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    """Parse minimal CLI + load yaml → flat argparse.Namespace.

    Resolution order (later wins):
      1. caller-supplied `defaults` (used when the yaml omits a key)
      2. yaml file values
      3. CLI overrides (only the small set in `allow_overrides`)
    """
    ap = _build_parser(prog, allow_overrides)
    cli = ap.parse_args(argv)

    cfg = _load_yaml(cli.config)
    flat = _flatten(cfg)

    merged: dict[str, Any] = {}
    if defaults:
        merged.update(defaults)
    merged.update(flat)

    # apply CLI overrides only when explicitly set
    if getattr(cli, "mode", None) is not None:
        merged["mode"] = cli.mode
    if getattr(cli, "dry_run", None):
        merged["dry_run"] = True
    if getattr(cli, "debug_detect", None):
        merged["debug_detect"] = True
    if getattr(cli, "hw_reset", None):
        merged["probe_hw_reset"] = True

    merged["config_path"] = cli.config
    return argparse.Namespace(**merged)


def visual_servo_config_from_args(args: argparse.Namespace, *,
                                  override_dt: Optional[float] = None,
                                  override_wheel_diameter: Optional[float] = None,
                                  override_wheel_base: Optional[float] = None):
    """Build a VisualServoConfig from a yaml-loaded Namespace.

    Centralised so pipeline.py and run_phase1_visual_servo.py construct the
    controller identically. `override_*` is only used when the caller has
    already resolved a geometry parameter (e.g. pipeline.py uses
    ControllerConfig's wheel_diameter to keep one source of truth).
    """
    from Driving.visual_servo_controller import VisualServoConfig

    tilt_range = (float(args.tilt_stop_range_deg[0]),
                  float(args.tilt_stop_range_deg[1]))

    return VisualServoConfig(
        img_w=args.img_w,
        img_h=args.img_h,
        kp_tilt=args.kp_tilt,
        ki_tilt=args.ki_tilt,
        kp_h=args.kp_h,
        ki_h=args.ki_h,
        kd_h=args.kd_h,
        kp_v=args.kp_v,
        v_max=args.v_max,
        omega_max=args.omega_max,
        tilt_min_deg=args.tilt_min_deg,
        tilt_max_deg=args.tilt_max_deg,
        tilt_max_rate_dps=args.tilt_max_rate_dps,
        d_stop_m=args.d_stop_m,
        tilt_stop_range_deg=tilt_range,
        tilt_brake_start_deg=getattr(args, "tilt_brake_start_deg", None),
        align_enabled=getattr(args, "align_enabled", False),
        align_v=getattr(args, "align_v", 0.05),
        align_tol_m=getattr(args, "align_tol_m", 0.02),
        align_debounce_frames=getattr(args, "align_debounce_frames", 5),
        align_timeout_s=getattr(args, "align_timeout_s", 10.0),
        wheel_diameter=override_wheel_diameter
            if override_wheel_diameter is not None else args.wheel_diameter,
        wheel_base=override_wheel_base
            if override_wheel_base is not None else args.wheel_base,
        coast_lost_frames=args.coast_frames,
        hold_lost_frames=args.hold_lost_frames,
        coast_speed_scale=args.coast_scale,
        search_creep_v=args.search_creep_v,
        search_timeout_s=args.search_timeout_s,
        horiz_dist_lp_alpha=args.horiz_dist_lp_alpha,
        tilt_err_deadband_px=args.tilt_err_deadband_px,
        stop_debounce_frames=args.stop_debounce_frames,
        dt=override_dt if override_dt is not None else args.dt,
    )
