"""CLI entry: ``python -m perception.detection.lead_aim_viewer``.

Two mutually-exclusive modes:

  --replay <path.npz>   load a log dumped by run_phase2_aiming --lead-log
  --live                spin up camera + detector + estimator + tracker
                        (no robot, no firing) and stream into the viewer

Live mode reuses the same CLI flags as run_phase2_aiming for camera /
detector / lead-aim parameters so the two stay in sync.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Lead-aim debug viewer (replay or live).",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", type=Path, default=None,
                     help="path to .npz produced by --lead-log")
    src.add_argument("--live", action="store_true",
                     help="spin up camera + tracker (no firing)")

    # Lead-aim params (used in both modes for visualization scaling; in
    # live mode they're also fed into the tracker).
    ap.add_argument("--lead-amplitude-m", type=float, default=0.25)
    ap.add_argument("--lead-half-period-min-s", type=float, default=3.0)
    ap.add_argument("--lead-half-period-max-s", type=float, default=6.0)
    ap.add_argument("--lead-safety-margin-m", type=float, default=0.03)
    ap.add_argument("--lead-total-delay-sec", type=float, default=0.7)

    # Live-mode camera + detector (mirror run_phase2_aiming).
    ap.add_argument("--backend", choices=("ultralytics", "hailo"),
                    default="hailo")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--hef", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--classes", type=int, nargs="+", default=None)
    ap.add_argument("--depth-roi-frac", type=float, default=0.4)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--camera-offset-x", type=float, default=0.157445)
    ap.add_argument("--camera-offset-y", type=float, default=-0.010)
    ap.add_argument("--camera-offset-z", type=float, default=-0.074)

    return ap.parse_args()


def _make_replay_source(args):
    # Lazy import — matplotlib pulled in only when actually launching viewer.
    from .data_source import ReplaySource
    src = ReplaySource(args.replay)
    # Override the in-file delay with CLI if user explicitly passed one
    # different from the default. Heuristic: always use CLI value so user
    # can re-explore prediction at a different lead time.
    src.set_delay(args.lead_total_delay_sec)
    # Use replay's stored amplitude / margin for scene scaling (run-time
    # truth) but fall back to CLI if missing.
    amp = float(src.meta.get("amplitude_m", args.lead_amplitude_m))
    margin = float(src.meta.get("safety_margin_m", args.lead_safety_margin_m))
    return src, amp, margin


def _make_live_source(args):
    # Heavy imports gated to live mode only.
    from contextlib import ExitStack
    from perception.detection.phase2_lead_aim import LeadAimParams
    from perception.detection.phase2_target import (
        CameraToPlateExtrinsic, Phase2TargetEstimator,
    )

    # Import camera + detector helpers from the runner module so the
    # selection logic (weights resolution, hailo HEF resolution, etc.)
    # stays single-sourced.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from run_phase2_aiming import (
        CAMERA, WARMUP_FRAMES,
        RealSenseCamera, build_detector_ctx, _PredictToDetectAdapter,
    )

    from .data_source import LiveSource

    params = LeadAimParams(
        amplitude_m=args.lead_amplitude_m,
        half_period_min_s=args.lead_half_period_min_s,
        half_period_max_s=args.lead_half_period_max_s,
        safety_margin_m=args.lead_safety_margin_m,
    )

    stack = ExitStack()
    detector = stack.enter_context(build_detector_ctx(args))
    camera = stack.enter_context(
        RealSenseCamera(CAMERA, hardware_reset_on_start=True)
    )
    camera.warmup(num_frames=WARMUP_FRAMES)
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
    src = LiveSource(camera, estimator, params, args.lead_total_delay_sec)
    src.start()
    return src, args.lead_amplitude_m, args.lead_safety_margin_m, stack


def main() -> int:
    args = _build_args()

    # Prefer Qt5Agg for interactive 3D, fall back to TkAgg if unavailable.
    import matplotlib
    if not os.environ.get("MPLBACKEND"):
        for backend in ("Qt5Agg", "TkAgg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue

    import matplotlib.pyplot as plt
    from .viewer import LeadAimViewer

    cleanup_stack = None
    try:
        if args.replay is not None:
            source, amp, margin = _make_replay_source(args)
        else:
            source, amp, margin, cleanup_stack = _make_live_source(args)

        viewer = LeadAimViewer(source, amp, margin)
        plt.show()
    finally:
        if hasattr(source, "stop_thread"):
            source.stop_thread()
        if cleanup_stack is not None:
            cleanup_stack.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
