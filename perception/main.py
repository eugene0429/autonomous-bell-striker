"""
Perception Module - Unified Entry Point
=======================================

Run modes:
  python main.py capture     → Data collection
  python main.py vio         → Custom VIO localization
  python main.py orbslam     → ORB-SLAM3 localization (default: --pi --no-imu, headless library)
  python main.py detect      → Target detection + 3D position estimation
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="RealSense D435i Perception Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mode descriptions:
  capture   Collect YOLO training data with RealSense camera
  vio       Real-time localization via custom Visual-Inertial Odometry
  orbslam   ORB-SLAM3 localization
            (default: production library = --pi + --no-imu + headless)
  detect    Target detection + depth-based 3D position estimation
        """,
    )
    parser.add_argument(
        "mode",
        choices=["capture", "vio", "detect", "orbslam"],
        help="Select run mode",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="YOLO model path (used in detect mode)",
    )
    # ── orbslam / vio 공용 ──
    parser.add_argument(
        "--imu", action="store_true",
        help="orbslam: enable IMU (default off). vio: ignored.",
    )
    parser.add_argument(
        "--no-imu", action="store_true",
        help="vio: disable IMU. orbslam: redundant (IMU is off by default).",
    )
    parser.add_argument(
        "--no-pi", action="store_true",
        help="orbslam: disable Pi-optimized yaml (default Pi mode on).",
    )
    parser.add_argument(
        "--pi", action="store_true",
        help="orbslam: redundant (Pi mode is on by default).",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="orbslam: legacy GUI test runner (with cv2 viewer + resource report).",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="vio: headless mode (terminal print). orbslam: redundant (always headless library).",
    )

    args, remaining = parser.parse_known_args()

    if args.mode == "capture":
        from data_collection.capture import RealsenseCapture, parse_args
        # Pass remaining args to capture's argparse
        sys.argv = [sys.argv[0]] + remaining
        capture_args = parse_args()
        capture = RealsenseCapture(capture_args)
        capture.start()

    elif args.mode == "vio":
        if args.headless:
            from vio.vio_runner import run_vio_headless
            run_vio_headless(use_imu=(not args.no_imu))
        else:
            from vio.vio_runner import run_vio
            run_vio(use_imu=(not args.no_imu))

    elif args.mode == "orbslam":
        import os
        os.environ.setdefault("ORBSLAM_NO_VIEWER", "1")

        # 기본값: --pi --no-imu --headless 와 동등 (production 라이브러리 모듈)
        use_imu = args.imu                  # default False
        pi_mode = not args.no_pi            # default True

        if args.gui:
            # 레거시 GUI 테스트 러너 (cv2 viewer + ResourceMonitor)
            from vio.orbslam_runner import run_orbslam
            os.environ.pop("ORBSLAM_NO_VIEWER", None)
            run_orbslam(use_imu=use_imu, pi_mode=pi_mode)
        else:
            # 새 production 모듈로 헤드리스 실행
            from vio.orbslam_localizer import (
                LocalizerConfig, _print_loop)
            _print_loop(LocalizerConfig(use_imu=use_imu, pi_mode=pi_mode))

    elif args.mode == "detect":
        print("[DETECT] Target detection + 3D position estimation mode")
        print("[TODO] Detection pipeline not yet implemented")
        print("  → See detection/detector.py, detection/position_estimator.py")


if __name__ == "__main__":
    main()
