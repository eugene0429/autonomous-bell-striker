# perception/vio — DEPRECATED (SLAM/VIO localization)

> ⚠️ **The code in this directory is not used in the current pipeline.**

Phase 1 driving was originally designed around **ORB-SLAM3 self-pose-based absolute-coordinate driving**,
but it was **abandoned** due to the **runtime instability of the Raspberry Pi 5 + ORB-SLAM combination**
(real-time tracking dropouts, drift, CPU saturation).

The current Phase 1 drives all the way to directly underneath the bell using **YOLO bbox + depth-based
active-tilt visual servoing** without SLAM — see [run_phase1_visual_servo.py](../../run_phase1_visual_servo.py),
[Driving/visual_servo_controller.py](../../Driving/visual_servo_controller.py).

The code here (`orbslam_*`, `vio_*`) and the `perception/Pangolin` and `perception/librealsense`
submodules are **preserved for reference only**. For detailed background, see the
implementation status banner in [../../docs/SW_ARCHITECTURE.md](../../docs/SW_ARCHITECTURE.md).
