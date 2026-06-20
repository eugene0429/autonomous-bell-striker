# perception/vio — DEPRECATED (SLAM/VIO 측위)

> ⚠️ **이 디렉터리의 코드는 현행 파이프라인에서 사용하지 않는다.**

Phase 1 주행은 원래 **ORB-SLAM3 self-pose 기반 절대 좌표 주행**으로 설계되었으나,
**Raspberry Pi 5 + ORB-SLAM 조합의 런타임 불안정성**(실시간 트래킹 끊김·드리프트·
CPU 포화)으로 **폐기**되었다.

현행 Phase 1 은 SLAM 없이 **YOLO bbox + depth 기반 active-tilt visual servoing**
으로 종 바로 아래까지 주행한다 — [run_phase1_visual_servo.py](../../run_phase1_visual_servo.py),
[Driving/visual_servo_controller.py](../../Driving/visual_servo_controller.py) 참고.

여기 코드(`orbslam_*`, `vio_*`)와 `perception/Pangolin`·`perception/librealsense`
서브모듈은 **참고용으로만 보존**한다. 자세한 배경은
[../../SW_ARCHITECTURE.md](../../SW_ARCHITECTURE.md) 의 구현 현황 배너 참고.
