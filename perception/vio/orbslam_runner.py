"""
ORB-SLAM3 Test Runner
Launches the rgbd_inertial_realsense_D435i binary as a subprocess and:
- Auto-generates yaml with actual camera calibration
- Displays frames saved by C++ to /tmp/orbslam_frame.jpg in a Python window
- Parses POSE: lines from stdout for trajectory visualization

Usage: python main.py orbslam
"""

import subprocess
import threading
import time
import tempfile
import shutil
import collections
import numpy as np
import cv2
import os
import psutil

# ORB-SLAM3 path — auto-detect based on platform
import platform
if platform.machine().startswith("aarch") or os.path.exists("/home/team1/ORB_SLAM3"):
    ORBSLAM3_DIR = "/home/team1/ORB_SLAM3"
else:
    ORBSLAM3_DIR = "/home/sim2real1/WALJU/deps/ORB_SLAM3"

BINARY_IMU    = os.path.join(ORBSLAM3_DIR, "Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i")
BINARY_NO_IMU = os.path.join(ORBSLAM3_DIR, "Examples/RGB-D/rgbd_realsense_D435i")

CONFIG_IMU    = os.path.join(ORBSLAM3_DIR, "Examples/RGB-D-Inertial/RealSense_D435i.yaml")
CONFIG_NO_IMU = os.path.join(ORBSLAM3_DIR, "Examples/RGB-D/RealSense_D435i.yaml")
CONFIG_PI     = os.path.join(ORBSLAM3_DIR, "Examples/RGB-D/RealSense_D435i_pi.yaml")

VOCAB      = os.path.join(ORBSLAM3_DIR, "Vocabulary/ORBvoc.txt")
FRAME_PATH = "/tmp/orbslam_frame.jpg"

STATE_NAMES = {-1: "NOT_READY", 0: "NO_IMAGE", 1: "INIT", 2: "OK", 3: "RECENTLY_LOST", 4: "LOST"}


# ──────────────────────────────────────────────
# 1. Read actual camera calibration
# ──────────────────────────────────────────────

_CALIB_CACHE_PATH = "/tmp/orbslam_calib_cache.npz"


def _flush_realsense(hardware_reset: bool = False, settle_sec: float = 2.0):
    """Open and close the camera from Python to flush dirty USB state.

    Parameters
    ----------
    hardware_reset : bool
        D435i 의 USB 디바이스를 power-cycle. 'failed to set power state' 같은
        suspend / 잔류 stream config 가 원인인 경우 가장 깨끗한 복구 수단.
        USB re-enumeration 이 ~5s 걸리므로 retry path 에서만 권장.
    settle_sec : float
        flush 후 카메라가 다시 안정될 때까지 대기 시간.
    """
    try:
        import pyrealsense2 as rs
        if hardware_reset:
            try:
                ctx = rs.context()
                devs = list(ctx.query_devices())
                for d in devs:
                    try:
                        name = d.get_info(rs.camera_info.name)
                    except Exception:
                        name = "?"
                    print(f"[ORBSLAM] hardware_reset {name}")
                    d.hardware_reset()
                # USB re-enumeration 대기 (Pi 환경에서 ~5s 필요)
                time.sleep(max(settle_sec, 5.0))
            except Exception as e:
                print(f"[ORBSLAM] hardware_reset failed: {e}")

        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        pipe.start(cfg)
        # 2 frame 만 받아 stream 가동 확인 (10→2: 0.6s 단축)
        for _ in range(2):
            pipe.wait_for_frames(timeout_ms=5000)
        pipe.stop()
        del pipe
        time.sleep(settle_sec)
    except Exception:
        time.sleep(settle_sec)
    return False


def _load_cached_calibration():
    """Load cached calibration if available."""
    try:
        data = np.load(_CALIB_CACHE_PATH, allow_pickle=True)
        calib = data["calib"].item()
        return calib
    except Exception:
        return None


def _save_calibration_cache(calib):
    """Save calibration to cache file."""
    try:
        np.savez(_CALIB_CACHE_PATH, calib=calib)
    except Exception:
        pass


def get_camera_calibration(width=640, height=480, fps=30, use_imu=True):
    """Read actual D435i calibration via pyrealsense2."""
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        if use_imu:
            config.enable_stream(rs.stream.accel)
            config.enable_stream(rs.stream.gyro)

        profile = pipeline.start(config)

        # Intrinsics/extrinsics can be read directly from profile without streaming
        color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ci = color_prof.get_intrinsics()

        R, t = None, None
        if use_imu:
            accel_prof = profile.get_stream(rs.stream.accel).as_motion_stream_profile()
            extr = accel_prof.get_extrinsics_to(color_prof)  # accel → color
            R = np.array(extr.rotation).reshape(3, 3)
            t = np.array(extr.translation)

        pipeline.stop()
        del pipeline
        time.sleep(4.0)  # Wait for camera USB release (before C++ binary opens it)

        calib = {
            "fx": ci.fx, "fy": ci.fy,
            "cx": ci.ppx, "cy": ci.ppy,
            "k1": ci.coeffs[0], "k2": ci.coeffs[1],
            "p1": ci.coeffs[2], "p2": ci.coeffs[3],
            "width": ci.width, "height": ci.height,
            "R_imu_cam": R, "t_imu_cam": t,
        }
        return calib

    except Exception as e:
        print(f"[ORBSLAM] Calibration failed: {e}, using defaults")
        return None


def build_yaml(calib, base_yaml_path, out_path, orb_nfeatures=None):
    """Read base yaml, apply actual calibration, and save.

    Parameters
    ----------
    orb_nfeatures : int | None
        설정 시 base yaml 의 `ORBextractor.nFeatures` 를 이 값으로 교체.
        None 이면 base yaml 값 그대로 사용.
    """
    with open(base_yaml_path, "r") as f:
        content = f.read()

    def replace(content, key, value):
        import re
        pattern = rf"({re.escape(key)}:\s*)[^\n]*"
        return re.sub(pattern, rf"\g<1>{value}", content)

    content = replace(content, "Camera1.fx", f"{calib['fx']:.6f}")
    content = replace(content, "Camera1.fy", f"{calib['fy']:.6f}")
    content = replace(content, "Camera1.cx", f"{calib['cx']:.6f}")
    content = replace(content, "Camera1.cy", f"{calib['cy']:.6f}")
    content = replace(content, "Camera1.k1", f"{calib['k1']:.8f}")
    content = replace(content, "Camera1.k2", f"{calib['k2']:.8f}")
    content = replace(content, "Camera1.p1", f"{calib['p1']:.8f}")
    content = replace(content, "Camera1.p2", f"{calib['p2']:.8f}")
    content = replace(content, "Camera.width",  str(calib['width']))
    content = replace(content, "Camera.height", str(calib['height']))

    if orb_nfeatures is not None:
        content = replace(content, "ORBextractor.nFeatures", str(int(orb_nfeatures)))

    # IMU→Camera extrinsics (T_b_c1: body=IMU, c1=color camera)
    R = calib["R_imu_cam"]
    t = calib["t_imu_cam"]
    if R is not None and t is not None:
        T_str = (
            "IMU.T_b_c1: !!opencv-matrix\n"
            "   rows: 4\n"
            "   cols: 4\n"
            "   dt: f\n"
            f"   data: [{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}, {t[0]:.6f},\n"
            f"         {R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}, {t[1]:.6f},\n"
            f"         {R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}, {t[2]:.6f},\n"
            f"         0.0, 0.0, 0.0, 1.0]"
        )
        import re
        content = re.sub(
            r"IMU\.T_b_c1:.*?(?=\n\n|\n#|\nIMU\.Insert)",
            T_str,
            content,
            flags=re.DOTALL,
        )

    with open(out_path, "w") as f:
        f.write(content)
    pass  # yaml saved silently


# ──────────────────────────────────────────────
# 2. Visualization helpers
# ──────────────────────────────────────────────

def draw_trajectory(traj_image, positions, scale=100, size=400):
    traj_image[:] = 40
    center = np.array([size // 2, size // 2])
    if len(positions) < 2:
        return traj_image
    for i in range(1, len(positions)):
        p0, p1 = positions[i - 1], positions[i]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        pt0 = (int(np.clip(center[0] + p0[0] * scale, 0, size - 1)),
               int(np.clip(center[1] - p0[2] * scale, 0, size - 1)))
        pt1 = (int(np.clip(center[0] + p1[0] * scale, 0, size - 1)),
               int(np.clip(center[1] - p1[2] * scale, 0, size - 1)))
        cv2.line(traj_image, pt0, pt1, (0, 255, 0), 1, cv2.LINE_AA)
    cur = positions[-1]
    cur_pt = (int(np.clip(center[0] + cur[0] * scale, 0, size - 1)),
              int(np.clip(center[1] - cur[2] * scale, 0, size - 1)))
    cv2.circle(traj_image, cur_pt, 5, (0, 0, 255), -1)
    cv2.putText(traj_image, "ORB-SLAM3  top-down", (5, size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1)
    cv2.putText(traj_image, "X→", (size - 25, size // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (128, 128, 128), 1)
    cv2.putText(traj_image, "Z↑", (size // 2 + 5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (128, 128, 128), 1)
    return traj_image


class ResourceMonitor:
    """
    Samples CPU/RAM of the ORB-SLAM3 C++ process every second.
    Prints a real-time feasibility analysis report for Pi 4/5 on session end.
    """
    # Pi performance ratio relative to current host (single-core IPC × clock comparison)
    # Based on desktop i7/Ryzen ~3.5GHz
    _PI_RATIO = {
        "Pi 4 (Cortex-A72 1.5GHz)": 0.18,  # ~5-6x slower per-core
        "Pi 5 (Cortex-A76 2.4GHz)": 0.38,  # ~2.5-3x slower per-core
    }
    _CPU_CORES = {
        "Pi 4 (Cortex-A72 1.5GHz)": 4,
        "Pi 5 (Cortex-A76 2.4GHz)": 4,
    }

    def __init__(self, proc: subprocess.Popen):
        self._proc    = proc
        self._stop    = threading.Event()
        self._samples: list = []   # (cpu_pct, rss_mb)
        self._fps_samples: list = []
        self._lock    = threading.Lock()
        self._host_cores = psutil.cpu_count(logical=False) or 4
        self._host_freq  = psutil.cpu_freq()
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def record_fps(self, fps: float):
        if fps > 0:
            with self._lock:
                self._fps_samples.append(fps)

    def _loop(self):
        try:
            ps = psutil.Process(self._proc.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop.is_set():
            try:
                # Total CPU/RSS for C++ process + child threads
                cpu = ps.cpu_percent(interval=1.0)
                rss = ps.memory_info().rss / 1024 / 1024
                with self._lock:
                    self._samples.append((cpu, rss))
            except psutil.NoSuchProcess:
                break

    def stop(self):
        self._stop.set()

    def latest(self):
        """Return latest (cpu_pct, rss_mb)"""
        with self._lock:
            return self._samples[-1] if self._samples else (0.0, 0.0)

    def report(self, pi_mode: bool) -> str:
        with self._lock:
            samples = list(self._samples)
            fps_s   = list(self._fps_samples)

        if not samples:
            return "[RESOURCE] No samples collected"

        cpus = [s[0] for s in samples]
        rss  = [s[1] for s in samples]
        avg_cpu = sum(cpus) / len(cpus)
        max_cpu = max(cpus)
        avg_rss = sum(rss)  / len(rss)
        max_rss = max(rss)
        avg_fps = sum(fps_s) / len(fps_s) if fps_s else 0.0

        # CPU frequency: .current may read abnormally low in power-save state → prefer .max
        freq = self._host_freq
        if freq:
            host_freq_mhz = freq.max if (freq.max and freq.max > 100) else freq.current
        else:
            host_freq_mhz = 3500.0
        if not host_freq_mhz or host_freq_mhz < 100:
            host_freq_mhz = 3500.0  # fallback if read fails

        # Actual host cores used (psutil cpu_percent sums across all cores)
        host_cores_used = avg_cpu / 100.0 * self._host_cores  # in "host-core" units

        lines = []
        lines.append("=" * 60)
        lines.append("  ORB-SLAM3 Resource Usage Report")
        lines.append("=" * 60)
        lines.append(f"  Config: {'Pi-optimized 424x240@15fps' if pi_mode else 'Default 640x480@30fps'}")
        lines.append(f"  Host CPU: {self._host_cores} cores @ {host_freq_mhz:.0f}MHz")
        lines.append("")
        lines.append(f"  [CPU]  avg {avg_cpu:5.1f}%   max {max_cpu:5.1f}%  (all-core sum, ≈{host_cores_used:.1f} cores used)")
        lines.append(f"  [RAM]  avg {avg_rss:5.0f}MB   max {max_rss:5.0f}MB")
        lines.append(f"  [FPS]  avg {avg_fps:5.1f} fps  (based on pose output)")
        lines.append("")
        lines.append("  ── Pi Real-time Feasibility Analysis ──")
        lines.append("  (Geekbench single-core ratio: Pi4=0.18×, Pi5=0.38×)")

        target_fps = 15.0 if pi_mode else 30.0

        for pi_name, ratio in self._PI_RATIO.items():
            pi_cores = self._CPU_CORES[pi_name]
            # Pi total compute capacity (in host-core units)
            pi_compute = pi_cores * ratio
            # feasible_fps = Pi_compute / (host_compute_per_frame)
            #              = avg_fps × pi_compute / host_cores_used
            if host_cores_used > 0 and avg_fps > 0:
                feasible_fps = avg_fps * pi_compute / host_cores_used
            else:
                feasible_fps = 0.0
            # Pi CPU usage (% of all Pi cores)
            est_cpu_pct = (host_cores_used / pi_compute * 100.0) if pi_compute > 0 else 999.0
            est_cpu_per_core = est_cpu_pct / pi_cores

            # Real-time criterion: must achieve at least 80% of target FPS
            rt_ok = feasible_fps >= target_fps * 0.8

            status = "✓ Real-time feasible" if rt_ok else "✗ Real-time not feasible"
            lines.append(f"\n  {pi_name}")
            lines.append(f"    Est. CPU usage: {est_cpu_pct:5.1f}% total  ({est_cpu_per_core:.0f}% / core)")
            lines.append(f"    Est. throughput: {feasible_fps:5.1f} fps (target: {target_fps:.0f} fps)")
            lines.append(f"    Result: {status}")
            if not rt_ok:
                # Estimate required nFeatures for real-time
                needed = host_cores_used * (target_fps / avg_fps) if avg_fps > 0 else 999
                feat_base = 500 if pi_mode else 1250
                suggested = max(100, int(feat_base * pi_compute / needed))
                if feasible_fps > 0:
                    lines.append(f"    → Reduce nFeatures to {suggested} or lower, or further reduce resolution")

        lines.append("")
        # RAM: safe zone set to <1GB considering OS + process overhead on Pi 4 min 4GB model
        ram_ok = max_rss < 1000
        lines.append(f"  [RAM] Recommended <1000MB  Current: {max_rss:.0f}MB  {'✓' if ram_ok else '✗ Insufficient (caution for Pi 4 2GB model)'}")
        lines.append("=" * 60)
        return "\n".join(lines)


def draw_overlay(image, positions, fps, tracking_state, cpu_pct=0.0, rss_mb=0.0):
    pos = positions[-1] if positions else np.zeros(3)
    state_str = STATE_NAMES.get(tracking_state, f"?({tracking_state})")
    ok = tracking_state == 2
    state_color = (0, 255, 0) if ok else (0, 100, 255) if tracking_state == 3 else (0, 50, 255)
    lines = [
        (f"ORB-SLAM3  FPS:{fps:.0f}  [{state_str}]", state_color),
        (f"Poses: {len(positions)}", (200, 200, 200)),
        (f"X={pos[0]:.3f}  Y={pos[1]:.3f}  Z={pos[2]:.3f} m", (200, 200, 200)),
        (f"CPU:{cpu_pct:.0f}%  RAM:{rss_mb:.0f}MB", (180, 220, 255)),
        ("'q' quit", (100, 100, 100)),
    ]
    for i, (text, col) in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
    return image


# ──────────────────────────────────────────────
# 3. Main runner
# ──────────────────────────────────────────────

def _pose_reader(proc, pose_buf, lock, stop_evt):
    for line in proc.stdout:
        if stop_evt.is_set():
            break
        line = line.strip()
        if line.startswith("POSE:"):
            try:
                vals = list(map(float, line.split()[1:]))
                if len(vals) == 7:
                    with lock:
                        pose_buf.append(np.array(vals))
            except ValueError:
                pass


def run_orbslam_headless(use_imu=True, pi_mode=False, _max_retries=3):
    """Headless ORB-SLAM3 runner — prints world-frame (x, y, theta) to terminal.

    World coordinate convention (camera starts at origin):
      - world X = camera Z (forward)
      - world Y = camera -X (left)
      - theta   = yaw angle (deg), CCW positive viewed from above
    """
    for attempt in range(_max_retries):
        _flush_realsense()
        rc = _run_orbslam_headless_once(use_imu=use_imu, pi_mode=pi_mode)
        if rc is None or rc == 0:
            return  # Normal exit (Ctrl+C or clean shutdown)
        print(f"\n[ORBSLAM] Crash (rc={rc}), retrying ({attempt+1}/{_max_retries})...")
    print(f"[ORBSLAM] Failed after {_max_retries} attempts")


def _run_orbslam_headless_once(use_imu=True, pi_mode=False):
    """Single attempt of headless ORB-SLAM3. Returns exit code (None=Ctrl+C)."""
    import math
    from scipy.spatial.transform import Rotation

    binary   = BINARY_IMU if use_imu else BINARY_NO_IMU
    base_cfg = CONFIG_IMU if use_imu else (CONFIG_PI if pi_mode else CONFIG_NO_IMU)
    mode_str = ("RGB-D-Inertial (IMU ON)" if use_imu
                else f"RGB-D (IMU OFF{'  Pi-optimized' if pi_mode else ''})")

    for path, name in [(binary, "binary"), (VOCAB, "Vocabulary"), (base_cfg, "base yaml")]:
        if not os.path.exists(path):
            print(f"[ORBSLAM] {name} not found: {path}")
            return 0

    # Use cached calibration to avoid opening camera twice (prevents USB contention)
    calib = _load_cached_calibration()
    if not calib:
        cal_w, cal_h, cal_fps = (640, 480, 15) if pi_mode else (640, 480, 30)
        calib = get_camera_calibration(width=cal_w, height=cal_h, fps=cal_fps, use_imu=use_imu)
        if calib:
            _save_calibration_cache(calib)
    tmp_dir = tempfile.mkdtemp(prefix="orbslam_")
    config_path = os.path.join(tmp_dir, "RealSense_D435i_calib.yaml")
    if calib:
        build_yaml(calib, base_cfg, config_path)
    else:
        shutil.copy(base_cfg, config_path)

    env = os.environ.copy()
    env["ORBSLAM_NO_VIEWER"] = "1"
    orbslam_libs = [
        os.path.join(ORBSLAM3_DIR, "lib"),
        os.path.join(ORBSLAM3_DIR, "Thirdparty/DBoW2/lib"),
        os.path.join(ORBSLAM3_DIR, "Thirdparty/g2o/lib"),
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(orbslam_libs + ([existing] if existing else []))

    stdout_path = os.path.join(tmp_dir, "stdout.log")
    stderr_path = os.path.join(tmp_dir, "stderr.log")
    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")

    proc = subprocess.Popen(
        [binary, VOCAB, config_path],
        stdout=stdout_f,
        stderr=stderr_f,
        env=env,
    )

    pose_buf = []
    lock = threading.Lock()
    stop_evt = threading.Event()
    last_state = [-1]

    def _file_pose_reader(path, pose_buf, lock, stop_evt):
        """Tail the stdout log file for POSE lines."""
        with open(path, "r") as f:
            while not stop_evt.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                line = line.strip()
                if line.startswith("POSE:"):
                    try:
                        vals = list(map(float, line.split()[1:]))
                        if len(vals) == 7:
                            with lock:
                                pose_buf.append(np.array(vals))
                    except ValueError:
                        pass

    # C++ stderr lines to suppress (warnings, sensor dumps, verbose info)
    _STDERR_SUPPRESS = (
        "optional parameter", "not found", "Sensor supports",
        "Description", "Current Value", "is not supported",
        "Discarding", "dropped frs", "Fail to track",
    )

    def _file_stderr_reader(path, stop_evt):
        """Tail the stderr log file for STATE lines only."""
        with open(path, "r") as f:
            while not stop_evt.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                line = line.rstrip()
                if line.startswith("STATE:"):
                    try:
                        last_state[0] = int(line.split(":")[1].strip())
                    except ValueError:
                        pass

    threading.Thread(target=_file_pose_reader, args=(stdout_path, pose_buf, lock, stop_evt), daemon=True).start()
    threading.Thread(target=_file_stderr_reader, args=(stderr_path, stop_evt), daemon=True).start()

    print(f"[ORBSLAM] {mode_str} (headless) — Ctrl+C to stop")
    print(f"{'state':>8s}  {'x_m':>8s}  {'y_m':>8s}  {'theta_deg':>10s}  {'fps':>5s}")

    positions = []
    fps = 0.0
    frame_cnt = 0
    t_fps = time.time()

    try:
        while proc.poll() is None:
            with lock:
                new_poses, pose_buf[:] = list(pose_buf), []

            for p in new_poses:
                positions.append(p)  # [x, y, z, qx, qy, qz, qw]
                frame_cnt += 1

                # Camera frame → world frame
                cam_pos = p[:3]
                quat = p[3:7]  # [qx, qy, qz, qw] — scipy uses [x,y,z,w]
                try:
                    R = Rotation.from_quat(quat).as_matrix()
                except Exception:
                    continue

                world_x = cam_pos[2]       # camera Z = forward
                world_y = -cam_pos[0]      # camera -X = left

                forward = R[:, 2]
                theta_rad = math.atan2(-forward[0], forward[2])
                theta_deg = math.degrees(theta_rad)

                state_str = STATE_NAMES.get(last_state[0], "?")
                print(f"\r{state_str:>8s}  {world_x:8.3f}  {world_y:8.3f}  {theta_deg:10.2f}  {fps:5.1f}", end="", flush=True)

            if len(positions) > 3000:
                positions = positions[1500:]

            elapsed = time.time() - t_fps
            if elapsed >= 1.0:
                fps = frame_cnt / elapsed
                frame_cnt = 0
                t_fps = time.time()

            if not new_poses:
                time.sleep(0.01)

        # Process exited on its own
        rc = proc.returncode

    except KeyboardInterrupt:
        print("\n[ORBSLAM] Stopped (Ctrl+C)")
        rc = None  # Signal normal exit to caller
    except Exception as e:
        print(f"\n[ORBSLAM] Python exception: {e}")
        rc = -1
    finally:
        stop_evt.set()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        stdout_f.close()
        stderr_f.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(FRAME_PATH):
            os.remove(FRAME_PATH)

        if positions:
            arr = np.array([p[:3] for p in positions])
            dist = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
            print(f"\n[ORBSLAM] {len(positions)} poses, {dist:.3f}m traveled")

    return rc


def run_orbslam(use_imu=True, pi_mode=False):
    binary   = BINARY_IMU if use_imu else BINARY_NO_IMU
    base_cfg = CONFIG_IMU if use_imu else (CONFIG_PI if pi_mode else CONFIG_NO_IMU)
    mode_str = ("RGB-D-Inertial (IMU ON)" if use_imu
                else f"RGB-D (IMU OFF{'  Pi-optimized 424x240@15fps' if pi_mode else ''})")

    # Check binary and files
    for path, name in [(binary, "binary"), (VOCAB, "Vocabulary"), (base_cfg, "base yaml")]:
        if not os.path.exists(path):
            print(f"[ORBSLAM] {name} not found: {path}")
            return

    # Generate yaml with actual calibration
    cal_w, cal_h, cal_fps = (640, 480, 15) if pi_mode else (640, 480, 30)
    calib = get_camera_calibration(width=cal_w, height=cal_h, fps=cal_fps, use_imu=use_imu)
    tmp_dir = tempfile.mkdtemp(prefix="orbslam_")
    config_path = os.path.join(tmp_dir, "RealSense_D435i_calib.yaml")
    if calib:
        build_yaml(calib, base_cfg, config_path)
    else:
        shutil.copy(base_cfg, config_path)

    print(f"[ORBSLAM] {mode_str} — 'q' to quit")

    env = os.environ.copy()
    orbslam_libs = [
        os.path.join(ORBSLAM3_DIR, "lib"),
        os.path.join(ORBSLAM3_DIR, "Thirdparty/DBoW2/lib"),
        os.path.join(ORBSLAM3_DIR, "Thirdparty/g2o/lib"),
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(orbslam_libs + ([existing] if existing else []))

    proc = subprocess.Popen(
        [binary, VOCAB, config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    monitor = ResourceMonitor(proc)
    pose_buf = []
    lock = threading.Lock()
    stop_evt = threading.Event()
    last_state = [-1]

    def _stderr_collector(p):
        for line in p.stderr:
            line = line.rstrip()
            if line.startswith("STATE:"):
                try:
                    last_state[0] = int(line.split(":")[1].strip())
                except ValueError:
                    pass

    threading.Thread(target=_pose_reader,     args=(proc, pose_buf, lock, stop_evt), daemon=True).start()
    threading.Thread(target=_stderr_collector, args=(proc,), daemon=False).start()

    traj_size = 400
    traj_image = np.zeros((traj_size, traj_size, 3), dtype=np.uint8)
    positions  = []
    fps        = 0.0
    frame_cnt  = 0
    t_fps      = time.time()
    blank_cam  = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_cam, "Waiting for ORB-SLAM3 frames...",
                (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)

    try:
        while proc.poll() is None:
            # Collect new poses
            with lock:
                new_poses, pose_buf[:] = list(pose_buf), []
            for p in new_poses:
                positions.append(p[:3])
                frame_cnt += 1
            if len(positions) > 3000:
                positions = positions[1500:]

            # FPS
            elapsed = time.time() - t_fps
            if elapsed >= 1.0:
                fps = frame_cnt / elapsed
                frame_cnt = 0
                t_fps = time.time()
                monitor.record_fps(fps)

            # Camera frame (C++ saves to /tmp/orbslam_frame.jpg)
            cam_img = cv2.imread(FRAME_PATH)
            cpu_pct, rss_mb = monitor.latest()
            if cam_img is None:
                cam_img = blank_cam
            else:
                cam_img = cv2.resize(cam_img, (640, 480))
                draw_overlay(cam_img, positions, fps, last_state[0], cpu_pct, rss_mb)

            # Trajectory
            if positions:
                draw_trajectory(traj_image, positions, scale=100, size=traj_size)

            traj_resized = cv2.resize(traj_image, (480, 480))
            combined = np.hstack([cam_img, traj_resized])

            if os.environ.get("DISPLAY"):
                cv2.imshow("ORB-SLAM3 Monitor", combined)
                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'):
                    print("[ORBSLAM] User quit")
                    break
            else:
                time.sleep(0.033)

    except KeyboardInterrupt:
        print("\n[ORBSLAM] Stopped (Ctrl+C)")
    finally:
        stop_evt.set()
        monitor.stop()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        cv2.destroyAllWindows()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(FRAME_PATH):
            os.remove(FRAME_PATH)

        print(monitor.report(pi_mode))

        if positions:
            arr = np.array(positions)
            dist = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
            print(f"\n[ORBSLAM] Results:")
            print(f"  Poses: {len(positions)}")
            print(f"  Distance traveled: {dist:.3f} m")
            print(f"  Final position: X={arr[-1,0]:.3f}  Y={arr[-1,1]:.3f}  Z={arr[-1,2]:.3f} m")
        else:
            print("[ORBSLAM] No poses recorded (tracking failed)")
        print("[ORBSLAM] Shutdown complete")
