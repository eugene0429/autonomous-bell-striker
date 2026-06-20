"""
ORB-SLAM3 Localization Module — production library.

`main.py orbslam --pi --no-imu --headless` 의 동작을 모듈 형태로 분리.
파이프라인이 import 해서 world-frame (x, y, θ) pose 를 stream 으로 받아 쓸 수 있다.

사용 예
------
    from vio.orbslam_localizer import OrbSlamLocalizer, LocalizerConfig
    from controller import DrivingController, ControllerConfig

    ctrl = DrivingController(ControllerConfig(wheel_diameter=0.10, wheel_base=0.30))

    with OrbSlamLocalizer() as loc:
        loc.wait_for_tracking(timeout=30.0)
        while True:
            pose = loc.get_pose()
            if pose is None or not pose["tracking_ok"]:
                continue
            out = ctrl.compute(pose["x"], pose["y"], pose["theta"], tx, ty)
            if out["reached"]:
                break
            send_to_motors(out["wheel_omega_left"], out["wheel_omega_right"])

월드 좌표계 (카메라 출발 자세 = origin)
--------------------------------------
    world_x =  camera_z   (forward)
    world_y = -camera_x   (left)
    theta   = yaw         (CCW positive viewed from above) [rad]
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# 스크립트로 직접 실행될 때 (`python vio/orbslam_localizer.py`) `vio` 패키지를
# 찾을 수 있도록 perception/ 을 sys.path 에 prepend.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# 기존 헬퍼 재사용 (yaml 빌드, 캘리브레이션 캐시, 경로 상수 등)
from vio.orbslam_runner import (
    BINARY_IMU,
    BINARY_NO_IMU,
    CONFIG_IMU,
    CONFIG_NO_IMU,
    CONFIG_PI,
    ORBSLAM3_DIR,
    STATE_NAMES,
    VOCAB,
    _flush_realsense,
    _load_cached_calibration,
    _save_calibration_cache,
    build_yaml,
    get_camera_calibration,
)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class LocalizerConfig:
    # ── 동작 모드 (main.py 의 --pi --no-imu --headless 와 동일이 default) ──
    use_imu: bool = False        # False == --no-imu
    pi_mode: bool = True         # True  == --pi   (RGB-D 640×480@15fps + nFeatures=1500 yaml)

    # ── 시작 / 종료 ──
    max_retries: int = 5                # subprocess 기동 실패 시 재시도 횟수
    startup_alive_timeout: float = 20.0 # 첫 POSE 라인 또는 subprocess death 까지 최대 대기
                                        # (vocab 로드 5–10s + 카메라 open 마진 포함)
    startup_settle_sec: float = 0.2     # 기동 직후 즉사 detection 용 최소 대기
    skip_flush_first_attempt: bool = False  # attempt 0 에서 _flush_realsense 생략 (~3s 단축).
                                            # 직전 세션이 깨끗했을 때만 안전 — 안정성 테스트 등
                                            # 반복 실행 환경에서는 False 가 안전.
    hw_reset_on_retry: bool = True       # retry 시 D435i hardware_reset → USB suspend 회복
    hw_reset_on_first_attempt: bool = True   # 첫 시도에서도 hardware_reset (USB 뺐다 꽂은 효과).
                                             # +5s 비용. 미적용 시 D435i cold-start race 로 첫
                                             # attempt 가 ~5 POSE 후 죽고 watchdog respawn 발생.
                                             # SLAM_DEBUG.md S6 참고.
    term_timeout_sec: float = 5.0       # stop() 시 SIGTERM → SIGKILL 전환 대기

    # ── 회생 watchdog (mid-run 사망 시 자동 재기동) ──
    auto_restart: bool = True           # subprocess 가 mid-run 에 죽으면 자동 respawn
    max_auto_restarts: int = 5          # 연속 자동 재기동 한도 (초과 시 watchdog 포기)
    watchdog_poll_sec: float = 0.5      # subprocess liveness 폴링 주기

    # ── tracking 대기 ──
    tracking_poll_sec: float = 0.05     # wait_for_tracking() / startup 폴링 주기
    tracking_stability_sec: float = 2.0 # wait_for_tracking() 가 첫 OK 후 안정성 확인 대기.
                                        # 이 윈도우 안에 subprocess 가 죽으면 watchdog 가
                                        # respawn → outer loop 가 다시 OK 를 기다림.
                                        # fragile bootstrap window 를 자동 흡수.

    # ── ORB-SLAM3 yaml 오버라이드 ──
    orb_nfeatures: Optional[int] = None # None → base yaml 값. 정수 지정 시 그 값으로 교체.
                                        # init 안정성 ↑ 필요할 때 1500 → 2000 등으로 올림.

    # ── 메모리 ──
    trajectory_cap: int = 5000          # 저장 trajectory 점 상한 (초과 시 절반 폐기)

    # ── 디버그 ──
    archive_dir: Optional[str] = None   # 설정 시 _cleanup_subprocess 가 tmp_dir 을
                                        # 삭제하지 않고 이 경로로 이동 (attempt_<N>/).
                                        # 죽은 attempt 들의 stdout/stderr 비교 분석용.


# ──────────────────────────────────────────────
# Localizer
# ──────────────────────────────────────────────
class OrbSlamLocalizer:
    """ORB-SLAM3 기반 측위 모듈 (헤드리스, 라이브러리 API)."""

    def __init__(self, cfg: Optional[LocalizerConfig] = None):
        self.cfg = cfg if cfg is not None else LocalizerConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._tmp_dir: Optional[str] = None
        self._stop_evt: Optional[threading.Event] = None
        self._stdout_f = None
        self._stderr_f = None

        self._lock = threading.Lock()
        self._latest_raw: Optional[np.ndarray] = None   # 최신 [x,y,z,qx,qy,qz,qw]
        self._tracking_state: int = -1                  # NOT_READY
        self._all_positions: List[np.ndarray] = []      # 카메라-프레임 (x,y,z) 누적

        # 회생 watchdog 상태
        self._stop_watchdog: Optional[threading.Event] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._restart_count: int = 0       # mid-run 자동 재기동 누적
        self._watchdog_dead: bool = False  # cap 초과로 watchdog 포기 상태
        self._restarting: bool = False     # watchdog 가 respawn 진행 중 (consumer 는 alive 로 봐야)

    # ── context manager ──
    def __enter__(self) -> "OrbSlamLocalizer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ── lifecycle ──
    def start(self) -> None:
        """ORB-SLAM3 subprocess 기동.

        고정 sleep 대신 "첫 POSE 라인 도착 OR subprocess death" 둘 중 하나가
        먼저 일어날 때까지 대기. vocab 로드 후 카메라 reopen 단계에서 죽는
        패턴(`failed to set power state` 등)도 startup 실패로 인식해 재시도한다.
        retry 시에는 D435i hardware_reset 으로 USB power state 를 강제로 깨움.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            # attempt 0: skip_flush_first_attempt=True 면 flush 자체 생략(~3s 단축).
            # hw_reset 은 attempt>0 의 hw_reset_on_retry 또는 attempt==0 의
            # hw_reset_on_first_attempt 가 True 일 때 적용 (USB 뺐다 꽂은 효과).
            hard = ((attempt > 0) and self.cfg.hw_reset_on_retry) or \
                   ((attempt == 0) and self.cfg.hw_reset_on_first_attempt)
            if not (attempt == 0 and self.cfg.skip_flush_first_attempt):
                _flush_realsense(hardware_reset=hard)

            try:
                self._spawn_once()
            except Exception as e:
                last_err = e
                self._cleanup_subprocess()
                time.sleep(1.0)
                continue

            # ── alive 신호 대기: 첫 POSE 라인 OR subprocess death ──
            ok = self._wait_alive_signal(self.cfg.startup_alive_timeout)
            if ok:
                self._start_watchdog()
                return

            print(f"[ORBSLAM] startup failed "
                  f"({attempt+1}/{self.cfg.max_retries}); retrying"
                  f"{' with hardware_reset' if not hard else ''}...")
            self._dump_failure_tail()
            self._cleanup_subprocess()
            time.sleep(1.0)

        msg = f"ORB-SLAM3 startup failed after {self.cfg.max_retries} attempts"
        if last_err is not None:
            msg += f": {last_err}"
        raise RuntimeError(msg)

    def _start_watchdog(self) -> None:
        """기동 성공 후 mid-run 사망 감시 스레드 가동."""
        if not self.cfg.auto_restart:
            return
        self._stop_watchdog = threading.Event()
        self._restart_count = 0
        self._watchdog_dead = False
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """subprocess 가 unexpected 종료되면 자동 respawn.

        - stop() 가 _stop_watchdog 를 set 하면 자연 종료
        - cap 초과 시 watchdog_dead=True 로 포기 → is_alive() 가 False 반환하게 됨
        """
        cap = self.cfg.max_auto_restarts
        while not self._stop_watchdog.is_set():
            time.sleep(self.cfg.watchdog_poll_sec)
            if self._stop_watchdog.is_set():
                return
            proc = self._proc
            if proc is None:
                continue
            rc = proc.poll()
            if rc is None:
                continue   # 살아있음
            # subprocess 죽음 — 회생 시도. is_alive() 가 즉시 False 로 보이지
            # 않게 _restarting 을 가장 먼저 set.
            self._restarting = True
            print(f"[ORBSLAM] subprocess died mid-run (rc={rc}, "
                  f"restart {self._restart_count + 1}/{cap})")
            self._dump_failure_tail()

            if self._restart_count >= cap:
                print(f"[ORBSLAM] watchdog exhausted ({cap} restarts) — giving up")
                self._watchdog_dead = True
                with self._lock:
                    self._latest_raw = None
                    self._tracking_state = -1
                self._cleanup_subprocess()
                return

            # respawn 시작 → consumer 에게는 alive 로 보이게 유지
            self._restarting = True
            try:
                # consumer 가 stale pose 를 보지 못하도록 즉시 reset
                with self._lock:
                    self._latest_raw = None
                    self._tracking_state = -1
                # reader 스레드 종료, subprocess/임시파일 정리
                if self._stop_evt is not None:
                    self._stop_evt.set()
                self._cleanup_subprocess()

                # respawn — 항상 hardware_reset (mid-run crash 의 가장 흔한 원인은 USB)
                time.sleep(1.0)
                try:
                    _flush_realsense(hardware_reset=True)
                    self._spawn_once()
                    if self._wait_alive_signal(self.cfg.startup_alive_timeout):
                        self._restart_count += 1
                        print(f"[ORBSLAM] watchdog respawn OK "
                              f"({self._restart_count}/{cap})")
                    else:
                        print(f"[ORBSLAM] watchdog respawn failed — first POSE never arrived")
                        self._cleanup_subprocess()
                        self._restart_count += 1
                except Exception as e:
                    print(f"[ORBSLAM] watchdog respawn exception: {e}")
                    self._cleanup_subprocess()
                    self._restart_count += 1
            finally:
                self._restarting = False

    def _wait_alive_signal(self, timeout: float) -> bool:
        """startup 단계에서 다음 중 하나를 기다린다:

        - 첫 POSE 라인 도착  → True (카메라+트래킹 정상)
        - subprocess death   → False (재시도 대상)
        - timeout 까지 둘 다 안 옴 → 살아있으면 True (vocab 로드가 길어졌을 뿐)

        startup_settle_sec 를 최소 대기 하한으로 사용해 즉시 종료된 케이스를
        놓치지 않도록 한다.
        """
        t0 = time.time()
        # 최소 대기: 즉시 종료(첫 0.x 초) 케이스 detection 보장
        time.sleep(max(0.0, self.cfg.startup_settle_sec))
        while time.time() - t0 < timeout:
            if not self.is_alive():
                return False
            with self._lock:
                if self._latest_raw is not None:
                    return True
            time.sleep(self.cfg.tracking_poll_sec)
        # timeout: 살아있으면 success 로 간주 (vocab 로드 / 트래킹 init 중일 수 있음)
        return self.is_alive()

    def _dump_failure_tail(self, n_chars: int = 600) -> None:
        """실패한 subprocess 의 stdout/stderr 끝부분을 출력 — 진단용."""
        if not self._tmp_dir:
            return
        for fn in ("stderr.log", "stdout.log"):
            p = os.path.join(self._tmp_dir, fn)
            try:
                with open(p) as f:
                    data = f.read().strip()
            except Exception:
                continue
            if not data:
                continue
            tail = data[-n_chars:]
            print(f"[ORBSLAM] {fn} tail:\n  | "
                  + tail.replace("\n", "\n  | "))

    def stop(self) -> None:
        """Subprocess 종료 + watchdog/reader 스레드 정지 + 임시 파일 정리."""
        # watchdog 먼저 멈춰야 stop() 의 cleanup 이 자동 respawn 을 트리거하지 않음
        if self._stop_watchdog is not None:
            self._stop_watchdog.set()
        if self._stop_evt is not None:
            self._stop_evt.set()
        self._cleanup_subprocess()

    # ── internal: subprocess setup ──
    def _spawn_once(self) -> None:
        c = self.cfg
        binary = BINARY_IMU if c.use_imu else BINARY_NO_IMU
        base_cfg = (CONFIG_IMU if c.use_imu
                    else (CONFIG_PI if c.pi_mode else CONFIG_NO_IMU))

        for path, name in [(binary, "binary"), (VOCAB, "Vocabulary"),
                           (base_cfg, "base yaml")]:
            if not os.path.exists(path):
                raise RuntimeError(f"{name} not found: {path}")

        # 캘리브레이션 (캐시 우선)
        calib = _load_cached_calibration()
        if not calib:
            cal_w, cal_h, cal_fps = (640, 480, 15) if c.pi_mode else (640, 480, 30)
            calib = get_camera_calibration(width=cal_w, height=cal_h, fps=cal_fps,
                                           use_imu=c.use_imu)
            if calib:
                _save_calibration_cache(calib)

        # tmp yaml
        self._tmp_dir = tempfile.mkdtemp(prefix="orbslam_")
        config_path = os.path.join(self._tmp_dir, "RealSense_D435i_calib.yaml")
        if calib:
            build_yaml(calib, base_cfg, config_path,
                       orb_nfeatures=self.cfg.orb_nfeatures)
        else:
            shutil.copy(base_cfg, config_path)

        # env (헤드리스 + ORB-SLAM3 lib path)
        env = os.environ.copy()
        env["ORBSLAM_NO_VIEWER"] = "1"
        libs = [
            os.path.join(ORBSLAM3_DIR, "lib"),
            os.path.join(ORBSLAM3_DIR, "Thirdparty/DBoW2/lib"),
            os.path.join(ORBSLAM3_DIR, "Thirdparty/g2o/lib"),
        ]
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(libs + ([existing] if existing else []))

        stdout_path = os.path.join(self._tmp_dir, "stdout.log")
        stderr_path = os.path.join(self._tmp_dir, "stderr.log")
        self._stdout_f = open(stdout_path, "w")
        self._stderr_f = open(stderr_path, "w")

        self._proc = subprocess.Popen(
            [binary, VOCAB, config_path],
            stdout=self._stdout_f,
            stderr=self._stderr_f,
            env=env,
        )

        # reader 스레드 (파일 tail 방식 — 현재 process 가 파일에 쓰는 동안 읽음)
        self._stop_evt = threading.Event()
        threading.Thread(target=self._tail_pose_lines,
                         args=(stdout_path,), daemon=True).start()
        threading.Thread(target=self._tail_state_lines,
                         args=(stderr_path,), daemon=True).start()

    def _cleanup_subprocess(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=self.cfg.term_timeout_sec)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

        for f in (self._stdout_f, self._stderr_f):
            try:
                if f is not None and not f.closed:
                    f.close()
            except Exception:
                pass
        self._stdout_f = None
        self._stderr_f = None

        if self._tmp_dir and os.path.exists(self._tmp_dir):
            if self.cfg.archive_dir:
                try:
                    os.makedirs(self.cfg.archive_dir, exist_ok=True)
                    n = len([d for d in os.listdir(self.cfg.archive_dir)
                             if d.startswith("attempt_")])
                    dst = os.path.join(self.cfg.archive_dir, f"attempt_{n+1:02d}")
                    shutil.move(self._tmp_dir, dst)
                    print(f"[ORBSLAM] archived {self._tmp_dir} → {dst}")
                except Exception as e:
                    print(f"[ORBSLAM] archive failed: {e}")
                    shutil.rmtree(self._tmp_dir, ignore_errors=True)
            else:
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = None

        # C++ 가 남기는 frame jpg
        frame_jpg = "/tmp/orbslam_frame.jpg"
        if os.path.exists(frame_jpg):
            try:
                os.remove(frame_jpg)
            except OSError:
                pass

    # ── internal: log readers ──
    def _tail_pose_lines(self, path: str) -> None:
        with open(path, "r") as f:
            while self._stop_evt is not None and not self._stop_evt.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                line = line.strip()
                if not line.startswith("POSE:"):
                    continue
                try:
                    vals = list(map(float, line.split()[1:]))
                except ValueError:
                    continue
                if len(vals) != 7:
                    continue
                arr = np.array(vals)
                with self._lock:
                    self._latest_raw = arr
                    self._all_positions.append(arr[:3].copy())
                    cap = self.cfg.trajectory_cap
                    if len(self._all_positions) > cap:
                        self._all_positions = self._all_positions[cap // 2:]

    def _tail_state_lines(self, path: str) -> None:
        with open(path, "r") as f:
            while self._stop_evt is not None and not self._stop_evt.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                line = line.rstrip()
                if not line.startswith("STATE:"):
                    continue
                try:
                    state = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    continue
                with self._lock:
                    self._tracking_state = state

    # ── public API ──
    def is_alive(self) -> bool:
        """subprocess 가 가용 상태인지 (respawn 중 포함).

        - watchdog 이 cap 초과로 포기 → False
        - watchdog 가 respawn 진행 중 → True (consumer 가 일시적 None 보고 빠져나가지 않게)
        - subprocess 가 죽었지만 watchdog 가 아직 detect 못 한 race window:
          → consumer 의 0.05s 폴링과 watchdog 의 0.5s 폴링 사이 gap 에서 False 가
            보이지 않도록 is_alive() 가 직접 death 감지 시 `_restarting=True` 로 표시
            (watchdog 가 곧 따라잡음). **단 watchdog 스레드가 실제로 가동 중일 때만**
            — startup 단계 (start() retry 루프) 에서는 watchdog 가 아직 없으므로
            False 를 반환해 retry 가 의도대로 동작하게 한다.
        """
        if self._watchdog_dead:
            return False
        if self._restarting:
            return True
        proc = self._proc
        if proc is None:
            return False
        if proc.poll() is None:
            return True
        # subprocess 죽음. _restarting flag 는 watchdog 가 가동 중일 때만 의미가 있음.
        # startup 단계에서는 watchdog_thread is None 이므로 그대로 False 반환 →
        # start() 의 retry 루프가 hw_reset 후 재시도하도록 둔다.
        wd = self._watchdog_thread
        watchdog_running = wd is not None and wd.is_alive()
        if watchdog_running and self.cfg.auto_restart \
                and self._restart_count < self.cfg.max_auto_restarts:
            self._restarting = True
            return True
        return False

    def is_tracking(self) -> bool:
        with self._lock:
            return self._tracking_state == 2  # OK

    def get_tracking_state(self) -> str:
        with self._lock:
            s = self._tracking_state
        return STATE_NAMES.get(s, f"?({s})")

    def get_pose(self) -> Optional[Dict]:
        """
        World-frame pose 반환.

        Returns
        -------
        dict | None
            x            : float   world X [m]   (= camera Z)
            y            : float   world Y [m]   (= -camera X)
            theta        : float   yaw [rad]     (CCW positive)
            theta_deg    : float   yaw [deg]
            tracking     : str     'NOT_READY' | 'NO_IMAGE' | 'INIT' | 'OK'
                                  | 'RECENTLY_LOST' | 'LOST'
            tracking_ok  : bool    (tracking == 'OK')
            raw          : np.ndarray (7,) [x, y, z, qx, qy, qz, qw] (camera frame)

        한 번도 POSE 라인이 들어오지 않았으면 None.
        """
        with self._lock:
            raw = self._latest_raw
            state = self._tracking_state
        if raw is None:
            return None

        # 카메라 프레임 → 월드 프레임
        # world_x =  camera_z (forward), world_y = -camera_x (left)
        cam_pos = raw[:3]
        quat = raw[3:7]  # scipy: [x, y, z, w]

        from scipy.spatial.transform import Rotation
        try:
            R = Rotation.from_quat(quat).as_matrix()
        except Exception:
            return None

        forward = R[:, 2]   # camera +z 방향이 월드에서 가리키는 방향
        theta_rad = float(math.atan2(-forward[0], forward[2]))

        return {
            "x":           float(cam_pos[2]),
            "y":           float(-cam_pos[0]),
            "theta":       theta_rad,
            "theta_deg":   math.degrees(theta_rad),
            "tracking":    STATE_NAMES.get(state, f"?({state})"),
            "tracking_ok": state == 2,
            "raw":         raw.copy(),
        }

    def wait_for_tracking(self, timeout: float = 60.0,
                          stability_sec: Optional[float] = None) -> bool:
        """tracking_state == OK 가 *안정적으로* 유지될 때까지 대기.

        첫 OK 관측 후 `stability_sec` 동안 subprocess 생존 + tracking_OK 유지를
        확인. fragile bootstrap window 에서 즉사하는 경우 watchdog 가 respawn
        하면 (is_alive() 가 True 유지) outer loop 가 자동으로 다시 OK 를 기다림.

        Parameters
        ----------
        timeout : float
            전체 대기 한도 (watchdog respawn 한 번 ≈ 35s 이므로 60s 권장).
        stability_sec : float | None
            None 이면 `LocalizerConfig.tracking_stability_sec` 사용.

        Returns
        -------
        bool  성공(안정적으로 tracking OK 도달) True / 타임아웃·dead False.
        """
        if stability_sec is None:
            stability_sec = self.cfg.tracking_stability_sec
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.is_alive():
                return False
            if not self.is_tracking():
                time.sleep(self.cfg.tracking_poll_sec)
                continue
            # 첫 tracking_OK 관측. 안정성 윈도우 검증.
            stable_start = time.time()
            broke = False
            while time.time() - stable_start < stability_sec:
                if not self.is_alive():
                    return False
                if not self.is_tracking():
                    broke = True
                    break
                time.sleep(self.cfg.tracking_poll_sec)
            if not broke:
                return True
            # subprocess 가 fragile window 안에 죽음 → watchdog respawn 대기 후
            # outer loop 가 다시 OK 를 기다림.
        return False

    def get_trajectory(self) -> np.ndarray:
        """누적 카메라-프레임 위치 (N, 3). 디버깅·로깅용."""
        with self._lock:
            return np.array(self._all_positions) if self._all_positions \
                else np.empty((0, 3))

    def get_total_distance(self) -> float:
        traj = self.get_trajectory()
        if len(traj) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


# ──────────────────────────────────────────────
# CLI: 라이브러리 단독 동작 확인
# ──────────────────────────────────────────────
def _print_loop(cfg: LocalizerConfig) -> None:
    """main.py 의 orbslam --headless 와 동등한 터미널 출력 루프."""
    print(f"[Localizer] starting "
          f"(IMU={'ON' if cfg.use_imu else 'OFF'}, "
          f"Pi={'ON' if cfg.pi_mode else 'OFF'}) — Ctrl+C to stop")
    print(f"{'state':>14s}  {'x_m':>8s}  {'y_m':>8s}  {'theta_deg':>10s}")
    with OrbSlamLocalizer(cfg) as loc:
        try:
            while loc.is_alive():
                pose = loc.get_pose()
                if pose is None:
                    state = loc.get_tracking_state()
                    print(f"\r{state:>14s}  {'--':>8s}  {'--':>8s}  {'--':>10s}",
                          end="", flush=True)
                else:
                    print(f"\r{pose['tracking']:>14s}  "
                          f"{pose['x']:8.3f}  {pose['y']:8.3f}  "
                          f"{pose['theta_deg']:10.2f}",
                          end="", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[Localizer] Ctrl+C — shutting down")
        finally:
            n = len(loc.get_trajectory())
            print(f"\n[Localizer] {n} poses, {loc.get_total_distance():.3f} m traveled")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="ORB-SLAM3 localization module — headless library demo.")
    ap.add_argument("--imu",   action="store_true",
                    help="enable IMU (default: off, == --no-imu)")
    ap.add_argument("--no-pi", action="store_true",
                    help="disable Pi-optimized yaml (default: pi mode on)")
    args = ap.parse_args()

    _print_loop(LocalizerConfig(
        use_imu=args.imu,
        pi_mode=not args.no_pi,
    ))
