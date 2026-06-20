"""
ORB-SLAM3 Localization Module — production library.

Separates the behavior of `main.py orbslam --pi --no-imu --headless` into a module.
A pipeline can import this and consume world-frame (x, y, θ) pose as a stream.

Usage example
-------------
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

World coordinate frame (camera starting pose = origin)
------------------------------------------------------
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

# When run directly as a script (`python vio/orbslam_localizer.py`), prepend
# perception/ to sys.path so the `vio` package can be found.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

# Reuse existing helpers (yaml build, calibration cache, path constants, etc.)
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
    # ── Operating mode (default equals main.py's --pi --no-imu --headless) ──
    use_imu: bool = False        # False == --no-imu
    pi_mode: bool = True         # True  == --pi   (RGB-D 640×480@15fps + nFeatures=1500 yaml)

    # ── Startup / shutdown ──
    max_retries: int = 5                # number of retries when subprocess startup fails
    startup_alive_timeout: float = 20.0 # max wait for first POSE line or subprocess death
                                        # (includes vocab load 5–10s + camera open margin)
    startup_settle_sec: float = 0.2     # minimum wait to detect instant death right after startup
    skip_flush_first_attempt: bool = False  # skip _flush_realsense on attempt 0 (~3s faster).
                                            # Safe only if the previous session was clean — for
                                            # repeated-run environments such as stability tests,
                                            # False is the safe choice.
    hw_reset_on_retry: bool = True       # D435i hardware_reset on retry → recover from USB suspend
    hw_reset_on_first_attempt: bool = True   # hardware_reset on the first attempt too (effect of
                                             # unplugging/replugging USB). +5s cost. Without it,
                                             # a D435i cold-start race kills the first attempt
                                             # after ~5 POSE and triggers a watchdog respawn.
                                             # See SLAM_DEBUG.md S6.
    term_timeout_sec: float = 5.0       # wait for SIGTERM → SIGKILL transition on stop()

    # ── Recovery watchdog (auto-restart on mid-run death) ──
    auto_restart: bool = True           # auto-respawn if the subprocess dies mid-run
    max_auto_restarts: int = 5          # limit on consecutive auto-restarts (watchdog gives up beyond it)
    watchdog_poll_sec: float = 0.5      # subprocess liveness polling interval

    # ── Tracking wait ──
    tracking_poll_sec: float = 0.05     # wait_for_tracking() / startup polling interval
    tracking_stability_sec: float = 2.0 # wait_for_tracking() stability-check window after first OK.
                                        # If the subprocess dies within this window, the watchdog
                                        # respawns → the outer loop waits for OK again.
                                        # Automatically absorbs the fragile bootstrap window.

    # ── ORB-SLAM3 yaml override ──
    orb_nfeatures: Optional[int] = None # None → base yaml value. If an integer is given, replace with it.
                                        # Raise from 1500 → 2000 etc. when more init stability is needed.

    # ── Memory ──
    trajectory_cap: int = 5000          # upper bound on stored trajectory points (discard half when exceeded)

    # ── Debug ──
    archive_dir: Optional[str] = None   # if set, _cleanup_subprocess moves tmp_dir to this path
                                        # (attempt_<N>/) instead of deleting it.
                                        # For comparative analysis of stdout/stderr of dead attempts.


# ──────────────────────────────────────────────
# Localizer
# ──────────────────────────────────────────────
class OrbSlamLocalizer:
    """ORB-SLAM3-based localization module (headless, library API)."""

    def __init__(self, cfg: Optional[LocalizerConfig] = None):
        self.cfg = cfg if cfg is not None else LocalizerConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._tmp_dir: Optional[str] = None
        self._stop_evt: Optional[threading.Event] = None
        self._stdout_f = None
        self._stderr_f = None

        self._lock = threading.Lock()
        self._latest_raw: Optional[np.ndarray] = None   # latest [x,y,z,qx,qy,qz,qw]
        self._tracking_state: int = -1                  # NOT_READY
        self._all_positions: List[np.ndarray] = []      # accumulated camera-frame (x,y,z)

        # Recovery watchdog state
        self._stop_watchdog: Optional[threading.Event] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._restart_count: int = 0       # accumulated mid-run auto-restarts
        self._watchdog_dead: bool = False  # watchdog-gave-up state (cap exceeded)
        self._restarting: bool = False     # watchdog respawn in progress (consumer should treat as alive)

    # ── context manager ──
    def __enter__(self) -> "OrbSlamLocalizer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ── lifecycle ──
    def start(self) -> None:
        """Start the ORB-SLAM3 subprocess.

        Instead of a fixed sleep, wait until whichever happens first: "first POSE
        line arrives OR subprocess death". A death during the camera-reopen step
        after vocab load (e.g. `failed to set power state`) is also treated as a
        startup failure and retried. On retry, D435i hardware_reset is used to
        forcibly wake the USB power state.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            # attempt 0: if skip_flush_first_attempt=True, skip the flush entirely (~3s faster).
            # hw_reset is applied when hw_reset_on_retry is True for attempt>0, or
            # hw_reset_on_first_attempt is True for attempt==0 (effect of replugging USB).
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

            # ── Wait for alive signal: first POSE line OR subprocess death ──
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
        """Start the mid-run death monitoring thread after a successful startup."""
        if not self.cfg.auto_restart:
            return
        self._stop_watchdog = threading.Event()
        self._restart_count = 0
        self._watchdog_dead = False
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """Auto-respawn when the subprocess terminates unexpectedly.

        - exits naturally when stop() sets _stop_watchdog
        - on cap exceeded, gives up by setting watchdog_dead=True → is_alive() returns False
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
                continue   # alive
            # subprocess died — attempt recovery. Set _restarting first so that
            # is_alive() does not immediately appear False.
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

            # start respawn → keep appearing alive to the consumer
            self._restarting = True
            try:
                # reset immediately so the consumer does not see a stale pose
                with self._lock:
                    self._latest_raw = None
                    self._tracking_state = -1
                # stop the reader thread, clean up subprocess/temp files
                if self._stop_evt is not None:
                    self._stop_evt.set()
                self._cleanup_subprocess()

                # respawn — always hardware_reset (USB is the most common cause of mid-run crash)
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
        """During startup, wait for one of the following:

        - first POSE line arrives  → True (camera + tracking OK)
        - subprocess death         → False (subject to retry)
        - neither arrives by timeout → True if still alive (vocab load just took long)

        Use startup_settle_sec as a minimum wait floor so that the
        instant-exit case is not missed.
        """
        t0 = time.time()
        # Minimum wait: guarantee detection of the instant-exit (first 0.x sec) case
        time.sleep(max(0.0, self.cfg.startup_settle_sec))
        while time.time() - t0 < timeout:
            if not self.is_alive():
                return False
            with self._lock:
                if self._latest_raw is not None:
                    return True
            time.sleep(self.cfg.tracking_poll_sec)
        # timeout: treat as success if alive (may be loading vocab / initializing tracking)
        return self.is_alive()

    def _dump_failure_tail(self, n_chars: int = 600) -> None:
        """Print the tail of the failed subprocess's stdout/stderr — for diagnostics."""
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
        """Terminate subprocess + stop watchdog/reader threads + clean up temp files."""
        # Stop the watchdog first so stop()'s cleanup does not trigger an auto-respawn
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

        # Calibration (cache first)
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

        # env (headless + ORB-SLAM3 lib path)
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

        # reader threads (file-tail approach — read while the running process writes to the file)
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

        # frame jpg left behind by C++
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
        """Whether the subprocess is available (including during respawn).

        - watchdog gave up on cap exceeded → False
        - watchdog respawn in progress → True (so the consumer does not bail out on a transient None)
        - race window where the subprocess died but the watchdog has not detected it yet:
          → so that False is not seen in the gap between the consumer's 0.05s polling
            and the watchdog's 0.5s polling, is_alive() marks `_restarting=True` when it
            detects death directly (the watchdog catches up soon). **But only when the
            watchdog thread is actually running** — during the startup phase (start()'s
            retry loop) the watchdog does not exist yet, so it returns False to let retry
            behave as intended.
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
        # subprocess died. The _restarting flag is only meaningful while the watchdog is running.
        # During startup, watchdog_thread is None, so return False as-is →
        # let start()'s retry loop retry after hw_reset.
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
        Return world-frame pose.

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

        None if no POSE line has ever arrived.
        """
        with self._lock:
            raw = self._latest_raw
            state = self._tracking_state
        if raw is None:
            return None

        # camera frame → world frame
        # world_x =  camera_z (forward), world_y = -camera_x (left)
        cam_pos = raw[:3]
        quat = raw[3:7]  # scipy: [x, y, z, w]

        from scipy.spatial.transform import Rotation
        try:
            R = Rotation.from_quat(quat).as_matrix()
        except Exception:
            return None

        forward = R[:, 2]   # direction the camera +z axis points to in the world
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
        """Wait until tracking_state == OK is held *stably*.

        After the first OK observation, confirm subprocess survival + tracking_OK
        is maintained for `stability_sec`. If it dies instantly in the fragile
        bootstrap window, once the watchdog respawns (is_alive() stays True) the
        outer loop automatically waits for OK again.

        Parameters
        ----------
        timeout : float
            overall wait limit (one watchdog respawn ≈ 35s, so 60s is recommended).
        stability_sec : float | None
            if None, use `LocalizerConfig.tracking_stability_sec`.

        Returns
        -------
        bool  True on success (stably reached tracking OK) / False on timeout or dead.
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
            # first tracking_OK observation. Validate the stability window.
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
            # subprocess died within the fragile window → after waiting for the
            # watchdog respawn, the outer loop waits for OK again.
        return False

    def get_trajectory(self) -> np.ndarray:
        """Accumulated camera-frame positions (N, 3). For debugging/logging."""
        with self._lock:
            return np.array(self._all_positions) if self._all_positions \
                else np.empty((0, 3))

    def get_total_distance(self) -> float:
        traj = self.get_trajectory()
        if len(traj) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


# ──────────────────────────────────────────────
# CLI: standalone library operation check
# ──────────────────────────────────────────────
def _print_loop(cfg: LocalizerConfig) -> None:
    """Terminal output loop equivalent to main.py's orbslam --headless."""
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
