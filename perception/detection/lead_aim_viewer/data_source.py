"""Data sources for the lead-aim viewer.

Two sources share a single ``DataSource`` protocol so the viewer is
agnostic:

  LiveSource  - spawns a background thread running camera + estimator +
                BellMotionTracker (no robot, no firing). Per tick the
                viewer pulls the latest snapshot from a thread-safe slot.
  ReplaySource - loads an .npz produced by LeadAimLogger; supports
                seek/pause/speed and recomputes z_pred / is_safe for a
                viewer-adjustable Δt.

FrameSnapshot is the single struct the viewer's scenes consume.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ───────────────────────── snapshot dataclass ───────────────────────────
@dataclass
class TrackerSnapshot:
    ready: bool
    v: float                  # m/s, nan if not ready
    z_center: float           # m,   nan if no endpoint
    endpoints_seen: int
    is_safe: bool
    z_pred: float             # m,   nan if not ready
    tau_endpoint: float       # s,   nan if not ready


@dataclass
class FrameSnapshot:
    t: float                                  # current time, seconds
    xyz_obs: Optional[np.ndarray]             # (3,) plate-frame, or None
    tracker: TrackerSnapshot
    recent_obs: np.ndarray                    # (N, 4) [t, x, y, z]
    recent_endpoints: np.ndarray              # (E, 2) [t, z]
    recent_shots: np.ndarray                  # (S, 5) [t, x, y, z, z_pred]
    delay_s: float                            # current Δt slider value
    is_live: bool
    t_range: Tuple[float, float]              # (t_min, t_max) for seek slider


# ─────────────────────────── ReplaySource ───────────────────────────────
class ReplaySource:
    """Plays back a .npz produced by LeadAimLogger.

    Time advances by wall_dt * speed each tick. Seek/pause/speed are
    plain attribute changes — the viewer wires widgets to them. The Δt
    slider re-derives z_pred / is_safe from the stored v and latest
    observed z, since these are the only fields that depend on Δt.
    """

    HISTORY_S = 10.0   # window length for recent_obs / recent_endpoints
    SHOT_FADE_S = 5.0  # window length for recent_shots

    def __init__(self, path: Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"replay file not found: {path}")
        z = np.load(path, allow_pickle=False)
        # frames/
        self.t = z["t"]
        self.x_obs = z["x_obs"]
        self.y_obs = z["y_obs"]
        self.z_obs = z["z_obs"]
        self.valid = z["valid"]
        # tracker/
        self.tr_ready = z["tr_ready"]
        self.tr_v = z["tr_v"]
        self.tr_zc = z["tr_zc"]
        self.tr_eps = z["tr_eps"]
        self.tr_safe = z["tr_safe"]
        self.tr_zpred = z["tr_zpred"]
        self.tr_tau = z["tr_tau"]
        # events/
        self.ep_t = z["ep_t"]
        self.ep_z = z["ep_z"]
        self.shot_t = z["shot_t"]
        self.shot_xyz = z["shot_xyz"]
        self.shot_zpred = z["shot_zpred"]
        # meta/
        self.meta: Dict[str, float] = dict(zip(
            (str(k) for k in z["meta_keys"]),
            (float(v) for v in z["meta_vals"]),
        ))
        # normalize timeline to start at 0 (logs use monotonic seconds)
        if self.t.size > 0:
            t0 = float(self.t[0])
            self.t = self.t - t0
            self.ep_t = self.ep_t - t0
            self.shot_t = self.shot_t - t0
        # playback state
        self.paused: bool = False
        self.speed: float = 1.0
        self._cursor_t: float = 0.0
        self._last_wall_t: float = time.monotonic()
        self.delay_s: float = float(self.meta.get("lead_total_delay_sec", 0.7))

    @property
    def is_live(self) -> bool:
        return False

    @property
    def t_range(self) -> Tuple[float, float]:
        if self.t.size == 0:
            return (0.0, 0.0)
        return (0.0, float(self.t[-1]))

    @property
    def cursor_t(self) -> float:
        return self._cursor_t

    def seek(self, t: float) -> None:
        lo, hi = self.t_range
        self._cursor_t = float(np.clip(t, lo, hi))
        self._last_wall_t = time.monotonic()

    def set_speed(self, s: float) -> None:
        self.speed = max(0.01, float(s))

    def set_delay(self, d: float) -> None:
        self.delay_s = max(0.0, float(d))

    def pause(self, on: bool) -> None:
        self.paused = bool(on)
        if not on:
            self._last_wall_t = time.monotonic()

    def step_frame(self, n: int = 1) -> None:
        """Advance cursor by n logged frames (replay-only convenience)."""
        if self.t.size == 0:
            return
        i = int(np.searchsorted(self.t, self._cursor_t, side="right"))
        j = int(np.clip(i + n - 1, 0, self.t.size - 1))
        self._cursor_t = float(self.t[j])

    def step(self) -> Optional[FrameSnapshot]:
        # Advance cursor by elapsed wall time × speed (unless paused).
        now = time.monotonic()
        if not self.paused:
            self._cursor_t = min(
                self._cursor_t + (now - self._last_wall_t) * self.speed,
                self.t_range[1],
            )
        self._last_wall_t = now
        return self._snapshot_at(self._cursor_t)

    def _snapshot_at(self, t_cursor: float) -> Optional[FrameSnapshot]:
        if self.t.size == 0:
            return None
        i = int(np.searchsorted(self.t, t_cursor, side="right") - 1)
        i = max(0, min(i, self.t.size - 1))

        # Current obs
        if bool(self.valid[i]):
            xyz_obs = np.array([self.x_obs[i], self.y_obs[i], self.z_obs[i]])
        else:
            xyz_obs = None

        # Recompute z_pred / is_safe with the (possibly slider-changed) Δt.
        # Linear extrapolation from latest valid z + tracker v + z_center.
        ready = bool(self.tr_ready[i])
        v = float(self.tr_v[i])
        zc = float(self.tr_zc[i])
        if ready and not np.isnan(v) and xyz_obs is not None:
            z_pred = float(xyz_obs[2]) + v * self.delay_s
            amp = float(self.meta.get("amplitude_m", 0.25))
            margin = float(self.meta.get("safety_margin_m", 0.03))
            is_safe = abs(z_pred - zc) <= amp - margin
            # τ_endpoint with current v / z
            z_next_ep = zc + (amp if v > 0 else -amp)
            tau = (z_next_ep - float(xyz_obs[2])) / v if v != 0.0 else float("nan")
        else:
            # Fall back to stored values if we can't recompute (e.g. obs
            # invalid this frame).
            z_pred = float(self.tr_zpred[i])
            is_safe = bool(self.tr_safe[i])
            tau = float(self.tr_tau[i])

        tracker = TrackerSnapshot(
            ready=ready, v=v, z_center=zc,
            endpoints_seen=int(self.tr_eps[i]),
            is_safe=is_safe, z_pred=z_pred, tau_endpoint=tau,
        )

        t_now = float(self.t[i])
        t_lo = t_now - self.HISTORY_S
        win = (self.t >= t_lo) & (self.t <= t_now) & self.valid
        recent_obs = np.column_stack([
            self.t[win], self.x_obs[win], self.y_obs[win], self.z_obs[win],
        ]) if np.any(win) else np.zeros((0, 4))

        ep_win = (self.ep_t >= t_lo) & (self.ep_t <= t_now)
        recent_endpoints = np.column_stack([
            self.ep_t[ep_win], self.ep_z[ep_win],
        ]) if np.any(ep_win) else np.zeros((0, 2))

        sh_win = (self.shot_t >= t_now - self.SHOT_FADE_S) & (self.shot_t <= t_now)
        if np.any(sh_win):
            sxyz = self.shot_xyz[sh_win]
            recent_shots = np.column_stack([
                self.shot_t[sh_win], sxyz[:, 0], sxyz[:, 1], sxyz[:, 2],
                self.shot_zpred[sh_win],
            ])
        else:
            recent_shots = np.zeros((0, 5))

        return FrameSnapshot(
            t=t_now, xyz_obs=xyz_obs, tracker=tracker,
            recent_obs=recent_obs, recent_endpoints=recent_endpoints,
            recent_shots=recent_shots,
            delay_s=self.delay_s, is_live=False, t_range=self.t_range,
        )


# ─────────────────────────────── LiveSource ─────────────────────────────
@dataclass
class _LiveBuffer:
    """Thread-safe shared state: producer writes, consumer reads latest."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    t0: float = 0.0
    samples: list = field(default_factory=list)         # [(t, x, y, z, valid)]
    endpoints: list = field(default_factory=list)       # [(t, z)]
    tracker_state: Optional[TrackerSnapshot] = None
    stop: bool = False


class LiveSource:
    """Background-thread live capture for the viewer.

    Uses the same camera + estimator + BellMotionTracker as
    run_phase2_lead_aim, but does NOT command the robot — no firing,
    no leveling angles. Pure read-only debugging.

    Construction takes already-built ``camera``, ``estimator``, and
    ``params`` so the viewer entry script owns the resource lifecycle.
    """

    HISTORY_S = 10.0

    def __init__(self, camera, estimator, params, delay_s: float):
        from perception.detection.phase2_lead_aim import BellMotionTracker
        self._camera = camera
        self._estimator = estimator
        self._params = params
        self._tracker = BellMotionTracker(params)
        self._buf = _LiveBuffer()
        self.delay_s = float(delay_s)
        self.paused = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def is_live(self) -> bool:
        return True

    @property
    def t_range(self) -> Tuple[float, float]:
        with self._buf.lock:
            if not self._buf.samples:
                return (0.0, 0.0)
            return (0.0, float(self._buf.samples[-1][0]))

    def start(self) -> None:
        self._thread.start()

    def stop_thread(self) -> None:
        with self._buf.lock:
            self._buf.stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def pause(self, on: bool) -> None:
        # Live: pause freezes the snapshot (producer keeps running).
        self.paused = bool(on)

    def seek(self, t: float) -> None:  # noqa: ARG002
        pass  # not meaningful for live

    def set_delay(self, d: float) -> None:
        self.delay_s = max(0.0, float(d))

    def set_speed(self, s: float) -> None:  # noqa: ARG002
        pass  # speed has no meaning live

    def step_frame(self, n: int = 1) -> None:  # noqa: ARG002
        pass

    def _run(self) -> None:
        last_eps = 0
        while True:
            with self._buf.lock:
                if self._buf.stop:
                    return
            try:
                color, depth_image, _ = self._camera.get_frames()
            except Exception:
                continue
            if color is None:
                continue
            try:
                p_plate = self._estimator.estimate(color, depth_image)
            except Exception:
                p_plate = None
            t_now = time.monotonic()
            with self._buf.lock:
                if self._buf.t0 == 0.0 and not self._buf.samples:
                    self._buf.t0 = t_now
                rel_t = t_now - self._buf.t0
            valid = p_plate is not None
            if valid:
                self._tracker.update(t_now, float(p_plate[2]))
            ts = self._snapshot_state()
            with self._buf.lock:
                if valid:
                    self._buf.samples.append((
                        rel_t, float(p_plate[0]), float(p_plate[1]),
                        float(p_plate[2]), True,
                    ))
                else:
                    self._buf.samples.append((rel_t, np.nan, np.nan, np.nan, False))
                # Trim history.
                cutoff = rel_t - self.HISTORY_S - 1.0
                while self._buf.samples and self._buf.samples[0][0] < cutoff:
                    self._buf.samples.pop(0)
                # Endpoint diff.
                if self._tracker.endpoints_seen > last_eps:
                    ep = self._tracker.last_endpoint()
                    if ep is not None:
                        self._buf.endpoints.append(
                            (ep[0] - self._buf.t0, float(ep[1]))
                        )
                    last_eps = self._tracker.endpoints_seen
                while self._buf.endpoints and self._buf.endpoints[0][0] < cutoff:
                    self._buf.endpoints.pop(0)
                self._buf.tracker_state = ts

    def _snapshot_state(self) -> TrackerSnapshot:
        tr = self._tracker
        ready = bool(tr.ready)
        v = float(tr.velocity) if tr.velocity is not None else float("nan")
        zc = float(tr.z_center) if tr.z_center is not None else float("nan")
        if ready:
            zp_opt = tr.predict_z(self.delay_s)
            tau_opt = tr.time_to_next_endpoint()
            zp = float(zp_opt) if zp_opt is not None else float("nan")
            tau = float(tau_opt) if tau_opt is not None else float("nan")
            safe = bool(tr.is_safe_to_fire(self.delay_s))
        else:
            zp = float("nan")
            tau = float("nan")
            safe = False
        return TrackerSnapshot(
            ready=ready, v=v, z_center=zc,
            endpoints_seen=int(tr.endpoints_seen),
            is_safe=safe, z_pred=zp, tau_endpoint=tau,
        )

    def step(self) -> Optional[FrameSnapshot]:
        with self._buf.lock:
            if not self._buf.samples or self._buf.tracker_state is None:
                return None
            if self.paused:
                # Use the most recent buffered snapshot; do not advance.
                pass
            samples = list(self._buf.samples)
            endpoints = list(self._buf.endpoints)
            ts = self._buf.tracker_state
        t_now = samples[-1][0]
        if samples[-1][4]:
            xyz_obs = np.array(samples[-1][1:4])
        else:
            xyz_obs = None
        t_lo = t_now - self.HISTORY_S
        valid_in_win = [s for s in samples if s[0] >= t_lo and s[4]]
        if valid_in_win:
            recent_obs = np.array([(s[0], s[1], s[2], s[3]) for s in valid_in_win])
        else:
            recent_obs = np.zeros((0, 4))
        ep_in_win = [(t, z) for (t, z) in endpoints if t >= t_lo]
        recent_endpoints = np.array(ep_in_win) if ep_in_win else np.zeros((0, 2))
        return FrameSnapshot(
            t=t_now, xyz_obs=xyz_obs, tracker=ts,
            recent_obs=recent_obs, recent_endpoints=recent_endpoints,
            recent_shots=np.zeros((0, 5)),     # viewer's live mode = no shots
            delay_s=self.delay_s, is_live=True,
            t_range=(0.0, t_now),
        )
