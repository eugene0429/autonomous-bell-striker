"""Per-run logger for lead-aim debugging.

Accumulates per-frame tracker snapshots + endpoint/shot events during a
``run_phase2_lead_aim`` execution, then dumps a single ``.npz`` for
later replay in the viewer.

Schema (see spec §2):
  frames/ t, z_obs, x_obs, y_obs, valid
  tracker/ ready, v, z_center, endpoints_seen, is_safe, z_pred,
           tau_endpoint
  events/ endpoint_t, endpoint_z, shot_t, shot_xyz, shot_z_pred
  meta/ params + CLI args (delay_s, amplitude, half periods, offsets, ...)

The recorder reads ONLY public accessors on BellMotionTracker — it
doesn't mutate tracker state or interfere with the firing loop.

Per-frame cost: dict appends + a few attribute reads. Negligible
versus camera + Hailo inference latency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from perception.detection.phase2_lead_aim import BellMotionTracker


class LeadAimLogger:
    """Accumulates lead-aim debug data; ``save()`` writes a single .npz.

    Wire-up (in ``run_phase2_lead_aim``):
        logger = LeadAimLogger(params, args)              # before loop
        logger.record_frame(t, p_plate, tracker, delay_s) # every frame
        logger.record_shot(t, aim_xyz, z_pred)            # at LOAD
        logger.save(path)                                 # in finally
    """

    def __init__(self, params, args) -> None:
        # frames/
        self.t: List[float] = []
        self.x_obs: List[float] = []
        self.y_obs: List[float] = []
        self.z_obs: List[float] = []
        self.valid: List[bool] = []
        # tracker/ (np.nan when not ready)
        self.tr_ready: List[bool] = []
        self.tr_v: List[float] = []
        self.tr_zc: List[float] = []
        self.tr_eps: List[int] = []
        self.tr_safe: List[bool] = []
        self.tr_zpred: List[float] = []
        self.tr_tau: List[float] = []
        # events/
        self.ep_t: List[float] = []
        self.ep_z: List[float] = []
        self.shot_t: List[float] = []
        self.shot_xyz: List[Tuple[float, float, float]] = []
        self.shot_zpred: List[float] = []
        # meta/
        self.meta: Dict[str, float] = self._build_meta(params, args)
        # state for endpoint diff detection
        self._last_eps_count: int = 0

    @staticmethod
    def _build_meta(params, args) -> Dict[str, float]:
        keys = [
            ("amplitude_m", getattr(params, "amplitude_m", float("nan"))),
            ("half_period_min_s", getattr(params, "half_period_min_s", float("nan"))),
            ("half_period_max_s", getattr(params, "half_period_max_s", float("nan"))),
            ("safety_margin_m", getattr(params, "safety_margin_m", float("nan"))),
            ("min_fit_samples", getattr(params, "min_fit_samples", float("nan"))),
            ("fit_window_samples", getattr(params, "fit_window_samples", float("nan"))),
            ("endpoint_v_eps_mps", getattr(params, "endpoint_v_eps_mps", float("nan"))),
        ]
        # Optional CLI args (defensive — viewer should tolerate missing).
        for a in ("lead_total_delay_sec", "lead_inter_shot_sec", "tilt_deg",
                  "launcher_offset_x", "launcher_offset_y", "launcher_offset_z",
                  "camera_offset_x", "camera_offset_y", "camera_offset_z"):
            keys.append((a, float(getattr(args, a, float("nan")))))
        return {k: float(v) for k, v in keys}

    # ── per-frame ingestion ──
    def record_frame(
        self,
        t: float,
        p_plate: Optional[np.ndarray],
        tracker: BellMotionTracker,
        delay_s: float,
    ) -> None:
        valid = p_plate is not None
        self.t.append(float(t))
        if valid:
            self.x_obs.append(float(p_plate[0]))
            self.y_obs.append(float(p_plate[1]))
            self.z_obs.append(float(p_plate[2]))
        else:
            self.x_obs.append(float("nan"))
            self.y_obs.append(float("nan"))
            self.z_obs.append(float("nan"))
        self.valid.append(bool(valid))

        ready = bool(tracker.ready)
        self.tr_ready.append(ready)
        self.tr_v.append(float(tracker.velocity) if tracker.velocity is not None else float("nan"))
        self.tr_zc.append(float(tracker.z_center) if tracker.z_center is not None else float("nan"))
        self.tr_eps.append(int(tracker.endpoints_seen))
        if ready:
            zp = tracker.predict_z(delay_s)
            self.tr_zpred.append(float(zp) if zp is not None else float("nan"))
            tau = tracker.time_to_next_endpoint()
            self.tr_tau.append(float(tau) if tau is not None else float("nan"))
            self.tr_safe.append(bool(tracker.is_safe_to_fire(delay_s)))
        else:
            self.tr_zpred.append(float("nan"))
            self.tr_tau.append(float("nan"))
            self.tr_safe.append(False)

        # endpoint diff: tracker bumped endpoints_seen this frame?
        if tracker.endpoints_seen > self._last_eps_count:
            ep = tracker.last_endpoint()
            if ep is not None:
                self.ep_t.append(float(ep[0]))
                self.ep_z.append(float(ep[1]))
            self._last_eps_count = tracker.endpoints_seen

    def record_shot(
        self,
        t: float,
        aim_xyz: np.ndarray,
        z_pred: Optional[float],
    ) -> None:
        self.shot_t.append(float(t))
        self.shot_xyz.append((float(aim_xyz[0]), float(aim_xyz[1]), float(aim_xyz[2])))
        self.shot_zpred.append(float(z_pred) if z_pred is not None else float("nan"))

    # ── dump ──
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_keys = np.array(list(self.meta.keys()))
        meta_vals = np.array(list(self.meta.values()), dtype=np.float64)
        shot_xyz_arr = (np.asarray(self.shot_xyz, dtype=np.float64).reshape(-1, 3)
                        if self.shot_xyz else np.zeros((0, 3), dtype=np.float64))
        np.savez_compressed(
            path,
            t=np.asarray(self.t, dtype=np.float64),
            x_obs=np.asarray(self.x_obs, dtype=np.float64),
            y_obs=np.asarray(self.y_obs, dtype=np.float64),
            z_obs=np.asarray(self.z_obs, dtype=np.float64),
            valid=np.asarray(self.valid, dtype=bool),
            tr_ready=np.asarray(self.tr_ready, dtype=bool),
            tr_v=np.asarray(self.tr_v, dtype=np.float64),
            tr_zc=np.asarray(self.tr_zc, dtype=np.float64),
            tr_eps=np.asarray(self.tr_eps, dtype=np.int32),
            tr_safe=np.asarray(self.tr_safe, dtype=bool),
            tr_zpred=np.asarray(self.tr_zpred, dtype=np.float64),
            tr_tau=np.asarray(self.tr_tau, dtype=np.float64),
            ep_t=np.asarray(self.ep_t, dtype=np.float64),
            ep_z=np.asarray(self.ep_z, dtype=np.float64),
            shot_t=np.asarray(self.shot_t, dtype=np.float64),
            shot_xyz=shot_xyz_arr,
            shot_zpred=np.asarray(self.shot_zpred, dtype=np.float64),
            meta_keys=meta_keys,
            meta_vals=meta_vals,
        )
