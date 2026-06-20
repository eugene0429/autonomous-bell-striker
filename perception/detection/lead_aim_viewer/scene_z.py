"""z(t) timeseries subplot renderer.

Shows the per-frame z observations, the current half-cycle's linear
velocity fit, the Δt-ahead prediction, detected endpoint markers,
estimated z_center, ±A lines, the safety band, and shot event markers.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .data_source import FrameSnapshot


class SceneZ:
    """One-axis owner. Owns persistent artists across frames."""

    def __init__(self, ax: Axes, amplitude_m: float, safety_margin_m: float):
        self.ax = ax
        self.amplitude_m = float(amplitude_m)
        self.safety_margin_m = float(safety_margin_m)

        self.ax.set_xlabel("t [s]")
        self.ax.set_ylabel("z [m]")
        self.ax.set_title("z(t)")
        self.ax.grid(True, alpha=0.3)

        # Persistent artists (created lazily on first update).
        self._obs_pts = None
        self._fit_line = None
        self._pred_marker = None
        self._ep_marker = None
        self._zc_line = None
        self._a_lo = None
        self._a_hi = None
        self._safe_band = None
        self._shot_lines = []

    def update(self, snap: FrameSnapshot) -> None:
        ax = self.ax

        # ── x range: 10 s window ending at current time ──
        t_now = snap.t
        x_lo = t_now - 10.0
        x_hi = t_now + max(snap.delay_s, 1.0)

        # ── obs scatter ──
        if self._obs_pts is not None:
            self._obs_pts.remove()
            self._obs_pts = None
        if snap.recent_obs.shape[0] > 0:
            ts = snap.recent_obs[:, 0]
            zs = snap.recent_obs[:, 3]
            span = max(1e-3, ts.max() - ts.min())
            c = (ts - ts.min()) / span  # newer = darker
            self._obs_pts = ax.scatter(
                ts, zs, c=c, cmap="Blues", s=14, vmin=0.0, vmax=1.0,
            )

        # ── current half-cycle v fit line ──
        if self._fit_line is not None:
            self._fit_line.remove()
            self._fit_line = None
        if (snap.tracker.ready and not np.isnan(snap.tracker.v)
                and snap.xyz_obs is not None):
            v = snap.tracker.v
            # fit line: starts at last endpoint t (if known and in window),
            # ends at t_now + delay_s.
            ep_t_lo = snap.recent_endpoints[-1, 0] if snap.recent_endpoints.shape[0] > 0 else x_lo
            t0 = max(x_lo, float(ep_t_lo))
            t1 = t_now + snap.delay_s
            z_now = float(snap.xyz_obs[2])
            z0 = z_now + v * (t0 - t_now)
            z1 = z_now + v * (t1 - t_now)
            (self._fit_line,) = ax.plot(
                [t0, t1], [z0, z1], color="tab:red", lw=1.5, alpha=0.8,
                label="v fit",
            )

        # ── z_pred marker ──
        if self._pred_marker is not None:
            self._pred_marker.remove()
            self._pred_marker = None
        if snap.tracker.ready and not np.isnan(snap.tracker.z_pred):
            self._pred_marker = ax.scatter(
                [t_now + snap.delay_s], [snap.tracker.z_pred],
                color="gold", s=120, marker="X", edgecolors="black",
                linewidths=0.5, zorder=5,
            )

        # ── endpoint markers ──
        if self._ep_marker is not None:
            self._ep_marker.remove()
            self._ep_marker = None
        if snap.recent_endpoints.shape[0] > 0:
            self._ep_marker = ax.scatter(
                snap.recent_endpoints[:, 0],
                snap.recent_endpoints[:, 1],
                color="black", s=60, marker="D", zorder=4,
            )

        # ── z_center, ±A lines, safety band ──
        for h in (self._zc_line, self._a_lo, self._a_hi, self._safe_band):
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
        self._zc_line = self._a_lo = self._a_hi = self._safe_band = None
        if not np.isnan(snap.tracker.z_center):
            zc = float(snap.tracker.z_center)
            self._zc_line = ax.axhline(zc, color="0.5", ls="--", lw=0.8)
            self._a_lo = ax.axhline(zc - self.amplitude_m, color="0.3",
                                    ls=":", lw=0.8)
            self._a_hi = ax.axhline(zc + self.amplitude_m, color="0.3",
                                    ls=":", lw=0.8)
            self._safe_band = ax.axhspan(
                zc - (self.amplitude_m - self.safety_margin_m),
                zc + (self.amplitude_m - self.safety_margin_m),
                color="tab:green", alpha=0.08,
            )

        # ── shot markers (replay only) ──
        for ln in self._shot_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._shot_lines = []
        for row in snap.recent_shots:
            t_shot = float(row[0])
            ln = ax.axvline(t_shot, color="orange", ls="-", lw=1.0, alpha=0.6)
            self._shot_lines.append(ln)
            zp = float(row[4])
            if not np.isnan(zp):
                ln2 = ax.axhline(zp, color="orange", ls=":", lw=0.5, alpha=0.4)
                self._shot_lines.append(ln2)

        ax.set_xlim(x_lo, x_hi)

        # Y range follows z_center if known; otherwise based on obs.
        if not np.isnan(snap.tracker.z_center):
            zc = float(snap.tracker.z_center)
            ax.set_ylim(zc - self.amplitude_m - 0.1,
                        zc + self.amplitude_m + 0.1)
        elif snap.recent_obs.shape[0] > 0:
            zs = snap.recent_obs[:, 3]
            pad = 0.1
            ax.set_ylim(float(zs.min()) - pad, float(zs.max()) + pad)
