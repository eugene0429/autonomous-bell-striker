"""3D subplot renderer: plate, bell trajectory, current pos, aim point, safety band.

All coordinates are in plate frame (m). +X forward, +Y left, +Z up.

Auto-rescales Z range every few seconds based on z_center estimate so the
safety band stays visible without the user zooming.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.axes import Axes

from .data_source import FrameSnapshot


class Scene3D:
    """One-axis owner. The viewer creates the 3D ``ax`` and passes it in."""

    PLATE_R = 0.06   # m, top plate radius (Rp from leveling sim)
    BAND_HALF = 0.5  # m, safety band rectangle half-extent in x and y

    def __init__(self, ax: Axes, amplitude_m: float, safety_margin_m: float):
        self.ax = ax
        self.amplitude_m = float(amplitude_m)
        self.safety_margin_m = float(safety_margin_m)

        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")
        self.ax.set_title("3D: plate frame")

        self._draw_plate()

        self._traj_scatter = None
        self._cur_scatter = None
        self._aim_scatter = None
        self._aim_line = None
        self._band_lo = None
        self._band_hi = None
        self._last_zrange_t = -1e9

    def _draw_plate(self) -> None:
        # Plate disc (filled triangle fan) at z=0.
        th = np.linspace(0.0, 2 * np.pi, 48)
        xs = self.PLATE_R * np.cos(th)
        ys = self.PLATE_R * np.sin(th)
        zs = np.zeros_like(th)
        self.ax.plot(xs, ys, zs, color="0.4", lw=1.2)
        # Origin marker + launcher exit direction (plate normal, length 0.1m).
        self.ax.plot([0, 0], [0, 0], [0, 0.1], color="0.4", lw=1.5)
        self.ax.scatter([0], [0], [0], color="0.4", s=20)

    def update(self, snap: FrameSnapshot) -> None:
        # Trajectory points colored by recency.
        if self._traj_scatter is not None:
            self._traj_scatter.remove()
            self._traj_scatter = None
        if snap.recent_obs.shape[0] > 0:
            xs = snap.recent_obs[:, 1]
            ys = snap.recent_obs[:, 2]
            zs = snap.recent_obs[:, 3]
            # color = age (older = lighter)
            ts = snap.recent_obs[:, 0]
            ages = (snap.t - ts) / max(1e-3, snap.t - ts.min() if ts.size > 1 else 1.0)
            self._traj_scatter = self.ax.scatter(
                xs, ys, zs, c=1.0 - ages, cmap="viridis",
                s=12, vmin=0.0, vmax=1.0,
            )

        # Current position.
        if self._cur_scatter is not None:
            self._cur_scatter.remove()
            self._cur_scatter = None
        if snap.xyz_obs is not None:
            self._cur_scatter = self.ax.scatter(
                [snap.xyz_obs[0]], [snap.xyz_obs[1]], [snap.xyz_obs[2]],
                color="red", s=80, marker="o", edgecolors="black",
                linewidths=0.5, label="current",
            )

        # Predicted aim point + dashed line from plate origin.
        if self._aim_scatter is not None:
            self._aim_scatter.remove()
            self._aim_scatter = None
        if self._aim_line is not None:
            self._aim_line.remove()
            self._aim_line = None
        if (snap.xyz_obs is not None and snap.tracker.ready
                and not np.isnan(snap.tracker.z_pred)):
            xa, ya = float(snap.xyz_obs[0]), float(snap.xyz_obs[1])
            za = float(snap.tracker.z_pred)
            self._aim_scatter = self.ax.scatter(
                [xa], [ya], [za], color="gold", s=120, marker="X",
                edgecolors="black", linewidths=0.5, label="aim (t+Δt)",
            )
            (self._aim_line,) = self.ax.plot(
                [0, xa], [0, ya], [0, za],
                color="gold", lw=0.8, ls="--", alpha=0.7,
            )

        # Safety band (two horizontal squares at z_center ± (A - margin)).
        for h in (self._band_lo, self._band_hi):
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
        self._band_lo = None
        self._band_hi = None
        if not np.isnan(snap.tracker.z_center):
            zc = float(snap.tracker.z_center)
            half = self.amplitude_m - self.safety_margin_m
            color = "tab:red" if not snap.tracker.is_safe else "tab:green"
            for z in (zc - half, zc + half):
                xs = np.array([[-self.BAND_HALF, self.BAND_HALF],
                               [-self.BAND_HALF, self.BAND_HALF]])
                ys = np.array([[-self.BAND_HALF, -self.BAND_HALF],
                               [self.BAND_HALF, self.BAND_HALF]])
                zs = np.full_like(xs, z)
                surf = self.ax.plot_surface(
                    xs, ys, zs, color=color, alpha=0.10,
                    linewidth=0, antialiased=False,
                )
                if z < zc:
                    self._band_lo = surf
                else:
                    self._band_hi = surf

        # Auto-rescale Z every 5 s based on z_center.
        if snap.t - self._last_zrange_t > 5.0:
            self._rescale_axes(snap)
            self._last_zrange_t = snap.t

    def _rescale_axes(self, snap: FrameSnapshot) -> None:
        if not np.isnan(snap.tracker.z_center):
            zc = float(snap.tracker.z_center)
        elif snap.xyz_obs is not None:
            zc = float(snap.xyz_obs[2])
        else:
            zc = 0.0
        self.ax.set_zlim(zc - self.amplitude_m - 0.1,
                         zc + self.amplitude_m + 0.1)
        # X/Y stay fixed: plate-relative range that comfortably covers a bell
        # ~1-3m forward and ±1m lateral.
        self.ax.set_xlim(-0.5, 3.5)
        self.ax.set_ylim(-1.5, 1.5)
