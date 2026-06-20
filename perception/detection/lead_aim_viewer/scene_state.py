"""Text status panel renderer (single ax.text monospace block)."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.axes import Axes

from .data_source import FrameSnapshot


class SceneState:
    def __init__(self, ax: Axes, amplitude_m: float, safety_margin_m: float):
        self.ax = ax
        self.amplitude_m = float(amplitude_m)
        self.safety_margin_m = float(safety_margin_m)
        ax.set_axis_off()
        self._text = ax.text(
            0.02, 0.98, "", family="monospace", fontsize=9,
            verticalalignment="top", transform=ax.transAxes,
        )

    def update(self, snap: FrameSnapshot) -> None:
        tr = snap.tracker
        v_cms = tr.v * 100.0 if not math.isnan(tr.v) else float("nan")
        v_arrow = ""
        if not math.isnan(tr.v):
            v_arrow = "↑ ascending" if tr.v > 0 else "↓ descending"
        zc = tr.z_center
        zp = tr.z_pred
        tau = tr.tau_endpoint
        if not (math.isnan(zp) or math.isnan(zc)):
            margin_to_ep = self.amplitude_m - abs(zp - zc)
        else:
            margin_to_ep = float("nan")
        mode = "LIVE" if snap.is_live else "REPLAY"
        t_lo, t_hi = snap.t_range
        valid_frac = ""
        if snap.recent_obs.shape[0] > 0:
            valid_frac = f"  ({snap.recent_obs.shape[0]} obs in 10s)"

        lines = [
            "─── tracker ───────────────",
            f"ready         : {tr.ready}",
            f"v             : {v_cms:+.2f} cm/s  {v_arrow}",
            f"z_center      : {zc:+.3f} m" if not math.isnan(zc) else
            "z_center      : --",
            f"endpoints_seen: {tr.endpoints_seen}",
            f"τ_endpoint    : {tau:+.2f} s" if not math.isnan(tau) else
            "τ_endpoint    : --",
            "",
            f"─── prediction (Δt={snap.delay_s:.2f}s) ─",
            f"z_pred        : {zp:+.3f} m" if not math.isnan(zp) else
            "z_pred        : --",
            f"is_safe       : {tr.is_safe}",
            (f"margin to ep  : {margin_to_ep*100:+.1f} cm  "
             f"(safety = {self.safety_margin_m*100:.1f} cm)")
            if not math.isnan(margin_to_ep) else
            "margin to ep  : --",
            "",
            "─── source ──────────────────",
            f"mode          : {mode}",
            f"t             : {snap.t:.2f} / {t_hi:.2f} s",
            f"obs valid     : {snap.xyz_obs is not None}{valid_frac}",
            f"shots         : {snap.recent_shots.shape[0]} in window",
        ]
        # Colorize based on critical flags. Matplotlib text doesn't do mixed
        # color, so use a single bg color for the whole panel when state is
        # alarming. Keep it subtle.
        if not tr.ready:
            self._text.set_color("0.5")
        elif not tr.is_safe:
            self._text.set_color("tab:red")
        else:
            self._text.set_color("black")
        self._text.set_text("\n".join(lines))
