"""LeadAimViewer: matplotlib figure + animation + widgets.

Composes the three scenes (3D, z(t), state panel) and wires playback
widgets (play/pause/step/rewind/speed/seek + Δt slider) to the data
source.

The viewer is data-source agnostic: it talks to LiveSource or
ReplaySource through the common ``step()`` interface. Live mode hides
the seek / step / speed widgets (no meaning).
"""

from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider

from .data_source import LiveSource, ReplaySource
from .scene_3d import Scene3D
from .scene_state import SceneState
from .scene_z import SceneZ

DataSource = Union[LiveSource, ReplaySource]


class LeadAimViewer:
    """Owns the figure, scenes, widgets, and animation timer."""

    TICK_INTERVAL_MS = 50   # ~20 Hz redraw cadence

    def __init__(self, source: DataSource,
                 amplitude_m: float, safety_margin_m: float):
        self.source = source
        self.amplitude_m = amplitude_m
        self.safety_margin_m = safety_margin_m

        self.fig = plt.figure(figsize=(16, 9), constrained_layout=False)
        self.fig.canvas.manager.set_window_title("Lead-Aim Viewer")
        gs = GridSpec(
            nrows=4, ncols=2, figure=self.fig,
            width_ratios=[1.4, 1.0],
            height_ratios=[3.0, 2.5, 0.6, 0.8],
            left=0.04, right=0.98, bottom=0.04, top=0.96,
            hspace=0.40, wspace=0.18,
        )
        # 3D scene spans left column rows 0-1.
        ax_3d = self.fig.add_subplot(gs[0:2, 0], projection="3d")
        # z(t) right, top.
        ax_z = self.fig.add_subplot(gs[0, 1])
        # state panel right, middle.
        ax_state = self.fig.add_subplot(gs[1, 1])
        # widget row (whole bottom).
        ax_widgets = self.fig.add_subplot(gs[3, :])
        ax_widgets.set_axis_off()

        self.scene_3d = Scene3D(ax_3d, amplitude_m, safety_margin_m)
        self.scene_z = SceneZ(ax_z, amplitude_m, safety_margin_m)
        self.scene_state = SceneState(ax_state, amplitude_m, safety_margin_m)

        self._build_widgets()
        self.anim = FuncAnimation(
            self.fig, self._on_tick, interval=self.TICK_INTERVAL_MS,
            blit=False, cache_frame_data=False,
        )

    def _build_widgets(self) -> None:
        is_live = self.source.is_live

        # Layout (axes coords in fig). Bottom row.
        ax_play = self.fig.add_axes([0.04, 0.06, 0.06, 0.04])
        self.btn_play = Button(ax_play, "Pause" if not getattr(self.source, "paused", False) else "Play")
        self.btn_play.on_clicked(self._on_play)

        if not is_live:
            ax_step = self.fig.add_axes([0.11, 0.06, 0.05, 0.04])
            self.btn_step = Button(ax_step, "Step >")
            self.btn_step.on_clicked(self._on_step)

            ax_rew = self.fig.add_axes([0.17, 0.06, 0.06, 0.04])
            self.btn_rewind = Button(ax_rew, "<< Rewind")
            self.btn_rewind.on_clicked(self._on_rewind)

            ax_speed = self.fig.add_axes([0.28, 0.06, 0.20, 0.04])
            self.sld_speed = Slider(ax_speed, "speed",
                                    valmin=0.1, valmax=4.0,
                                    valinit=1.0, valstep=0.1)
            self.sld_speed.on_changed(lambda v: self.source.set_speed(float(v)))

            lo, hi = self.source.t_range
            ax_seek = self.fig.add_axes([0.52, 0.06, 0.28, 0.04])
            self.sld_seek = Slider(ax_seek, "seek [s]",
                                   valmin=float(lo), valmax=max(float(hi), float(lo) + 1e-3),
                                   valinit=float(lo))
            self.sld_seek.on_changed(lambda v: self.source.seek(float(v)))

        ax_delay = self.fig.add_axes([0.84, 0.06, 0.13, 0.04])
        init_delay = float(getattr(self.source, "delay_s", 0.7))
        self.sld_delay = Slider(ax_delay, "Δt [s]",
                                valmin=0.1, valmax=2.0,
                                valinit=init_delay, valstep=0.05)
        self.sld_delay.on_changed(lambda v: self.source.set_delay(float(v)))

    # ── widget callbacks ──
    def _on_play(self, _event) -> None:
        new_state = not getattr(self.source, "paused", False)
        self.source.pause(new_state)
        self.btn_play.label.set_text("Play" if new_state else "Pause")

    def _on_step(self, _event) -> None:
        # Step requires being paused.
        self.source.pause(True)
        self.btn_play.label.set_text("Play")
        if hasattr(self.source, "step_frame"):
            self.source.step_frame(1)

    def _on_rewind(self, _event) -> None:
        self.source.seek(self.source.t_range[0])
        if hasattr(self, "sld_seek"):
            self.sld_seek.set_val(self.source.t_range[0])

    # ── animation tick ──
    def _on_tick(self, _frame_idx) -> None:
        snap = self.source.step()
        if snap is None:
            return
        self.scene_3d.update(snap)
        self.scene_z.update(snap)
        self.scene_state.update(snap)
        # Keep seek slider in sync (replay only) without re-triggering callback.
        if not snap.is_live and hasattr(self, "sld_seek"):
            saved = self.sld_seek.eventson
            self.sld_seek.eventson = False
            try:
                self.sld_seek.set_val(snap.t)
            finally:
                self.sld_seek.eventson = saved
