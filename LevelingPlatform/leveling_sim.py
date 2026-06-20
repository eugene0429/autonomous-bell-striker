"""
3-DOF Leveling Platform Simulator (3-RSS parallel mechanism)

Geometry
--------
- 3 motors placed on a base circle (radius Rb) at 120 deg spacing.
- Each motor rotates a crank arm of length La in the vertical plane
  that contains the base radius direction (axis of rotation is tangential).
- A coupler rod of length Lc connects the crank tip to the top plate.
  * Crank-coupler joint at A: revolute (axis parallel to the motor axis,
    i.e., tangential) -- no angular-range limit modeled here.
  * Coupler-plate joint at P: spherical (RC-style ball joint) -- the rod
    can tilt away from the bracket axis by at most BALL_MAX_DEG.
  This is a 3-RRS parallel mechanism.
- The top plate has 3 attachment points on a circle of radius Rp, at
  the same 120 deg angles as the base (in the plate body frame).

Given a desired platform orientation (unit normal vector n) and a
commanded center height h, this script solves the inverse kinematics
for the three motor angles and renders the mechanism in 3D.

Motor angle convention
----------------------
theta = 0 corresponds to the home pose: crank points radially inward
(horizontal) and the coupler is vertical. Positive theta rotates the crank
tip upward. With this convention:
    A_i = B_i - La * (cos(theta) r_hat + sin(theta) z_hat)

Closed-form IK per leg
----------------------
    d_i = P_i - B_i
    u   = d_i . r_hat_i        (radial component)
    v   = d_i_z                (vertical component)
    k   = (|d_i|^2 + La^2 - Lc^2) / (2 La)
    u cos th + v sin th = -k
 => th_i = atan2(v, u) - acos( -k / sqrt(u^2 + v^2) )
    (the '-' branch picks the "knee-out" assembly so theta = 0 at home)

Run
---
    python leveling_sim.py
Drag the sliders to tilt the platform. The title shows the solved
motor angles in degrees. If a pose is unreachable (|k| > sqrt(u^2+v^2))
the title turns red.
"""

import argparse
import sys
import threading
from pathlib import Path as _P
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider

# 모터 스트리머 — __main__ 에서 --port 주면 활성, 아니면 None (sim-only).
_streamer: "Optional[MotorStreamer]" = None

# -------------------- Geometry (edit to match your hardware) --------------------
Rb = 0.10775   # base pivot radius [m]
La = 0.035   # crank (motor arm) length [m]
Lc = 0.111   # coupler rod length [m]
# Home pose assumption: crank points radially inward (horizontal) and
# coupler points straight up. This forces:
Rp = Rb - La   # top plate joint radius [m]  -> 0.06
H0 = Lc        # nominal platform center height [m] -> 0.12

PHI = np.deg2rad([0.0, 120.0, 240.0])   # motor angular positions

# Motor encoder resolution: 4096 steps per full revolution (2*pi)
MOTOR_STEPS = 4096
MOTOR_STEP_RAD = 2.0 * np.pi / MOTOR_STEPS

# RC ball joint (only at the plate end P): the ball sits inside a bracket
# cup; the rod can tilt away from the bracket's opening axis by at most
# BALL_MAX_DEG before hitting the cup rim. Typical RC ball joints allow
# ~25-35 deg.
BALL_MAX_DEG = 30.0

# P-side bracket opening axis (the only ball joint in the mechanism):
# glued to the top plate with the cup facing down, so the opening axis in
# the plate body frame is -z (when the plate is level the bracket looks
# straight down). In world frame the rod direction *entering* the bracket
# is -coupler_dir; comparing it to -R @ z_hat is the same as comparing
# coupler_dir to +R @ z_hat (the plate's body +z in world coords).


# -------------------- Math helpers --------------------
def plate_center_offset(R):
    """
    3-RRS geometric constraint. Given a (yaw-free) platform rotation R,
    return the plate-center horizontal offset (cx, cy) such that each
    P_i = (cx, cy, *) + R @ p_body_i lies exactly in its motor's
    r_hat-z vertical plane (i.e., P_i . t_hat_i = 0).

    Derivation:
        For each leg i let a_i = Rp * (R r_hat_i) . t_hat_i.
        The 3 constraints become  -cx sin(phi_i) + cy cos(phi_i) = -a_i.
        With PHI = {0, 120, 240}, sum(t_hat_i) = 0 and
        sum(t_hat_i t_hat_i^T) = (3/2) I_2, which gives the closed form
        below. Yaw-free R (as produced by rot_from_normal) guarantees
        sum(a_i) = 0 so the 3 equations are consistent.
    """
    a = np.zeros(3)
    for i, phi in enumerate(PHI):
        r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
        t_hat = np.array([-np.sin(phi), np.cos(phi), 0.0])
        a[i] = Rp * float(np.dot(R @ r_hat, t_hat))
    cx =  (2.0 / 3.0) * float(np.sum(np.sin(PHI) * a))
    cy = -(2.0 / 3.0) * float(np.sum(np.cos(PHI) * a))
    return cx, cy


def rot_from_normal(n):
    """Rotation matrix mapping +z to the unit vector n (shortest arc)."""
    n = np.asarray(n, dtype=float)
    n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, n))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180 deg flip around x
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def inverse_kinematics(normal, height, ball_max_deg=None):
    """
    Returns:
        thetas       : (3,) motor angles [rad] (NaN where unreachable)
        A            : (3,3) crank tip positions
        P            : (3,3) top joint positions
        B            : (3,3) base pivot positions
        ok           : bool - all legs reachable (length AND P-side ball
                       joint limit)
        ball_angles  : (3,) deflection angle [deg] of the coupler from the
                       plate-side bracket axis, one per leg. NaN where the
                       length constraint is infeasible. (The A-side joint
                       is revolute, so no angular limit is modeled there.)
    """
    if ball_max_deg is None:
        ball_max_deg = BALL_MAX_DEG

    R = rot_from_normal(normal)
    # 3-RRS constraint: plate center shifts horizontally with tilt so that
    # each P_i stays in its motor's r_hat-z plane.
    cx, cy = plate_center_offset(R)
    c = np.array([cx, cy, height])
    z_hat = np.array([0.0, 0.0, 1.0])
    plate_up_world = R @ z_hat  # plate's body +z in world coords

    B = np.zeros((3, 3))
    P = np.zeros((3, 3))
    A = np.zeros((3, 3))
    thetas = np.full(3, np.nan)
    ball_angles = np.full(3, np.nan)
    ok = True

    for i, phi in enumerate(PHI):
        r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
        B[i] = Rb * r_hat
        # top joint in plate body frame, then rotated & translated
        p_body = np.array([Rp * np.cos(phi), Rp * np.sin(phi), 0.0])
        P[i] = c + R @ p_body

        d = P[i] - B[i]
        u = float(np.dot(d, r_hat))
        v = float(d[2])
        k = (float(np.dot(d, d)) + La * La - Lc * Lc) / (2.0 * La)
        rho = np.hypot(u, v)
        if rho < 1e-12 or abs(k) > rho:
            ok = False
            continue
        # '-' branch -> home pose is theta = 0
        # (crank horizontal pointing inward, coupler vertical)
        th = np.arctan2(v, u) - np.arccos(-k / rho)
        # wrap to [-pi, pi]
        th = (th + np.pi) % (2.0 * np.pi) - np.pi
        # Quantize to nearest encoder step (4096 counts / revolution)
        th = np.round(th / MOTOR_STEP_RAD) * MOTOR_STEP_RAD
        thetas[i] = th
        A[i] = B[i] - La * (np.cos(th) * r_hat + np.sin(th) * z_hat)

        # P-side ball joint deflection (A-side is revolute: no check).
        coupler_dir = (P[i] - A[i]) / Lc
        cos_P = float(np.clip(np.dot(coupler_dir, plate_up_world), -1.0, 1.0))
        ang_P = np.rad2deg(np.arccos(cos_P))
        ball_angles[i] = ang_P
        if ang_P > ball_max_deg:
            ok = False

    return thetas, A, P, B, ok, ball_angles


def _platform_joints(nx, ny, zc):
    """Given pose state (nx, ny, zc), return P_i (3x3) and the normal n.
    The plate center is placed using the 3-RRS offset so every P_i lies in
    its motor's r_hat-z plane."""
    s2 = nx * nx + ny * ny
    nz = np.sqrt(max(1.0 - s2, 0.0))
    n = np.array([nx, ny, nz])
    R = rot_from_normal(n)
    cx, cy = plate_center_offset(R)
    c = np.array([cx, cy, zc])
    P = np.zeros((3, 3))
    for i, phi in enumerate(PHI):
        p_body = np.array([Rp * np.cos(phi), Rp * np.sin(phi), 0.0])
        P[i] = c + R @ p_body
    return P, n


def forward_kinematics(thetas_q, guess_n, guess_zc):
    """
    Given the three (quantized) motor angles, solve for the platform pose
    via Newton's method. Returns (n_actual, zc_actual, ok).
    """
    # crank tips from quantized angles (motor-angle convention: theta=0 at home)
    A = np.zeros((3, 3))
    for i, phi in enumerate(PHI):
        r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
        A[i] = Rb * r_hat - La * (np.cos(thetas_q[i]) * r_hat
                                  + np.sin(thetas_q[i]) * np.array([0, 0, 1.0]))

    def residual(state):
        P, _ = _platform_joints(state[0], state[1], state[2])
        return np.array([np.sum((P[i] - A[i])**2) - Lc * Lc for i in range(3)])

    state = np.array([guess_n[0], guess_n[1], guess_zc], dtype=float)
    eps = 1e-7
    for _ in range(25):
        r = residual(state)
        if np.linalg.norm(r) < 1e-12:
            break
        J = np.zeros((3, 3))
        for k in range(3):
            s2 = state.copy(); s2[k] += eps
            J[:, k] = (residual(s2) - r) / eps
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return guess_n, guess_zc, False
        state += dx
        if np.linalg.norm(dx) < 1e-12:
            break
    _, n_act = _platform_joints(state[0], state[1], state[2])
    ok = np.linalg.norm(residual(state)) < 1e-8
    return n_act, float(state[2]), ok


# =====================================================================
#  MotorStreamer — GUI 드래그 → 실모터 연속 구동 (AIMF streaming)
#
#  - GUI 스레드: 매 update() 마다 push((x,y,z)) — 한 슬롯에 덮어쓰기만 한다.
#  - Worker 스레드: mc.aim_fast() 는 비블로킹 (펌웨어가 syncWrite 후 즉시 OK).
#    USB CDC RTT (~3 ms) 만에 다음 명령 송신 가능.
#  - drop-old: 워커가 송신하는 사이에 들어온 중간 타겟은 폐기, 가장 최신
#    타겟만 다음 회차에 송신.
#  - 결과: 60 Hz 드래그를 stutter 없이 연속 추종. Dynamixel 서보의 내부
#    프로파일 컨트롤러가 매 GOAL_POSITION 갱신마다 매끄럽게 재orientation.
# =====================================================================
class MotorStreamer:
    def __init__(self, mc, ik, log=print, use_fast: bool = True):
        self._mc = mc
        self._ik = ik
        self._log = log
        self._latest: Optional[Tuple[float, float, float]] = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._use_fast = use_fast

    def set_fast(self, fast: bool) -> None:
        """AIMF (비블로킹) vs AIM (블로킹) 전환. 진단/디버깅용."""
        self._use_fast = bool(fast)
        mode = "AIMF (fast)" if self._use_fast else "AIM (blocking)"
        self._log(f"[motor-streamer] mode = {mode}")

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="motor-streamer", daemon=True)
        self._thread.start()

    def push(self, target) -> None:
        with self._lock:
            self._latest = (float(target[0]), float(target[1]), float(target[2]))
        self._event.set()

    def stop(self) -> None:
        self._stop.set()
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._event.wait()
            self._event.clear()
            if self._stop.is_set():
                break
            with self._lock:
                target = self._latest
                self._latest = None
            if target is None:
                continue
            try:
                out = self._ik.aim_at(target)
                if out.get("angles_steps") is None:
                    continue   # IK unreachable — skip
                if self._use_fast:
                    self._mc.aim_fast(out)   # AIMF 비블로킹
                else:
                    self._mc.aim(out)        # AIM 블로킹 (진단용)
            except Exception as e:
                self._log(f"[motor-streamer] {type(e).__name__}: {e}")


# =====================================================================
#  LAYOUT — all axes positioned manually for a clean, non-overlapping UI
#
#  +-----------------------------------------+------------------+----+
#  |                                         |   XY picker      | Z  |
#  |           3D view                       |   (square)       |bar |
#  |                                         |                  |    |
#  +-----------------------------------------+------------------+----+
#  |           (3D continues)                | sliders + info   |    |
#  +-----------------------------------------+------------------+----+
# =====================================================================
fig = plt.figure(figsize=(14, 9))

# ---- 3D view (left) ----
ax = fig.add_axes([0.02, 0.06, 0.52, 0.90], projection='3d')

# ---- XY picker (right-top, square) ----
# To keep it square: fig is 14 x 9 in, so frac_w 0.28 => 3.92 in
# frac_h = 3.92 / 9 = 0.436
_xy_l, _xy_b, _xy_w, _xy_h = 0.58, 0.50, 0.28, 0.44
ax_xy = fig.add_axes([_xy_l, _xy_b, _xy_w, _xy_h])

# ---- Z bar (right of XY, same vertical span) ----
ax_z = fig.add_axes([_xy_l + _xy_w + 0.03, _xy_b, 0.02, _xy_h])

# ---- Sliders (below pickers) ----
_sl_l, _sl_w = 0.64, 0.26
ax_h    = fig.add_axes([_sl_l, 0.38, _sl_w, 0.022])
ax_Rb   = fig.add_axes([_sl_l, 0.32, _sl_w, 0.022])
ax_La   = fig.add_axes([_sl_l, 0.26, _sl_w, 0.022])
ax_Lc   = fig.add_axes([_sl_l, 0.20, _sl_w, 0.022])
ax_ball = fig.add_axes([_sl_l, 0.14, _sl_w, 0.022])
s_h    = Slider(ax_h,    'height',   0.04, 0.30, valinit=H0, valfmt='%.3f m')
s_Rb   = Slider(ax_Rb,   'Rb',       0.04, 0.25, valinit=Rb, valfmt='%.3f m')
s_La   = Slider(ax_La,   'La',       0.01, 0.12, valinit=La, valfmt='%.3f m')
s_Lc   = Slider(ax_Lc,   'Lc',       0.04, 0.30, valinit=Lc, valfmt='%.3f m')
s_ball = Slider(ax_ball, 'ball max', 5.0,  60.0, valinit=BALL_MAX_DEG, valfmt='%.1f deg')

err_text = ax_xy.text(0.02, 0.98, '', transform=ax_xy.transAxes,
                      va='top', ha='left', fontsize=8, family='monospace',
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85),
                      zorder=10)

# ---- 3D artists ----
base_poly,  = ax.plot([], [], [], 'k-', lw=2)
plate_poly, = ax.plot([], [], [], 'b-', lw=2)
crank_lines   = [ax.plot([], [], [], 'r-', lw=3)[0] for _ in range(3)]
coupler_lines = [ax.plot([], [], [], 'g-', lw=2)[0] for _ in range(3)]
pivot_pts,   = ax.plot([], [], [], 'ko', ms=5)
joint_pts,   = ax.plot([], [], [], 'bo', ms=5)
tip_pts,     = ax.plot([], [], [], 'rs', ms=5)
normal_line, = ax.plot([], [], [], 'm-', lw=2)
aim_line,    = ax.plot([], [], [], 'm--', lw=1.2)
actual_line, = ax.plot([], [], [], 'c-', lw=1.2)

lim = max(Rb, Rp) * 3.0
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
ax.set_box_aspect((1, 1, 1))
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
title = ax.set_title('', pad=10)

# ---- XY picker setup ----
XY_LIM = 1.0
Z_LIM  = (2.5, 3.5)
target_state = {'x': 0.0, 'y': 0.0, 'z': 3.0}

ax_xy.set_xlim(-XY_LIM, XY_LIM); ax_xy.set_ylim(-XY_LIM, XY_LIM)
ax_xy.set_aspect('equal')
ax_xy.set_title('Target X, Y  (click / drag)', fontsize=10, pad=6)
ax_xy.set_xlabel('X [m]', fontsize=9); ax_xy.set_ylabel('Y [m]', fontsize=9)
ax_xy.tick_params(labelsize=8)
ax_xy.grid(True, alpha=0.3)

WS_N = 60
_gx = np.linspace(-XY_LIM, XY_LIM, WS_N)
_gy = np.linspace(-XY_LIM, XY_LIM, WS_N)
ws_img = ax_xy.imshow(np.zeros((WS_N, WS_N)),
                      extent=[-XY_LIM, XY_LIM, -XY_LIM, XY_LIM],
                      origin='lower', cmap='Greens', vmin=0, vmax=1,
                      alpha=0.35, zorder=0)
_th = np.linspace(0, 2 * np.pi, 64)
base_circle, = ax_xy.plot(Rb * np.cos(_th), Rb * np.sin(_th), 'k-', lw=1)
ax_xy.axhline(0, color='gray', lw=0.5); ax_xy.axvline(0, color='gray', lw=0.5)
xy_marker, = ax_xy.plot([target_state['x']], [target_state['y']],
                        'm*', ms=14, zorder=5)
xy_hit, = ax_xy.plot([], [], 'co', ms=7, zorder=5)

# ---- Z bar setup ----
ax_z.set_xlim(0, 1); ax_z.set_ylim(*Z_LIM)
ax_z.set_xticks([])
ax_z.set_title('Z [m]', fontsize=9, pad=6)
ax_z.tick_params(labelsize=8)
ax_z.yaxis.tick_right(); ax_z.yaxis.set_label_position('right')
z_marker, = ax_z.plot([0.5], [target_state['z']], 'm*', ms=14)


def close_loop(pts):
    return np.vstack([pts, pts[0]])


def update(_=None):
    h = s_h.val
    T = np.array([target_state['x'], target_state['y'], target_state['z']])
    # Commanded aim direction: from nominal on-axis point (0,0,h) to T.
    v = T - np.array([0.0, 0.0, h])
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = v / nv

    thetas, A, P, B, ok, ball_angles = inverse_kinematics(n, h, s_ball.val)
    # Actual plate center (shifted horizontally by the 3-RRS constraint).
    cx_off, cy_off = plate_center_offset(rot_from_normal(n))
    c = np.array([cx_off, cy_off, h])

    bp = close_loop(B)
    base_poly.set_data(bp[:, 0], bp[:, 1]); base_poly.set_3d_properties(bp[:, 2])
    pp = close_loop(P)
    plate_poly.set_data(pp[:, 0], pp[:, 1]); plate_poly.set_3d_properties(pp[:, 2])

    for i in range(3):
        if np.isnan(thetas[i]):
            crank_lines[i].set_data([], []);  crank_lines[i].set_3d_properties([])
            coupler_lines[i].set_data([], []); coupler_lines[i].set_3d_properties([])
            continue
        xs = [B[i, 0], A[i, 0]]; ys = [B[i, 1], A[i, 1]]; zs = [B[i, 2], A[i, 2]]
        crank_lines[i].set_data(xs, ys); crank_lines[i].set_3d_properties(zs)
        xs = [A[i, 0], P[i, 0]]; ys = [A[i, 1], P[i, 1]]; zs = [A[i, 2], P[i, 2]]
        coupler_lines[i].set_data(xs, ys); coupler_lines[i].set_3d_properties(zs)

    pivot_pts.set_data(B[:, 0], B[:, 1]); pivot_pts.set_3d_properties(B[:, 2])
    joint_pts.set_data(P[:, 0], P[:, 1]); joint_pts.set_3d_properties(P[:, 2])
    valid = ~np.isnan(thetas)
    tip_pts.set_data(A[valid, 0], A[valid, 1]); tip_pts.set_3d_properties(A[valid, 2])

    nend = c + 0.06 * n
    normal_line.set_data([c[0], nend[0]], [c[1], nend[1]])
    normal_line.set_3d_properties([c[2], nend[2]])

    # commanded aim line from platform center to target (target is off-screen)
    aim_line.set_data([c[0], T[0]], [c[1], T[1]])
    aim_line.set_3d_properties([c[2], T[2]])

    # --- Forward kinematics with quantized motor angles -> actual aim ---
    hit = np.array([np.nan, np.nan, np.nan])
    err_xy = np.nan; err_ang = np.nan
    if ok and not np.any(np.isnan(thetas)):
        n_act, zc_act, fk_ok = forward_kinematics(thetas, n, h)
        if fk_ok and n_act[2] > 1e-6:
            cx_act, cy_act = plate_center_offset(rot_from_normal(n_act))
            c_act = np.array([cx_act, cy_act, zc_act])
            # ray c_act + t*n_act  intersected with plane z = T[2]
            t = (T[2] - c_act[2]) / n_act[2]
            hit = c_act + t * n_act
            err_xy = float(np.linalg.norm(hit[:2] - T[:2]))
            cosang = float(np.clip(np.dot(n, n_act), -1, 1))
            err_ang = np.rad2deg(np.arccos(cosang))
            actual_line.set_data([c_act[0], hit[0]], [c_act[1], hit[1]])
            actual_line.set_3d_properties([c_act[2], hit[2]])
        else:
            actual_line.set_data([], []); actual_line.set_3d_properties([])
    else:
        actual_line.set_data([], []); actual_line.set_3d_properties([])

    # P-side ball joint deflection summary (max across 3 legs)
    if np.all(np.isnan(ball_angles)):
        ball_txt = '  ball P   : N/A'
    else:
        max_P = np.nanmax(ball_angles)
        flag = '  !!' if max_P > s_ball.val else ''
        ball_txt = (
            f'  ball P   : {max_P:5.2f} deg (max over 3 legs){flag}\n'
            f'  ball lim : {s_ball.val:5.2f} deg'
        )
    # 3-RRS plate-center horizontal shift
    ball_txt += (f'\n  c shift  : ({c[0]*1000:+6.2f}, {c[1]*1000:+6.2f}) mm')

    if np.isnan(hit[0]):
        xy_hit.set_data([], [])
        err_text.set_text('  aim error: N/A\n' + ball_txt)
    else:
        xy_hit.set_data([hit[0]], [hit[1]])
        err_text.set_text(
            f'  target  : ({T[0]:+.4f}, {T[1]:+.4f}, {T[2]:.2f})\n'
            f'  actual  : ({hit[0]:+.4f}, {hit[1]:+.4f}, {hit[2]:.2f})\n'
            f'  err XY  : {err_xy*1000:7.3f} mm\n'
            f'  err ang : {err_ang:7.4f} deg\n'
            f'  Rp(auto): {Rp:.4f} m\n'
            + ball_txt
        )

    deg = np.rad2deg(thetas)
    tilt_deg = np.rad2deg(np.arccos(np.clip(n[2], -1, 1)))
    txt = (f'theta1={deg[0]:+6.2f}  theta2={deg[1]:+6.2f}  theta3={deg[2]:+6.2f} [deg]'
           f'   (tilt={tilt_deg:.1f} deg)')
    title.set_text(txt)
    title.set_color('black' if ok else 'red')
    fig.canvas.draw_idle()

    # 실모터 스트리밍 (drop-old). IK 불가능한 포즈는 건너뜀.
    if _streamer is not None and ok and not np.any(np.isnan(thetas)):
        _streamer.push((target_state['x'], target_state['y'], target_state['z']))


def recompute_workspace():
    """For current height + link lengths + target Z, compute which
    target (x,y) points are reachable, and update the shading."""
    h = s_h.val
    z = target_state['z']
    c = np.array([0.0, 0.0, h])
    mask = np.zeros((WS_N, WS_N), dtype=float)
    for iy, y in enumerate(_gy):
        for ix, x in enumerate(_gx):
            v = np.array([x, y, z]) - c
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                mask[iy, ix] = 1.0
                continue
            n = v / nv
            _, _, _, _, ok, _ = inverse_kinematics(n, h, s_ball.val)
            mask[iy, ix] = 1.0 if ok else 0.0
    ws_img.set_data(mask)


def on_params(_=None):
    """Link length / height slider handler: mutates globals, refreshes view."""
    global Rb, La, Lc, Rp, H0
    Rb = s_Rb.val
    La = s_La.val
    Lc = s_Lc.val
    Rp = max(Rb - La, 1e-4)
    H0 = Lc
    base_circle.set_data(Rb*np.cos(_th), Rb*np.sin(_th))
    recompute_workspace()
    update()


def _on_click(event):
    if event.inaxes is ax_xy and event.xdata is not None:
        target_state['x'] = float(np.clip(event.xdata, -XY_LIM, XY_LIM))
        target_state['y'] = float(np.clip(event.ydata, -XY_LIM, XY_LIM))
        xy_marker.set_data([target_state['x']], [target_state['y']])
        update()
    elif event.inaxes is ax_z and event.ydata is not None:
        target_state['z'] = float(np.clip(event.ydata, *Z_LIM))
        z_marker.set_data([0.5], [target_state['z']])
        recompute_workspace()
        update()

fig.canvas.mpl_connect('button_press_event', _on_click)
# drag support
def _on_motion(event):
    if event.button == 1:
        _on_click(event)
fig.canvas.mpl_connect('motion_notify_event', _on_motion)

s_h.on_changed(on_params)
s_Rb.on_changed(on_params)
s_La.on_changed(on_params)
s_Lc.on_changed(on_params)
s_ball.on_changed(on_params)

recompute_workspace()
update()


# -------------------- Programmatic entry point --------------------
def solve(normal, height=H0):
    """Given a direction vector, return motor angles [deg]."""
    thetas, _A, _P, _B, ok, _ba = inverse_kinematics(normal, height)
    return np.rad2deg(thetas), ok


def aim_at(target, height=H0):
    """Point the platform at a 3D world point. Returns motor angles [deg]."""
    T = np.asarray(target, dtype=float)
    c = np.array([0.0, 0.0, height])
    v = T - c
    n = v / np.linalg.norm(v)
    return solve(n, height)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="3-RRS leveling sim + optional live motor streaming")
    ap.add_argument("--port", default=None,
                    help="OpenRB serial port (예: /dev/cu.usbmodem11301). "
                         "생략 시 sim-only.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--home", action="store_true",
                    help="모터 연결 직후 HOME 송신")
    ap.add_argument("--verbose", action="store_true",
                    help="시리얼 송수신 라인을 stderr 로 출력")
    args = ap.parse_args()

    # quick sanity check
    ang, _ok = solve([0, 0, 1], H0)
    print('neutral pose motor angles [deg]:', ang, 'reachable:', _ok)

    if args.port is None:
        plt.show()
    else:
        # leveling_ik / leveling_motor 는 같은 디렉토리에 있으므로 import 경로 추가
        sys.path.insert(0, str(_P(__file__).resolve().parent))
        from leveling_ik    import LevelingConfig, LevelingIK         # noqa: E402
        from leveling_motor import LevelingMotorClient, MotorClientConfig  # noqa: E402

        mc_ik = LevelingIK(LevelingConfig())
        mc = LevelingMotorClient(MotorClientConfig(
            port=args.port, baud=args.baud, verbose=args.verbose))
        mc.connect()
        try:
            if not mc.ping():
                print("[FAIL] OpenRB PING 실패", file=sys.stderr)
                sys.exit(2)
            if args.home:
                print("[motor] HOME ...")
                mc.home()

            _streamer = MotorStreamer(mc, mc_ik, use_fast=True)
            _streamer.start()
            print(f"[motor] streaming to {args.port} "
                  f"(AIMF non-blocking, drop-old)")

            # ── Fast 모드 토글 (AIMF vs AIM 진단용) ───────────────
            #   체크 ON  → AIMF (비블로킹, 스트리밍에 적합)
            #   체크 OFF → AIM  (블로킹, 모터가 실제 응답하는지 검증용)
            ax_fast = fig.add_axes([0.64, 0.04, 0.20, 0.07])
            fast_check = CheckButtons(ax_fast, ["Fast mode (AIMF)"], [True])

            def _on_fast_toggle(_label):
                fast = fast_check.get_status()[0]
                if _streamer is not None:
                    _streamer.set_fast(fast)
            fast_check.on_clicked(_on_fast_toggle)

            # 초기 타겟 1회 송신
            update()

            def _on_close(_evt):
                if _streamer is not None:
                    _streamer.stop()
            fig.canvas.mpl_connect("close_event", _on_close)

            plt.show()
        finally:
            if _streamer is not None:
                _streamer.stop()
            try:
                mc.disconnect()
            except Exception:
                pass
