"""
3-RRS Leveling Platform — inverse kinematics module.

Target 3D point (x, y, z) → compute the 3 motor angles needed to align the
plate normal toward that direction.

Vehicle / hardware model
-----------------
- 3 motors at 120° spacing on a base of radius Rb.
- Each motor has a crank of length La (rotating at point B, axis of rotation is tangential).
- A coupler of length Lc connects the crank tip A to the plate joint P.
  * A: revolute joint (tangential axis) → the coupler stays in the motor's r̂-z vertical plane.
  * P: spherical (RC ball) joint, allows deflection up to BALL_MAX_DEG relative to the plate +z axis.
- The top plate is attached on a circle of radius Rp = Rb - La.
- Home pose (all motors θ=0): crank horizontal (inward), coupler vertical.
  Plate center = (0, 0, H0=Lc).

3-RRS center offset
----------------
When the plate tilts, the center slides slightly horizontally so that every P_i
stays in its own motor's r̂-z plane. _plate_center_offset() computes this in
closed form.

Usage example
------
    from leveling_ik import LevelingIK, LevelingConfig

    cfg = LevelingConfig()                  # defaults match the build spec
    ik  = LevelingIK(cfg)
    out = ik.aim_at((0.10, 0.00, 3.0))      # 3D point in the plate-base frame
    if out['ok']:
        send_to_motors(out['angles_steps'])  # encoder step (0..motor_steps-1)
    else:
        # length infeasible or ball joint limit exceeded → realign base and retry
        ...

CLI
---
    python3 leveling_ik.py --target 0.10 0.00 3.0

Dependencies
-----
Uses numpy only. Headless (Pi5 OK).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class LevelingConfig:
    # ── Mechanism parameters ──
    Rb: float = 0.108                 # base pivot radius [m]
    La: float = 0.035                 # crank length [m]
    Lc: float = 0.111                 # coupler length [m]

    # ── Motor parameters ──
    motor_phis_deg: Tuple[float, float, float] = (0.0, 120.0, 240.0)  # azimuths
    motor_steps: int = 4096          # encoder counts / revolution

    # ── Limits ──
    ball_max_deg: float = 30.0       # P-side ball joint deflection limit

    # ── Output ──
    quantize: bool = True            # whether to round to encoder steps

    # ── Derived values ──
    @property
    def Rp(self) -> float:
        """Plate joint radius (forced by the home pose: crank horizontal + coupler vertical)."""
        return self.Rb - self.La

    @property
    def H0(self) -> float:
        """Plate center height at the home pose."""
        return self.Lc

    @property
    def motor_step_rad(self) -> float:
        return 2.0 * np.pi / self.motor_steps


# ──────────────────────────────────────────────
# IK module
# ──────────────────────────────────────────────
class LevelingIK:
    """3-RRS leveling platform inverse kinematics."""

    def __init__(self, cfg: Optional[LevelingConfig] = None):
        self.cfg = cfg if cfg is not None else LevelingConfig()
        self._phi = np.deg2rad(
            np.asarray(self.cfg.motor_phis_deg, dtype=float))

    # ── Main API ──
    def aim_at(
        self,
        target,
        height: Optional[float] = None,
    ) -> Dict:
        """
        Compute motor angles so the plate center (0, 0, height) aims at the target 3D point.

        Parameters
        ----------
        target  (x, y, z) [m] in the plate-base frame
                Note: camera-frame results must be transformed by the caller beforehand.
        height  plate center z [m] (None → cfg.H0)

        Returns
        -------
        dict — same as aim_normal().
        """
        h = self.cfg.H0 if height is None else height
        T = np.asarray(target, dtype=float)
        v = T - np.array([0.0, 0.0, h])
        nv = float(np.linalg.norm(v))
        n = np.array([0.0, 0.0, 1.0]) if nv < 1e-9 else v / nv
        return self.aim_normal(n, height=h)

    def aim_normal(
        self,
        normal,
        height: Optional[float] = None,
    ) -> Dict:
        """
        Compute motor angles by directly supplying the plate normal (unit vector).

        Returns
        -------
        dict
            angles_deg     list[float]|None   motor angles [deg] (None if unreachable)
            angles_rad     list[float]|None   motor angles [rad]
            angles_steps   list[int]|None     encoder step (rounded)
            ok             bool               length OK AND ball limit OK
            ball_deg       list[float|None]   ball deflection of each leg [deg]
            c_shift_m      tuple[float,float] plate center horizontal shift [m]
            normal         list[float]        commanded plate normal (unit)
        """
        h = self.cfg.H0 if height is None else height
        thetas, ok, ball = self._inverse_kinematics(normal, h)
        cx, cy = self._plate_center_offset(self._rot_from_normal(normal))

        reachable = not bool(np.any(np.isnan(thetas)))
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        step_rad = self.cfg.motor_step_rad

        return {
            "angles_deg":   ([float(np.rad2deg(t)) for t in thetas]
                             if reachable else None),
            "angles_rad":   [float(t) for t in thetas] if reachable else None,
            "angles_steps": ([int(round(t / step_rad)) for t in thetas]
                             if reachable else None),
            "ok":           bool(ok),
            "ball_deg":     [float(b) if not np.isnan(b) else None for b in ball],
            "c_shift_m":    (float(cx), float(cy)),
            "normal":       [float(n[0]), float(n[1]), float(n[2])],
        }

    # ── Internal: rotation / center shift / IK ──
    @staticmethod
    def _rot_from_normal(n) -> np.ndarray:
        """Shortest-arc rotation from +z to unit vector n (yaw-free)."""
        n = np.asarray(n, dtype=float)
        n = n / np.linalg.norm(n)
        z = np.array([0.0, 0.0, 1.0])
        c = float(np.dot(z, n))
        if c > 1.0 - 1e-12:
            return np.eye(3)
        if c < -1.0 + 1e-12:
            return np.diag([1.0, -1.0, -1.0])
        v = np.cross(z, n)
        s = float(np.linalg.norm(v))
        vx = np.array([[0.0, -v[2],  v[1]],
                       [v[2],  0.0, -v[0]],
                       [-v[1], v[0],  0.0]])
        return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    def _plate_center_offset(self, R) -> Tuple[float, float]:
        """
        For yaw-free R, the plate center horizontal shift (cx, cy) that makes
        every P_i lie in its motor's r̂-z plane.
        """
        Rp = self.cfg.Rp
        a = np.zeros(3)
        for i, phi in enumerate(self._phi):
            r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
            t_hat = np.array([-np.sin(phi), np.cos(phi), 0.0])
            a[i] = Rp * float(np.dot(R @ r_hat, t_hat))
        cx =  (2.0 / 3.0) * float(np.sum(np.sin(self._phi) * a))
        cy = -(2.0 / 3.0) * float(np.sum(np.cos(self._phi) * a))
        return cx, cy

    def _inverse_kinematics(
        self,
        normal,
        height: float,
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        normal + height → (thetas_rad, ok, ball_deg).

        thetas_rad  : (3,) motor angles [rad] (NaN for unreachable legs)
        ok          : all legs reachable AND all ball deflections ≤ ball_max_deg
        ball_deg    : (3,) P-side ball deflection [deg] (NaN for unreachable legs)
        """
        c = self.cfg
        Rp = c.Rp
        R = self._rot_from_normal(normal)
        cx, cy = self._plate_center_offset(R)
        center = np.array([cx, cy, height])
        z_hat = np.array([0.0, 0.0, 1.0])
        plate_up = R @ z_hat

        thetas = np.full(3, np.nan)
        ball = np.full(3, np.nan)
        ok = True
        step_rad = c.motor_step_rad

        for i, phi in enumerate(self._phi):
            r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
            B_i = c.Rb * r_hat
            p_body = np.array([Rp * np.cos(phi), Rp * np.sin(phi), 0.0])
            P_i = center + R @ p_body

            d = P_i - B_i
            u = float(np.dot(d, r_hat))
            v = float(d[2])
            k = (float(np.dot(d, d)) + c.La * c.La - c.Lc * c.Lc) / (2.0 * c.La)
            rho = float(np.hypot(u, v))
            if rho < 1e-12 or abs(k) > rho:
                ok = False
                continue

            # '-' branch → θ=0 at the home pose
            th = np.arctan2(v, u) - np.arccos(-k / rho)
            th = (th + np.pi) % (2.0 * np.pi) - np.pi
            if c.quantize:
                th = round(th / step_rad) * step_rad

            A_i = B_i - c.La * (np.cos(th) * r_hat + np.sin(th) * z_hat)
            coupler_dir = (P_i - A_i) / c.Lc
            cos_P = float(np.clip(np.dot(coupler_dir, plate_up), -1.0, 1.0))
            ang_P = float(np.rad2deg(np.arccos(cos_P)))

            thetas[i] = th
            ball[i] = ang_P
            if ang_P > c.ball_max_deg:
                ok = False

        return thetas, ok, ball


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="3-RRS leveling platform inverse kinematics")
    ap.add_argument("--target", nargs=3, type=float, required=True,
                    metavar=("X", "Y", "Z"),
                    help="target 3D point in plate-base frame [m]")
    ap.add_argument("--height", type=float, default=None,
                    help="plate center height [m] (default cfg.H0=Lc)")
    ap.add_argument("--ball_max", type=float, default=None,
                    help="P-side ball joint limit [deg] (default cfg.ball_max_deg)")
    ap.add_argument("--no-quantize", action="store_true",
                    help="skip encoder step quantization")
    args = ap.parse_args()

    cfg = LevelingConfig()
    if args.ball_max is not None:
        cfg.ball_max_deg = args.ball_max
    if args.no_quantize:
        cfg.quantize = False

    ik = LevelingIK(cfg)
    r = ik.aim_at(tuple(args.target), height=args.height)

    if r["angles_deg"] is None:
        print("UNREACHABLE (length constraint violated on at least one leg)")
        sys.exit(2)

    a = r["angles_deg"]
    s = r["angles_steps"]
    b = r["ball_deg"]
    cx, cy = r["c_shift_m"]
    h = cfg.H0 if args.height is None else args.height
    print(f"target        : ({args.target[0]:+.4f}, {args.target[1]:+.4f}, "
          f"{args.target[2]:+.4f}) m")
    print(f"plate height  : {h:.4f} m")
    print(f"normal        : ({r['normal'][0]:+.5f}, {r['normal'][1]:+.5f}, "
          f"{r['normal'][2]:+.5f})")
    print(f"motor angles  : {a[0]:+8.4f}   {a[1]:+8.4f}   {a[2]:+8.4f}   [deg]")
    print(f"encoder steps : {s[0]:+8d}   {s[1]:+8d}   {s[2]:+8d}   "
          f"(0..{cfg.motor_steps-1})")
    print(f"ball P defl.  : {b[0]:8.4f}   {b[1]:8.4f}   {b[2]:8.4f}   "
          f"[deg] (lim={cfg.ball_max_deg})")
    print(f"center shift  : ({cx*1000:+.3f}, {cy*1000:+.3f}) mm")
    print(f"feasible      : {r['ok']}")
    sys.exit(0 if r["ok"] else 1)
