"""
3-RRS Leveling Platform — inverse kinematics module.

목표 3D 점 (x, y, z) → 플레이트 normal 을 그 방향으로 정렬하기 위한
3개의 모터 각도를 계산.

차량 / 하드웨어 모델
-----------------
- 베이스 반경 Rb 위에 120° 간격으로 3개의 모터.
- 각 모터에 길이 La 의 크랭크 (B 점에서 회전, 회전축은 접선 방향).
- 길이 Lc 의 커플러가 크랭크 끝 A 와 플레이트 조인트 P 를 연결.
  * A: 회전 조인트 (접선축) → 커플러는 모터의 r̂-z 수직 평면에 머무름.
  * P: 구면 (RC 볼) 조인트, 플레이트 +z 축 기준 BALL_MAX_DEG 까지 굴곡 허용.
- 상부 플레이트는 반경 Rp = Rb - La 의 원 위에 부착.
- 홈 자세 (모든 모터 θ=0): 크랭크 수평 (안쪽), 커플러 수직.
  플레이트 중심 = (0, 0, H0=Lc).

3-RRS 중심 오프셋
----------------
플레이트가 기울어지면 모든 P_i 가 자기 모터의 r̂-z 평면 안에 머물도록
중심이 수평으로 약간 미끄러진다. _plate_center_offset() 가 이를 닫힌형
으로 산출.

사용 예
------
    from leveling_ik import LevelingIK, LevelingConfig

    cfg = LevelingConfig()                  # 기본값이 빌드 사양과 일치
    ik  = LevelingIK(cfg)
    out = ik.aim_at((0.10, 0.00, 3.0))      # plate-base frame 의 3D 점
    if out['ok']:
        send_to_motors(out['angles_steps'])  # 인코더 step (0..motor_steps-1)
    else:
        # 길이 불가 또는 볼 조인트 한계 초과 → 베이스 재정렬 후 재시도
        ...

CLI
---
    python3 leveling_ik.py --target 0.10 0.00 3.0

의존성
-----
numpy 만 사용. 헤드리스 (Pi5 OK).
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
    # ── 기구 파라미터 ──
    Rb: float = 0.108                 # base pivot radius [m]
    La: float = 0.035                 # crank length [m]
    Lc: float = 0.111                 # coupler length [m]

    # ── 모터 파라미터 ──
    motor_phis_deg: Tuple[float, float, float] = (0.0, 120.0, 240.0)  # azimuths
    motor_steps: int = 4096          # encoder counts / revolution

    # ── 한계 ──
    ball_max_deg: float = 30.0       # P-side ball joint deflection limit

    # ── 출력 ──
    quantize: bool = True            # 인코더 step 으로 round 할지

    # ── 파생값 ──
    @property
    def Rp(self) -> float:
        """플레이트 조인트 반경 (홈 자세에 의해 강제: crank 수평 + 커플러 수직)."""
        return self.Rb - self.La

    @property
    def H0(self) -> float:
        """홈 자세에서 플레이트 중심 높이."""
        return self.Lc

    @property
    def motor_step_rad(self) -> float:
        return 2.0 * np.pi / self.motor_steps


# ──────────────────────────────────────────────
# IK module
# ──────────────────────────────────────────────
class LevelingIK:
    """3-RRS 평탄화 플랫폼 역기구학."""

    def __init__(self, cfg: Optional[LevelingConfig] = None):
        self.cfg = cfg if cfg is not None else LevelingConfig()
        self._phi = np.deg2rad(
            np.asarray(self.cfg.motor_phis_deg, dtype=float))

    # ── 메인 API ──
    def aim_at(
        self,
        target,
        height: Optional[float] = None,
    ) -> Dict:
        """
        플레이트 중심 (0, 0, height) 에서 target 3D 점을 향하도록 모터 각도 산출.

        Parameters
        ----------
        target  플레이트-베이스 프레임의 (x, y, z) [m]
                ※ 카메라 좌표계 결과는 호출 측에서 미리 변환해 둘 것.
        height  플레이트 중심 z [m] (None → cfg.H0)

        Returns
        -------
        dict — aim_normal() 과 동일.
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
        플레이트 normal (unit vector) 을 직접 입력해 모터 각도 산출.

        Returns
        -------
        dict
            angles_deg     list[float]|None   모터 각도 [deg] (도달 불가면 None)
            angles_rad     list[float]|None   모터 각도 [rad]
            angles_steps   list[int]|None     인코더 step (round)
            ok             bool               길이 OK AND 볼 한계 OK
            ball_deg       list[float|None]   각 leg 의 볼 굴곡 [deg]
            c_shift_m      tuple[float,float] 플레이트 중심 수평 시프트 [m]
            normal         list[float]        명령된 plate normal (unit)
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

    # ── 내부: 회전 / 중심 시프트 / IK ──
    @staticmethod
    def _rot_from_normal(n) -> np.ndarray:
        """+z → unit vector n 으로의 최단호 회전 (yaw-free)."""
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
        Yaw-free R 에 대해, 모든 P_i 가 모터의 r̂-z 평면에 들도록 만드는
        플레이트 중심 수평 시프트 (cx, cy).
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

        thetas_rad  : (3,) 모터 각도 [rad] (도달 불가 leg 는 NaN)
        ok          : 모든 leg 도달 AND 모든 볼 굴곡 ≤ ball_max_deg
        ball_deg    : (3,) P-side 볼 굴곡 [deg] (도달 불가 leg 는 NaN)
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

            # '-' branch → 홈 자세에서 θ=0
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
