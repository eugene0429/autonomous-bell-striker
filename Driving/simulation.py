"""
Tank-style Vehicle 2D Navigation Simulation
- 잔디 환경에서 목표 지점(x, y)으로 주행하는 탱크 방식 로버 시뮬레이션
- SLAM 위치 추정 오차 모델링 + 외란(disturbance) 모델링 포함
- 목표: 특정 오차 범위 내 도달
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import Tuple, List
import time

# macOS 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class SimConfig:
    # 시뮬레이션
    dt: float = 0.067           # 시간 스텝 (s) - 15Hz (Pi5 + D435i SLAM 기준)
    max_time: float = 60.0      # 최대 시뮬레이션 시간 (s)

    # 로버 물리 파라미터
    wheel_base: float = 0.3     # 좌우 바퀴 간격 (m)
    max_speed: float = 0.3      # 최대 선속도 (m/s)
    max_omega: float = 1.0      # 최대 각속도 (rad/s)

    # 목표 지점
    target_x: float = 5.0
    target_y: float = 4.0
    goal_tolerance: float = 0.3  # 목표 도달 판정 반경 (m)

    # 시작 위치
    start_x: float = 0.0
    start_y: float = 0.0
    start_theta: float = 0.0    # rad

    # 외란 (잔디 환경)
    disturbance_v_std: float = 0.08      # 선속도 외란 표준편차 (m/s)
    disturbance_omega_std: float = 0.15  # 각속도 외란 표준편차 (rad/s)
    slip_factor_mean: float = 0.90       # 잔디 슬립 (평균 90% 전달)
    slip_factor_std: float = 0.05        # 슬립 변동

    # SLAM 오차 모델
    slam_noise_xy_std: float = 0.03      # 위치 측정 가우시안 노이즈 (m) - SLAM은 맵 최적화로 VIO보다 낮음
    slam_noise_theta_std: float = 0.015  # 헤딩 측정 노이즈 (rad)
    slam_drift_rate: float = 0.003       # 드리프트 누적 속도 (m/s) - 맵 기반 보정으로 VIO보다 느림
    slam_drift_theta_rate: float = 0.001 # 헤딩 드리프트 속도 (rad/s)

    # SLAM relocalization 실패
    slam_reloc_failure_prob: float = 0.01      # relocalization 실패 확률 (특징점 부족 등)
    slam_reloc_failure_noise: float = 0.3      # 실패 시 위치 오차 크기 (m)

    # SLAM 신뢰도 필터
    slam_jump_threshold: float = 0.5   # 한 스텝에 이 이상 점프하면 outlier (m)
    slam_jump_theta_threshold: float = 0.3  # 헤딩 점프 threshold (rad, ~17°)
    slam_lowconf_speed_scale: float = 0.3   # 신뢰도 낮을 때 속도 배율
    slam_reject_holdoff: int = 3       # outlier 감지 후 무시할 프레임 수

    # 시리얼 통신 (Pi → OpenRB)
    serial_rate_hz: float = 15.0      # 명령 전송 주기 (Hz)
    cmd_v_deadzone: float = 0.02      # 이 이하 선속도는 0으로 처리 (m/s)
    cmd_omega_deadzone: float = 0.05  # 이 이하 각속도는 0으로 처리 (rad/s)
    cmd_v_resolution: float = 0.01    # 선속도 양자화 단위 (m/s)
    cmd_omega_resolution: float = 0.01  # 각속도 양자화 단위 (rad/s)

    # 제어기 게인
    kp_linear: float = 0.8       # 거리 비례 게인
    kp_angular: float = 2.5      # 각도 비례 게인
    ki_angular: float = 0.1      # 각도 적분 게인
    kd_angular: float = 0.3      # 각도 미분 게인
    slowdown_radius: float = 1.0 # 감속 시작 반경 (m)


# ──────────────────────────────────────────────
# Vehicle Model (Tank / Differential Drive)
# ──────────────────────────────────────────────
class TankVehicle:
    """탱크 방식(차동 구동) 로버 모델"""

    def __init__(self, x: float, y: float, theta: float, wheel_base: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.L = wheel_base

    def update(self, v_cmd: float, omega_cmd: float, dt: float,
               disturbance: 'DisturbanceModel') -> Tuple[float, float]:
        """
        제어 명령(v, omega)을 받아 실제 위치를 업데이트.
        외란이 적용된 실제 동작을 반영.
        """
        v_actual, omega_actual = disturbance.apply(v_cmd, omega_cmd)

        self.x += v_actual * np.cos(self.theta) * dt
        self.y += v_actual * np.sin(self.theta) * dt
        self.theta += omega_actual * dt
        self.theta = self._wrap_angle(self.theta)

        return v_actual, omega_actual

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @property
    def state(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta


# ──────────────────────────────────────────────
# Disturbance Model (잔디 환경 외란)
# ──────────────────────────────────────────────
class DisturbanceModel:
    """
    잔디 환경에서의 외란 모델링:
    - 바퀴 슬립 (잔디 위에서 구동력 손실)
    - 가우시안 노이즈 (불규칙 지형)
    - 방향 교란 (풀, 돌멩이 등)
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def apply(self, v_cmd: float, omega_cmd: float) -> Tuple[float, float]:
        c = self.cfg

        # 슬립: 잔디에서 바퀴가 미끄러져 명령 대비 실제 속도 감소
        slip = np.random.normal(c.slip_factor_mean, c.slip_factor_std)
        slip = np.clip(slip, 0.7, 1.0)

        # 선속도 외란
        v_noise = np.random.normal(0, c.disturbance_v_std)
        v_actual = v_cmd * slip + v_noise

        # 각속도 외란 (좌우 바퀴 슬립 차이로 인한 방향 교란)
        omega_noise = np.random.normal(0, c.disturbance_omega_std)
        omega_actual = omega_cmd * slip + omega_noise

        return v_actual, omega_actual


# ──────────────────────────────────────────────
# SLAM Error Model (SLAM 측위 오차)
# ──────────────────────────────────────────────
class SLAMModel:
    """
    SLAM 위치 추정 오차 모델:
    - 가우시안 측정 노이즈 (맵 최적화로 VIO 대비 낮음)
    - 시간에 따라 누적되는 드리프트 (맵 기반 보정으로 느리게 누적)
    - relocalization 실패 시 큰 위치 오차 발생
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.drift_x = 0.0
        self.drift_y = 0.0
        self.drift_theta = 0.0
        self.time_elapsed = 0.0

    def estimate(self, true_x: float, true_y: float, true_theta: float,
                 dt: float) -> Tuple[float, float, float]:
        c = self.cfg
        self.time_elapsed += dt
        # 드리프트 누적 (랜덤 워크)
        self.drift_x += np.random.normal(0, c.slam_drift_rate * dt)
        self.drift_y += np.random.normal(0, c.slam_drift_rate * dt)
        self.drift_theta += np.random.normal(0, c.slam_drift_theta_rate * dt)

        # 측정 노이즈
        noise_x = np.random.normal(0, c.slam_noise_xy_std)
        noise_y = np.random.normal(0, c.slam_noise_xy_std)
        noise_theta = np.random.normal(0, c.slam_noise_theta_std)

        # Relocalization 실패: 특징점 부족 등으로 큰 위치 오차 발생
        if np.random.random() < c.slam_reloc_failure_prob:
            noise_x += np.random.normal(0, c.slam_reloc_failure_noise)
            noise_y += np.random.normal(0, c.slam_reloc_failure_noise)

        est_x = true_x + self.drift_x + noise_x
        est_y = true_y + self.drift_y + noise_y
        est_theta = true_theta + self.drift_theta + noise_theta

        return est_x, est_y, est_theta

    @property
    def current_drift(self) -> Tuple[float, float, float]:
        return self.drift_x, self.drift_y, self.drift_theta


# ──────────────────────────────────────────────
# SLAM Confidence Filter (이상치 제거 + 신뢰도 판정)
# ──────────────────────────────────────────────
class SLAMFilter:
    """
    SLAM 추정값의 이상치를 걸러내고 신뢰도를 판정.
    - 한 스텝에 비현실적으로 큰 위치 점프 → reject (이전 값 유지)
    - reject 연속 발생 시 신뢰도 저하 → 제어기에 감속 신호
    - 실제 로버에서는 이 필터가 SLAM raw 출력과 제어기 사이에 위치
    """

    MAX_CONSECUTIVE_REJECTS = 10  # 이 이상 연속 reject 시 강제 수용 (reset)

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.prev_est = None          # 이전 필터 출력 (x, y, theta)
        self.reject_count = 0         # 연속 reject 횟수
        self.holdoff_remaining = 0    # reject 후 남은 holdoff 프레임
        self.confidence = 1.0         # 0.0 ~ 1.0 신뢰도
        self.total_rejects = 0        # 누적 reject 횟수 (로그용)

    def update(self, raw_x: float, raw_y: float, raw_theta: float,
               dt: float) -> Tuple[float, float, float, float]:
        """
        SLAM raw 추정값을 ���터링.
        Returns: (filtered_x, filtered_y, filtered_theta, confidence)
        """
        c = self.cfg

        if self.prev_est is None:
            self.prev_est = (raw_x, raw_y, raw_theta)
            self.confidence = 1.0
            return raw_x, raw_y, raw_theta, self.confidence

        px, py, pt = self.prev_est

        # 연속 reject가 너무 많으면 강제 수용 (로버가 이동해서 frozen 위치와 괴리)
        if self.reject_count >= self.MAX_CONSECUTIVE_REJECTS:
            self.prev_est = (raw_x, raw_y, raw_theta)
            self.reject_count = 0
            self.holdoff_remaining = 0
            self.confidence = 0.3  # 낮은 신뢰도로 재시작
            return raw_x, raw_y, raw_theta, self.confidence

        # 위치 점프 크기 계산
        jump_xy = np.sqrt((raw_x - px)**2 + (raw_y - py)**2)
        jump_theta = abs(((raw_theta - pt) + np.pi) % (2 * np.pi) - np.pi)

        # 물리적으로 가능한 최대 이동: max_speed * dt * 안전 마진
        max_possible_jump = c.max_speed * dt * 3.0

        is_outlier = (jump_xy > max(c.slam_jump_threshold, max_possible_jump) or
                      jump_theta > c.slam_jump_theta_threshold)

        if is_outlier or self.holdoff_remaining > 0:
            # Outlier → 이전 값 유지
            if is_outlier:
                self.reject_count += 1
                self.total_rejects += 1
                self.holdoff_remaining = c.slam_reject_holdoff
            self.holdoff_remaining = max(0, self.holdoff_remaining - 1)

            # 신뢰도 감소
            self.confidence = max(0.1, self.confidence - 0.2)

            # 이전 값 유지 (dead reckoning 느낌)
            return px, py, pt, self.confidence
        else:
            # 정상 → 값 수용, 신뢰도 회복
            self.reject_count = 0
            self.confidence = min(1.0, self.confidence + 0.05)
            self.prev_est = (raw_x, raw_y, raw_theta)
            return raw_x, raw_y, raw_theta, self.confidence


# ──────────────────────────────────────────────
# Serial Command Protocol (Pi → OpenRB 시뮬레이션)
# ──────────────────────────────────────────────
class SerialCommandSim:
    """
    Pi5 → OpenRB 시리얼 통신 시뮬레이션.
    실제 구현 시 참고할 프로토콜:
      패킷: [0xFF][0xFE][v_high][v_low][omega_high][omega_low][checksum]
      - v, omega: signed int16, 단위 mm/s, mrad/s
      - checksum: XOR of payload bytes

    시뮬레이션에서는:
    - 전송 주기 제한 (serial_rate_hz)
    - 데드존 처리 (모터 떨림 방지)
    - 양자화 (실제 시리얼에서의 정수 변환 반영)
    - rate limiting (급격한 명령 변화 완화)
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.send_interval = 1.0 / cfg.serial_rate_hz
        self.time_since_send = 0.0
        self.last_sent_v = 0.0
        self.last_sent_omega = 0.0
        self.packet_count = 0

    def process(self, v_cmd: float, omega_cmd: float,
                dt: float) -> Tuple[float, float, bool]:
        """
        제어기 출력을 시리얼 전송용으로 변환.
        Returns: (processed_v, processed_omega, was_sent_this_step)
        """
        c = self.cfg
        self.time_since_send += dt

        if self.time_since_send < self.send_interval:
            # 아직 전송 타이밍 아님 → 이전 명령 유지
            return self.last_sent_v, self.last_sent_omega, False

        # 전송 타이밍 도달
        self.time_since_send = 0.0

        # 데드존 처리
        if abs(v_cmd) < c.cmd_v_deadzone:
            v_cmd = 0.0
        if abs(omega_cmd) < c.cmd_omega_deadzone:
            omega_cmd = 0.0

        # 양자화 (시리얼 int16 변환 시뮬레이션)
        v_cmd = round(v_cmd / c.cmd_v_resolution) * c.cmd_v_resolution
        omega_cmd = round(omega_cmd / c.cmd_omega_resolution) * c.cmd_omega_resolution

        self.last_sent_v = v_cmd
        self.last_sent_omega = omega_cmd
        self.packet_count += 1

        return v_cmd, omega_cmd, True

    def build_packet(self, v: float, omega: float) -> bytes:
        """
        실제 OpenRB로 보낼 바이트 패킷 생성 (참고용).
        실제 로버 코드에서 이 형식으로 serial.write() 호출.
        """
        v_int = int(np.clip(v * 1000, -32768, 32767))     # mm/s
        omega_int = int(np.clip(omega * 1000, -32768, 32767))  # mrad/s

        v_high = (v_int >> 8) & 0xFF
        v_low = v_int & 0xFF
        omega_high = (omega_int >> 8) & 0xFF
        omega_low = omega_int & 0xFF
        checksum = v_high ^ v_low ^ omega_high ^ omega_low

        return bytes([0xFF, 0xFE, v_high, v_low, omega_high, omega_low, checksum])


# ──────────────────────────────────────────────
# Navigation Controller (PID 기반)
# ──────────────────────────────────────────────
class NavigationController:
    """
    SLAM 추정 위치 기반으로 목표 지점까지 주행하는 제어기.
    - 거리 비례 선속도 제어
    - PID 각속도 제어
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.prev_angle_error = 0.0
        self.integral_angle_error = 0.0

    def compute(self, est_x: float, est_y: float, est_theta: float,
                target_x: float, target_y: float,
                slam_confidence: float = 1.0) -> Tuple[float, float]:
        c = self.cfg

        dx = target_x - est_x
        dy = target_y - est_y
        distance = np.sqrt(dx**2 + dy**2)
        desired_heading = np.arctan2(dy, dx)

        # 헤딩 오차
        angle_error = desired_heading - est_theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # PID 각속도
        self.integral_angle_error += angle_error * c.dt
        self.integral_angle_error = np.clip(self.integral_angle_error, -2.0, 2.0)
        derivative = (angle_error - self.prev_angle_error) / c.dt
        self.prev_angle_error = angle_error

        omega = (c.kp_angular * angle_error +
                 c.ki_angular * self.integral_angle_error +
                 c.kd_angular * derivative)
        omega = np.clip(omega, -c.max_omega, c.max_omega)

        # 선속도: 거리 비례 + 감속 + 헤딩 오차 클 때 감속
        speed_factor = min(1.0, distance / c.slowdown_radius)
        heading_factor = max(0.2, 1.0 - abs(angle_error) / np.pi)
        v = c.kp_linear * distance * speed_factor * heading_factor
        v = np.clip(v, 0, c.max_speed)

        # SLAM 신뢰도 기반 감속: confidence 낮으면 속도 줄임
        if slam_confidence < 0.8:
            conf_scale = c.slam_lowconf_speed_scale + (1.0 - c.slam_lowconf_speed_scale) * (slam_confidence / 0.8)
            v *= conf_scale
            omega *= conf_scale

        return v, omega


# ──────────────────────────────────────────────
# Simulation Runner
# ──────────────────────────────────────────────
@dataclass
class SimLog:
    time: List[float] = field(default_factory=list)
    true_x: List[float] = field(default_factory=list)
    true_y: List[float] = field(default_factory=list)
    true_theta: List[float] = field(default_factory=list)
    est_x: List[float] = field(default_factory=list)
    est_y: List[float] = field(default_factory=list)
    est_theta: List[float] = field(default_factory=list)
    filtered_x: List[float] = field(default_factory=list)
    filtered_y: List[float] = field(default_factory=list)
    filtered_theta: List[float] = field(default_factory=list)
    slam_confidence: List[float] = field(default_factory=list)
    cmd_v: List[float] = field(default_factory=list)
    cmd_omega: List[float] = field(default_factory=list)
    serial_v: List[float] = field(default_factory=list)
    serial_omega: List[float] = field(default_factory=list)
    actual_v: List[float] = field(default_factory=list)
    actual_omega: List[float] = field(default_factory=list)
    distance_to_goal: List[float] = field(default_factory=list)
    slam_drift_x: List[float] = field(default_factory=list)
    slam_drift_y: List[float] = field(default_factory=list)


def run_simulation(cfg: SimConfig, seed: int = None) -> Tuple[SimLog, bool]:
    if seed is not None:
        np.random.seed(seed)

    vehicle = TankVehicle(cfg.start_x, cfg.start_y, cfg.start_theta, cfg.wheel_base)
    disturbance = DisturbanceModel(cfg)
    slam =SLAMModel(cfg)
    slam_filter = SLAMFilter(cfg)
    serial_cmd = SerialCommandSim(cfg)
    controller = NavigationController(cfg)
    log = SimLog()

    t = 0.0
    reached = False

    while t < cfg.max_time:
        true_x, true_y, true_theta = vehicle.state

        # SLAM 위치 추정 (raw)
        est_x, est_y, est_theta = slam.estimate(true_x, true_y, true_theta, cfg.dt)

        # SLAM 신뢰도 필터 (outlier rejection)
        filt_x, filt_y, filt_theta, confidence = slam_filter.update(
            est_x, est_y, est_theta, cfg.dt)

        # 제어 명령 계산 (필터링된 위치 + 신뢰도 기반 감속)
        v_cmd, omega_cmd = controller.compute(filt_x, filt_y, filt_theta,
                                               cfg.target_x, cfg.target_y,
                                               slam_confidence=confidence)

        # 시리얼 프로토콜 시뮬레이션 (데드존 + 양자화 + rate limiting)
        v_serial, omega_serial, was_sent = serial_cmd.process(
            v_cmd, omega_cmd, cfg.dt)

        # 차량 업데이트 (시리얼로 실제 전송된 명령 기준 + 외란 적용)
        v_actual, omega_actual = vehicle.update(
            v_serial, omega_serial, cfg.dt, disturbance)

        # 실제 목표까지 거리
        dist = np.sqrt((true_x - cfg.target_x)**2 + (true_y - cfg.target_y)**2)

        # 로그 기록
        log.time.append(t)
        log.true_x.append(true_x)
        log.true_y.append(true_y)
        log.true_theta.append(true_theta)
        log.est_x.append(est_x)
        log.est_y.append(est_y)
        log.est_theta.append(est_theta)
        log.filtered_x.append(filt_x)
        log.filtered_y.append(filt_y)
        log.filtered_theta.append(filt_theta)
        log.slam_confidence.append(confidence)
        log.cmd_v.append(v_cmd)
        log.cmd_omega.append(omega_cmd)
        log.serial_v.append(v_serial)
        log.serial_omega.append(omega_serial)
        log.actual_v.append(v_actual)
        log.actual_omega.append(omega_actual)
        log.distance_to_goal.append(dist)
        log.slam_drift_x.append(slam.drift_x)
        log.slam_drift_y.append(slam.drift_y)

        # 목표 도달 판정 (실제 위치 기준)
        if dist < cfg.goal_tolerance:
            reached = True
            break

        t += cfg.dt

    print(f"[SLAM Filter] Total outlier rejects: {slam_filter.total_rejects}")
    print(f"[Serial] Packets sent: {serial_cmd.packet_count}")

    return log, reached


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
def plot_results(log: SimLog, cfg: SimConfig, reached: bool):
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Tank Vehicle Navigation Simulation (Grass Terrain)',
                 fontsize=14, fontweight='bold')

    # ── 1) 2D 경로 ──
    ax1 = fig.add_subplot(3, 3, (1, 4))
    ax1.set_facecolor('#c8e6c9')

    ax1.plot(log.true_x, log.true_y, 'b-', linewidth=1.5, label='True Path', alpha=0.8)
    ax1.plot(log.est_x, log.est_y, 'r--', linewidth=1.0, label='SLAM Raw', alpha=0.4)
    ax1.plot(log.filtered_x, log.filtered_y, 'm-', linewidth=1.0,
             label='SLAM Filtered', alpha=0.7)

    ax1.plot(cfg.start_x, cfg.start_y, 'gs', markersize=12, label='Start', zorder=5)

    goal_circle = plt.Circle((cfg.target_x, cfg.target_y), cfg.goal_tolerance,
                              color='red', fill=False, linewidth=2, linestyle='--')
    ax1.add_patch(goal_circle)
    ax1.plot(cfg.target_x, cfg.target_y, 'r*', markersize=15, label='Target', zorder=5)

    ax1.plot(log.true_x[-1], log.true_y[-1], 'b^', markersize=10,
             label=f'Final (err={log.distance_to_goal[-1]:.3f}m)', zorder=5)

    step = max(1, len(log.time) // 15)
    for i in range(0, len(log.time), step):
        dx = 0.15 * np.cos(log.true_theta[i])
        dy = 0.15 * np.sin(log.true_theta[i])
        ax1.annotate('', xy=(log.true_x[i] + dx, log.true_y[i] + dy),
                     xytext=(log.true_x[i], log.true_y[i]),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('2D Navigation Path')
    ax1.legend(loc='upper left', fontsize=7)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    status = "REACHED" if reached else "NOT REACHED"
    status_color = 'green' if reached else 'red'
    ax1.text(0.98, 0.02, f'Goal: {status}\nTime: {log.time[-1]:.1f}s',
             transform=ax1.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))

    # ── 2) 목표까지 거리 ──
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(log.time, log.distance_to_goal, 'b-', linewidth=1.2)
    ax2.axhline(y=cfg.goal_tolerance, color='r', linestyle='--',
                label=f'Tolerance ({cfg.goal_tolerance}m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Distance to Goal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── 3) 제어 → 시리얼 → 실제 (v) ──
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(log.time, log.cmd_v, 'b-', label='Controller v', alpha=0.5, linewidth=0.8)
    ax3.plot(log.time, log.serial_v, 'g-', label='Serial v', alpha=0.8, linewidth=1.2)
    ax3.plot(log.time, log.actual_v, 'b--', label='Actual v', alpha=0.5, linewidth=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Linear Velocity (m/s)')
    ax3.set_title('Command Pipeline: v')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ── 4) SLAM 드리프트 ──
    ax4 = fig.add_subplot(3, 3, 5)
    ax4.plot(log.time, log.slam_drift_x, 'r-', label='Drift X')
    ax4.plot(log.time, log.slam_drift_y, 'g-', label='Drift Y')
    drift_mag = [np.sqrt(dx**2 + dy**2) for dx, dy in
                 zip(log.slam_drift_x, log.slam_drift_y)]
    ax4.plot(log.time, drift_mag, 'k--', label='|Drift|', alpha=0.6)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Drift (m)')
    ax4.set_title('SLAM Drift Accumulation')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 5) 위치 추정 오차 (raw vs filtered) ──
    ax5 = fig.add_subplot(3, 3, 6)
    raw_pos_error = [np.sqrt((tx - ex)**2 + (ty - ey)**2)
                     for tx, ex, ty, ey in
                     zip(log.true_x, log.est_x, log.true_y, log.est_y)]
    filt_pos_error = [np.sqrt((tx - fx)**2 + (ty - fy)**2)
                      for tx, fx, ty, fy in
                      zip(log.true_x, log.filtered_x, log.true_y, log.filtered_y)]

    ax5.plot(log.time, raw_pos_error, 'r-', label='Raw SLAM Error', alpha=0.5, linewidth=0.8)
    ax5.plot(log.time, filt_pos_error, 'b-', label='Filtered Error', alpha=0.8, linewidth=1.2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position Error (m)')
    ax5.set_title('SLAM Error: Raw vs Filtered')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── 6) SLAM Confidence ──
    ax6 = fig.add_subplot(3, 3, 7)
    ax6.plot(log.time, log.slam_confidence, 'purple', linewidth=1.2)
    ax6.axhline(y=0.8, color='orange', linestyle='--', label='Slowdown threshold', alpha=0.7)
    ax6.fill_between(log.time, 0, log.slam_confidence, alpha=0.15, color='purple')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Confidence')
    ax6.set_title('SLAM Confidence (Filter)')
    ax6.set_ylim(-0.05, 1.1)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ── 7) 제어 → 시리얼 → 실제 (omega) ──
    ax7 = fig.add_subplot(3, 3, 8)
    ax7.plot(log.time, log.cmd_omega, 'r-', label='Controller ω', alpha=0.5, linewidth=0.8)
    ax7.plot(log.time, log.serial_omega, 'g-', label='Serial ω', alpha=0.8, linewidth=1.2)
    ax7.plot(log.time, log.actual_omega, 'r--', label='Actual ω', alpha=0.5, linewidth=0.8)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Angular Velocity (rad/s)')
    ax7.set_title('Command Pipeline: ω')
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)

    # ── 8) 헤딩 오차 ──
    ax8 = fig.add_subplot(3, 3, 9)
    heading_error = [abs(((tt - et + np.pi) % (2 * np.pi)) - np.pi)
                     for tt, et in zip(log.true_theta, log.est_theta)]
    filt_heading_error = [abs(((tt - ft + np.pi) % (2 * np.pi)) - np.pi)
                          for tt, ft in zip(log.true_theta, log.filtered_theta)]
    ax8.plot(log.time, np.degrees(heading_error), 'r-',
             label='Raw Heading Error', alpha=0.5, linewidth=0.8)
    ax8.plot(log.time, np.degrees(filt_heading_error), 'b-',
             label='Filtered Heading Error', alpha=0.8, linewidth=1.2)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Heading Error (deg)')
    ax8.set_title('Heading Error: Raw vs Filtered')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/baeg-yujin/Desktop/project/Capstone Design/Driving/simulation_result.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
# Monte Carlo Analysis (다회 시뮬레이션)
# ──────────────────────────────────────────────
def monte_carlo(cfg: SimConfig, n_runs: int = 50):
    """여러 번 시뮬레이션을 돌려 성공률과 최종 오차 분포 분석"""
    final_errors = []
    success_count = 0
    arrival_times = []

    print(f"\n{'='*50}")
    print(f"Monte Carlo Simulation ({n_runs} runs)")
    print(f"{'='*50}")

    for i in range(n_runs):
        log, reached = run_simulation(cfg, seed=i * 42 + 7)
        final_err = log.distance_to_goal[-1]
        final_errors.append(final_err)
        if reached:
            success_count += 1
            arrival_times.append(log.time[-1])

    final_errors = np.array(final_errors)
    success_rate = success_count / n_runs * 100

    print(f"Success Rate:      {success_rate:.1f}% ({success_count}/{n_runs})")
    print(f"Final Error Mean:  {final_errors.mean():.4f} m")
    print(f"Final Error Std:   {final_errors.std():.4f} m")
    print(f"Final Error Max:   {final_errors.max():.4f} m")
    if arrival_times:
        print(f"Avg Arrival Time:  {np.mean(arrival_times):.2f} s")
    print(f"{'='*50}\n")

    # 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Monte Carlo Analysis ({n_runs} runs)', fontweight='bold')

    axes[0].hist(final_errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=cfg.goal_tolerance, color='r', linestyle='--',
                    label=f'Tolerance ({cfg.goal_tolerance}m)')
    axes[0].set_xlabel('Final Distance to Goal (m)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Final Error Distribution (Success: {success_rate:.0f}%)')
    axes[0].legend()

    if arrival_times:
        axes[1].hist(arrival_times, bins=20, color='forestgreen',
                     edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Arrival Time (s)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Arrival Time Distribution')
    else:
        axes[1].text(0.5, 0.5, 'No successful runs', ha='center', va='center',
                     fontsize=14, transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig('/Users/baeg-yujin/Desktop/project/Capstone Design/Driving/monte_carlo_result.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
# Real-time Animation
# ──────────────────────────────────────────────
def run_animation(cfg: SimConfig, seed: int = 42):
    """
    인터랙티브 애니메이션 시뮬레이션.
    1) 설정 화면: 좌클릭으로 출발 위치, 우클릭으로 목표 위치 지정
    2) ▶ Play 버튼을 누르면 시뮬레이션 시작
    """
    from matplotlib.widgets import Button

    # ── 1단계: 인터랙티브 설정 화면 ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)

    state = {
        'start': None,
        'target': None,
        'running': False,
        'start_marker': None,
        'target_marker': None,
        'target_circle': None,
    }

    def draw_setup():
        ax.clear()
        ax.set_facecolor('#c8e6c9')
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)

        title_parts = ['[Setup] Left-click: Start | Right-click: Target']
        if state['start'] is not None:
            sx, sy = state['start']
            ax.plot(sx, sy, 'gs', markersize=14, zorder=5, label=f'Start ({sx:.1f}, {sy:.1f})')
            title_parts.append(f'Start=({sx:.1f}, {sy:.1f})')
        if state['target'] is not None:
            tx, ty = state['target']
            ax.plot(tx, ty, 'r*', markersize=18, zorder=5, label=f'Target ({tx:.1f}, {ty:.1f})')
            goal_circle = plt.Circle((tx, ty), cfg.goal_tolerance,
                                      color='red', fill=False, linewidth=2, linestyle='--')
            ax.add_patch(goal_circle)
            title_parts.append(f'Target=({tx:.1f}, {ty:.1f})')

        ax.set_title(' | '.join(title_parts))
        if state['start'] is not None or state['target'] is not None:
            ax.legend(loc='upper left', fontsize=9)
        fig.canvas.draw_idle()

    def on_click(event):
        if state['running']:
            return
        if event.inaxes != ax:
            return

        if event.button == 1:  # Left-click: Start
            state['start'] = (event.xdata, event.ydata)
        elif event.button == 3:  # Right-click: Target
            state['target'] = (event.xdata, event.ydata)

        draw_setup()

    def on_play(event):
        if state['start'] is None or state['target'] is None:
            ax.set_title('[Error] Please set both Start and Target positions!',
                         color='red', fontweight='bold')
            fig.canvas.draw_idle()
            return
        state['running'] = True

    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    ax_btn = plt.axes([0.4, 0.03, 0.2, 0.06])
    btn_play = Button(ax_btn, '▶ Play', color='#4CAF50', hovercolor='#66BB6A')
    btn_play.label.set_fontsize(14)
    btn_play.label.set_fontweight('bold')
    btn_play.on_clicked(on_play)

    draw_setup()

    # Play 버튼이 눌릴 때까지 대기
    while not state['running']:
        plt.pause(0.1)
        if not plt.fignum_exists(fig.number):
            return  # 창이 닫히면 종료

    fig.canvas.mpl_disconnect(cid)

    # ── 2단계: 시뮬레이션 실행 ──
    sx, sy = state['start']
    tx, ty = state['target']
    cfg.start_x, cfg.start_y = sx, sy
    cfg.target_x, cfg.target_y = tx, ty

    np.random.seed(seed)
    vehicle = TankVehicle(cfg.start_x, cfg.start_y, cfg.start_theta, cfg.wheel_base)
    disturbance = DisturbanceModel(cfg)
    slam =SLAMModel(cfg)
    slam_filter = SLAMFilter(cfg)
    serial_cmd = SerialCommandSim(cfg)
    controller = NavigationController(cfg)

    # Play 버튼 숨기기
    ax_btn.set_visible(False)

    true_path_x, true_path_y = [], []
    est_path_x, est_path_y = [], []

    t = 0.0
    while t < cfg.max_time:
        if not plt.fignum_exists(fig.number):
            return  # 창이 닫히면 종료

        true_x, true_y, true_theta = vehicle.state
        est_x, est_y, est_theta = slam.estimate(true_x, true_y, true_theta, cfg.dt)
        filt_x, filt_y, filt_theta, confidence = slam_filter.update(
            est_x, est_y, est_theta, cfg.dt)
        v_cmd, omega_cmd = controller.compute(filt_x, filt_y, filt_theta,
                                               cfg.target_x, cfg.target_y,
                                               slam_confidence=confidence)
        v_serial, omega_serial, _ = serial_cmd.process(v_cmd, omega_cmd, cfg.dt)
        vehicle.update(v_serial, omega_serial, cfg.dt, disturbance)

        true_path_x.append(true_x)
        true_path_y.append(true_y)
        est_path_x.append(est_x)
        est_path_y.append(est_y)

        dist = np.sqrt((true_x - cfg.target_x)**2 + (true_y - cfg.target_y)**2)

        # 매 5프레임마다 화면 갱신
        if int(t / cfg.dt) % 5 == 0:
            ax.clear()
            ax.set_facecolor('#c8e6c9')

            ax.plot(true_path_x, true_path_y, 'b-', linewidth=1.5, label='True')
            ax.plot(est_path_x, est_path_y, 'r--', linewidth=1.0, label='SLAM Est.', alpha=0.6)

            # 로버 표시
            rover_size = 0.15
            corners = np.array([[-rover_size, -rover_size/2],
                                [rover_size, -rover_size/2],
                                [rover_size, rover_size/2],
                                [-rover_size, rover_size/2]])
            R = np.array([[np.cos(true_theta), -np.sin(true_theta)],
                          [np.sin(true_theta), np.cos(true_theta)]])
            rotated = corners @ R.T + np.array([true_x, true_y])
            rover_patch = plt.Polygon(rotated, closed=True, facecolor='navy',
                                       edgecolor='black', alpha=0.8, zorder=10)
            ax.add_patch(rover_patch)

            # 방향 화살표
            arrow_len = 0.25
            ax.annotate('', xy=(true_x + arrow_len * np.cos(true_theta),
                                true_y + arrow_len * np.sin(true_theta)),
                        xytext=(true_x, true_y),
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
                        zorder=11)

            # 목표
            goal_circle = plt.Circle((cfg.target_x, cfg.target_y), cfg.goal_tolerance,
                                      color='red', fill=False, linewidth=2, linestyle='--')
            ax.add_patch(goal_circle)
            ax.plot(cfg.target_x, cfg.target_y, 'r*', markersize=15, zorder=5)
            ax.plot(cfg.start_x, cfg.start_y, 'gs', markersize=10, zorder=5)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f't={t:.1f}s | dist={dist:.3f}m | v={v_serial:.2f} | ω={omega_serial:.2f} | conf={confidence:.2f}')
            ax.set_aspect('equal')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            margin = 1.0
            all_x = true_path_x + [cfg.target_x, cfg.start_x]
            all_y = true_path_y + [cfg.target_y, cfg.start_y]
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

        if dist < cfg.goal_tolerance:
            print(f"Goal reached at t={t:.2f}s, error={dist:.4f}m")
            break

        t += cfg.dt

    plt.show()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tank Vehicle Navigation Simulator')
    parser.add_argument('--mode', choices=['single', 'animate', 'monte_carlo'],
                        default='single', help='Simulation mode')
    parser.add_argument('--target_x', type=float, default=5.0)
    parser.add_argument('--target_y', type=float, default=4.0)
    parser.add_argument('--tolerance', type=float, default=0.3)
    parser.add_argument('--max_time', type=float, default=60.0)
    parser.add_argument('--runs', type=int, default=50, help='Monte Carlo runs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--slip', type=float, default=0.90,
                        help='Mean slip factor (0-1, lower = more slip)')
    parser.add_argument('--slam_noise', type=float, default=0.03,
                        help='SLAM position noise std (m)')
    parser.add_argument('--slam_drift', type=float, default=0.003,
                        help='SLAM drift rate (m/s)')
    args = parser.parse_args()

    cfg = SimConfig(
        target_x=args.target_x,
        target_y=args.target_y,
        goal_tolerance=args.tolerance,
        max_time=args.max_time,
        slip_factor_mean=args.slip,
        slam_noise_xy_std=args.slam_noise,
        slam_drift_rate=args.slam_drift,
    )

    print(f"Target: ({cfg.target_x}, {cfg.target_y})")
    print(f"Tolerance: {cfg.goal_tolerance}m")
    print(f"Slip: {cfg.slip_factor_mean:.0%} | SLAM noise: {cfg.slam_noise_xy_std}m "
          f"| SLAM drift: {cfg.slam_drift_rate}m/s")

    if args.mode == 'single':
        log, reached = run_simulation(cfg, seed=args.seed)
        print(f"\nResult: {'REACHED' if reached else 'NOT REACHED'}")
        print(f"Final error: {log.distance_to_goal[-1]:.4f}m")
        print(f"Time: {log.time[-1]:.2f}s")
        plot_results(log, cfg, reached)

    elif args.mode == 'animate':
        run_animation(cfg, seed=args.seed)

    elif args.mode == 'monte_carlo':
        monte_carlo(cfg, n_runs=args.runs)
