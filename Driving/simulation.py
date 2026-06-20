"""
Tank-style Vehicle 2D Navigation Simulation
- Tank-style rover simulation that drives to a target point (x, y) on grass terrain
- Includes SLAM position estimation error modeling + disturbance modeling
- Goal: reach the target within a specified error tolerance
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import Tuple, List
import time

# macOS Korean font configuration
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class SimConfig:
    # Simulation
    dt: float = 0.067           # Time step (s) - 15Hz (based on Pi5 + D435i SLAM)
    max_time: float = 60.0      # Maximum simulation time (s)

    # Rover physical parameters
    wheel_base: float = 0.3     # Distance between left/right wheels (m)
    max_speed: float = 0.3      # Maximum linear velocity (m/s)
    max_omega: float = 1.0      # Maximum angular velocity (rad/s)

    # Target point
    target_x: float = 5.0
    target_y: float = 4.0
    goal_tolerance: float = 0.3  # Goal-reached decision radius (m)

    # Start position
    start_x: float = 0.0
    start_y: float = 0.0
    start_theta: float = 0.0    # rad

    # Disturbance (grass terrain)
    disturbance_v_std: float = 0.08      # Linear-velocity disturbance standard deviation (m/s)
    disturbance_omega_std: float = 0.15  # Angular-velocity disturbance standard deviation (rad/s)
    slip_factor_mean: float = 0.90       # Grass slip (mean 90% transmission)
    slip_factor_std: float = 0.05        # Slip variation

    # SLAM error model
    slam_noise_xy_std: float = 0.03      # Position-measurement Gaussian noise (m) - SLAM is lower than VIO due to map optimization
    slam_noise_theta_std: float = 0.015  # Heading-measurement noise (rad)
    slam_drift_rate: float = 0.003       # Drift accumulation rate (m/s) - slower than VIO due to map-based correction
    slam_drift_theta_rate: float = 0.001 # Heading drift rate (rad/s)

    # SLAM relocalization failure
    slam_reloc_failure_prob: float = 0.01      # Relocalization failure probability (e.g. insufficient feature points)
    slam_reloc_failure_noise: float = 0.3      # Position error magnitude on failure (m)

    # SLAM confidence filter
    slam_jump_threshold: float = 0.5   # Jump larger than this in one step is an outlier (m)
    slam_jump_theta_threshold: float = 0.3  # Heading jump threshold (rad, ~17°)
    slam_lowconf_speed_scale: float = 0.3   # Speed scale when confidence is low
    slam_reject_holdoff: int = 3       # Number of frames to ignore after outlier detection

    # Serial communication (Pi → OpenRB)
    serial_rate_hz: float = 15.0      # Command transmission rate (Hz)
    cmd_v_deadzone: float = 0.02      # Linear velocity below this is treated as 0 (m/s)
    cmd_omega_deadzone: float = 0.05  # Angular velocity below this is treated as 0 (rad/s)
    cmd_v_resolution: float = 0.01    # Linear-velocity quantization unit (m/s)
    cmd_omega_resolution: float = 0.01  # Angular-velocity quantization unit (rad/s)

    # Controller gains
    kp_linear: float = 0.8       # Distance proportional gain
    kp_angular: float = 2.5      # Angle proportional gain
    ki_angular: float = 0.1      # Angle integral gain
    kd_angular: float = 0.3      # Angle derivative gain
    slowdown_radius: float = 1.0 # Deceleration onset radius (m)


# ──────────────────────────────────────────────
# Vehicle Model (Tank / Differential Drive)
# ──────────────────────────────────────────────
class TankVehicle:
    """Tank-style (differential drive) rover model"""

    def __init__(self, x: float, y: float, theta: float, wheel_base: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.L = wheel_base

    def update(self, v_cmd: float, omega_cmd: float, dt: float,
               disturbance: 'DisturbanceModel') -> Tuple[float, float]:
        """
        Receive control commands (v, omega) and update the actual position.
        Reflects the actual motion with disturbance applied.
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
# Disturbance Model (grass terrain disturbance)
# ──────────────────────────────────────────────
class DisturbanceModel:
    """
    Disturbance modeling on grass terrain:
    - Wheel slip (loss of traction on grass)
    - Gaussian noise (irregular terrain)
    - Directional disturbance (grass, pebbles, etc.)
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def apply(self, v_cmd: float, omega_cmd: float) -> Tuple[float, float]:
        c = self.cfg

        # Slip: wheels slip on grass, reducing actual speed relative to command
        slip = np.random.normal(c.slip_factor_mean, c.slip_factor_std)
        slip = np.clip(slip, 0.7, 1.0)

        # Linear-velocity disturbance
        v_noise = np.random.normal(0, c.disturbance_v_std)
        v_actual = v_cmd * slip + v_noise

        # Angular-velocity disturbance (directional disturbance from left/right wheel slip difference)
        omega_noise = np.random.normal(0, c.disturbance_omega_std)
        omega_actual = omega_cmd * slip + omega_noise

        return v_actual, omega_actual


# ──────────────────────────────────────────────
# SLAM Error Model (SLAM positioning error)
# ──────────────────────────────────────────────
class SLAMModel:
    """
    SLAM position estimation error model:
    - Gaussian measurement noise (lower than VIO due to map optimization)
    - Drift accumulating over time (accumulates slowly due to map-based correction)
    - Large position error occurs on relocalization failure
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
        # Drift accumulation (random walk)
        self.drift_x += np.random.normal(0, c.slam_drift_rate * dt)
        self.drift_y += np.random.normal(0, c.slam_drift_rate * dt)
        self.drift_theta += np.random.normal(0, c.slam_drift_theta_rate * dt)

        # Measurement noise
        noise_x = np.random.normal(0, c.slam_noise_xy_std)
        noise_y = np.random.normal(0, c.slam_noise_xy_std)
        noise_theta = np.random.normal(0, c.slam_noise_theta_std)

        # Relocalization failure: large position error from insufficient feature points, etc.
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
# SLAM Confidence Filter (outlier rejection + confidence evaluation)
# ──────────────────────────────────────────────
class SLAMFilter:
    """
    Filter out outliers in SLAM estimates and evaluate confidence.
    - Unrealistically large position jump in one step → reject (keep previous value)
    - Confidence drops on consecutive rejects → deceleration signal to the controller
    - On a real rover, this filter sits between the SLAM raw output and the controller
    """

    MAX_CONSECUTIVE_REJECTS = 10  # Force acceptance after this many consecutive rejects (reset)

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.prev_est = None          # Previous filter output (x, y, theta)
        self.reject_count = 0         # Consecutive reject count
        self.holdoff_remaining = 0    # Remaining holdoff frames after reject
        self.confidence = 1.0         # 0.0 ~ 1.0 confidence
        self.total_rejects = 0        # Cumulative reject count (for logging)

    def update(self, raw_x: float, raw_y: float, raw_theta: float,
               dt: float) -> Tuple[float, float, float, float]:
        """
        Filter the SLAM raw estimate.
        Returns: (filtered_x, filtered_y, filtered_theta, confidence)
        """
        c = self.cfg

        if self.prev_est is None:
            self.prev_est = (raw_x, raw_y, raw_theta)
            self.confidence = 1.0
            return raw_x, raw_y, raw_theta, self.confidence

        px, py, pt = self.prev_est

        # Too many consecutive rejects → force acceptance (rover has moved, diverging from the frozen position)
        if self.reject_count >= self.MAX_CONSECUTIVE_REJECTS:
            self.prev_est = (raw_x, raw_y, raw_theta)
            self.reject_count = 0
            self.holdoff_remaining = 0
            self.confidence = 0.3  # Restart with low confidence
            return raw_x, raw_y, raw_theta, self.confidence

        # Compute position jump magnitude
        jump_xy = np.sqrt((raw_x - px)**2 + (raw_y - py)**2)
        jump_theta = abs(((raw_theta - pt) + np.pi) % (2 * np.pi) - np.pi)

        # Physically possible maximum movement: max_speed * dt * safety margin
        max_possible_jump = c.max_speed * dt * 3.0

        is_outlier = (jump_xy > max(c.slam_jump_threshold, max_possible_jump) or
                      jump_theta > c.slam_jump_theta_threshold)

        if is_outlier or self.holdoff_remaining > 0:
            # Outlier → keep previous value
            if is_outlier:
                self.reject_count += 1
                self.total_rejects += 1
                self.holdoff_remaining = c.slam_reject_holdoff
            self.holdoff_remaining = max(0, self.holdoff_remaining - 1)

            # Reduce confidence
            self.confidence = max(0.1, self.confidence - 0.2)

            # Keep previous value (dead-reckoning-like behavior)
            return px, py, pt, self.confidence
        else:
            # Normal → accept value, recover confidence
            self.reject_count = 0
            self.confidence = min(1.0, self.confidence + 0.05)
            self.prev_est = (raw_x, raw_y, raw_theta)
            return raw_x, raw_y, raw_theta, self.confidence


# ──────────────────────────────────────────────
# Serial Command Protocol (Pi → OpenRB simulation)
# ──────────────────────────────────────────────
class SerialCommandSim:
    """
    Pi5 → OpenRB serial communication simulation.
    Protocol to reference for the real implementation:
      Packet: [0xFF][0xFE][v_high][v_low][omega_high][omega_low][checksum]
      - v, omega: signed int16, units mm/s, mrad/s
      - checksum: XOR of payload bytes

    In the simulation:
    - Transmission rate limit (serial_rate_hz)
    - Dead-zone handling (prevents motor chatter)
    - Quantization (reflects integer conversion in real serial)
    - Rate limiting (smooths abrupt command changes)
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
        Convert controller output for serial transmission.
        Returns: (processed_v, processed_omega, was_sent_this_step)
        """
        c = self.cfg
        self.time_since_send += dt

        if self.time_since_send < self.send_interval:
            # Not yet transmission time → keep previous command
            return self.last_sent_v, self.last_sent_omega, False

        # Transmission time reached
        self.time_since_send = 0.0

        # Dead-zone handling
        if abs(v_cmd) < c.cmd_v_deadzone:
            v_cmd = 0.0
        if abs(omega_cmd) < c.cmd_omega_deadzone:
            omega_cmd = 0.0

        # Quantization (serial int16 conversion simulation)
        v_cmd = round(v_cmd / c.cmd_v_resolution) * c.cmd_v_resolution
        omega_cmd = round(omega_cmd / c.cmd_omega_resolution) * c.cmd_omega_resolution

        self.last_sent_v = v_cmd
        self.last_sent_omega = omega_cmd
        self.packet_count += 1

        return v_cmd, omega_cmd, True

    def build_packet(self, v: float, omega: float) -> bytes:
        """
        Build the byte packet to send to the real OpenRB (for reference).
        Real rover code calls serial.write() in this format.
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
# Navigation Controller (PID based)
# ──────────────────────────────────────────────
class NavigationController:
    """
    Controller that drives to the target point based on the SLAM-estimated position.
    - Distance-proportional linear-velocity control
    - PID angular-velocity control
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

        # Heading error
        angle_error = desired_heading - est_theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # PID angular velocity
        self.integral_angle_error += angle_error * c.dt
        self.integral_angle_error = np.clip(self.integral_angle_error, -2.0, 2.0)
        derivative = (angle_error - self.prev_angle_error) / c.dt
        self.prev_angle_error = angle_error

        omega = (c.kp_angular * angle_error +
                 c.ki_angular * self.integral_angle_error +
                 c.kd_angular * derivative)
        omega = np.clip(omega, -c.max_omega, c.max_omega)

        # Linear velocity: distance-proportional + deceleration + decelerate on large heading error
        speed_factor = min(1.0, distance / c.slowdown_radius)
        heading_factor = max(0.2, 1.0 - abs(angle_error) / np.pi)
        v = c.kp_linear * distance * speed_factor * heading_factor
        v = np.clip(v, 0, c.max_speed)

        # SLAM-confidence-based deceleration: reduce speed when confidence is low
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

        # SLAM position estimation (raw)
        est_x, est_y, est_theta = slam.estimate(true_x, true_y, true_theta, cfg.dt)

        # SLAM confidence filter (outlier rejection)
        filt_x, filt_y, filt_theta, confidence = slam_filter.update(
            est_x, est_y, est_theta, cfg.dt)

        # Compute control command (filtered position + confidence-based deceleration)
        v_cmd, omega_cmd = controller.compute(filt_x, filt_y, filt_theta,
                                               cfg.target_x, cfg.target_y,
                                               slam_confidence=confidence)

        # Serial protocol simulation (dead zone + quantization + rate limiting)
        v_serial, omega_serial, was_sent = serial_cmd.process(
            v_cmd, omega_cmd, cfg.dt)

        # Vehicle update (based on the command actually sent over serial + disturbance applied)
        v_actual, omega_actual = vehicle.update(
            v_serial, omega_serial, cfg.dt, disturbance)

        # Actual distance to the target
        dist = np.sqrt((true_x - cfg.target_x)**2 + (true_y - cfg.target_y)**2)

        # Log recording
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

        # Goal-reached decision (based on actual position)
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

    # ── 1) 2D path ──
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

    # ── 2) Distance to goal ──
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(log.time, log.distance_to_goal, 'b-', linewidth=1.2)
    ax2.axhline(y=cfg.goal_tolerance, color='r', linestyle='--',
                label=f'Tolerance ({cfg.goal_tolerance}m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Distance to Goal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── 3) Controller → serial → actual (v) ──
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(log.time, log.cmd_v, 'b-', label='Controller v', alpha=0.5, linewidth=0.8)
    ax3.plot(log.time, log.serial_v, 'g-', label='Serial v', alpha=0.8, linewidth=1.2)
    ax3.plot(log.time, log.actual_v, 'b--', label='Actual v', alpha=0.5, linewidth=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Linear Velocity (m/s)')
    ax3.set_title('Command Pipeline: v')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ── 4) SLAM drift ──
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

    # ── 5) Position estimation error (raw vs filtered) ──
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

    # ── 7) Controller → serial → actual (omega) ──
    ax7 = fig.add_subplot(3, 3, 8)
    ax7.plot(log.time, log.cmd_omega, 'r-', label='Controller ω', alpha=0.5, linewidth=0.8)
    ax7.plot(log.time, log.serial_omega, 'g-', label='Serial ω', alpha=0.8, linewidth=1.2)
    ax7.plot(log.time, log.actual_omega, 'r--', label='Actual ω', alpha=0.5, linewidth=0.8)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Angular Velocity (rad/s)')
    ax7.set_title('Command Pipeline: ω')
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)

    # ── 8) Heading error ──
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
# Monte Carlo Analysis (multiple simulation runs)
# ──────────────────────────────────────────────
def monte_carlo(cfg: SimConfig, n_runs: int = 50):
    """Run the simulation multiple times to analyze success rate and final error distribution"""
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

    # Result visualization
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
    Interactive animation simulation.
    1) Setup screen: left-click to set start position, right-click to set target position
    2) Press the ▶ Play button to start the simulation
    """
    from matplotlib.widgets import Button

    # ── Step 1: interactive setup screen ──
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

    # Wait until the Play button is pressed
    while not state['running']:
        plt.pause(0.1)
        if not plt.fignum_exists(fig.number):
            return  # Exit if the window is closed

    fig.canvas.mpl_disconnect(cid)

    # ── Step 2: run the simulation ──
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

    # Hide the Play button
    ax_btn.set_visible(False)

    true_path_x, true_path_y = [], []
    est_path_x, est_path_y = [], []

    t = 0.0
    while t < cfg.max_time:
        if not plt.fignum_exists(fig.number):
            return  # Exit if the window is closed

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

        # Refresh the screen every 5 frames
        if int(t / cfg.dt) % 5 == 0:
            ax.clear()
            ax.set_facecolor('#c8e6c9')

            ax.plot(true_path_x, true_path_y, 'b-', linewidth=1.5, label='True')
            ax.plot(est_path_x, est_path_y, 'r--', linewidth=1.0, label='SLAM Est.', alpha=0.6)

            # Draw the rover
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

            # Direction arrow
            arrow_len = 0.25
            ax.annotate('', xy=(true_x + arrow_len * np.cos(true_theta),
                                true_y + arrow_len * np.sin(true_theta)),
                        xytext=(true_x, true_y),
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
                        zorder=11)

            # Target
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
