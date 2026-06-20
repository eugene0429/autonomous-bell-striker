from __future__ import annotations

import unittest
from typing import List, Tuple

from Driving.drive_to import SafetyConfig, SafetySupervisor


class _Clock:
    def __init__(self, t0: float = 0.0):
        self.t = t0
    def __call__(self) -> float:
        return self.t
    def advance(self, dt: float) -> None:
        self.t += dt


def _ok(x: float, y: float):
    return {"x": x, "y": y, "theta": 0.0, "tracking_ok": True, "tracking": "OK"}

def _lost():
    return {"x": 0.0, "y": 0.0, "theta": 0.0, "tracking_ok": False, "tracking": "LOST"}


class TestSupervisorOKPath(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.logs: List[str] = []
        self.sup = SafetySupervisor(
            cfg=SafetyConfig(),
            now=self.clock,
            log=self.logs.append,
        )

    def test_first_ok_returns_ok(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.assertEqual(self.logs, [])

    def test_consecutive_ok_within_velocity_returns_ok(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(0.067)             # 15 Hz period
        self.assertEqual(self.sup.check(_ok(0.01, 0.0)), "OK")
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(0.02, 0.0)), "OK")
        self.assertEqual(self.logs, [])


class TestSupervisorLostEscalation(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.logs: List[str] = []
        self.sup = SafetySupervisor(
            cfg=SafetyConfig(lost_quiet_sec=0.5, lost_warn_sec=3.0,
                             warn_log_period=0.5),
            now=self.clock,
            log=self.logs.append,
        )

    def test_short_lost_under_quiet_threshold_holds_silently(self):
        self.sup.check(_ok(0.0, 0.0))
        self.clock.advance(0.1)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.clock.advance(0.3)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(self.logs, [])  # silent

    def test_lost_in_warn_window_logs_and_holds(self):
        self.sup.check(_ok(0.0, 0.0))
        self.clock.advance(0.6)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(len(self.logs), 1)
        self.assertIn("tracking lost", self.logs[0])
        # next check 0.1s later: still under warn_log_period → no new log
        self.clock.advance(0.1)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(len(self.logs), 1)
        # now 0.5s after first warn → new log line
        self.clock.advance(0.5)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(len(self.logs), 2)

    def test_lost_beyond_total_threshold_aborts(self):
        self.sup.check(_ok(0.0, 0.0))
        # quiet (0.5) + warn (3.0) = 3.5s total before abort
        self.clock.advance(3.6)
        self.assertEqual(self.sup.check(_lost()), "ABORT")
        self.assertIn("tracking lost", self.sup.reason)

    def test_recovery_clears_lost_state(self):
        self.sup.check(_ok(0.0, 0.0))
        self.clock.advance(1.0)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.clock.advance(0.1)
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.logs.clear()
        # should not log "lost" anymore
        self.clock.advance(0.6)
        # treat next pose as fresh; no new lost
        self.assertEqual(self.sup.check(_ok(0.001, 0.0)), "OK")
        self.assertEqual(self.logs, [])


class TestSupervisorPoseJump(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.logs: List[str] = []
        # max_linear_vel=0.3, jump_factor=3 → at dt=0.067s, threshold = 0.06m
        self.sup = SafetySupervisor(
            cfg=SafetyConfig(max_linear_vel=0.3, jump_factor=3.0,
                             jump_outlier_max=3),
            now=self.clock,
            log=self.logs.append,
        )

    def test_single_jump_holds_then_recovers(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(0.067)                    # threshold ≈ 0.06 m
        self.assertEqual(self.sup.check(_ok(1.0, 0.0)), "HOLD")  # 1m jump
        self.clock.advance(0.067)
        # next plausible pose (close to last_ok=(0,0)) → OK, counter resets
        self.assertEqual(self.sup.check(_ok(0.01, 0.0)), "OK")

    def test_three_consecutive_jumps_abort(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        for i in range(3):
            self.clock.advance(0.067)
            res = self.sup.check(_ok(10.0 + i, 0.0))
            if i < 2:
                self.assertEqual(res, "HOLD")
            else:
                self.assertEqual(res, "ABORT")
        self.assertIn("pose jump", self.sup.reason)

    def test_non_consecutive_jump_does_not_accumulate(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(5.0, 0.0)), "HOLD")     # jump 1
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(0.01, 0.0)), "OK")       # reset
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(5.0, 0.0)), "HOLD")     # jump 1 again, not 2
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(0.02, 0.0)), "OK")

    def test_long_dt_disables_jump_check(self):
        # If dt >= 1s (e.g. after a long pause), don't classify as a jump.
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(1.5)
        self.assertEqual(self.sup.check(_ok(2.0, 0.0)), "OK")


from Driving.drive_to import RunArgs, _run_loop


class _FakeLocalizer:
    def __init__(self, poses):
        self._poses = list(poses)
    def get_pose(self):
        if not self._poses:
            return None
        return self._poses.pop(0)
    def is_alive(self):
        return True


class _FakeMotor:
    def __init__(self):
        self.calls: List[Tuple[float, float]] = []
    def drive(self, wL, wR):
        self.calls.append((wL, wR))


class _FakeController:
    """Returns a fixed (small) command and reports reached after N calls."""
    def __init__(self, reach_after: int):
        self.calls = 0
        self._reach_after = reach_after
    def compute(self, x, y, theta, tx, ty):
        self.calls += 1
        return {
            "wheel_omega_left": 1.0,
            "wheel_omega_right": 1.0,
            "v": 0.1, "omega": 0.0,
            "distance": 0.05 if self.calls >= self._reach_after else 1.0,
            "angle_error": 0.0,
            "reached": self.calls >= self._reach_after,
        }


class _SleepNoop:
    def __init__(self): self.calls = 0
    def __call__(self, _seconds): self.calls += 1


class TestRunLoop(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(x=1.0, y=0.0, rate=15.0, timeout=10.0,
                        port="/dev/null", baud=115200,
                        dry_run=True, verbose=False)
        defaults.update(overrides)
        return RunArgs(**defaults)

    def test_reaches_target_returns_zero(self):
        clock = _Clock()
        sup = SafetySupervisor(SafetyConfig(), now=clock, log=lambda _m: None)
        loc = _FakeLocalizer([_ok(0.0, 0.0), _ok(0.01, 0.0), _ok(0.02, 0.0)])
        motor = _FakeMotor()
        ctrl = _FakeController(reach_after=3)
        sleep = _SleepNoop()

        rc = _run_loop(self._args(), loc, ctrl, motor, sup,
                       now=clock, sleep=sleep)
        self.assertEqual(rc, 0)
        # last call must be (0, 0) per "send zero on reach" requirement
        self.assertEqual(motor.calls[-1], (0.0, 0.0))

    def test_abort_from_supervisor_returns_two(self):
        clock = _Clock()
        sup = SafetySupervisor(
            SafetyConfig(lost_quiet_sec=0.0, lost_warn_sec=0.0),
            now=clock, log=lambda _m: None,
        )
        # First pose triggers ABORT (lost from frame 1, total threshold = 0)
        loc = _FakeLocalizer([_lost()])
        motor = _FakeMotor()
        ctrl = _FakeController(reach_after=999)
        sleep = _SleepNoop()

        rc = _run_loop(self._args(), loc, ctrl, motor, sup,
                       now=clock, sleep=sleep)
        self.assertEqual(rc, 2)

    def test_timeout_returns_one(self):
        clock = _Clock()
        sup = SafetySupervisor(SafetyConfig(), now=clock, log=lambda _m: None)
        # Endless OK frames; controller never reaches.
        loc = _FakeLocalizer([_ok(0.0, 0.0)] * 10000)
        motor = _FakeMotor()
        ctrl = _FakeController(reach_after=10**9)
        # advance the clock inside sleep so timeout actually fires
        period = 1.0 / 15.0

        def sleep_fn(_secs):
            clock.advance(period)

        args = self._args(timeout=0.5)   # <= 8 iterations at 15 Hz
        rc = _run_loop(args, loc, ctrl, motor, sup, now=clock, sleep=sleep_fn)
        self.assertEqual(rc, 1)

    def test_hold_sends_zero_velocity(self):
        clock = _Clock()
        sup = SafetySupervisor(SafetyConfig(lost_quiet_sec=0.5),
                               now=clock, log=lambda _m: None)
        loc = _FakeLocalizer([_ok(0.0, 0.0), _lost(), _lost()])
        motor = _FakeMotor()
        ctrl = _FakeController(reach_after=999)

        period = 1.0 / 15.0
        def sleep_fn(_secs): clock.advance(period)

        # Three frames; we expect the second and third to issue HOLD -> drive(0,0)
        args = self._args(timeout=0.25)   # 0.25 / period ~= 3 iterations
        _ = _run_loop(args, loc, ctrl, motor, sup, now=clock, sleep=sleep_fn)
        # First frame is OK -> nonzero drive; subsequent HOLD frames must each
        # issue (0,0). Counting >= 2 zeros distinguishes the HOLD branch from
        # the single zero the timeout-exit path would emit on its own.
        self.assertEqual(motor.calls[0], (1.0, 1.0))
        self.assertGreaterEqual(motor.calls.count((0.0, 0.0)), 2)
