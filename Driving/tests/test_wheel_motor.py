from __future__ import annotations

import unittest

from Driving.wheel_motor import WheelMotorClient, WheelMotorConfig


class TestWheelMotorConfig(unittest.TestCase):
    def test_defaults(self):
        c = WheelMotorConfig()
        self.assertEqual(c.port, "/dev/ttyACM0")
        self.assertEqual(c.baud, 115200)
        self.assertEqual(c.max_wheel_mrad_s, 30000)
        self.assertEqual(c.deadzone_mrad_s, 5)
        self.assertEqual(c.direction_signs, (+1, +1))
        self.assertFalse(c.verbose)
        self.assertFalse(c.dry_run)


class TestWheelMotorClientConstruction(unittest.TestCase):
    def test_can_instantiate_with_dry_run(self):
        client = WheelMotorClient(WheelMotorConfig(dry_run=True))
        self.assertTrue(client.cfg.dry_run)
        self.assertEqual(client.sent_lines, [])


class TestDriveQuantization(unittest.TestCase):
    def _client(self, **cfg_overrides):
        cfg = WheelMotorConfig(dry_run=True, **cfg_overrides)
        return WheelMotorClient(cfg)

    def test_zero_zero_emits_zero_zero(self):
        c = self._client()
        c.drive(0.0, 0.0)
        self.assertEqual(c.sent_lines, ["DRIVE 0 0"])

    def test_quantizes_to_mrad_per_sec(self):
        c = self._client()
        c.drive(1.234, -2.345)
        self.assertEqual(c.sent_lines, ["DRIVE 1234 -2345"])

    def test_rounds_to_nearest_mrad(self):
        c = self._client()
        c.drive(0.0014, -0.0016)   # 1.4 → 1, -1.6 → -2
        # both inside deadzone (|w| < 5 mrad) → forced to 0
        self.assertEqual(c.sent_lines, ["DRIVE 0 0"])

    def test_deadzone_zeros_both_when_both_below(self):
        c = self._client()
        c.drive(0.003, -0.004)   # 3 and -4 mrad, both < 5 → 0 0
        self.assertEqual(c.sent_lines, ["DRIVE 0 0"])

    def test_deadzone_does_not_zero_when_one_side_above(self):
        c = self._client()
        c.drive(0.003, 1.0)   # 3 mrad (inside) and 1000 mrad (outside) → keep 3
        self.assertEqual(c.sent_lines, ["DRIVE 3 1000"])

    def test_clamps_to_max(self):
        c = self._client()
        c.drive(50.0, -50.0)
        self.assertEqual(c.sent_lines, ["DRIVE 30000 -30000"])

    def test_direction_signs_flip_right_wheel(self):
        c = self._client(direction_signs=(+1, -1))
        c.drive(1.0, 1.0)
        self.assertEqual(c.sent_lines, ["DRIVE 1000 -1000"])

    def test_multiple_calls_accumulate_in_sent_lines(self):
        c = self._client()
        c.drive(0.0, 0.0)
        c.drive(1.0, -1.0)
        c.drive(0.0, 0.0)
        self.assertEqual(c.sent_lines, ["DRIVE 0 0", "DRIVE 1000 -1000", "DRIVE 0 0"])


class TestPingStop(unittest.TestCase):
    def test_ping_returns_true_in_dry_run(self):
        c = WheelMotorClient(WheelMotorConfig(dry_run=True))
        self.assertTrue(c.ping())
        self.assertEqual(c.sent_lines, ["PING"])

    def test_stop_returns_true_in_dry_run(self):
        c = WheelMotorClient(WheelMotorConfig(dry_run=True))
        self.assertTrue(c.stop())
        self.assertEqual(c.sent_lines, ["STOP"])


class TestLifecycle(unittest.TestCase):
    def test_context_manager_does_not_raise_in_dry_run(self):
        with WheelMotorClient(WheelMotorConfig(dry_run=True)) as c:
            c.drive(1.0, 1.0)
        # On exit, disconnect() should send STOP. In dry-run, that
        # appears in sent_lines.
        self.assertEqual(c.sent_lines[-1], "STOP")

    def test_disconnect_when_never_connected_is_safe(self):
        c = WheelMotorClient(WheelMotorConfig(dry_run=True))
        c.disconnect()   # should not raise even though connect() never called
        # in dry-run, disconnect sends STOP unconditionally
        self.assertEqual(c.sent_lines, ["STOP"])

    def test_connect_dry_run_is_noop(self):
        c = WheelMotorClient(WheelMotorConfig(dry_run=True))
        c.connect()      # must not try to import or open pyserial
        self.assertIsNone(c._ser)
