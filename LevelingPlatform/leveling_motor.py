"""
Leveling Platform — Pi5 ↔ OpenRB-150 serial client.

[LevelingIK.aim_at()](leveling_ik.py) 의 출력 (`angles_steps`) 을 OpenRB-150 으로
보내 3개의 Dynamixel(또는 동등 모터) 을 절대 위치로 이동시키는 클라이언트.

═══════════════════════════════════════════════════════════════════════
  Wire Protocol — ASCII / line-oriented / '\n'-terminated
═══════════════════════════════════════════════════════════════════════

Transport
---------
- USB CDC virtual COM (Linux: `/dev/ttyACM0` 등)
- Baud   : 115200  (안정적·호환 좋음. 필요 시 1 Mbps 까지 상향 가능)
- 8N1, no flow control
- Encoding: 7-bit ASCII
- Line terminator: LF only ('\n' / 0x0A) — CR 무시

Command (Pi → OpenRB)
---------------------
    PING\n
        keep-alive / health check.

    AIM <s1> <s2> <s3>\n
        모터 1/2/3 을 절대 인코더 step 위치로 동시 이동.
        s_i : signed decimal integer.
              MOTOR_STEPS=4096 기준 [-2048..+2047] 가 typical.
              펌웨어 측에서 한계 위반 시 ERR OUT_OF_RANGE 반환.
        OpenRB 는 모든 모터의 모션 완료 (또는 timeout/abort) 후에 'OK'/'ERR' 반환.

    HOME\n
        모든 모터를 step=0 으로 이동.

    STATUS\n
        현재 위치/플래그 조회 (블로킹 X, 즉시 응답).

    STOP\n
        모든 모터 즉시 정지 (현재 명령을 취소).

Response (OpenRB → Pi)
----------------------
    OK\n
        명령 처리 완료. AIM/HOME 의 경우 모션 완료까지 블로킹 후 반환.
    ERR <reason>\n
        실패. reason 예: PARSE / OUT_OF_RANGE / TIMEOUT / NOT_HOMED / BUSY / HW
    PONG\n
        PING 의 응답.
    S <wL> <wR> <s1> <s2> <s3> <s4> <s5> <rpmT> <rpmB> <flags>\n
        STATUS 의 응답. (펌웨어 v1.2+ 의 11-필드 포맷)
            wL/wR        : 휠 ID 6/7 실측 속도 [mrad/s]
            s1/s2/s3     : 레벨링 LVL_1/2/3 step
            s4           : 카메라 틸트 ID 4 step
            s5           : 로더 ID 5 step
            rpmT/rpmB    : T-motor TOP/BOTTOM 현재 RPM 명령값
            flags        : bit0=wheel watchdog tripped
                           bit1=leveling moving
                           bit2=tilt moving
                           bit3=loader moving
                           bit4=tmotor active (>100 RPM)
                           bit5=error latched
                           bit6=homed
                           bit7=estop

Synchronization rules
---------------------
- 동기식 request-response. Pi 는 OK/ERR/PONG/S 라인을 받기 전 다음 명령 안 보냄.
- OpenRB 는 한 번에 한 명령만 처리. 추가 명령은 큐에 쌓지 말고 BUSY 로 거절.
- AIM/HOME 은 모션 완료까지 응답 보류 (가장 단순한 동기 모델).
- 모터 stall / timeout 검출 시 자동으로 STOP 후 ERR 반환.

Frame size & timing
-------------------
- 가장 긴 일반 명령: "AIM -2048 -2048 -2048\n" = 23 bytes ≈ 2 ms @ 115200
- 가장 긴 응답:      "S -30000 30000 -2048 -2048 -2048 -2048 -2048 10000 10000 255\n"
                     ≈ 60 bytes ≈ 5 ms (STATUS, v1.2+ 11-필드 포맷)
- 모션 latency 포함 한 사이클 ≤ 100 ms 가 일반 (Phase 2 의 매 타격 직전 1 회 호출)

═══════════════════════════════════════════════════════════════════════
  Python API
═══════════════════════════════════════════════════════════════════════

    from leveling_ik    import LevelingIK, LevelingConfig
    from leveling_motor import LevelingMotorClient, MotorClientConfig

    ik = LevelingIK(LevelingConfig())

    with LevelingMotorClient(MotorClientConfig(port="/dev/ttyACM0")) as mc:
        mc.ping()                                    # 헬스 체크
        mc.home()                                    # 시작 시 home
        out = ik.aim_at((0.10, 0.00, 3.0))
        if not out["ok"]:
            raise RuntimeError("IK unreachable")
        mc.aim(out)                                  # 모터 이동 + OK 대기
        print(mc.status())  # {wheel_mrad, leveling_steps, tilt_step,
                            #  loader_step, tmotor_rpm, flags,
                            #  watchdog, leveling_moving, tilt_moving,
                            #  loader_moving, tmotor_active, error,
                            #  homed, estop, moving}
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class MotorClientConfig:
    # ── Transport ──
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    read_timeout_sec: float = 5.0           # AIM/HOME 모션 완료 대기 포함
    write_timeout_sec: float = 1.0
    open_settle_sec: float = 2.0            # 포트 open 후 OpenRB 안정화 대기

    # ── 회전 방향 / 영점 보정 ──
    # IK 가 주는 angles_steps 에 적용: cmd_steps[i] = sign[i] * angles_steps[i] + offset[i]
    direction_signs: Tuple[int, int, int] = (+1, +1, +1)
    home_offsets_steps: Tuple[int, int, int] = (0, 0, 0)

    # ── 안전 한계 (펌웨어가 cross-check 하지만 클라이언트도 미리 검사) ──
    motor_min_step: int = -2048
    motor_max_step: int = +2047

    # ── 디버그 ──
    verbose: bool = False                   # 송수신 라인을 stderr 로 출력
    dry_run: bool = False                   # True 면 직접 시리얼 안 보내고 출력만


# ──────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────
class MotorProtocolError(RuntimeError):
    """OpenRB 펌웨어가 ERR 응답을 줬거나 응답 형식이 깨짐."""


class MotorTimeoutError(RuntimeError):
    """timeout 안에 응답이 들어오지 않음."""


# ──────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────
class LevelingMotorClient:
    """OpenRB-150 으로 3-RRS 레벨링 모터 명령 송신."""

    def __init__(self, cfg: Optional[MotorClientConfig] = None):
        self.cfg = cfg if cfg is not None else MotorClientConfig()
        self._ser = None  # pyserial.Serial
        self._last_cmd: str = ""  # dry-run 응답 합성용

    # ── lifecycle ──
    def connect(self) -> None:
        if self.cfg.dry_run:
            self._log("[dry-run] skip serial open")
            return
        import serial  # lazy import — pyserial 이 없는 환경도 dry_run 으로 import 가능
        self._ser = serial.Serial(
            port=self.cfg.port,
            baudrate=self.cfg.baud,
            timeout=self.cfg.read_timeout_sec,
            write_timeout=self.cfg.write_timeout_sec,
        )
        # OpenRB 가 USB-CDC reset 후 안정화될 때까지 대기 + 버퍼 비우기
        time.sleep(self.cfg.open_settle_sec)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self._log(f"[open] {self.cfg.port} @ {self.cfg.baud}")

    def disconnect(self) -> None:
        if self._ser is None:
            return
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._ser.close()
        except Exception:
            pass
        self._ser = None
        self._log("[close]")

    def __enter__(self) -> "LevelingMotorClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # ── public API ──
    def ping(self) -> bool:
        return self._command("PING") == "PONG"

    def aim(self, ik_result: Dict) -> bool:
        """LevelingIK.aim_at() 출력 dict 를 받아 AIM 명령으로 송신.

        Raises
        ------
        ValueError              IK 가 unreachable 이거나 (angles_steps is None)
                                step 값이 한계 밖일 때.
        MotorProtocolError      OpenRB 가 ERR 응답.
        MotorTimeoutError       응답이 timeout 안에 안 옴.
        """
        steps = ik_result.get("angles_steps")
        if steps is None or len(steps) != 3:
            raise ValueError("ik_result['angles_steps'] is missing/invalid "
                             "(IK probably unreachable)")
        cmd_steps = self._apply_calibration(steps)
        for i, s in enumerate(cmd_steps):
            if not (self.cfg.motor_min_step <= s <= self.cfg.motor_max_step):
                raise ValueError(
                    f"motor {i+1} step {s} out of range "
                    f"[{self.cfg.motor_min_step}, {self.cfg.motor_max_step}]")
        return self._command(
            f"AIM {cmd_steps[0]} {cmd_steps[1]} {cmd_steps[2]}") == "OK"

    def aim_fast(self, ik_result: Dict) -> bool:
        """AIM 의 streaming 버전 — 펌웨어가 syncWrite 만 하고 즉시 OK.

        용도
        ----
        GUI 드래그·연속 추종처럼 모션 완료를 기다리지 않고 다음 명령을
        바로 송신해야 할 때. AIM 과 응답 의미가 다르다:
            AIM       OK = 모터가 목표에 도달 (tolerance 내)
            aim_fast  OK = 명령이 수신·발송되었음 (아직 이동 중일 수 있음)

        "도달 보장" 이 필요한 시퀀스 (예: 조준 후 사격) 에는 AIM 을 쓸 것.
        """
        steps = ik_result.get("angles_steps")
        if steps is None or len(steps) != 3:
            raise ValueError("ik_result['angles_steps'] is missing/invalid "
                             "(IK probably unreachable)")
        cmd_steps = self._apply_calibration(steps)
        for i, s in enumerate(cmd_steps):
            if not (self.cfg.motor_min_step <= s <= self.cfg.motor_max_step):
                raise ValueError(
                    f"motor {i+1} step {s} out of range "
                    f"[{self.cfg.motor_min_step}, {self.cfg.motor_max_step}]")
        return self._command(
            f"AIMF {cmd_steps[0]} {cmd_steps[1]} {cmd_steps[2]}") == "OK"

    def home(self) -> bool:
        return self._command("HOME") == "OK"

    def stop(self) -> bool:
        return self._command("STOP") == "OK"

    def status(self) -> Dict:
        """STATUS 명령 응답을 dict 로 파싱.

        펌웨어 응답 포맷 (v1.2+):
            S <wL> <wR> <s1> <s2> <s3> <s4> <s5> <rpmT> <rpmB> <flags>
        """
        resp = self._command("STATUS")
        parts = resp.split()
        if len(parts) != 11 or parts[0] != "S":
            raise MotorProtocolError(f"bad STATUS reply: {resp!r}")
        try:
            wL, wR = int(parts[1]), int(parts[2])
            leveling = (int(parts[3]), int(parts[4]), int(parts[5]))
            tilt_step = int(parts[6])
            loader_step = int(parts[7])
            rpm_top, rpm_bot = int(parts[8]), int(parts[9])
            flags = int(parts[10])
        except ValueError as e:
            raise MotorProtocolError(f"bad STATUS numbers: {resp!r}") from e
        return {
            "wheel_mrad":      (wL, wR),
            "leveling_steps":  leveling,
            "tilt_step":       tilt_step,
            "loader_step":     loader_step,
            "tmotor_rpm":      (rpm_top, rpm_bot),
            "flags":           flags,
            "watchdog":        bool(flags & (1 << 0)),
            "leveling_moving": bool(flags & (1 << 1)),
            "tilt_moving":     bool(flags & (1 << 2)),
            "loader_moving":   bool(flags & (1 << 3)),
            "tmotor_active":   bool(flags & (1 << 4)),
            "error":           bool(flags & (1 << 5)),
            "homed":           bool(flags & (1 << 6)),
            "estop":           bool(flags & (1 << 7)),
            "moving":          bool(flags & 0b00001110),  # leveling|tilt|loader
        }

    # ── 내부 ──
    def _apply_calibration(self, ik_steps) -> List[int]:
        c = self.cfg
        return [int(c.direction_signs[i] * ik_steps[i] + c.home_offsets_steps[i])
                for i in range(3)]

    def _command(self, cmd: str) -> str:
        """한 줄 명령 송신 후 한 줄 응답 반환. ERR 응답은 예외로 변환."""
        self._send_line(cmd)
        resp = self._recv_line()
        if resp.startswith("ERR"):
            raise MotorProtocolError(f"controller refused {cmd!r}: {resp}")
        return resp

    def _send_line(self, line: str) -> None:
        self._last_cmd = line
        payload = (line + "\n").encode("ascii")
        self._log(f"→ {line}")
        if self.cfg.dry_run:
            return
        if self._ser is None:
            raise RuntimeError("not connected (call connect() or use context manager)")
        self._ser.write(payload)
        self._ser.flush()

    def _recv_line(self) -> str:
        if self.cfg.dry_run:
            fake = self._dry_run_response(self._last_cmd)
            self._log(f"← (dry-run) {fake}")
            return fake
        if self._ser is None:
            raise RuntimeError("not connected")
        raw = self._ser.readline()       # blocks up to read_timeout_sec
        if not raw:
            raise MotorTimeoutError(
                f"no response within {self.cfg.read_timeout_sec}s")
        line = raw.decode("ascii", errors="replace").rstrip("\r\n")
        self._log(f"← {line}")
        return line

    @staticmethod
    def _dry_run_response(cmd: str) -> str:
        """프로토콜 명세에 부합하는 가짜 응답 — dry-run/유닛테스트용."""
        if cmd == "PING":
            return "PONG"
        if cmd == "STATUS":
            # 모션 종료 + homed 상태로 가정 (flags = 0b10000)
            return "S 0 0 0 16"
        return "OK"  # AIM / HOME / STOP

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[motor] {msg}", file=sys.stderr)


# ──────────────────────────────────────────────
# CLI — IK + serial send 통합 한 발 송신 (벤치 검증용)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    from leveling_ik import LevelingConfig, LevelingIK

    ap = argparse.ArgumentParser(
        description="Leveling Platform: IK → OpenRB serial single-shot")
    ap.add_argument("--target", nargs=3, type=float, required=True,
                    metavar=("X", "Y", "Z"),
                    help="target 3D point in plate-base frame [m]")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--home", action="store_true",
                    help="send HOME before AIM")
    ap.add_argument("--dry-run", action="store_true",
                    help="시리얼 미접속, 명령 라인만 출력")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ik = LevelingIK(LevelingConfig())
    out = ik.aim_at(tuple(args.target))
    if out["angles_deg"] is None:
        print("IK UNREACHABLE — aborting", file=sys.stderr)
        sys.exit(2)
    print(f"target  : ({args.target[0]:+.4f}, {args.target[1]:+.4f}, "
          f"{args.target[2]:+.4f}) m")
    print(f"angles  : {out['angles_deg']} deg")
    print(f"steps   : {out['angles_steps']}")
    print(f"feasible: {out['ok']}")
    if not out["ok"]:
        print("⚠ ball joint limit exceeded — proceeding anyway", file=sys.stderr)

    cfg = MotorClientConfig(
        port=args.port,
        baud=args.baud,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    with LevelingMotorClient(cfg) as mc:
        if not args.dry_run:
            assert mc.ping(), "PING failed"
        if args.home:
            mc.home()
        mc.aim(out)
        st = mc.status()
        print(f"final status: {st}")
