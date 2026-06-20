/*
 * Integrated Controller — ESC + Dynamixel (Leveling Platform Logic) + Loader
 * Controller : OpenRB-150
 *
 * ── ESC (T-Motor F90 / HobbyWing XRotor) ──────────────────────────
 *   Motors : 2x T-Motor F90 2806.5 1300KV
 *   ESCs   : 2x HobbyWing XRotor Pro 50A
 *   Library: Servo (built-in Arduino)
 *
 *   Wiring:
 *     ESC 1 signal wire → OpenRB D5  (+ GND)
 *     ESC 2 signal wire → OpenRB D3  (+ GND)
 *     ESC power leads   → LiPo battery (3S)
 *     ⚠️  Do NOT connect ESC 5V BEC to OpenRB 5V — only share GND.
 *
 * ── Dynamixel X-Series (Leveling Platform) ─────────────────────────
 *   Motors : 3x Dynamixel X-series (ID 1, 2, 3)
 *   Library: Dynamixel2Arduino
 *
 *   Step range : -2048 ~ +2047  (4096 steps / revolution)
 *   Offsets    : Motor1=+1024(90°), Motor2=+1024(90°), Motor3=0
 *
 *   Wiring:
 *     Dynamixel bus → OpenRB Serial1 (built-in half-duplex port)
 *
 * ── Dynamixel XL330 (Ball Loader) ──────────────────────────────────
 *   Motors : 1x Dynamixel XL330 (ID 4)
 *   Mode   : Velocity Control Mode (Continuous rotation)
 *
 * ── SAFETY FEATURES (ESC) ──────────────────────────────────────────
 *   1. Arming sequence required before any throttle command
 *   2. Soft ramp — throttle always starts from 0, ramps gradually
 *   3. Maximum throttle cap (configurable)
 *   4. Maximum ramp rate per loop step (configurable)
 *   5. Watchdog timeout — motors cut if no command within WATCHDOG_MS
 *   6. Emergency stop ("estop") cuts both ESC motors instantly
 * ───────────────────────────────────────────────────────────────────
 *
 * Serial commands (115200 baud, Newline line ending):
 *
 *   ── ESC ──────────────────────────────────────────────────────────
 *   arm                           Arm both ESCs
 *   disarm                        Disarm ESCs, cut throttle
 *   estop                         EMERGENCY STOP — instant cut to zero
 *   esc:<ch>:<throttle%>          Set ESC throttle 0~100  e.g. "esc:1:50"
 *   esc:both:<throttle%>          Set both ESCs           e.g. "esc:both:60"
 *   esc:status                    Print ESC state
 *
 *   ── Dynamixel (Leveling Platform) ───────────────────────────────
 *   PING                          → PONG
 *   AIM <s1> <s2> <s3>            Move all 3 motors to step positions
 *                                  e.g. "AIM 100 -200 500"
 *   HOME                          Move all motors to step 0 (with offsets)
 *   STATUS                        Print step positions + flags byte
 *   STOP                          Disable/re-enable torque (hold position)
 *   dxl:<id>:on                   Enable torque on motor  e.g. "dxl:2:on"
 *   dxl:<id>:off                  Disable torque on motor e.g. "dxl:1:off"
 *   dxl:all:on                    Enable torque on all DXL motors
 *   dxl:all:off                   Disable torque on all DXL motors
 *   dxl:status                    Print Dynamixel positions (step + raw)
 *
 *   ── Loader (XL330) ───────────────────────────────────────────────
 *   loader:<speed>                Set loader velocity e.g. "loader:150" or "loader:0"
 *
 *   ── General ──────────────────────────────────────────────────────
 *   status                        Print both ESC and Dynamixel status
 */

#include <Servo.h>
#include <Dynamixel2Arduino.h>

// ══════════════════════════════════════════════
//  USER SETTINGS
// ══════════════════════════════════════════════

// ── ESC ──────────────────────────────────────
#define ESC1_PIN          10
#define ESC2_PIN          11

#define THROTTLE_MIN      1000    // µs — zero throttle
#define THROTTLE_MAX      2000    // µs — full throttle
#define THROTTLE_ARM      1000    // µs — arming pulse

#define MAX_THROTTLE_PCT  90      // Hard cap (0~100%)
#define RAMP_RATE_PCT     5       // Max change per 50ms step
#define ARM_DURATION_MS   3000    // Arming pulse duration
#define WATCHDOG_MS       20000    // Cut if no command for this long

// ── Dynamixel (Leveling Platform) ─────────────
static const long     SERIAL_BAUD          = 115200;
static const long     DXL_BAUD             = 1000000;   // 운영 baud (1 Mbps)
static const long     DXL_BAUD_FACTORY     = 57600;     // 출고 baud — auto-upgrade 경로
static const float    DXL_PROTOCOL         = 2.0;

static const int      NUM_MOTORS           = 3;
static const uint8_t  MOTOR_IDS[NUM_MOTORS] = {1, 2, 3};

// 오프셋: 4096 step 기준 90도 = 1024 steps
// 모터 1: +1024(90°), 모터 2: +1024(90°), 모터 3: 0
static const int      OFFSET_STEPS[NUM_MOTORS] = {-50, 0, 0};

static const int      STEP_MIN             = -2048;
static const int      STEP_MAX             = +2047;
static const int32_t  CENTER_RAW           = 2048;

static const unsigned long MOTION_TIMEOUT_MS  = 4000;
static const int32_t       ARRIVED_TOLERANCE  = 10;
static const uint16_t      MOTION_POLL_MS     = 5;

static const uint32_t PROFILE_VEL          = 200;
static const uint32_t PROFILE_ACC          = 50;

// ── Loader Motor (XL330) ──────────────────────
static const uint8_t  LOADER_MOTOR_ID      = 4;

// ══════════════════════════════════════════════

#define DXL_SERIAL   Serial1
#define DEBUG_SERIAL Serial

Servo esc1, esc2;
Dynamixel2Arduino dxl(DXL_SERIAL, -1);
using namespace ControlTableItem;

// ── ESC state ─────────────────────────────────
bool     armed            = false;
bool     estopped         = false;
int      targetPct[3]     = {0, 0, 0};
int      currentPct[3]    = {0, 0, 0};
uint32_t lastCommandTime  = 0;

// ── DXL state ─────────────────────────────────
struct MotorState {
  int32_t target_raw = CENTER_RAW;
  bool    moving     = false;
};
static MotorState g_motor[NUM_MOTORS];
static bool       g_error_latched = false;
static bool       g_homed         = false;

// ══════════════════════════════════════════════
//  DXL Unit Conversion (Offset Applied)
// ══════════════════════════════════════════════

static inline int32_t stepToRaw(int motor_idx, int step) {
  int adjusted = step + OFFSET_STEPS[motor_idx];
  return (int32_t)constrain(adjusted, STEP_MIN, STEP_MAX) + CENTER_RAW;
}

static inline int stepFromRaw(int motor_idx, int32_t raw) {
  return (int)(raw - CENTER_RAW) - OFFSET_STEPS[motor_idx];
}

// ══════════════════════════════════════════════
//  DXL Motor Control
// ══════════════════════════════════════════════

static bool driveMotor(int motor_idx, int target_step) {
  int32_t raw = stepToRaw(motor_idx, target_step);
  if (!dxl.setGoalPosition(MOTOR_IDS[motor_idx], raw)) return false;
  g_motor[motor_idx].target_raw = raw;
  g_motor[motor_idx].moving     = true;
  return true;
}

static bool stopAllDXL() {
  for (int i = 0; i < NUM_MOTORS; ++i) {
    dxl.torqueOff(MOTOR_IDS[i]);
    dxl.torqueOn(MOTOR_IDS[i]);
    g_motor[i].moving = false;
  }
  dxl.setGoalVelocity(LOADER_MOTOR_ID, 0); // 로더 모터도 정지
  return true;
}

// waitMotionComplete 함수는 비동기식 처리를 위해 제거됨
static int currentStep(int motor_idx) {
  int32_t raw = dxl.getPresentPosition(MOTOR_IDS[motor_idx]);
  return stepFromRaw(motor_idx, raw);
}

// ── Flags byte (moving bits 0-2, error bit 3, homed bit 4) ──
static inline uint8_t flagsByte() {
  uint8_t f = 0;
  for (int i = 0; i < NUM_MOTORS; ++i) if (g_motor[i].moving) f |= (1u << i);
  if (g_error_latched) f |= 0x08;
  if (g_homed)         f |= 0x10;
  return f;
}

// ══════════════════════════════════════════════
//  DXL Command Handlers
// ══════════════════════════════════════════════

static void handleAIM(const char *args) {
  int s[NUM_MOTORS];
  if (sscanf(args, "%d %d %d", &s[0], &s[1], &s[2]) != NUM_MOTORS) {
    DEBUG_SERIAL.println("ERR PARSE"); return;
  }
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (s[i] < STEP_MIN || s[i] > STEP_MAX) { DEBUG_SERIAL.println("ERR OUT_OF_RANGE"); return; }
  }
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (!driveMotor(i, s[i])) { DEBUG_SERIAL.println("ERR HW"); return; }
  }
  
  // 비동기 처리: 모터에 명령만 내리고 즉시 OK 응답
  DEBUG_SERIAL.println("OK");
}

static void handleHOME() {
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (!driveMotor(i, 0)) { DEBUG_SERIAL.println("ERR HW"); return; }
  }
  
  // 비동기 처리: 즉시 OK 응답
  g_homed = true;
  g_error_latched = false;
  DEBUG_SERIAL.println("OK");
}

static void replyDXLSTATUS() {
  // STATUS 커맨드: "S <step1> <step2> <step3> <flags>"
  DEBUG_SERIAL.print("S ");
  for (int i = 0; i < NUM_MOTORS; ++i) {
    DEBUG_SERIAL.print(currentStep(i));
    DEBUG_SERIAL.print(' ');
  }
  DEBUG_SERIAL.println(flagsByte());
}

static void printDXLVerbose() {
  // dxl:status 커맨드: 사람이 읽기 쉬운 포맷
  DEBUG_SERIAL.println("  ── Dynamixel Status (Leveling Platform) ──");
  for (int i = 0; i < NUM_MOTORS; ++i) {
    int32_t raw  = dxl.getPresentPosition(MOTOR_IDS[i]);
    int     step = stepFromRaw(i, raw);
    DEBUG_SERIAL.print("  [DXL ID "); DEBUG_SERIAL.print(MOTOR_IDS[i]);
    DEBUG_SERIAL.print("] step: "); DEBUG_SERIAL.print(step);
    DEBUG_SERIAL.print("  raw: "); DEBUG_SERIAL.print(raw);
    DEBUG_SERIAL.print("  moving: "); DEBUG_SERIAL.println(g_motor[i].moving ? "YES" : "NO");
  }
  
  // 로더 모터 상태 출력
  int32_t loader_vel = dxl.getPresentVelocity(LOADER_MOTOR_ID);
  DEBUG_SERIAL.print("  [DXL ID "); DEBUG_SERIAL.print(LOADER_MOTOR_ID);
  DEBUG_SERIAL.print(" (Loader)] Velocity: "); DEBUG_SERIAL.println(loader_vel);
  
  DEBUG_SERIAL.print("  Flags: 0x"); DEBUG_SERIAL.print(flagsByte(), HEX);
  DEBUG_SERIAL.print("  homed: "); DEBUG_SERIAL.print(g_homed ? "YES" : "NO");
  DEBUG_SERIAL.print("  error_latched: "); DEBUG_SERIAL.println(g_error_latched ? "YES" : "NO");
}

// dxl:<id>:on/off  또는  dxl:all:on/off
void handleDXLCommand(String cmd) {
  if (cmd.equalsIgnoreCase("status")) { printDXLVerbose(); return; }

  // ── all:on / all:off ──
  if (cmd.startsWith("all:") || cmd.startsWith("ALL:")) {
    String payload = cmd.substring(4);
    payload.trim();
    if (payload.equalsIgnoreCase("on") || payload.equalsIgnoreCase("off")) {
      bool torqueOn = payload.equalsIgnoreCase("on");
      for (int i = 0; i < NUM_MOTORS; i++)
        torqueOn ? dxl.torqueOn(MOTOR_IDS[i]) : dxl.torqueOff(MOTOR_IDS[i]);
      
      torqueOn ? dxl.torqueOn(LOADER_MOTOR_ID) : dxl.torqueOff(LOADER_MOTOR_ID);
        
      DEBUG_SERIAL.print("  [DXL] All motors torque ");
      DEBUG_SERIAL.println(torqueOn ? "ON" : "OFF");
      DEBUG_SERIAL.println("Ready.");
    } else {
      DEBUG_SERIAL.println("  Unknown dxl:all command. Use: dxl:all:on | dxl:all:off");
    }
    return;
  }

  // ── <id>:on / <id>:off ──
  int colonIdx = cmd.indexOf(':');
  if (colonIdx == -1) { DEBUG_SERIAL.println("  Invalid DXL command."); return; }

  uint8_t id      = (uint8_t)cmd.substring(0, colonIdx).toInt();
  String  payload = cmd.substring(colonIdx + 1);
  payload.trim();

  // ID 유효성 검사
  bool validId = (id == LOADER_MOTOR_ID);
  for (int i = 0; i < NUM_MOTORS; i++) if (MOTOR_IDS[i] == id) { validId = true; break; }
  if (!validId) {
    DEBUG_SERIAL.print("  Unknown DXL ID: "); DEBUG_SERIAL.println(id);
    return;
  }

  if (payload.equalsIgnoreCase("on") || payload.equalsIgnoreCase("off")) {
    bool torqueOn = payload.equalsIgnoreCase("on");
    torqueOn ? dxl.torqueOn(id) : dxl.torqueOff(id);
    DEBUG_SERIAL.print("  [DXL ID "); DEBUG_SERIAL.print(id);
    DEBUG_SERIAL.println(torqueOn ? "] Torque ON" : "] Torque OFF");
    DEBUG_SERIAL.println("Ready.");
    return;
  }

  DEBUG_SERIAL.println("  Unknown DXL command. Use: dxl:<id>:on | dxl:<id>:off | dxl:all:on/off");
}

void handleLoaderCommand(String cmd) {
  int speed = cmd.toInt(); // 문자열이 아니거나 비어있으면 0 반환 (정지)
  dxl.setGoalVelocity(LOADER_MOTOR_ID, speed);
  DEBUG_SERIAL.print("  [Loader] Target Velocity -> ");
  DEBUG_SERIAL.println(speed);
  DEBUG_SERIAL.println("Ready.");
}

// ── 최상위 DXL 커맨드 디스패처 (PING/AIM/HOME/STATUS/STOP) ──
static bool dispatchDXLTopLevel(String &input) {
  // PING
  if (input.equalsIgnoreCase("PING")) {
    DEBUG_SERIAL.println("PONG"); return true;
  }
  // HOME
  if (input.equalsIgnoreCase("HOME")) {
    handleHOME(); return true;
  }
  // STATUS
  if (input.equalsIgnoreCase("STATUS")) {
    replyDXLSTATUS(); return true;
  }
  // STOP
  if (input.equalsIgnoreCase("STOP")) {
    stopAllDXL(); DEBUG_SERIAL.println("OK"); return true;
  }
  // AIM <s1> <s2> <s3>
  if (input.startsWith("AIM ") || input.startsWith("aim ")) {
    handleAIM(input.substring(4).c_str()); return true;
  }
  return false;
}

// ══════════════════════════════════════════════
//  ESC Helpers
// ══════════════════════════════════════════════

int pctToMicros(int pct) {
  pct = constrain(pct, 0, 100);
  return map(pct, 0, 100, THROTTLE_MIN, THROTTLE_MAX);
}

void writeESC(uint8_t ch, int pct) {
  pct = constrain(pct, 0, MAX_THROTTLE_PCT);
  currentPct[ch] = pct;
  int us = pctToMicros(pct);
  if (ch == 1) esc1.writeMicroseconds(us);
  if (ch == 2) esc2.writeMicroseconds(us);
}

void cutMotors() {
  writeESC(1, 0);
  writeESC(2, 0);
  targetPct[1] = 0;
  targetPct[2] = 0;
}

void armESCs() {
  DEBUG_SERIAL.println("  [ESC] Arming — do NOT apply throttle...");
  esc1.writeMicroseconds(THROTTLE_ARM);
  esc2.writeMicroseconds(THROTTLE_ARM);
  delay(ARM_DURATION_MS);
  armed    = true;
  estopped = false;
  lastCommandTime = millis();
  DEBUG_SERIAL.println("  [ESC] Armed. Ready for throttle.");
  DEBUG_SERIAL.println("Ready.");
}

void disarmESCs() {
  cutMotors();
  armed = false;
  DEBUG_SERIAL.println("  [ESC] Disarmed. Send 'arm' to re-enable.");
  DEBUG_SERIAL.println("Ready.");
}

void emergencyStop() {
  cutMotors();
  estopped = true;
  armed    = false;
  DEBUG_SERIAL.println("  !! EMERGENCY STOP — all ESC motors cut.");
  DEBUG_SERIAL.println("  Send 'arm' to re-arm after resolving the issue.");
}

void printESCStatus() {
  DEBUG_SERIAL.println("  ── ESC Status ──");
  DEBUG_SERIAL.print  ("  Armed    : "); DEBUG_SERIAL.println(armed    ? "YES" : "NO");
  DEBUG_SERIAL.print  ("  E-Stopped: "); DEBUG_SERIAL.println(estopped ? "YES" : "NO");
  for (uint8_t ch = 1; ch <= 2; ch++) {
    DEBUG_SERIAL.print("  [ESC ");
    DEBUG_SERIAL.print(ch);
    DEBUG_SERIAL.print("] Target: ");
    DEBUG_SERIAL.print(targetPct[ch]);
    DEBUG_SERIAL.print("%  Current: ");
    DEBUG_SERIAL.print(currentPct[ch]);
    DEBUG_SERIAL.print("%  (");
    DEBUG_SERIAL.print(pctToMicros(currentPct[ch]));
    DEBUG_SERIAL.println("µs)");
  }
}

void handleESCCommand(String cmd) {
  if (cmd.equalsIgnoreCase("status")) { printESCStatus(); return; }

  if (cmd.startsWith("both:") || cmd.startsWith("BOTH:")) {
    int pct = cmd.substring(5).toInt();
    if (pct < 0 || pct > 100) { DEBUG_SERIAL.println("  Out of range. Use 0~100."); return; }
    if (!armed) { DEBUG_SERIAL.println("  NOT ARMED. Send 'arm' first."); return; }
    pct = constrain(pct, 0, MAX_THROTTLE_PCT);
    targetPct[1] = pct;
    targetPct[2] = pct;
    DEBUG_SERIAL.print("  [ESC] Both → "); DEBUG_SERIAL.print(pct); DEBUG_SERIAL.println("%");
    DEBUG_SERIAL.println("Ready.");
    return;
  }

  int colonIdx = cmd.indexOf(':');
  if (colonIdx != -1) {
    uint8_t ch  = (uint8_t)cmd.substring(0, colonIdx).toInt();
    int     pct = cmd.substring(colonIdx + 1).toInt();
    if (ch != 1 && ch != 2) { DEBUG_SERIAL.println("  Invalid ESC channel. Use 1 or 2."); return; }
    if (pct < 0 || pct > 100) { DEBUG_SERIAL.println("  Out of range. Use 0~100."); return; }
    if (!armed) { DEBUG_SERIAL.println("  NOT ARMED. Send 'arm' first."); return; }
    pct = constrain(pct, 0, MAX_THROTTLE_PCT);
    targetPct[ch] = pct;
    DEBUG_SERIAL.print("  [ESC "); DEBUG_SERIAL.print(ch);
    DEBUG_SERIAL.print("] Target → "); DEBUG_SERIAL.print(pct); DEBUG_SERIAL.println("%");
    DEBUG_SERIAL.println("Ready.");
    return;
  }

  DEBUG_SERIAL.println("  Unknown ESC command. Use: esc:<ch>:<pct> | esc:both:<pct> | esc:status");
}

// ══════════════════════════════════════════════
//  DXL Baud Auto-Upgrade
// ══════════════════════════════════════════════

/**
 * ensureDxlBaud — 부팅 시 모터 baud 를 DXL_BAUD 에 맞춤.
 *   1) target baud 로 ping → 성공이면 즉시 통과 (이후 부팅의 정상 경로).
 *   2) factory baud 로 fallback → 발견되는 모터의 EEPROM 에 target baud 기입.
 *   3) target baud 로 재연결.
 *   EEPROM 기입은 1회성 — 이후 부팅부터는 (1) 에서 통과.
 */
static bool ensureDxlBaud() {
  const uint8_t all_ids[] = {MOTOR_IDS[0], MOTOR_IDS[1], MOTOR_IDS[2], LOADER_MOTOR_ID};
  const uint8_t n = sizeof(all_ids) / sizeof(all_ids[0]);

  dxl.begin(DXL_BAUD);
  for (uint8_t i = 0; i < n; i++) {
    if (dxl.ping(all_ids[i])) return true;
  }

  dxl.begin(DXL_BAUD_FACTORY);
  bool any_found = false;
  for (uint8_t i = 0; i < n; i++) {
    if (!dxl.ping(all_ids[i])) continue;
    any_found = true;
    dxl.torqueOff(all_ids[i]);
    dxl.setBaudrate(all_ids[i], DXL_BAUD);  // EEPROM write
    delay(20);
  }
  if (!any_found) return false;

  delay(100);
  dxl.begin(DXL_BAUD);
  for (uint8_t i = 0; i < n; i++) {
    if (dxl.ping(all_ids[i])) return true;
  }
  return false;
}

// ══════════════════════════════════════════════
//  Setup
// ══════════════════════════════════════════════

void setup() {
  DEBUG_SERIAL.begin(SERIAL_BAUD);
  while (!DEBUG_SERIAL && millis() < 3000);

  // ── ESC 초기화 ──
  esc1.attach(ESC1_PIN, THROTTLE_MIN, THROTTLE_MAX);
  esc2.attach(ESC2_PIN, THROTTLE_MIN, THROTTLE_MAX);
  esc1.writeMicroseconds(THROTTLE_MIN);
  esc2.writeMicroseconds(THROTTLE_MIN);

  DEBUG_SERIAL.println("=== Integrated Controller: ESC + Dynamixel (Leveling Platform) + Loader ===");
  DEBUG_SERIAL.println();

  DEBUG_SERIAL.println("[ESC]");
  DEBUG_SERIAL.print  ("  Max throttle cap : "); DEBUG_SERIAL.print(MAX_THROTTLE_PCT); DEBUG_SERIAL.println("%");
  DEBUG_SERIAL.print  ("  Ramp rate        : "); DEBUG_SERIAL.print(RAMP_RATE_PCT);    DEBUG_SERIAL.println("% per step");
  DEBUG_SERIAL.print  ("  Watchdog timeout : "); DEBUG_SERIAL.print(WATCHDOG_MS);      DEBUG_SERIAL.println("ms");
  DEBUG_SERIAL.println("  ⚠️  Send 'arm' before sending any throttle.");
  DEBUG_SERIAL.println();

  // ── Dynamixel 초기화 ──
  DEBUG_SERIAL.println("[Dynamixel — Leveling Platform]");
  dxl.setPortProtocolVersion(DXL_PROTOCOL);
  if (!ensureDxlBaud()) {
    DEBUG_SERIAL.println("  ERROR: No DXL motor responded at 1Mbps or 57600.");
    DEBUG_SERIAL.println("  Check power (12V), TTL wiring, and motor IDs.");
    g_error_latched = true;
  }

  for (int i = 0; i < NUM_MOTORS; i++) {
    uint8_t id = MOTOR_IDS[i];
    if (!dxl.ping(id)) {
      DEBUG_SERIAL.print("  ERROR: Cannot find motor ID "); DEBUG_SERIAL.println(id);
      g_error_latched = true;
    } else {
      DEBUG_SERIAL.print("  Found motor ID "); DEBUG_SERIAL.println(id);
      dxl.torqueOff(id);
      dxl.setOperatingMode(id, OP_POSITION);
      dxl.writeControlTableItem(PROFILE_VELOCITY,     id, PROFILE_VEL);
      dxl.writeControlTableItem(PROFILE_ACCELERATION, id, PROFILE_ACC);
      dxl.torqueOn(id);
      g_motor[i].target_raw = dxl.getPresentPosition(id);
    }
  }
  
  // ── Loader Motor (XL330) 초기화 ──
  DEBUG_SERIAL.println();
  DEBUG_SERIAL.println("[Dynamixel — Loader Motor (XL330)]");
  if (!dxl.ping(LOADER_MOTOR_ID)) {
    DEBUG_SERIAL.print("  ERROR: Cannot find Loader motor ID "); DEBUG_SERIAL.println(LOADER_MOTOR_ID);
  } else {
    DEBUG_SERIAL.print("  Found Loader motor ID "); DEBUG_SERIAL.println(LOADER_MOTOR_ID);
    dxl.torqueOff(LOADER_MOTOR_ID);
    // 속도 제어 모드(Velocity Control Mode)로 설정
    dxl.setOperatingMode(LOADER_MOTOR_ID, OP_VELOCITY);
    dxl.torqueOn(LOADER_MOTOR_ID);
    DEBUG_SERIAL.println("  Operating Mode set to OP_VELOCITY.");
  }

  DEBUG_SERIAL.println();
  DEBUG_SERIAL.println("Ready.");
  DEBUG_SERIAL.println("Commands:");
  DEBUG_SERIAL.println("  ESC    : arm | disarm | estop | esc:<ch>:<pct> | esc:both:<pct> | esc:status");
  DEBUG_SERIAL.println("  Level  : PING | AIM <s1> <s2> <s3> | HOME | STATUS | STOP");
  DEBUG_SERIAL.println("  Loader : loader:<speed>  (e.g., loader:150, loader:-150, loader:0)");
  DEBUG_SERIAL.println("  DXL    : dxl:<id>:on/off | dxl:all:on/off | dxl:status");
  DEBUG_SERIAL.println("  GEN    : status");
}

// ══════════════════════════════════════════════
//  Main Loop
// ══════════════════════════════════════════════

void loop() {
  uint32_t now = millis();

  // ── 워치독: 명령 없으면 ESC 차단 ──
  if (armed && (now - lastCommandTime > WATCHDOG_MS)) {
    DEBUG_SERIAL.println("  WATCHDOG: No command received. Cutting ESC motors.");
    cutMotors();
    armed = false;
  }

  // ── 비동기 다이나믹셀 상태 업데이트 (100ms 간격) ──
  static uint32_t lastPoll = 0;
  if (now - lastPoll > 100) {
    lastPoll = now;
    for (int i = 0; i < NUM_MOTORS; ++i) {
      if (g_motor[i].moving) {
        bool hw_moving = (bool)dxl.readControlTableItem(MOVING, MOTOR_IDS[i]);
        int32_t present = dxl.getPresentPosition(MOTOR_IDS[i]);
        bool arrived = (abs(present - g_motor[i].target_raw) <= ARRIVED_TOLERANCE);
        if (!hw_moving || arrived) {
          g_motor[i].moving = false;
        }
      }
    }
  }

  // ── 소프트 램프: 목표 스로틀로 서서히 증가 ──
  for (uint8_t ch = 1; ch <= 2; ch++) {
    if (currentPct[ch] != targetPct[ch]) {
      int diff = targetPct[ch] - currentPct[ch];
      int step = constrain(diff, -RAMP_RATE_PCT, RAMP_RATE_PCT);
      writeESC(ch, currentPct[ch] + step);
    }
  }

  // ── Serial 입력 ──
  if (!DEBUG_SERIAL.available()) {
    delay(50);
    return;
  }

  String input = DEBUG_SERIAL.readStringUntil('\n');
  input.trim();
  if (input.length() == 0) return;

  lastCommandTime = now;

  // ── 최상위 커맨드 ──────────────────────────

  // estop — 최우선
  if (input.equalsIgnoreCase("estop")) { emergencyStop(); return; }

  // arm / disarm
  if (input.equalsIgnoreCase("arm"))    { armESCs();    return; }
  if (input.equalsIgnoreCase("disarm")) { disarmESCs(); return; }

  // status — ESC + DXL 동시 출력
  if (input.equalsIgnoreCase("status")) {
    printESCStatus();
    printDXLVerbose();
    return;
  }

  // ── DXL 최상위 커맨드 (PING / AIM / HOME / STATUS / STOP) ──
  if (dispatchDXLTopLevel(input)) return;

  // ── Loader 서브시스템 ──
  if (input.startsWith("loader:") || input.startsWith("LOADER:")) {
    handleLoaderCommand(input.substring(7));
    return;
  }

  // ── ESC 서브시스템 ──
  if (input.startsWith("esc:") || input.startsWith("ESC:")) {
    handleESCCommand(input.substring(4));
    return;
  }

  // ── DXL 서브시스템 (torque on/off 등) ──
  if (input.startsWith("dxl:") || input.startsWith("DXL:")) {
    handleDXLCommand(input.substring(4));
    return;
  }

  DEBUG_SERIAL.println("  Unknown command.");
  DEBUG_SERIAL.println("  Use: arm | disarm | estop | status");
  DEBUG_SERIAL.println("       PING | AIM <s1> <s2> <s3> | HOME | STATUS | STOP");
  DEBUG_SERIAL.println("       loader:<speed> | esc:<cmd> | dxl:<cmd>");
}
