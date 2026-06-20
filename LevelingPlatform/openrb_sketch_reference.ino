/*
 * Leveling Platform Controller — OpenRB-150 reference sketch.
 *
 * For the serial protocol with the Pi5 (LevelingMotorClient), see the
 * header docstring of [leveling_motor.py](leveling_motor.py).
 *
 * This sketch only provides the *protocol parsing/dispatch skeleton*.
 * The actual motor control (writing Dynamixel goalPosition, detecting motion
 * completion, etc.) should be filled in where MARK [TODO-FW] appears.
 *
 * Build env
 *   - Arduino IDE 2.x  +  OpenRB-150 board package
 *   - Library: DYNAMIXEL2Arduino (for motor control, to be filled in by a teammate)
 *   - Serial : USB CDC, 115200 8N1
 *
 * Wiring assumption
 *   - 3 motors (ID 1, 2, 3) daisy-chained on the OpenRB-150 DYNAMIXEL port.
 *   - 4096 step / rev per motor (XL330 / XM430, etc.).
 *
 * Protocol summary  (all lines '\n' terminated, 7-bit ASCII)
 *
 *   Pi → OpenRB                    OpenRB → Pi
 *   ───────────                    ──────────
 *   PING                           PONG
 *   AIM <s1> <s2> <s3>             OK   |  ERR <reason>
 *   HOME                           OK   |  ERR <reason>
 *   STATUS                         S <s1> <s2> <s3> <flags>
 *   STOP                           OK
 *
 *   flags bits: 0=motor1 moving, 1=2, 2=3, 3=error, 4=homed
 */

#include <Arduino.h>

// MARK [TODO-FW] — actual motor library
// #include <Dynamixel2Arduino.h>
// Dynamixel2Arduino dxl(Serial1, /*DIR_PIN=*/-1);

// ───────────────────────────────────────────────
//   Config
// ───────────────────────────────────────────────
static const long   SERIAL_BAUD       = 115200;
static const int    NUM_MOTORS        = 3;
static const int    MOTOR_IDS[NUM_MOTORS] = {1, 2, 3};
static const int    STEP_MIN          = -2048;
static const int    STEP_MAX          = +2047;
static const unsigned long MOTION_TIMEOUT_MS = 4000;
static const int    READBUF_LEN       = 64;     // max length of one line

// ───────────────────────────────────────────────
//   State (last absolute step commanded by IK + flags)
// ───────────────────────────────────────────────
struct MotorState {
  int  target_step  = 0;
  bool moving       = false;
};
static MotorState  g_motor[NUM_MOTORS];
static bool        g_error_latched = false;
static bool        g_homed         = false;
static char        g_buf[READBUF_LEN];
static uint8_t     g_buf_len = 0;

// ───────────────────────────────────────────────
//   Reply helpers
// ───────────────────────────────────────────────
static inline void replyOK()                         { Serial.println("OK"); }
static inline void replyPONG()                       { Serial.println("PONG"); }
static inline void replyERR(const char *reason)      {
  Serial.print("ERR ");
  Serial.println(reason);
}
static inline uint8_t flagsByte() {
  uint8_t f = 0;
  for (int i = 0; i < NUM_MOTORS; ++i) if (g_motor[i].moving) f |= (1u << i);
  if (g_error_latched) f |= 0x08;
  if (g_homed)         f |= 0x10;
  return f;
}
static void replySTATUS() {
  Serial.print("S ");
  for (int i = 0; i < NUM_MOTORS; ++i) {
    Serial.print(currentStep(i));   // MARK [TODO-FW]: read actual position
    Serial.print(' ');
  }
  Serial.println(flagsByte());
}

// ───────────────────────────────────────────────
//   MARK [TODO-FW] — motor control (to be filled in by a teammate)
// ───────────────────────────────────────────────
static int currentStep(int motor_idx) {
  // Read via dxl.getPresentPosition(MOTOR_IDS[motor_idx]) or similar,
  // convert to the signed step domain (-2048..+2047), and return it.
  // Temporary stub: return the last commanded position as-is.
  return g_motor[motor_idx].target_step;
}

static bool driveMotor(int motor_idx, int target_step) {
  // dxl.setGoalPosition(MOTOR_IDS[motor_idx], target_step + offset, UNIT_RAW)
  // Just issue the command and return true immediately (waitMotionComplete handles waiting for motion completion).
  g_motor[motor_idx].target_step = target_step;
  g_motor[motor_idx].moving      = true;
  return true;
}

static bool stopAllMotors() {
  // Stop immediately via dxl.torqueOff(MOTOR_IDS[i]) or similar.
  for (int i = 0; i < NUM_MOTORS; ++i) g_motor[i].moving = false;
  return true;
}

static bool waitMotionComplete(unsigned long timeout_ms) {
  // Poll until all motors reach their goal (or use dxl.getMoving()).
  unsigned long t0 = millis();
  while (millis() - t0 < timeout_ms) {
    bool any = false;
    for (int i = 0; i < NUM_MOTORS; ++i) {
      // bool moving_i = dxl.readControlTableItem(MOVING, MOTOR_IDS[i]);
      bool moving_i = false;            // stub: treat as completed immediately
      g_motor[i].moving = moving_i;
      any |= moving_i;
    }
    if (!any) return true;
    delay(2);
  }
  // timeout
  stopAllMotors();
  g_error_latched = true;
  return false;
}

// ───────────────────────────────────────────────
//   Command handlers
// ───────────────────────────────────────────────
static void handleAIM(const char *args) {
  int s[NUM_MOTORS];
  if (sscanf(args, "%d %d %d", &s[0], &s[1], &s[2]) != NUM_MOTORS) {
    replyERR("PARSE");
    return;
  }
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (s[i] < STEP_MIN || s[i] > STEP_MAX) { replyERR("OUT_OF_RANGE"); return; }
  }
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (!driveMotor(i, s[i])) { replyERR("HW"); return; }
  }
  if (!waitMotionComplete(MOTION_TIMEOUT_MS)) {
    replyERR("TIMEOUT");
    return;
  }
  replyOK();
}

static void handleHOME() {
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (!driveMotor(i, 0)) { replyERR("HW"); return; }
  }
  if (!waitMotionComplete(MOTION_TIMEOUT_MS)) {
    replyERR("TIMEOUT");
    return;
  }
  g_homed = true;
  replyOK();
}

static void handleSTOP() {
  stopAllMotors();
  replyOK();
}

static void dispatch(char *line) {
  // line has the '\n' stripped and is 0-terminated.
  if      (strncmp(line, "PING",  4) == 0)  replyPONG();
  else if (strncmp(line, "AIM ",  4) == 0)  handleAIM(line + 4);
  else if (strncmp(line, "HOME",  4) == 0)  handleHOME();
  else if (strncmp(line, "STATUS", 6) == 0) replySTATUS();
  else if (strncmp(line, "STOP",  4) == 0)  handleSTOP();
  else replyERR("PARSE");
}

// ───────────────────────────────────────────────
//   Setup / loop
// ───────────────────────────────────────────────
void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) {/* wait for USB-CDC ready */ }

  // MARK [TODO-FW] — motor initialization (Dynamixel TTL comms, torque on, etc.)
  // dxl.begin(1000000);
  // dxl.setPortProtocolVersion(2.0);
  // for (int i = 0; i < NUM_MOTORS; ++i) {
  //   dxl.torqueOff(MOTOR_IDS[i]);
  //   dxl.setOperatingMode(MOTOR_IDS[i], OP_POSITION);
  //   dxl.torqueOn(MOTOR_IDS[i]);
  // }
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;                      // ignore CR
    if (c == '\n') {
      g_buf[g_buf_len] = '\0';
      if (g_buf_len > 0) dispatch(g_buf);
      g_buf_len = 0;
    } else if (g_buf_len < READBUF_LEN - 1) {
      g_buf[g_buf_len++] = c;
    } else {
      // overflow → discard line + ERR
      g_buf_len = 0;
      replyERR("OVERFLOW");
    }
  }
}
