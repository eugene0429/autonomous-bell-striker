// ======================================================
// OpenRB + MDD10A
// 2 Encoder Motors (encoder signals NOT used)
// Arrow-key teleop sketch
//
// Physical Motor on D8/D7  = R motor
// Physical Motor on A3/A4  = L motor
//
// Controls:
//   Up    / W : forward
//   Down  / X : backward
//   Left  / A : tank turn left
//   Right / D : tank turn right
//   Space / S : stop
//   + / =     : speed up (step)
//   - / _     : speed down (step)
//   P         : print state
//   H / ?     : help
//
// Notes:
//   Arrow keys are read as ANSI escape sequences (ESC [ A/B/C/D),
//   which require a raw-mode serial terminal (screen, minicom,
//   PuTTY raw, etc.). The Arduino IDE Serial Monitor does NOT send
//   raw key presses, so use the WASD/X fallback there.
// ======================================================


// ---------- Right motor pins: physical motor on D8/D7 ----------
const int R_PWM = 8;    // OpenRB D8 -> MDD10A PWM
const int R_DIR = 7;    // OpenRB D7 -> MDD10A DIR

// ---------- Left motor pins: physical motor on A3/A4 ----------
const int L_PWM = A3;   // OpenRB A3 -> MDD10A PWM
const int L_DIR = A4;   // OpenRB A4 -> MDD10A DIR


// ---------- Direction settings ----------
// If a motor rotates opposite to expected direction, change HIGH to LOW.
const bool L_FORWARD_DIR = HIGH;
const bool R_FORWARD_DIR = HIGH;


// ---------- Safety / speed ----------
const int PWM_LIMIT   = 120;  // hard cap, never exceed
const int PWM_STEP    = 10;   // +/- step
const int PWM_DEFAULT = 60;   // initial cruise PWM


// ---------- State ----------
int currentSpeed = PWM_DEFAULT;
int lPWMCommand = 0;
int rPWMCommand = 0;


// ======================================================
// Motor primitives
// ======================================================
void setMotor(int pwmPin, int dirPin, int value, bool forwardDir) {
  value = constrain(value, -PWM_LIMIT, PWM_LIMIT);

  if (value > 0) {
    digitalWrite(dirPin, forwardDir);
    analogWrite(pwmPin, value);
  } else if (value < 0) {
    digitalWrite(dirPin, !forwardDir);
    analogWrite(pwmPin, -value);
  } else {
    analogWrite(pwmPin, 0);
  }
}

void setLeft(int value) {
  value = constrain(value, -PWM_LIMIT, PWM_LIMIT);
  lPWMCommand = value;
  setMotor(L_PWM, L_DIR, value, L_FORWARD_DIR);
}

void setRight(int value) {
  value = constrain(value, -PWM_LIMIT, PWM_LIMIT);
  rPWMCommand = value;
  setMotor(R_PWM, R_DIR, value, R_FORWARD_DIR);
}

void stopAll() {
  setLeft(0);
  setRight(0);
}


// ======================================================
// High-level motions
// ======================================================
void moveForward()  { setLeft( currentSpeed); setRight( currentSpeed); }
void moveBackward() { setLeft(-currentSpeed); setRight(-currentSpeed); }
void turnLeft()     { setLeft(-currentSpeed); setRight( currentSpeed); }
void turnRight()    { setLeft( currentSpeed); setRight(-currentSpeed); }


// ======================================================
// Speed adjustment
// ======================================================
void changeSpeed(int delta) {
  int prev = currentSpeed;
  currentSpeed = constrain(currentSpeed + delta, 0, PWM_LIMIT);

  // Re-apply current heading at the new speed if already moving.
  if (lPWMCommand > 0 && rPWMCommand > 0)      moveForward();
  else if (lPWMCommand < 0 && rPWMCommand < 0) moveBackward();
  else if (lPWMCommand < 0 && rPWMCommand > 0) turnLeft();
  else if (lPWMCommand > 0 && rPWMCommand < 0) turnRight();

  Serial.print("SPEED,");
  Serial.print(prev);
  Serial.print("->");
  Serial.println(currentSpeed);
}


// ======================================================
// Reporting
// ======================================================
void printState() {
  Serial.print("STATE,L=");
  Serial.print(lPWMCommand);
  Serial.print(",R=");
  Serial.print(rPWMCommand);
  Serial.print(",SPEED=");
  Serial.println(currentSpeed);
}

void printHelp() {
  Serial.println();
  Serial.println("====================================");
  Serial.println("OpenRB + MDD10A  Arrow-key teleop");
  Serial.println("------------------------------------");
  Serial.println("Pins:");
  Serial.println("  Left  motor: PWM=A3, DIR=A4");
  Serial.println("  Right motor: PWM=D8, DIR=D7");
  Serial.println("------------------------------------");
  Serial.println("Keys:");
  Serial.println("  Up    / W : forward");
  Serial.println("  Down  / X : backward");
  Serial.println("  Left  / A : turn left");
  Serial.println("  Right / D : turn right");
  Serial.println("  Space / S : stop");
  Serial.println("  + / -     : speed up / down");
  Serial.println("  P         : print state");
  Serial.println("  H         : help");
  Serial.println("====================================");
  Serial.println();
}


// ======================================================
// Input handling: ANSI escape state machine + key fallback
// ======================================================
enum EscState { ESC_NONE, ESC_GOT_ESC, ESC_GOT_BRACKET };
EscState escState = ESC_NONE;

void handleArrow(char code) {
  switch (code) {
    case 'A': moveForward();  Serial.println("CMD,UP");    break;
    case 'B': moveBackward(); Serial.println("CMD,DOWN");  break;
    case 'C': turnRight();    Serial.println("CMD,RIGHT"); break;
    case 'D': turnLeft();     Serial.println("CMD,LEFT");  break;
    default: break;
  }
}

void handleKey(char c) {
  switch (c) {
    case 'w': case 'W':
      moveForward();  Serial.println("CMD,FWD");  break;
    case 'x': case 'X':
      moveBackward(); Serial.println("CMD,BACK"); break;
    case 'a': case 'A':
      turnLeft();     Serial.println("CMD,LEFT"); break;
    case 'd': case 'D':
      turnRight();    Serial.println("CMD,RIGHT");break;
    case 's': case 'S':
    case ' ':
      stopAll();      Serial.println("CMD,STOP"); break;
    case '+': case '=':
      changeSpeed(+PWM_STEP); break;
    case '-': case '_':
      changeSpeed(-PWM_STEP); break;
    case 'p': case 'P':
      printState(); break;
    case 'h': case 'H': case '?':
      printHelp(); break;
    case '\r': case '\n':
      break;  // ignore line endings
    default:
      // Unknown key: silent. Avoids spam from terminal control bytes.
      break;
  }
}

void handleByte(int b) {
  if (b < 0) return;

  if (escState == ESC_GOT_ESC) {
    if (b == '[') { escState = ESC_GOT_BRACKET; return; }
    // ESC followed by something other than '[' -> abort sequence,
    // treat current byte as a normal key.
    escState = ESC_NONE;
    handleKey((char)b);
    return;
  }

  if (escState == ESC_GOT_BRACKET) {
    escState = ESC_NONE;
    handleArrow((char)b);
    return;
  }

  if (b == 0x1B) {  // ESC
    escState = ESC_GOT_ESC;
    return;
  }

  handleKey((char)b);
}


// ======================================================
// Setup
// ======================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(L_PWM, OUTPUT);
  pinMode(L_DIR, OUTPUT);
  pinMode(R_PWM, OUTPUT);
  pinMode(R_DIR, OUTPUT);

  stopAll();

  printHelp();
}


// ======================================================
// Loop
// ======================================================
void loop() {
  while (Serial.available() > 0) {
    handleByte(Serial.read());
  }
}
