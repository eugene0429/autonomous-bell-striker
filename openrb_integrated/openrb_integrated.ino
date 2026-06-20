/**
 * ============================================================
 *  openrb_integrated.ino
 *  Pi5 ↔ OpenRB-150 integrated firmware (protocol v1)
 *
 *  Actuators
 *    Wheels XC430 ×2 : ID 6 (LEFT), ID 7 (RIGHT)  — Velocity Mode
 *    Leveling DXL ×3 : ID 1, 2, 3                 — Position Mode
 *    Camera tilt     : ID 4                        — Position Mode
 *    Feeder (loader) : ID 5                        — Position Mode
 *    T-motor ESC ×2 : PWM 1000~2000 µs (Servo library)
 *
 *  Communication
 *    USB CDC 115200 8N1, line-based ASCII ≤64 bytes
 *    DRIVE → fire-and-forget (no reply), 200 ms watchdog
 *    others → sync (single-line reply)
 *
 *  Velocity unit conversion
 *    Protocol : mrad/s  (signed int, ±30000)
 *    XC430    : velocity unit = 0.229 rpm/unit
 *               mrad/s → unit = mrad/s × 0.041576
 *                 (= 1/1000 / (2π/60) / 0.229)
 *    Sign     : ID6 forward = positive unit  (WHEEL_DIR_SIGN_L = +1)
 *               ID7 forward = negative unit  (WHEEL_DIR_SIGN_R = -1)
 *
 *  Pin assignment
 *    T-motor TOP    : pin 9
 *    T-motor BOTTOM : pin 10
 *    (wheels are on the DXL TTL bus — no separate GPIO)
 *
 *  Dependencies
 *    Dynamixel2Arduino  (ROBOTIS)
 *    Servo              (Arduino built-in)
 *
 *  Change history
 *    v2.7 : Loader (ID 5) vibration reduction — velocity P↓ + I↑ rebalance.
 *           - LOADER_VEL_P_GAIN 2000 → 400, LOADER_VEL_I_GAIN(new) = 3200.
 *           - Symptom: enough force to lift the ball, but excessive vibration
 *             (judder) while rotating. Cause was the P-gain (=2000) pushed
 *             extremely high to break through stall, which also made PWM
 *             swing wildly on every velocity ripple during rotation.
 *           - Fix: lower P to remove oscillation, and boost I (integral
 *             wind-up) to handle the static-friction breakthrough instead.
 *             VELOCITY_I_GAIN is now written every time in both loaderRotateBy
 *             and initLoaderVelMotor (guarding against EEPROM default revert
 *             on reboot).
 *    v2.6 : FF ↓ + P ↑ (further suppress arrival wobble + reinforce static sag)
 *           - DXL_FEEDFORWARD_ACC_GAIN 280 → 220, DXL_POSITION_P_GAIN 1500 →
 *             1800. D kept at 600.
 *           - Symptom: v2.5's (FF=280, D=600) reduced the wobble but did not
 *             fully eliminate it + slight sag from weight observed after stop.
 *           - Fix:
 *               1) Lower FF one more step (220) to further reduce the PWM
 *                  impulse strength caused by trajectory acc step-change.
 *               2) Raise P 1500 → 1800 for stronger holding torque even on
 *                  the small position error at stop. Directly compensates sag.
 *           - Balance: FF↓ and P↑ do not conflict. FF supplements PWM during
 *             the trajectory-tracking phase, P provides steady-state holding
 *             stiffness — different acting phases.
 *
 *    v2.5 : FF/D gain balance tuning (catch the wobble near arrival)
 *           - DXL_FEEDFORWARD_ACC_GAIN 400 → 280, DXL_POSITION_D_GAIN 400 → 600.
 *           - Symptom: v2.4's FF 200 lacked smoothness → raising it to 400
 *             produced overshoot + slight wobble after arrival. Because FF
 *             turns the step-change of trajectory acceleration directly into a
 *             PWM impulse, stronger FF triggers ringing near arrival.
 *           - Fix: lower FF one step to reduce impulse strength, and raise D
 *             one step to absorb the residual impulse. New P/D/FF balance
 *             point (P=1500, D=600, FF=280).
 *           - Variability: the sweet spot shifts within a ±20% range depending
 *             on weight and mechanical characteristics. If it wobbles again,
 *             lower FF further to 220; if it gets sluggish, nudge FF back up
 *             to 320.
 *
 *    v2.4 : Introduce feedforward ACC gain (smooth micro-trajectory transitions)
 *           - FEEDFORWARD_2ND_GAIN applied uniformly, default 0 → 200.
 *             Added ff_acc_gain parameter to initPosMotor.
 *           - Symptom: v2.3's D gain caught the ringing, but the "stepping
 *             feel" was severe especially when descending under load (gravity
 *             helping). Each 16 ms AIMF tick's micro-trajectory arrived faster
 *             than expected due to gravity assist → before P/D could react the
 *             next trajectory started fresh → the same start-end repeated every
 *             cycle, felt as a step.
 *           - Fix: add the trajectory generator's desired acceleration directly
 *             to PWM to compensate ahead of time, before P/D catch up. The
 *             transition between micro-trajectories becomes smooth.
 *           - 200 is a conservative starting value. Can be raised to 400~800 if
 *             insufficient. If stronger response is needed, also introduce
 *             FEEDFORWARD_1ST_GAIN (velocity FF).
 *
 *    v2.3 : Introduce position D gain (suppress overshoot / ringing)
 *           - POSITION_D_GAIN applied uniformly, default 0 → 400.
 *             Added position_d_gain parameter to initPosMotor.
 *           - Symptom: with v2.2's raised PROFILE_VEL/ACC + v2.1's P gain 1500,
 *             responsiveness improved but P=1500/D=0 is underdamped, so ringing
 *             occurred under weight load. When a new trajectory started every
 *             AIMF tick (16 ms), the previous ringing had not fully died and
 *             acted as an impulse → felt as "stepping + vibration".
 *           - Fix: add -velocity feedback via the D term → immediate damping of
 *             overshoot. 400 gives D/P ≈ 0.27, slightly on the over-damped
 *             side. Vibration clearly reduced.
 *           - Can be raised to 600~800 if insufficient. But too high amplifies
 *             encoder noise into high-frequency chatter — check by ear.
 *
 *    v2.2 : Raise Position Mode profile acceleration (ease AIMF streaming stutter)
 *           - DXL_PROFILE_VEL 400 → 700, DXL_PROFILE_ACC 80 → 250.
 *           - Symptom: stutter / jerky tracking under weight load when 60 Hz
 *             AIMF streaming from a GUI drag. Cause: with ACC=80 the time to
 *             accelerate to max velocity is ~320 ms, so within each 16 ms tick
 *             (= new goal arrival) the acceleration phase could not finish and
 *             got overwritten by the next trajectory, repeating the same
 *             initial motion each time.
 *           - With ACC 250 the time to reach max vel is ~180 ms. The
 *             acceleration phase takes a larger fraction of one tick, so it
 *             enters the normal-speed region → smooth tracking. VEL also raised
 *             to 700 so a large goal-gap is partially covered within one cycle.
 *           - If insufficient, raise ACC further to 300~500, and if it hums /
 *             vibrates add a separate POSITION_D_GAIN (default 0) of 400~600.
 *
 *    v2.1 : Strengthen Position Mode P gain + formalize STATUS reply format
 *           - Dynamixel POSITION_P_GAIN raised uniformly from default 800 →
 *             1500 (Kp_eff = P/128: 6.25 → 11.7). Applied to LVL_1/2/3 + TILT.
 *             Added position_p_gain parameter to initPosMotor (default = new
 *             constant DXL_POSITION_P_GAIN).
 *           - Symptom: on HOME the residual error could not get within
 *             DXL_ARRIVED_TOL=10 (~0.88°), so waitMotion → ERR TIMEOUT repeated.
 *             Via STATUS, residual-error boundary observed BEFORE
 *             (-201, -102, 305) → AFTER (-6, -2, 5).
 *           - Fix: raise P gain to converge residual error within tolerance. If
 *             insufficient, DXL_POSITION_P_GAIN can be raised further to
 *             2000~3000.
 *           - The STATUS reply format had grown to 11 fields
 *             (S wL wR s1 s2 s3 s4 s5 rpmT rpmB flags) with the v1.2/v1.4 wheel
 *             and loader additions, but the leveling_motor.py parser still used
 *             the old 5-field format. Synced the Python-side status() parser
 *             (same-commit).
 *
 *    v2.0 : Loader (ID 5) "free by default, briefly rotate on LOAD" workflow
 *           - Changed so that after the user manually free-rotates the loader to
 *             load a ball and presses LOAD, the motor briefly wakes, rotates to
 *             current position + 90°, and returns to torque off.
 *           - initLoaderVelMotor: ends with torque OFF at boot (never turned
 * on).
 *           - loaderRotateBy: on entry torqueOn → ensure P-gain MOVE → read
 *             current position as baseline (the cumulative tracker
 *             g_loader_goal_raw is not used, kept only for STATUS reporting) →
 *             poll → vel=0 → torqueOff.
 *           - handleStop: the loader finishes with torqueOff (so the user can
 * free it by hand).
 *           - Removed 2-stage P-gain (MOVE/HOLD) — HOLD gain is meaningless
 *             while torque is off. Keep only a single LOADER_VEL_P_GAIN(=2000).
 *           - Simplified ensureLoaderReady: HW err check only (no need to
 *             pre-check torque state, since LOAD turns it on directly).
 *
 *    v1.9 : Loader (ID 5) Position → Velocity Mode switch (overcome friction)
 *           - Position Mode (including Step Mode) computes PWM = position_error
 *             × Kp / 128, so as the motor approaches the goal the error shrinks
 *             and PWM comes out of saturation. If the residual torque is then
 *             smaller than static friction, it stops just short of the goal.
 *             Same phenomenon as commanding a small angle change in Dynamixel
 *             Wizard and the motor not turning.
 *           - Velocity Mode has integral wind-up in the speed controller, so
 *             when the motor stalls PWM ramps all the way to max and breaks
 *             through the friction.
 *           - Changed the loader init to initVelMotor. Cleaned up the
 *             setup/handleRecover branching.
 *           - Unified the loader rotation in LOAD/STRIKE into a
 *             loaderRotateBy(delta) helper: setGoalVelocity(LOADER_VEL) →
 *             position polling → setGoalVelocity(0). The cumulative goal tracker
 *             is kept → exactly +90° advance per cycle, no cumulative drift
 *             across cycles (coasting overshoot absorbed by a constant offset).
 *           - handleStop: the loader stops actively via setGoalVelocity(0).
 *           - Removed uses of LOADER_PROFILE_VEL/ACC and the op_mode branch in
 *             initPosMotor (the params themselves kept at defaults for
 *             compatibility).
 *           - recoverLoader: changed to Velocity Mode init.
 *           - Fixed a drift bug where the tracker ran +1024 step ahead after a
 *             TIMEOUT. loaderShutdownCheckAndRecover now always resyncs
 *             g_loader_goal_raw to present position regardless of HW err.
 *             Otherwise, under insufficient load, every LOAD would forever take
 *             a larger goal and fall into a "neither reboots nor accepts
 *             commands" state.
 *           - Hardened recoverLoader: wait after reboot 250→500 ms, retry
 *             initVelMotor 3 times (200 ms apart). Absorbs cold reboot race.
 *           - Split out initLoaderVelMotor: boost the loader VELOCITY_P_GAIN.
 *             At default 100 the P-term yields only 9% PWM on stall, so it
 *             times out waiting for I-gain wind-up. The boost saturates PWM
 *             immediately on stall.
 *           - 2-stage P-gain (MOVE=2000 / HOLD=200) — the Velocity loop has no
 *             D-gain, so in the high-Kp idle state small disturbances cause
 *             oscillation. loaderRotateBy raises it to MOVE at motion start and
 *             returns it to HOLD at the end. handleStop also returns it to HOLD
 *             (covering the STOP-during-motion case). Fixed the issue where a
 *             light touch with a finger made it vibrate.
 *           - ensureLoaderReady() pre-check on LOAD/STRIKE entry: reboot on HW
 *             err, re-enable on torque off. Blocks the case where overload
 *             residue from a previous LOAD swallowed the next LOAD.
 *
 *    v1.8 : Loader (ID 5) rotation resolution + reverse-rotation bug + torque fix
 *           - LOADER_CYCLE_STEP 2047 (≈180°) → 1024 (90°). 1 LOAD = 90° CCW.
 *           - Init only the loader in OP_EXTENDED_POSITION mode. In Position
 *             Mode (0..4095), when the goal crossed the 4095 → 0 boundary the
 *             motor reverse-rotated the long way (CW 270°) instead of the short
 *             angle (CCW 90°) — fixed. In Extended mode the cumulative raw
 *             increases monotonically, so it is always 90° CCW, one direction.
 *           - Init only the loader with PROFILE_VELOCITY/ACCELERATION = 0 (Step
 *             Mode). The previous 400/80 was at ~17% PWM at 50 ms after start,
 *             not enough to overcome the static friction of a heavy ball. 0/0
 *             applies the goal immediately → instant saturation up to the PWM
 *             Limit. Same behavior as Dynamixel Wizard's defaults (0/0).
 *           - Added profile_vel/profile_acc parameters to initPosMotor so the
 *             profile can be set per motor (default = DXL_PROFILE_*).
 *           - When LOAD/STRIKE receives WAIT_TIMEOUT, it reads
 *             HARDWARE_ERROR_STATUS and, if nonzero (overload etc.),
 *             automatically performs reboot → re-init → goal-tracker recapture.
 *             Reply: "ERR OVERLOAD 0x<bits>". The user can resend LOAD and it
 *             works normally. (Before: after torque self-shutdown commands
 *             appeared swallowed — required an OpenRB reset every time.)
 *           - Added cumulative goal tracker g_loader_goal_raw. Removes the issue
 *             where accumulating (cur + step) caused a DXL_ARRIVED_TOL(=10)
 *             arrival error to build up per cycle and shift the phase after
 *             multiple rotations.
 *           - Capture the loader reference point in setup(). Treats the boot-time
 *             PRESENT_POSITION as the reference (0°) so every LOAD accumulates
 *             +90° CCW from there. The motor does not move anywhere at boot.
 *           - Resync the tracker to current position when handleStop() ends. So
 *             the next LOAD accumulates exactly +90° from the STOP-interrupted
 *             point.
 *           - The 2-stage LOAD in handleStrike() uses the same tracker.
 *
 *    v1.0 : Initial integration (DC motor + FF/PI)
 *    v1.1 : Incorporate leveling_motor improvements
 *           (DXL offset, dual motion-complete decision,
 *            PROFILE_VEL/ACC, setup ping check)
 *    v1.2 : Wheels DC+encoder → XC430 (ID 6,7) Velocity Mode replacement
 *           Removed: MDD10A pins, AB encoder ISR, FF+PI control loop,
 *                 WheelCtrl struct
 *           Added: wheelSetVelocity(), mradToUnit() conversion,
 *                 XC430 watchdog (velocity 0 write)
 *    v1.3 : Apply Sync Write
 *           wheelSetVelocity() → simultaneous write of ID 6·7 Goal Velocity
 *           handleAim() / handleHome() → simultaneous write of ID 1·2·3 Goal Position
 *           wheelStop() → simultaneous write of ID 6·7 velocity 0
 *    v1.4 : Leveling performance + abort semantics
 *           - DXL_BAUDRATE 57600 → 1 Mbps + auto-upgrade at boot
 *           - waitMotion() : SyncRead (LVL_1·2·3 together), removed MOVING register
 *           - PROFILE_VEL/ACC 200/50 → 400/80
 *           - Changed waitMotion() return to 3-state WAIT_ARRIVED/TIMEOUT/ABORTED
 *           - Motion handlers reply ERR ABORTED on ABORTED (STOP identifiable)
 *           - handleStrike() delay() → drainable wait + estop check between stages
 *    v1.7 : Harden motor init + RECOVER command
 *           - ensureDxlBaud() : fixed a bug that returned immediately on the
 *             first successful ping. Now checks every ID, finds motors not at
 *             the target baud via factory baud, upgrades EEPROM, and re-verifies.
 *             Automatically recovers a mixed-baud state.
 *           - Per-motor ping retry (default 3 times, 20 ms apart). Mitigates the
 *             race at cold boot where a motor has not yet stabilized.
 *           - On boot failure, send an explicit per-motor log:
 *               "ERR INIT_ID<n>"  (PING / mode / torque setup failure)
 *           - Track failed motors via g_motor_init_failed[8].
 *           - New RECOVER command: re-init only the failed motors.
 *               Reply:  "OK"                — all recovered (or no failed motor)
 *                      "ERR INIT a,b,c"   — list of IDs still failing
 *           - initPosMotor / initVelMotor helpers share setup/RECOVER code.
 *
 *    v1.6 : Fix SyncWrite packet cache bug
 *           - Dynamixel2Arduino caches the packet in InfoSyncWriteInst_t, so
 *             from the second call onward data changes were not reflected to the
 *             motor.
 *           - Force is_info_changed = true on every call of syncMoveLevel /
 *             wheelSetVelocity / wheelStop → packet re-encoding.
 *           - Symptom: in an AIMF rapid stream only the first waypoint applied,
 *             and subsequent commands were not reflected to the motor despite an
 *             OK reply.
 *
 *    v1.5 : Streaming AIMF + estop semantic consistency + force Drive Mode
 *           - Added a non-blocking AIMF command (no waitMotion, immediate OK)
 *           - Resolves stutter in GUI drag / continuous tracking
 *           - The arrival-guaranteed sequence keeps the existing AIM
 *           - Motion-start handlers (AIM/AIMF/HOME/TILT/LOAD/STRIKE) clear
 *             g_estop=false on entry.
 *             Cause: SAMD21 USB-CDC reopen is not reset → a cleanup STOP from
 *             the previous session left g_estop latched into the next session →
 *             the first HOME/AIM returned ABORTED on waitMotion's first
 *             iteration.
 *           - STOP's motion-interrupt meaning is preserved (waitMotion checks
 *             g_estop after drainSerial).
 *           - In setup(), force the Drive Mode of Position Mode motors (ID 1~5)
 *             to 0 (Velocity-based profile). With Time-based profile, AIMF
 *             streaming resets the timer every time and the motor effectively
 *             stops.
 * ============================================================
 */

#include <Dynamixel2Arduino.h>
#include <Servo.h>

using namespace ControlTableItem;

// ═══════════════════════════════════════════════════════════════
//  ★ User configuration area ★
// ═══════════════════════════════════════════════════════════════

// ── DXL bus ──────────────────────────────────────────────────
//   Operating baud = 1 Mbps. If a motor is at factory baud (57600), setup()'s
//   ensureDxlBaud() writes 1 Mbps into EEPROM once and then reconnects.
#define DXL_SERIAL Serial1
#define DXL_DIR_PIN -1
#define DXL_BAUDRATE 1000000UL       // operating baud (1 Mbps)
#define DXL_BAUDRATE_FACTORY 57600UL // auto-upgrade path
#define DXL_PROTOCOL 2.0f

// ── DXL ID ───────────────────────────────────────────────────
#define ID_LVL_1 1
#define ID_LVL_2 2
#define ID_LVL_3 3
#define ID_TILT 4
#define ID_LOAD 5
//   ID_WHEEL_L / ID_WHEEL_R refer to the *physical* left/right wheels (left/right
//   relative to the bot's travel direction, not the ID sticker on the motor
//   case). The wiring/mounting is swapped, so ID 7 is physically left and ID 6
//   is physically right — if the bot is disassembled and rewired so the swap is
//   corrected, just swap 6 ↔ 7 again.
#define ID_WHEEL_L 7
#define ID_WHEEL_R 6

// ── Wheel direction sign ─────────────────────────────────────
//   Tuned so each motor turns the correct way on forward (positive mrad/s).
//   The two motors are mounted facing each other about the bot's centerline,
//   so the left/right signs are usually opposite.
#define WHEEL_DIR_SIGN_L 1 // +1 or -1
#define WHEEL_DIR_SIGN_R -1

// ── Camera tilt (ID 4) direction sign ────────────────────────
//   Convention: positive step = camera up (matches the controller's
//   _step_from_deg and the 0°→90° sweep).
//   Verified as +1 so that positive=up for this board's physical mounting. If
//   the motor turns the wrong way, just swap +1 ↔ -1.
//   Applied identically in both handleTilt / handleTiltAsync so the sync/async
//   paths match. (watchdog hold / STOP rewrite present position as-is, so they
//   are sign-independent.)
#define TILT_DIR_SIGN +1 // +1 or -1

// ── Leveling (ID 1,2,3) direction sign ───────────────────────
//   Convention: positive step = positive rotation direction in the leveling_sim
//   visualization. All three motors rotate exactly opposite to the sim when
//   mounted on the bot, hence -1.
//   Applied symmetrically in stepToRaw / rawToStep → the step axes of AIM input
//   and STATUS readback match the sim axis. If only one motor is off, just
//   change that SIGN.
#define LVL_DIR_SIGN_1 -1
#define LVL_DIR_SIGN_2 -1
#define LVL_DIR_SIGN_3 -1

// ── XC430 velocity unit conversion ───────────────────────────
//   1 unit = 0.229 rpm = 0.229 × 2π/60 rad/s ≈ 0.023980 rad/s
//   mrad/s → unit : mrad/s / 1000 / 0.023980 ≈ mrad/s × 0.041701
//   (datasheet exact value: 0.229 rpm/unit)
#define MRAD_TO_UNIT 0.041701f

// ── XC430 velocity upper limit (unit) ─────────────────────────
//   XC430-W150-R no-load max ≈ 60 rpm ≈ 262 unit
//   Limited to 230 unit including a safety margin
#define WHEEL_UNIT_MAX 230

// ── Position Mode DXL offset (ID 1~5, step units) ────────────
//   4096 step = 1 revolution, 90° = 1024 step
static const int DXL_OFFSET[8] = {
    0, // [0] unused
    0, // [1] LVL_1  (+90°)
    0, // [2] LVL_2  (+90°)
    0, // [3] LVL_3
    0, // [4] TILT
    0, // [5] LOAD
    0, // [6] unused (physical right wheel — Velocity Mode)
    0  // [7] unused (physical left wheel — Velocity Mode)
};

// ── Position Mode motion decision ─────────────────────────────
//   The MOVING register check is removed — even at 1 Mbps each read
//   transaction adds up to ~300 µs, so decide by position tolerance only.
#define DXL_ARRIVED_TOL 10
#define DXL_POLL_MS 5
#define DXL_PROFILE_VEL 700 // (v1.3: 200, v1.4: 400, v2.2: 700) — AIMF tracking ↑
#define DXL_PROFILE_ACC 250 // (v1.3:  50, v1.4:  80, v2.2: 250) — ease stutter
// Meaning of the two values above (XL/XC X-series datasheet):
//   PROFILE_VEL : 0.229 rpm/unit. 700 unit ≈ 160 rpm ≈ 2.67 rev/s.
//   PROFILE_ACC : 214.577 rev/min² per unit. 250 unit ≈ 894 rev/min² ≈
//                 14.9 rev/s². → time to accelerate to max vel ≈ 180 ms.
// AIMF streaming delivers a new goal at 60 Hz (16 ms), so if ACC is too low
// each mini-trajectory gets overwritten mid-acceleration and repeats the same
// initial motion → felt as stutter. Raising it to 250 gets closer to the normal
// speed within one tick → smoother tracking. Can be raised to 300~500 depending
// on weight load, and if it hums/vibrates add POSITION_D_GAIN, 0 → 400~600.
// Feedforward 2nd-order (Acceleration) gain (RAM, addr 88). default 0 → 200.
// (v2.4)
//   Adds the desired acceleration the trajectory generator computes each step
//   directly to PWM. P/D are error-based reactive, whereas FF applies PWM
//   proactively to follow the trajectory shape → smoother transition between
//   micro-trajectories.
//   Especially effective "when descending": where gravity is a helping force,
//   in the region where the trajectory desired acc is negative, FF's negative
//   PWM compensates ahead of time for the overshoot where the motor arrived
//   faster than the trajectory.
//   Raising 200→400 produced overshoot / wobble after arrival. Lowered to 280
//   + D ↑ to absorb residual impulse (v2.5), but wobble still remained. Lowered
//   further to 220 (v2.6) — reduces FF impulse strength one more step. If the
//   step feel gets strong again, nudge back up to around 250.
#define DXL_FEEDFORWARD_ACC_GAIN 220
// Position D gain (RAM, addr 80). default 0 → 400. (v2.3)
//   The D term is the derivative of position_error → effectively -velocity
//   feedback. With only a large P gain, the acceleration P produces overshoots
//   static friction/inertia → easily leads to ringing. If D is 0 the ringing
//   does not die cleanly and acts as an impulse at the start of the next AIMF
//   tick's new trajectory → felt as "stepping + vibration".
//   400 gives D/P ≈ 0.27 against P=1500, slightly on the over-damped side.
//   Residual wobble from the FF impulse appeared, so strengthened one step to
//   600 (D/P ≈ 0.40) (v2.5).
//   Can be raised to 800 if needed. Too high produces high-frequency chatter
//   (encoder noise amplified by the D term) — if you hear a "tsk-tsk-tsk" or
//   "whine" tone from the motor, lower it one step immediately.
#define DXL_POSITION_D_GAIN 600
// Position P gain (RAM, addr 84). default 800 → 1500. (v2.1)
//   Kp_eff = P_GAIN / 128. PWM ∝ position_error × Kp_eff. If the residual error
//   near the goal cannot get within ±DXL_ARRIVED_TOL(=10 step ≈ 0.88°),
//   waitMotion TIMEOUT. With the 3-RRS mechanism friction + aggressive
//   PROFILE_VEL/ACC setting, the default 800 was observed in many cases to
//   leave residual error of ±15~30 (diagnosed via STATUS on HOME). Raised to
//   1500 to strengthen convergence (v2.1).
//   v2.6: slight sag from weight observed after stop → raise stiffness 1500 →
//   1800. Can be raised further to 2000~3000 if needed. Too high → humming /
//   overshoot.
#define DXL_POSITION_P_GAIN 1800
// ── Loader (ID 5) Velocity Mode rotation ─────────────────────
//   Position Mode (even Step Mode) uses PWM = error × Kp / 128, so as the motor
//   approaches the goal PWM comes out of saturation and torque drops sharply →
//   it cannot overcome static friction and stops just short of the target.
//   Velocity Mode has integral wind-up in the speed PI, so when the motor
//   stalls PWM ramps all the way to max and breaks through the friction.
//
//   LOADER_VEL          : Velocity command raw unit (0.229 rpm/unit, signed).
//                         200 unit ≈ 45.8 rpm → 90° (1024 step) ≈ 0.33 s.
//                         (Within the XM430 default Velocity Limit of 230. To go
//                          faster you must also raise the EEPROM Velocity Limit.)
//   LOADER_VEL_P_GAIN   : velocity P-gain during motion.
//   LOADER_VEL_I_GAIN   : velocity I-gain during motion.
//
//   v2.7 vibration reduction: the old P-gain=2000 (20× the default 100) broke
//   through static friction by saturating PWM instantly with the P-term alone
//   on stall, but the same gain *during rotation* made PWM swing wildly on
//   every small velocity ripple, so the motor juddered (vibrated). Enough force
//   but excessive vibration → lower P (remove oscillation) and hand the
//   static-friction breakthrough to I (integral wind-up): on stall ∫error
//   accumulates/saturates and pushes PWM all the way to break through static
//   friction (slightly slower than P but smoother). Boosted above the default
//   I=1920 to preserve breakthrough speed. If you want less vibration, lower P
//   further (e.g. 200); if force/response is lacking, raise I (e.g. 4000).
//
//   v2.0: the loader is in "torque off by default, briefly torque on for LOAD"
//   mode. After the user free-rotates by hand to load a ball and presses LOAD,
//   the motor briefly wakes, rotates to "current position + 90°", and returns
//   to torque off. The vibration / holding-stall problem itself disappears.
#define LOADER_VEL 400
#define LOADER_VEL_P_GAIN 500
#define LOADER_VEL_I_GAIN 4000
// Overshoot prevention — slow down in the final segment + active brake after stop.
//   LOADER_VEL_SLOW   : speed in the deceleration segment (raw). The overshoot
//                       from polling latency is determined by this speed.
//                       80 raw ≈ 18 rpm = 1.25 step/ms.
//   LOADER_BRAKE_ZONE : switch to SLOW when the steps remaining to next are at
//                       or below this value. 180 step ≈ 16° — ~18% of 1024 step
//                       (90°).
//   LOADER_BRAKE_MS   : wait between setGoalVelocity(0) and torqueOff. During
//                       this the velocity PI loop absorbs inertia as an active
//                       brake.
#define LOADER_VEL_SLOW 80
#define LOADER_BRAKE_ZONE 180
#define LOADER_BRAKE_MS 80UL

// ── T-motor ESC ──────────────────────────────────────────────
#define PIN_ESC_TOP 9
#define PIN_ESC_BOT 10
#define ESC_MIN_US 1000
#define ESC_MAX_US 2000
#define TMOTOR_MAX_RPM 10000

// ── Protocol constants ───────────────────────────────────────
#define LINE_MAX 64
#define WATCHDOG_MS 200UL
#define MOTION_TIMEOUT 4000UL
#define WHEEL_MRAD_MAX 30000
#define WHEEL_DEADZONE 5
#define DXL_STEP_MIN -2047
#define DXL_STEP_MAX 2047
// Loader (ID 5) rotation amount per LOAD (raw step). Since 4096 step = 360°,
// 1024 step = 90°. With DRIVE_MODE bit0=0 (Normal), + value = CCW.
#define LOADER_CYCLE_STEP 1024

// ═══════════════════════════════════════════════════════════════
//  Global objects / variables
// ═══════════════════════════════════════════════════════════════

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
Servo escTop, escBot;

// Wheels
uint32_t g_last_drive_ms = 0;
bool g_watchdog_tripped = false;
bool g_drive_active = false;

// TILT_ASYNC fire-and-forget watchdog (200 ms hold-in-place)
uint32_t g_last_tilt_async_ms = 0;

// T-motor
uint16_t g_rpmTop = 0, g_rpmBot = 0;

// Position Mode DXL
bool g_leveling_moving = false;
bool g_tilt_moving = false;
bool g_loader_moving = false;
bool g_err_latched = false;
bool g_homed = false;
bool g_estop = false;
int32_t g_dxl_target[8] = {0};
bool g_motion_busy = false; // guard while waitMotion() is running

// Loader cumulative goal position (raw, multi-turn). +LOADER_CYCLE_STEP added on
// each LOAD. The loader is in Velocity Mode (v1.9) — present position is
// reported multi-turn, so it increases monotonically without wrap. loaderRotateBy
// uses this value as the poll termination condition (present ≥ next - tol).
//   The initial value is overwritten in setup() with the boot-time
//   PRESENT_POSITION — "the angle the motor was sitting at when powered on" is
//   the reference, and every LOAD accumulates +90° CCW from there. The motor
//   does not move anywhere at boot.
int32_t g_loader_goal_raw = 0;

// Track motors that failed init (ping + mode + torqueOn) in setup().
// The RECOVER command retries only the IDs whose flag is set.
// Index = DXL ID (1..7).
bool g_motor_init_failed[8] = {false};

// Serial buffer
char g_buf[LINE_MAX + 2];
uint8_t g_buf_idx = 0;

// ── Sync Write buffers ────────────────────────────────────────
// Goal Velocity (addr 104, 4 bytes signed) — wheels ID 6·7
DYNAMIXEL::InfoSyncWriteInst_t g_sw_vel;
DYNAMIXEL::XELInfoSyncWrite_t g_sw_vel_xel[2];
int32_t g_sw_vel_data[2]; // [0]=L, [1]=R

// Goal Position (addr 116, 4 bytes) — leveling ID 1·2·3
DYNAMIXEL::InfoSyncWriteInst_t g_sw_pos;
DYNAMIXEL::XELInfoSyncWrite_t g_sw_pos_xel[3];
int32_t g_sw_pos_data[3]; // [0]=LVL1 [1]=LVL2 [2]=LVL3

// ── Sync Read buffers ─────────────────────────────────────────
// Present Position (addr 132, 4 bytes) — leveling ID 1·2·3
// Used for the 3-axis simultaneous position polling in waitMotion() → transaction 3→1.
DYNAMIXEL::InfoSyncReadInst_t g_sr_pos;
DYNAMIXEL::XELInfoSyncRead_t g_sr_pos_xel[3];
int32_t g_sr_pos_data[3]; // [0]=LVL1 [1]=LVL2 [2]=LVL3

// ── waitMotion return codes ───────────────────────────────────
// ARRIVED  : all motors arrived within tolerance
// TIMEOUT  : time exceeded (possible HW problem)
// ABORTED  : STOP received while waiting → g_estop transitions to true
enum WaitResult : uint8_t {
  WAIT_ARRIVED = 0,
  WAIT_TIMEOUT = 1,
  WAIT_ABORTED = 2,
};

// ═══════════════════════════════════════════════════════════════
//  Wheels (XC430 Velocity Mode)
// ═══════════════════════════════════════════════════════════════

/**
 * mrad/s → XC430 velocity unit conversion
 *   below deadzone → 0, result clamped to ±WHEEL_UNIT_MAX
 */
int32_t mradToUnit(int32_t mrad, int dir_sign) {
  if (abs(mrad) < WHEEL_DEADZONE)
    return 0;
  float unit = (float)mrad * MRAD_TO_UNIT * (float)dir_sign;
  return (int32_t)constrain((long)unit, -WHEEL_UNIT_MAX, WHEEL_UNIT_MAX);
}

/** Velocity command to both wheels — Sync Write (ID 6·7 together) */
void wheelSetVelocity(int32_t mrad_L, int32_t mrad_R) {
  g_sw_vel_data[0] = mradToUnit(mrad_L, WHEEL_DIR_SIGN_L);
  g_sw_vel_data[1] = mradToUnit(mrad_R, WHEEL_DIR_SIGN_R);
  g_sw_vel.is_info_changed = true; // re-encode packet each call (avoid cache reuse)
  dxl.syncWrite(&g_sw_vel);
}

/** Immediate stop of both wheels — Sync Write (velocity = 0 together) */
void wheelStop() {
  g_sw_vel_data[0] = 0;
  g_sw_vel_data[1] = 0;
  g_sw_vel.is_info_changed = true;
  dxl.syncWrite(&g_sw_vel);
}

/**
 * Read the current actual velocity in mrad/s (for STATUS reply)
 *   getPresentVelocity() = signed velocity unit
 */
int32_t wheelReadMrad(uint8_t id, int dir_sign) {
  int32_t unit = (int32_t)dxl.getPresentVelocity(id);
  return (int32_t)((float)unit / MRAD_TO_UNIT / (float)dir_sign);
}

// ═══════════════════════════════════════════════════════════════
//  T-motor ESC
// ═══════════════════════════════════════════════════════════════

uint16_t rpmToUs(uint16_t rpm) {
  if (rpm == 0)
    return ESC_MIN_US;
  uint16_t r = min(rpm, (uint16_t)TMOTOR_MAX_RPM);
  return (uint16_t)(ESC_MIN_US +
                    ((uint32_t)r * (ESC_MAX_US - ESC_MIN_US)) / TMOTOR_MAX_RPM);
}

void setTmotor(uint16_t rpmT, uint16_t rpmB) {
  escTop.writeMicroseconds(rpmToUs(rpmT));
  escBot.writeMicroseconds(rpmToUs(rpmB));
  g_rpmTop = rpmT;
  g_rpmBot = rpmB;
}

// ═══════════════════════════════════════════════════════════════
//  Position Mode DXL utilities (ID 1~5)
// ═══════════════════════════════════════════════════════════════

// Direction-sign lookup for leveling ID 1..3. Other IDs are +1 (no-op).
static inline int32_t lvlDirSign(uint8_t id) {
  switch (id) {
  case ID_LVL_1: return LVL_DIR_SIGN_1;
  case ID_LVL_2: return LVL_DIR_SIGN_2;
  case ID_LVL_3: return LVL_DIR_SIGN_3;
  default:       return 1;
  }
}

int32_t stepToRaw(uint8_t id, int32_t step) {
  int32_t signed_step = lvlDirSign(id) * step;
  int32_t adj = constrain(signed_step + DXL_OFFSET[id], DXL_STEP_MIN, DXL_STEP_MAX);
  return adj + 2048;
}

int32_t rawToStep(uint8_t id, int32_t raw) {
  return lvlDirSign(id) * ((raw - 2048) - DXL_OFFSET[id]);
}

int32_t dxlReadStep(uint8_t id) {
  return rawToStep(id, (int32_t)dxl.getPresentPosition(id));
}

bool dxlMove(uint8_t id, int32_t step) {
  int32_t raw = stepToRaw(id, step);
  if (!dxl.setGoalPosition(id, (uint32_t)raw))
    return false;
  g_dxl_target[id] = raw;
  return true;
}

/**
 * Leveling 3-axis Sync Write — send Goal Position simultaneously
 *   s1/s2/s3: target step for each axis
 *   true on success, false on failure
 */
bool syncMoveLevel(int32_t s1, int32_t s2, int32_t s3) {
  g_sw_pos_data[0] = stepToRaw(ID_LVL_1, s1);
  g_sw_pos_data[1] = stepToRaw(ID_LVL_2, s2);
  g_sw_pos_data[2] = stepToRaw(ID_LVL_3, s3);
  g_dxl_target[ID_LVL_1] = g_sw_pos_data[0];
  g_dxl_target[ID_LVL_2] = g_sw_pos_data[1];
  g_dxl_target[ID_LVL_3] = g_sw_pos_data[2];
  // Force packet re-encoding on every call. If not set, the library reuses the
  // previous packet from cache and data updates after the first syncWrite are
  // not reflected to the motor (the cause of only the first waypoint applying
  // in an AIMF rapid stream).
  g_sw_pos.is_info_changed = true;
  return dxl.syncWrite(&g_sw_pos);
}

// Forward declaration of dispatch() — called when draining serial inside waitMotion()
void dispatch(char *line);

// Forward declaration of motor init helpers — handleRecover() calls them above their definition
static bool pingRetry(uint8_t id, uint8_t retries = 3);
static bool initPosMotor(uint8_t id, uint8_t op_mode = OP_POSITION,
                         uint32_t profile_vel = DXL_PROFILE_VEL,
                         uint32_t profile_acc = DXL_PROFILE_ACC,
                         uint16_t position_p_gain = DXL_POSITION_P_GAIN,
                         uint16_t position_d_gain = DXL_POSITION_D_GAIN,
                         uint16_t ff_acc_gain = DXL_FEEDFORWARD_ACC_GAIN);
static bool initVelMotor(uint8_t id);
static bool initLoaderVelMotor();
static bool ensureLoaderReady();
static bool recoverLoader();
static bool drainableWait(uint32_t ms);

/**
 * Serial drain inside waitMotion()
 *   So the serial buffer (64 bytes) does not overflow even during a blocking
 *   wait, consume received bytes immediately and process completed lines via
 *   dispatch().
 *
 *   Recursion safety:
 *     During waitMotion() the Pi is waiting for the current sync command's
 *     OK/ERR, so it does not send AIM·HOME·TILT·LOAD·STRIKE (= the commands that
 *     call waitMotion). Therefore the dispatch() → waitMotion() recursion does
 *     not actually occur.
 */
static void drainSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r')
      continue;
    if (c == '\n') {
      g_buf[g_buf_idx] = '\0';
      if (g_buf_idx > 0)
        dispatch(g_buf);
      g_buf_idx = 0;
    } else if (g_buf_idx < LINE_MAX) {
      g_buf[g_buf_idx++] = c;
    } else {
      g_buf_idx = 0;
      while (Serial.available() && Serial.read() != '\n')
        ;
      sendErr("OVERFLOW");
    }
  }
}

/**
 * motion-complete decision (position tolerance only)
 *   - LVL_1·2·3 batch : update all 3 axes' present position at once via one
 *     SyncRead.
 *   - others (TILT/LOAD/single axis) : per-id getPresentPosition.
 *   - call drainSerial() each polling interval → prevent buffer overflow.
 *
 * Returns
 *   WAIT_ARRIVED  : all arrived within tolerance
 *   WAIT_TIMEOUT  : timeout expired
 *   WAIT_ABORTED  : STOP received while waiting (g_estop=true) → explicitly
 * notify the caller (prevents the fake success where STOP makes target equal
 * current and it looks like ARRIVED)
 */
WaitResult waitMotion(const uint8_t *ids, uint8_t n,
                      uint32_t timeout_ms = MOTION_TIMEOUT) {
  bool done[8] = {false};
  g_motion_busy = true;
  uint32_t start = millis();

  // Fast path: 3-axis leveling batch polling (transaction 3→1)
  const bool lvl_batch = (n == 3 && ids[0] == ID_LVL_1 && ids[1] == ID_LVL_2 &&
                          ids[2] == ID_LVL_3);

  while (millis() - start < timeout_ms) {
    drainSerial();

    // If drainSerial() called handleStop(), g_estop=true.
    // STOP changes g_dxl_target to current so arrived would look true on the
    // next poll, so explicitly intercept here to block the fake success.
    if (g_estop) {
      g_motion_busy = false;
      return WAIT_ABORTED;
    }

    bool any_pending = false;

    if (lvl_batch) {
      g_sr_pos.is_info_changed = true;
      uint8_t recv = dxl.syncRead(&g_sr_pos);
      if (recv == 3) {
        for (uint8_t i = 0; i < 3; i++) {
          if (done[i])
            continue;
          int32_t present = g_sr_pos_data[i];
          bool arrived =
              (abs(present - g_dxl_target[ids[i]]) <= DXL_ARRIVED_TOL);
          if (arrived)
            done[i] = true;
          else
            any_pending = true;
        }
      } else {
        any_pending = true; // communication failure — retry
      }
    } else {
      for (uint8_t i = 0; i < n; i++) {
        if (done[i])
          continue;
        uint8_t id = ids[i];
        int32_t present = (int32_t)dxl.getPresentPosition(id);
        bool arrived = (abs(present - g_dxl_target[id]) <= DXL_ARRIVED_TOL);
        if (arrived)
          done[i] = true;
        else
          any_pending = true;
      }
    }

    if (!any_pending) {
      g_motion_busy = false;
      return WAIT_ARRIVED;
    }
    delay(DXL_POLL_MS);
  }
  g_motion_busy = false;
  return WAIT_TIMEOUT;
}

/**
 * Convert a waitMotion result into an ERR reply (distinguishes ABORTED from
 * TIMEOUT). Common to handleAim/Home/Tilt/Load/Strike.
 */
void replyByResult(WaitResult r) {
  switch (r) {
  case WAIT_ARRIVED:
    sendOk();
    break;
  case WAIT_TIMEOUT:
    sendErr("TIMEOUT");
    break;
  case WAIT_ABORTED:
    sendErr("ABORTED");
    break;
  }
}

// ═══════════════════════════════════════════════════════════════
//  Reply helpers
// ═══════════════════════════════════════════════════════════════

void sendOk() { Serial.println("OK"); }
void sendErr(const char *reason) {
  Serial.print("ERR ");
  Serial.println(reason);
  g_err_latched = true;
}

// ═══════════════════════════════════════════════════════════════
//  Command handlers
// ═══════════════════════════════════════════════════════════════

void handlePing() {
  g_err_latched = false;
  Serial.println("PONG");
}

void handleStatus() {
  int32_t wL = wheelReadMrad(ID_WHEEL_L, WHEEL_DIR_SIGN_L);
  int32_t wR = wheelReadMrad(ID_WHEEL_R, WHEEL_DIR_SIGN_R);

  int32_t s[6];
  for (uint8_t id = 1; id <= 5; id++)
    s[id] = dxlReadStep(id);

  uint8_t flags = 0;
  if (g_watchdog_tripped)
    flags |= (1 << 0);
  if (g_leveling_moving)
    flags |= (1 << 1);
  if (g_tilt_moving)
    flags |= (1 << 2);
  if (g_loader_moving)
    flags |= (1 << 3);
  if (g_rpmTop > 100 || g_rpmBot > 100)
    flags |= (1 << 4);
  if (g_err_latched)
    flags |= (1 << 5);
  if (g_homed)
    flags |= (1 << 6);
  if (g_estop)
    flags |= (1 << 7);

  // Output format: S wL wR p1 p2 p3 p4 p5 rpmT rpmB flags   (11 fields, v1.2+)
  Serial.print("S ");
  Serial.print(wL);
  Serial.print(' ');
  Serial.print(wR);
  Serial.print(' ');
  for (uint8_t id = 1; id <= 5; id++) {
    Serial.print(s[id]);
    Serial.print(' ');
  }
  Serial.print(g_rpmTop);
  Serial.print(' ');
  Serial.print(g_rpmBot);
  Serial.print(' ');
  Serial.println(flags);
}

void handleStop() {
  wheelStop();
  g_drive_active = false;
  g_watchdog_tripped = false;

  setTmotor(0, 0);

  // Position Mode motors (ID 1~4): fix goal to current position → lock in place.
  for (uint8_t id = 1; id <= 4; id++) {
    int32_t cur = (int32_t)dxl.getPresentPosition(id);
    dxl.setGoalPosition(id, (uint32_t)cur);
    g_dxl_target[id] = cur;
  }
  // The loader (ID 5) is in Velocity Mode — clear the command with vel 0 +
  // torqueOff to return to a state where the user can free-rotate by hand again
  // (same as the v2.0 default free state). The next LOAD takes the current
  // present as its baseline anyway, so resync the tracker too.
  dxl.setGoalVelocity(ID_LOAD, 0);
  dxl.torqueOff(ID_LOAD);
  g_loader_goal_raw = (int32_t)dxl.getPresentPosition(ID_LOAD);
  g_dxl_target[ID_LOAD] = g_loader_goal_raw;

  g_leveling_moving = false;
  g_tilt_moving = false;
  g_loader_moving = false;
  g_estop = true;
  g_err_latched = false;
  sendOk();
}

void handleDrive(char *args) {
  // DRIVE is fire-and-forget (protocol §4 "no reply"). It never sends a line
  // even on the error path — if a stale ERR line is left in the buffer, the
  // next sync command's (PING/STOP/TILT) readline() reads it instead of
  // OK/PONG and the responses desync. So parse failures are silently dropped,
  // and out-of-range is clamped instead of ERR (same pattern as TILT_ASYNC).
  long vL, vR;
  if (sscanf(args, "%ld %ld", &vL, &vR) != 2) {
    return;
  }
  vL = constrain(vL, (long)-WHEEL_MRAD_MAX, (long)WHEEL_MRAD_MAX);
  vR = constrain(vR, (long)-WHEEL_MRAD_MAX, (long)WHEEL_MRAD_MAX);

  wheelSetVelocity((int32_t)vL, (int32_t)vR);
  g_last_drive_ms = millis();
  g_watchdog_tripped = false;
  g_drive_active = true;
  g_estop = false;
  // no reply (fire-and-forget)
}

void handleAim(char *args) {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  long s1, s2, s3;
  if (sscanf(args, "%ld %ld %ld", &s1, &s2, &s3) != 3) {
    sendErr("PARSE");
    return;
  }
  if (s1 < DXL_STEP_MIN || s1 > DXL_STEP_MAX || s2 < DXL_STEP_MIN ||
      s2 > DXL_STEP_MAX || s3 < DXL_STEP_MIN || s3 > DXL_STEP_MAX) {
    sendErr("RANGE");
    return;
  }

  // Explicit motion start — release the previous session's STOP latch.
  // (SAMD21 USB-CDC reopen is not reset, so globals persist across sessions)
  g_estop = false;

  if (!syncMoveLevel((int32_t)s1, (int32_t)s2, (int32_t)s3)) {
    sendErr("HW");
    return;
  }

  g_leveling_moving = true;
  const uint8_t ids[] = {ID_LVL_1, ID_LVL_2, ID_LVL_3};
  WaitResult r = waitMotion(ids, 3);
  g_leveling_moving = false;
  replyByResult(r);
}

/**
 * handleAimF — non-blocking (streaming) AIM.
 *   Updates only GOAL_POSITION via syncWrite and returns OK immediately.
 *   Dynamixel servos natively support overwriting GOAL_POSITION during motion
 *   (re-orientation), so they track without stutter even with continuous 60 Hz
 *   updates like a GUI drag.
 *
 *   Difference from AIM: no waitMotion() call → the host can send the next
 *   command after one RTT (~3 ms). Use AIM for sequences that need "arrival
 *   guarantee".
 *
 *   g_leveling_moving is left true (user is in the middle of commanding), and
 *   STOP/HOME clear it.
 */
void handleAimF(char *args) {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  long s1, s2, s3;
  if (sscanf(args, "%ld %ld %ld", &s1, &s2, &s3) != 3) {
    sendErr("PARSE");
    return;
  }
  if (s1 < DXL_STEP_MIN || s1 > DXL_STEP_MAX || s2 < DXL_STEP_MIN ||
      s2 > DXL_STEP_MAX || s3 < DXL_STEP_MIN || s3 > DXL_STEP_MAX) {
    sendErr("RANGE");
    return;
  }
  g_estop = false; // explicit motion start
  if (!syncMoveLevel((int32_t)s1, (int32_t)s2, (int32_t)s3)) {
    sendErr("HW");
    return;
  }
  g_leveling_moving = true;
  sendOk(); // ← no waitMotion, immediate reply
}

void handleHome() {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  g_estop = false; // explicit motion start
  if (!syncMoveLevel(0, 0, 0)) {
    sendErr("HW");
    return;
  }

  g_leveling_moving = true;
  const uint8_t ids[] = {ID_LVL_1, ID_LVL_2, ID_LVL_3};
  WaitResult r = waitMotion(ids, 3);
  g_leveling_moving = false;
  if (r != WAIT_ARRIVED) {
    replyByResult(r);
    return;
  }
  g_homed = true;
  sendOk();
}

void handleTilt(char *args) {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  long s4;
  if (sscanf(args, "%ld", &s4) != 1) {
    sendErr("PARSE");
    return;
  }
  if (s4 < DXL_STEP_MIN || s4 > DXL_STEP_MAX) {
    sendErr("RANGE");
    return;
  }
  g_estop = false; // explicit motion start
  if (!dxlMove(ID_TILT, (int32_t)(TILT_DIR_SIGN * s4))) {
    sendErr("HW");
    return;
  }

  g_tilt_moving = true;
  const uint8_t ids[] = {ID_TILT};
  WaitResult r = waitMotion(ids, 1);
  g_tilt_moving = false;
  // Refresh TILT_ASYNC watchdog so the hold-rewrite doesn't fire immediately
  // after a sync TILT (semantically harmless, but wastes a setGoalPosition).
  g_last_tilt_async_ms = millis();
  replyByResult(r);
}

void handleTiltAsync(char *args) {
  // Fire-and-forget tilt setpoint. No motion-complete poll, no reply.
  // 200 ms watchdog (g_last_tilt_async_ms) holds-in-place if stream stalls.
  long s4;
  if (sscanf(args, "%ld", &s4) != 1) {
    return;   // silently drop parse errors — f&f path
  }
  if (s4 < DXL_STEP_MIN) s4 = DXL_STEP_MIN;
  if (s4 > DXL_STEP_MAX) s4 = DXL_STEP_MAX;
  if (g_motion_busy) {
    return;   // sync TILT motion-complete in progress — drop
  }
  g_estop = false;
  dxlMove(ID_TILT, (int32_t)(TILT_DIR_SIGN * s4));
  g_last_tilt_async_ms = millis();
}

void handleSpin(char *args) {
  long rT, rB;
  if (sscanf(args, "%ld %ld", &rT, &rB) != 2) {
    sendErr("PARSE");
    return;
  }
  if (rT < 0 || rB < 0 || rT > TMOTOR_MAX_RPM || rB > TMOTOR_MAX_RPM) {
    sendErr("RANGE");
    return;
  }
  setTmotor((uint16_t)rT, (uint16_t)rB);
  sendOk();
}

/**
 * loaderShutdownCheckAndRecover — follow-up handling right after a TIMEOUT.
 *
 *   1) Resync tracker (always): a timeout means loaderRotateBy had already
 *      advanced g_loader_goal_raw by +delta before polling → the tracker is
 *      ahead of the actual position. Leaving it makes the next LOAD goal
 *      actual+2048, which becomes a heavier task and is eventually never
 *      reached. Roll it back to present so the next LOAD is normalized to
 *      "current spot + 90°".
 *
 *   2) Read HARDWARE_ERROR_STATUS — if nonzero (overload/overheat etc.), clear
 *      it via reboot + re-init. In Velocity Mode the overload bit often does
 *      not trip within the 4-second timeout (XL/XM series default ~5 sec+), so
 *      a return of 0 is common. Even so, the tracker resync alone makes the
 *      next LOAD work normally.
 *
 *   Returns:
 *      0 : no HW error, only tracker resynced. Caller sends a normal TIMEOUT reply.
 *      1 : HW error detected + reboot recovery succeeded. Caller sends OVERLOAD reply.
 *     -1 : HW error detected + reboot recovery failed. Caller sends INIT_FAIL reply.
 */
static int8_t loaderShutdownCheckAndRecover(int32_t *hw_err_out) {
  // (1) Always: roll the tracker back to where the motor actually arrived.
  g_loader_goal_raw = (int32_t)dxl.getPresentPosition(ID_LOAD);
  g_dxl_target[ID_LOAD] = g_loader_goal_raw;

  // (2) Check HW err + reboot if needed.
  int32_t hw_err = dxl.readControlTableItem(HARDWARE_ERROR_STATUS, ID_LOAD);
  if (hw_err_out)
    *hw_err_out = hw_err;
  if (hw_err == 0)
    return 0;
  return recoverLoader() ? 1 : -1;
}

/**
 * loaderRotateBy — rotate the loader to "current physical position + delta_step".
 *
 *   v2.0 workflow:
 *     1. torqueOn — normally torque is off so the motor can free-rotate. Wake it
 *        here.
 *     2. Ensure P-gain MOVE (after a reboot it may be at EEPROM default).
 *     3. Re-read the current present_position and use it as the baseline. The
 *        position the user turned to by hand is reflected. The cumulative
 *        tracker (g_loader_goal_raw) is not used.
 *     4. setGoalVelocity(LOADER_VEL) → drainSerial polling → done when present
 *        reaches baseline + delta_step.
 *     5. setGoalVelocity(0) → torqueOff. Back to a state where the user can
 * free-rotate by hand again.
 *
 *   Returns: WaitResult — used by the caller to send a reply.
 *     WAIT_ARRIVED : arrived normally, motor in torque off state.
 *     WAIT_TIMEOUT : failed to arrive in time (motor also returned to torque off).
 *     WAIT_ABORTED : interrupted by STOP (g_estop=true), motor torque off.
 */
static WaitResult loaderRotateBy(int32_t delta_step) {
  // 1. torque ON — wake the motor from the normal free state.
  if (!dxl.torqueOn(ID_LOAD))
    return WAIT_TIMEOUT;
  // 2. PI-gain (may have reverted to EEPROM default e.g. on reboot, so set every time).
  //    Low P = suppress vibration during rotation, high I = break through static friction via wind-up on stall.
  dxl.writeControlTableItem(VELOCITY_P_GAIN, ID_LOAD, LOADER_VEL_P_GAIN);
  dxl.writeControlTableItem(VELOCITY_I_GAIN, ID_LOAD, LOADER_VEL_I_GAIN);

  // 3. Current position = baseline. The arbitrary angle the user turned to by hand is the start point.
  int32_t start = (int32_t)dxl.getPresentPosition(ID_LOAD);
  int32_t next = start + delta_step;
  g_loader_goal_raw = next;     // for STATUS reporting
  g_dxl_target[ID_LOAD] = next; // for STATUS reporting

  if (!dxl.setGoalVelocity(ID_LOAD, LOADER_VEL)) {
    dxl.torqueOff(ID_LOAD);
    return WAIT_TIMEOUT;
  }

  g_motion_busy = true;
  g_loader_moving = true;
  uint32_t t0 = millis();
  WaitResult result = WAIT_TIMEOUT;
  bool braking = false;

  while (millis() - t0 < MOTION_TIMEOUT) {
    drainSerial();
    if (g_estop) {
      result = WAIT_ABORTED;
      break;
    }
    int32_t present = (int32_t)dxl.getPresentPosition(ID_LOAD);
    // Enter deceleration segment: the last LOADER_BRAKE_ZONE step at SLOW speed.
    // Minimizes the effect of polling latency (5ms × 6 step/ms = 30 step ≈ 2.6°).
    if (!braking && present >= next - LOADER_BRAKE_ZONE) {
      dxl.setGoalVelocity(ID_LOAD, LOADER_VEL_SLOW);
      braking = true;
    }
    if (present >= next - DXL_ARRIVED_TOL) {
      result = WAIT_ARRIVED;
      break;
    }
    delay(DXL_POLL_MS);
  }

  // 5. Stop: after the vel=0 command, hold briefly so the PI loop active-brakes →
  //    then torqueOff. Immediate torqueOff would overshoot from inertial coasting.
  dxl.setGoalVelocity(ID_LOAD, 0);
  drainableWait(LOADER_BRAKE_MS);
  dxl.torqueOff(ID_LOAD);
  g_loader_moving = false;
  g_motion_busy = false;
  return result;
}

void handleLoad() {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  g_estop = false; // explicit motion start

  // Pre-check: if overload residue from a previous LOAD remains (HW err /
  // torque off), automatically handle reboot/re-enable here. Otherwise the LOAD
  // command appears "swallowed" (writing only velocity while torque is off does
  // not move the motor).
  if (!ensureLoaderReady()) {
    sendErr("LOADER_NOT_READY");
    return;
  }

  WaitResult r = loaderRotateBy(LOADER_CYCLE_STEP);

  if (r == WAIT_TIMEOUT) {
    int32_t hw_err = 0;
    int8_t rc = loaderShutdownCheckAndRecover(&hw_err);
    if (rc == 1) {
      Serial.print("ERR OVERLOAD 0x");
      Serial.println((unsigned)hw_err, HEX);
      g_err_latched = true;
      return;
    }
    if (rc == -1) {
      sendErr("OVERLOAD_REINIT_FAIL");
      return;
    }
    // rc == 0: no HW error → normal TIMEOUT reply
  }
  replyByResult(r);
}

/**
 * drainable wait — replaces delay(). Waits for ms while continuing to process
 * serial. If g_estop becomes true while waiting, returns false immediately
 * (interrupted by STOP). Returns true = normal expiry.
 */
static bool drainableWait(uint32_t ms) {
  uint32_t t0 = millis();
  while (millis() - t0 < ms) {
    drainSerial();
    if (g_estop)
      return false;
    delay(1);
  }
  return true;
}

void handleStrike(char *args) {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  long rpm, hold_ms;
  if (sscanf(args, "%ld %ld", &rpm, &hold_ms) != 2) {
    sendErr("PARSE");
    return;
  }
  if (rpm < 0 || rpm > TMOTOR_MAX_RPM || hold_ms < 0) {
    sendErr("RANGE");
    return;
  }
  g_estop = false; // explicit motion start

  // ── 1) Spin-up ──────────────────────────────────────────
  setTmotor((uint16_t)rpm, (uint16_t)rpm);
  // Not delay() but a drainable wait — a STOP arriving during it takes effect immediately.
  if (!drainableWait((uint32_t)hold_ms)) {
    // STOP has already done setTmotor(0,0) + sent sendOk.
    // Here, return ERR ABORTED as STRIKE's own reply.
    sendErr("ABORTED");
    return;
  }

  // ── 2) LOAD ─────────────────────────────────────────────
  // Pre-check (absorb HW err / torque off residue)
  if (!ensureLoaderReady()) {
    setTmotor(0, 0);
    sendErr("LOADER_NOT_READY");
    return;
  }
  // Velocity Mode + position polling — avoids Position Mode's PWM desaturation issue.
  WaitResult r = loaderRotateBy(LOADER_CYCLE_STEP);

  if (r != WAIT_ARRIVED) {
    setTmotor(0, 0); // safety: guarantee spin-down
    if (r == WAIT_TIMEOUT) {
      int32_t hw_err = 0;
      int8_t rc = loaderShutdownCheckAndRecover(&hw_err);
      if (rc == 1) {
        Serial.print("ERR OVERLOAD 0x");
        Serial.println((unsigned)hw_err, HEX);
        g_err_latched = true;
        return;
      }
      if (rc == -1) {
        sendErr("OVERLOAD_REINIT_FAIL");
        return;
      }
    }
    replyByResult(r); // normal TIMEOUT / ABORTED reply
    return;
  }

  // ── 3) Spin-down ────────────────────────────────────────
  setTmotor(0, 0);
  sendOk();
}

/**
 * handleRecover — retry init for motors that failed init at boot.
 *   Only IDs with g_motor_init_failed[id] = true are targeted (successful motors
 *   are not touched). Automatically determines whether an ID belongs to
 *   Position/Velocity.
 *
 *   Reply
 *     OK                  — no failed motor, or all recovered
 *     ERR INIT a,b,c      — list of IDs still failing after the attempt (comma-separated)
 *
 *   Note: replies BUSY while a motion is in progress (g_motion_busy). Doing
 *   torqueOff/On during motion can affect the SyncWrite/Read timing of other
 *   motors.
 */
void handleRecover() {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }

  // Position Mode: ID 1~4. The loader (ID 5) is included in vel_ids below.
  const uint8_t pos_ids[] = {ID_LVL_1, ID_LVL_2, ID_LVL_3, ID_TILT};
  const uint8_t vel_ids[] = {ID_LOAD, ID_WHEEL_L, ID_WHEEL_R};
  uint8_t still_failed[8];
  uint8_t fail_count = 0;

  for (uint8_t i = 0; i < 4; i++) {
    uint8_t id = pos_ids[i];
    if (!g_motor_init_failed[id])
      continue;
    if (initPosMotor(id)) {
      g_motor_init_failed[id] = false;
    } else {
      still_failed[fail_count++] = id;
    }
  }
  for (uint8_t i = 0; i < 3; i++) {
    uint8_t id = vel_ids[i];
    if (!g_motor_init_failed[id])
      continue;
    bool ok = (id == ID_LOAD) ? initLoaderVelMotor() : initVelMotor(id);
    if (ok) {
      g_motor_init_failed[id] = false;
    } else {
      still_failed[fail_count++] = id;
    }
  }

  if (fail_count == 0) {
    g_err_latched = false;
    sendOk();
    return;
  }

  Serial.print("ERR INIT ");
  for (uint8_t i = 0; i < fail_count; i++) {
    Serial.print(still_failed[i]);
    if (i + 1 < fail_count)
      Serial.print(',');
  }
  Serial.println();
  g_err_latched = true;
}

// ═══════════════════════════════════════════════════════════════
//  Line dispatcher
// ═══════════════════════════════════════════════════════════════
void dispatch(char *line) {
  while (*line == ' ')
    line++;
  if (*line == '\0')
    return;

  char cmd[16] = {0};
  char *p = line;
  uint8_t ci = 0;
  while (*p && *p != ' ' && ci < 15)
    cmd[ci++] = *p++;
  cmd[ci] = '\0';
  for (uint8_t i = 0; cmd[i]; i++)
    if (cmd[i] >= 'a' && cmd[i] <= 'z')
      cmd[i] -= 32;
  while (*p == ' ')
    p++;

  if (!strcmp(cmd, "PING"))
    handlePing();
  else if (!strcmp(cmd, "STATUS"))
    handleStatus();
  else if (!strcmp(cmd, "STOP"))
    handleStop();
  else if (!strcmp(cmd, "DRIVE"))
    handleDrive(p);
  else if (!strcmp(cmd, "AIM"))
    handleAim(p);
  else if (!strcmp(cmd, "AIMF"))
    handleAimF(p);
  else if (!strcmp(cmd, "HOME"))
    handleHome();
  else if (!strcmp(cmd, "TILT"))
    handleTilt(p);
  else if (!strcmp(cmd, "TILT_ASYNC"))
    handleTiltAsync(p);
  else if (!strcmp(cmd, "SPIN"))
    handleSpin(p);
  else if (!strcmp(cmd, "LOAD"))
    handleLoad();
  else if (!strcmp(cmd, "STRIKE"))
    handleStrike(p);
  else if (!strcmp(cmd, "RECOVER"))
    handleRecover();
  else
    sendErr("PARSE");
}

// ── Motor init helpers ─────────────────────────────────────────
// Shared by setup() and handleRecover(). Retries + per-motor diagnostic log.

/**
 * pingRetry — retry ping N times at short intervals.
 *   Mitigates the race when a motor's firmware has not yet stabilized right
 *   after cold boot. At 1Mbps a single motor ping is ~300 µs, adding only
 *   retries × 20 ms of latency.
 */
static bool pingRetry(uint8_t id, uint8_t retries) {
  for (uint8_t k = 0; k < retries; k++) {
    if (dxl.ping(id))
      return true;
    delay(20);
  }
  return false;
}

/**
 * initPosMotor — initialize one Position-family motor.
 *   Sequence: ping → torqueOff → DriveMode(0)/OpMode/Profile → torqueOn.
 *   Checking the response of torqueOn is needed to catch a motor shut down by
 *   hardware-error (on overload/overheat/electrical shock the motor
 *   auto-torque-offs and refuses to re-enable).
 *
 *   op_mode = OP_POSITION         : single rotation (0..4095 wrap) — leveling/tilt
 *           = OP_EXTENDED_POSITION : multi-turn cumulative — loader only
 *
 *   profile_vel / profile_acc : if each is 0, Step Mode (profile disabled, goal
 *     applied immediately → max PWM right from the start). Normal motors use
 *     DXL_PROFILE_VEL/ACC for smooth trapezoidal-trajectory tracking. For a
 *     high-static-friction load like the loader, 0/0.
 *
 *   position_p_gain : the P term of the Position PID (RAM, addr 84). Set larger
 *     than the default 800 to converge the residual error near the goal within
 *     ±DXL_ARRIVED_TOL. Too large → humming/overshoot. Uses the default
 *     DXL_POSITION_P_GAIN (1500).
 *
 *   position_d_gain : the D term of the Position PID (RAM, addr 80). default 0 →
 *     default DXL_POSITION_D_GAIN (400). Suppresses the overshoot/ringing of the
 *     acceleration produced by P → reduces the felt "stepping + vibration" under
 *     weight load / AIMF streaming.
 *
 *   ff_acc_gain : Feedforward 2nd-order (acceleration) gain (RAM, addr 88).
 *     default 0 → default DXL_FEEDFORWARD_ACC_GAIN (200). Adds the trajectory
 *     generator's desired acceleration to PWM ahead of time to compensate
 *     tracking lag/overshoot between micro-trajectories. Effective under AIMF
 *     streaming + load (especially gravity descent where it is a helping force).
 */
static bool initPosMotor(uint8_t id, uint8_t op_mode, uint32_t profile_vel,
                         uint32_t profile_acc, uint16_t position_p_gain,
                         uint16_t position_d_gain, uint16_t ff_acc_gain) {
  if (!pingRetry(id))
    return false;
  dxl.torqueOff(id);
  dxl.writeControlTableItem(DRIVE_MODE, id, 0);
  dxl.setOperatingMode(id, op_mode);
  dxl.writeControlTableItem(PROFILE_VELOCITY, id, profile_vel);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, id, profile_acc);
  dxl.writeControlTableItem(POSITION_D_GAIN, id, position_d_gain);
  dxl.writeControlTableItem(POSITION_P_GAIN, id, position_p_gain);
  dxl.writeControlTableItem(FEEDFORWARD_2ND_GAIN, id, ff_acc_gain);
  if (!dxl.torqueOn(id))
    return false;
  g_dxl_target[id] = (int32_t)dxl.getPresentPosition(id);
  return true;
}

/**
 * initVelMotor — initialize one Velocity Mode motor (wheel).
 *   ping → torqueOff → OpMode(VEL) → torqueOn → goal 0.
 */
static bool initVelMotor(uint8_t id) {
  if (!pingRetry(id))
    return false;
  dxl.torqueOff(id);
  dxl.setOperatingMode(id, OP_VELOCITY);
  if (!dxl.torqueOn(id))
    return false;
  dxl.setGoalVelocity(id, 0);
  return true;
}

/**
 * initLoaderVelMotor — loader (ID 5) specific Velocity Mode init.
 *   v2.0: ends in torque OFF state. When handling a LOAD command, loaderRotateBy
 *   cycles briefly through torqueOn → rotate → torqueOff. Normally the user can
 *   free-rotate the loader by hand to load a ball.
 *
 *   Sequence: ping → torqueOff → OpMode(VEL) → write P-gain to RAM → goal_vel=0
 *   → stay torqueOff. Aside from ping, torque is never turned on, so the motor
 *   never moves (cold-boot safe).
 */
static bool initLoaderVelMotor() {
  if (!pingRetry(ID_LOAD))
    return false;
  dxl.torqueOff(ID_LOAD);
  dxl.setOperatingMode(ID_LOAD, OP_VELOCITY);
  dxl.writeControlTableItem(VELOCITY_P_GAIN, ID_LOAD, LOADER_VEL_P_GAIN);
  dxl.writeControlTableItem(VELOCITY_I_GAIN, ID_LOAD, LOADER_VEL_I_GAIN);
  // Set goal_vel to 0 in advance — so the motor does not suddenly jump at the next torqueOn.
  dxl.setGoalVelocity(ID_LOAD, 0);
  // No torque ON. Stays in free-wheel state until the next LOAD explicitly turns it on.
  return true;
}

/**
 * ensureLoaderReady — pre-check right before entering LOAD.
 *   HARDWARE_ERROR_STATUS != 0 → recoverLoader (reboot + re-init). If a previous
 *   LOAD left the motor self-shut-down from overload, the next torqueOn itself
 *   fails.
 *
 *   v2.0: does not check torque state — normally torque off is the normal state,
 *   and loaderRotateBy explicitly torqueOns on entry.
 *
 *   returns true = can operate normally, false = recovery failed (caller sends ERR).
 */
static bool ensureLoaderReady() {
  int32_t hw_err = dxl.readControlTableItem(HARDWARE_ERROR_STATUS, ID_LOAD);
  if (hw_err != 0) {
    return recoverLoader();
  }
  return true;
}

/**
 * recoverLoader — auto-recover the loader (ID 5) from a Hardware shutdown state.
 *   Called when the motor set Hardware Error Status and self-shut-down torque
 *   due to overheat/overload etc. reboot clears all volatile state + the error
 *   flag → re-run init in Velocity Mode → recapture g_loader_goal_raw to the
 *   current position (so the next LOAD accumulates from where the motor stopped
 *   short of the goal).
 *
 *   delay(500) + initVelMotor retried 3 times: the XL/XM datasheet recommends
 *   200 ms+, but on cold reboot a case was observed where it responds reliably
 *   only after 500 ms. Even if the first init fails, retry up to 3 times at
 *   200 ms intervals to absorb the race.
 */
static bool recoverLoader() {
  dxl.reboot(ID_LOAD);
  delay(500);
  for (uint8_t attempt = 0; attempt < 3; attempt++) {
    if (initLoaderVelMotor()) {
      g_motor_init_failed[ID_LOAD] = false;
      g_loader_goal_raw = (int32_t)dxl.getPresentPosition(ID_LOAD);
      g_dxl_target[ID_LOAD] = g_loader_goal_raw;
      return true;
    }
    delay(200);
  }
  g_motor_init_failed[ID_LOAD] = true;
  return false;
}

/**
 * ensureDxlBaud — bring all IDs to the target baud.
 *   v1.7: fixed the pre-v1.6 bug that returned immediately on the first
 *   successful ping.
 *   1) ping all IDs at target baud → mark responding motors.
 *   2) re-search only the motors that did not respond at target via factory baud.
 *   3) write target baud into the EEPROM of motors found at factory.
 *   4) reconnect at target baud → final verification log.
 *
 *   Returns: true if at least one responds at target baud.
 *         (false means the bus itself is dead → ERR INIT_BAUD)
 */
static bool ensureDxlBaud() {
  const uint8_t all_ids[] = {ID_LVL_1, ID_LVL_2,   ID_LVL_3,  ID_TILT,
                             ID_LOAD,  ID_WHEEL_L, ID_WHEEL_R};
  const uint8_t n = sizeof(all_ids) / sizeof(all_ids[0]);
  bool ok_at_target[7] = {false};

  // (1) full check at target baud
  dxl.begin(DXL_BAUDRATE);
  delay(20);
  uint8_t ok_count = 0;
  for (uint8_t i = 0; i < n; i++) {
    if (pingRetry(all_ids[i], 2)) {
      ok_at_target[i] = true;
      ok_count++;
    }
  }
  if (ok_count == n)
    return true; // normal path — all motors already at target baud

  // (2) search + upgrade only the missing motors at factory baud
  dxl.begin(DXL_BAUDRATE_FACTORY);
  delay(20);
  for (uint8_t i = 0; i < n; i++) {
    if (ok_at_target[i])
      continue;
    if (!pingRetry(all_ids[i], 2))
      continue;
    dxl.torqueOff(all_ids[i]);
    dxl.setBaudrate(all_ids[i], DXL_BAUDRATE); // EEPROM write
    delay(30);
    Serial.print("INFO BAUD_UPGRADED ");
    Serial.println(all_ids[i]);
  }

  // (3) reconnect at target baud + final verification
  delay(100);
  dxl.begin(DXL_BAUDRATE);
  delay(20);
  bool any_alive = false;
  for (uint8_t i = 0; i < n; i++) {
    if (pingRetry(all_ids[i], 2)) {
      any_alive = true;
    } else {
      Serial.print("ERR INIT_PING_ID");
      Serial.println(all_ids[i]);
    }
  }
  return any_alive;
}

// ═══════════════════════════════════════════════════════════════
//  setup
// ═══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  while (!Serial)
    ; // remove for standalone battery operation

  // T-motor ESC arming
  escTop.attach(PIN_ESC_TOP);
  escBot.attach(PIN_ESC_BOT);
  escTop.writeMicroseconds(ESC_MIN_US);
  escBot.writeMicroseconds(ESC_MIN_US);
  delay(2000);

  // DXL bus — ensure operating baud (auto-upgrade if a motor is at factory baud)
  dxl.setPortProtocolVersion(DXL_PROTOCOL);
  if (!ensureDxlBaud()) {
    g_err_latched = true;
    Serial.println("ERR INIT_BAUD");
  }

  // Position Mode: ID 1~4 (leveling + tilt)
  // initPosMotor forces Drive Mode = 0 (Velocity-based profile).
  // Because with a Time-based profile, AIMF 50 ms streaming resets the timer
  // every time and the motor effectively stops (see the v1.5 note).
  // The loader (ID 5) is in Velocity Mode — Position Mode's PWM∝error limit
  // cannot overcome static friction (see the v1.9 note).
  const uint8_t pos_ids[] = {ID_LVL_1, ID_LVL_2, ID_LVL_3, ID_TILT};
  for (uint8_t i = 0; i < 4; i++) {
    uint8_t id = pos_ids[i];
    if (initPosMotor(id)) {
      g_motor_init_failed[id] = false;
    } else {
      g_motor_init_failed[id] = true;
      g_err_latched = true;
      Serial.print("ERR INIT_ID");
      Serial.println(id);
    }
  }

  // Velocity Mode: ID 5 (loader) — 2-stage VELOCITY_P_GAIN (MOVE/HOLD) variant
  if (initLoaderVelMotor()) {
    g_motor_init_failed[ID_LOAD] = false;
  } else {
    g_motor_init_failed[ID_LOAD] = true;
    g_err_latched = true;
    Serial.print("ERR INIT_ID");
    Serial.println(ID_LOAD);
  }

  // Capture the loader (ID 5) reference point — define the boot-time current
  // position as 0°.
  // No setGoalPosition call: the motor does not move anywhere and stays in place.
  // Thereafter every LOAD accumulates +90° CCW from this reference point.
  if (!g_motor_init_failed[ID_LOAD]) {
    g_loader_goal_raw = (int32_t)dxl.getPresentPosition(ID_LOAD);
    g_dxl_target[ID_LOAD] = g_loader_goal_raw;
  }

  // Velocity Mode: ID 6~7 (XC430 wheels)
  const uint8_t whl_ids[] = {ID_WHEEL_L, ID_WHEEL_R};
  for (uint8_t i = 0; i < 2; i++) {
    uint8_t id = whl_ids[i];
    if (initVelMotor(id)) {
      g_motor_init_failed[id] = false;
    } else {
      g_motor_init_failed[id] = true;
      g_err_latched = true;
      Serial.print("ERR INIT_ID");
      Serial.println(id);
    }
  }

  // ── Initialize Sync Write structs (once) ─────────────────
  // Thereafter wheelSetVelocity/wheelStop/syncMoveLevel update only the data
  // values and call dxl.syncWrite()
  g_sw_vel_xel[0].id = ID_WHEEL_L;
  g_sw_vel_xel[0].p_data = (uint8_t *)&g_sw_vel_data[0];
  g_sw_vel_xel[1].id = ID_WHEEL_R;
  g_sw_vel_xel[1].p_data = (uint8_t *)&g_sw_vel_data[1];
  g_sw_vel.addr = 104; // Goal Velocity
  g_sw_vel.addr_length = 4;
  g_sw_vel.p_xels = g_sw_vel_xel;
  g_sw_vel.xel_count = 2;

  g_sw_pos_xel[0].id = ID_LVL_1;
  g_sw_pos_xel[0].p_data = (uint8_t *)&g_sw_pos_data[0];
  g_sw_pos_xel[1].id = ID_LVL_2;
  g_sw_pos_xel[1].p_data = (uint8_t *)&g_sw_pos_data[1];
  g_sw_pos_xel[2].id = ID_LVL_3;
  g_sw_pos_xel[2].p_data = (uint8_t *)&g_sw_pos_data[2];
  g_sw_pos.addr = 116; // Goal Position
  g_sw_pos.addr_length = 4;
  g_sw_pos.p_xels = g_sw_pos_xel;
  g_sw_pos.xel_count = 3;

  // Sync Read: leveling 3-axis PRESENT_POSITION (waitMotion fast path)
  g_sr_pos_xel[0].id = ID_LVL_1;
  g_sr_pos_xel[0].p_recv_buf = (uint8_t *)&g_sr_pos_data[0];
  g_sr_pos_xel[1].id = ID_LVL_2;
  g_sr_pos_xel[1].p_recv_buf = (uint8_t *)&g_sr_pos_data[1];
  g_sr_pos_xel[2].id = ID_LVL_3;
  g_sr_pos_xel[2].p_recv_buf = (uint8_t *)&g_sr_pos_data[2];
  g_sr_pos.addr = 132; // Present Position
  g_sr_pos.addr_length = 4;
  g_sr_pos.p_xels = g_sr_pos_xel;
  g_sr_pos.xel_count = 3;

  g_last_drive_ms = millis();
}

// ═══════════════════════════════════════════════════════════════
//  loop
// ═══════════════════════════════════════════════════════════════
void loop() {
  // 1. Serial receive & line assembly (shared with the inside of waitMotion).
  //    NOTE: inside drainSerial(), handleDrive / handleTiltAsync may update
  //    g_last_*_ms = millis(), so `now` must be captured *after* drainSerial().
  //    Otherwise a uint32_t underflow (now < g_last_*_ms → near 4B) exceeds the
  //    WATCHDOG_MS threshold every iteration, and an arriving DRIVE gets
  //    overwritten by wheelStop() in the same iteration.
  drainSerial();

  uint32_t now = millis();

  // 2. DRIVE 200 ms watchdog
  //    The XC430 internal loop keeps running, so an explicit velocity=0 write is needed.
  if (!g_estop && g_drive_active) {
    if ((now - g_last_drive_ms) > WATCHDOG_MS && !g_watchdog_tripped) {
      wheelStop();
      g_watchdog_tripped = true;
    }
  }

  // 3. TILT_ASYNC 200 ms watchdog → hold current position.
  //    Unlike DRIVE (which forces velocity=0), tilt must not snap to a
  //    default angle — 0° would let the camera fall. Snapshot present
  //    position and rewrite it as the goal.
  if (g_last_tilt_async_ms != 0 &&
      (now - g_last_tilt_async_ms) > WATCHDOG_MS) {
    int32_t present = (int32_t)dxl.getPresentPosition(ID_TILT);
    dxl.setGoalPosition(ID_TILT, (uint32_t)present);
    g_last_tilt_async_ms = 0;   // re-armed on next TILT_ASYNC
  }
}
