# ORB-SLAM3 stability debug plan

In the D435i + ORB-SLAM3 RGB-D (no-IMU, pi_mode) combination, after the first attempt the C++ subprocess dies within 10–15 frames (≈ 1.5–2.5s @ 6fps), and the watchdog repeatedly restarts it on a 30s cycle.

## 1. Problem definition

**Symptom**: run `python orbslam_localizer.py` → vocab load (~25s) → camera open → POSE printed 10–15 times → C++ binary exits → watchdog restarts after hardware_reset → dies the same way → infinite repeat on a 30s cycle.

**Kernel signals**:
```
uvcvideo 3-1:1.2: Failed to set UVC probe control : -32 (exp. 48)
usb 3-1: USB disconnect, device number N
usb 3-1: new SuperSpeed USB device number N+1
```
errno -32 = EPIPE (USB control endpoint STALL).

## 2. Facts known so far

- **The first attempt right after unplugging and replugging the USB tends to work relatively well** → residual USB state / driver attach state is a variable.
- Cable: genuine USB 3.2. Port: USB 3.0 SuperSpeed (Bus 003 or 005, both 5000M).
- Load-reduction attempts (`nFeatures 1000→500`, `Camera.fps 15→6`) were applied but the symptom did not change → ARM CPU load is likely not the direct cause.
- The watchdog's `_flush_realsense(hardware_reset=True)` also fails to recover it → not resolved by simple USB re-enumeration.
- The udev rule `99-realsense-libusb.rules` is installed (permissions 0666). However, **the uvcvideo kernel module is still loaded and attached to the D435i interface** (confirmed via lsmod, journalctl).
- One prior run moved to Bus 5-1 and streamed normally — same cable / same code.
- Host is a Pi 5 (kernel 6.8.0-1053-raspi, BCM2712).
- **If it happens to survive at some point during the repeated restarts, it then runs stably from there on** — there is a narrow race window in the startup stage, and whether it passes is stochastic. Once it enters stable streaming mode, it no longer dies within the same session.

## 3. Core hypotheses (in order of likelihood)

| # | Hypothesis | Basis | Effect if refuted |
|---|------|------|-------------|
| H1 | The kernel `uvcvideo` attaches to the D435i interface and races librealsense over the control endpoint → SET_CUR PROBE STALL → reset | The UVC probe error message itself. Confirmed uvcvideo module attach. The udev rule only grants permissions and does not blacklist UVC | If it still fails after uvcvideo unbind/blacklist, reject H1 |
| H2 | D435i firmware / librealsense version mismatch → SET_CUR PROBE is fired in a form the firmware rejects | Sometimes works even on the same cable/port (depends on firmware state) | If it's the same after firmware / librealsense update, reject |
| H3 | Marginal SuperSpeed signal integrity of the USB cable/port — negotiation passes for the first ~2s but additional SET_CUR causes STALL | Sometimes recovers when moving ports. Known signal-quality issues with Pi 5 USB-C | If it's the same after trying all other ports/cables, reject |
| H4 | The ORB-SLAM3 C++ binary requests a non-standard sequence from librealsense (sensor option set, etc.) → firmware rejects | If standalone librealsense is stable, this hypothesis is strengthened | If it dies with standalone librealsense too, reject |
| H5 | Insufficient Pi 5 USB-C power (D435i peak ≥ 900mA + spike when the emitter turns on) | Disconnect pattern after a certain point | Weakened if `vcgencmd get_throttled` is 0x0 |

## 4. Step-by-step debugging plan

Each step is broken into small pieces so it finishes **within 30 minutes**. Record results in §5 as you go.

### S0. Baseline information gathering (once, 5 min)
- **Goal**: System/firmware/module state snapshot — basis for hypothesis branching.
- **Run**:
  ```bash
  vcgencmd get_throttled
  rs-enumerate-devices -s        # FW version, USB type, serial
  rs-fw-update -l                # whether a firmware update is available
  lsusb -t                       # current attach state + speed
  lsmod | grep uvc
  journalctl -k --since "10 min ago" | grep -E "usb|uvc" | tail -50
  ```
- **Decision**: throttled value 0x0 (normal) / non-zero (under-voltage, etc.). FW version 5.13.x or higher is the latest.
- **Branch**: if there's under-voltage, prioritize H5. If FW is 5.12 or lower, prioritize H2.

### S1. 60-second standalone librealsense test without ORB-SLAM3 (10 min, **most important**)
- **Goal**: Separate whether the problem is on the ORB-SLAM3 side or the camera/USB side.
- **Run**:
  ```bash
  # 6 fps RGB-D stream for 60 seconds. Exactly the same stream combination as SLAM.
  rs-capture &        # or a short python script (below)
  sleep 60
  pkill -f rs-capture
  journalctl -k --since "2 min ago" | grep -E "uvc|usb 3-1|usb 5-1|disconnect"
  ```
  Python mini-script (reproducing the exact conditions):
  ```python
  import pyrealsense2 as rs, time
  p = rs.pipeline(); c = rs.config()
  c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
  c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
  p.start(c); t0=time.time()
  for i in range(60*6):
      p.wait_for_frames(timeout_ms=5000)
      if i % 30 == 0: print(f"{time.time()-t0:5.1f}s  frame {i}")
  p.stop()
  ```
- **Decision**:
  - **No disconnect for 60s** → the cause is on the ORB-SLAM3 binary side (H4). Branch to § S5.
  - **Disconnect around 30s** → camera/USB/UVC conflict (H1/H2/H3/H5). Branch to § S2.

### S2. uvcvideo separation (15 min, verifying H1)
- **Goal**: Confirm whether the conflict between the kernel UVC and librealsense over the control endpoint is the cause.
- **Run (a)**: Temporary unbind — with the camera already attached:
  ```bash
  # Detach the D435i's video interface from uvcvideo
  for i in /sys/bus/usb/drivers/uvcvideo/*-*; do
      echo $(basename "$i") | sudo tee /sys/bus/usb/drivers/uvcvideo/unbind 2>/dev/null
  done
  lsusb -t | grep -A1 8086    # confirm Driver=[none]
  ```
  Then re-run the 60s test from S1.
- **Run (b)**: Block the module itself (permanent):
  ```bash
  # /etc/modprobe.d/blacklist-uvc-realsense.conf
  echo "blacklist uvcvideo" | sudo tee /etc/modprobe.d/blacklist-uvc-realsense.conf
  sudo rmmod uvcvideo uvc 2>/dev/null
  # May affect other UVC cameras (webcams, etc.). Confirm with (a) before applying permanently.
  ```
- **Decision**:
  - Stable after unbind → H1 confirmed. Permanent fix via librealsense `rs2_set_devices_changed_callback` at SLAM start, or by strengthening the udev rule (`ENV{ID_USB_INTERFACE_NUM}=="..." ATTR{authorized}="0"`).
  - Still dies after unbind → reject H1, go to S3.

### S3. Cable/port swap matrix (10 min, verifying H3)
- **Goal**: Isolate the signal integrity problem.
- **Run**: Test each of the combinations below for 30s:
  | Port | Cable | Result |
  |------|--------|------|
  | Bus 003 (3-1) | current cable | |
  | Bus 005 (5-1) | current cable | |
  | Bus 003 | another USB-C cable (if available) | |
  | Bus 005 | another cable | |
  | Bus 002 or 004 (USB 2.0) | current cable | |
- **Decision**: If it dies on USB 2.0 too, the cable/port is irrelevant. If it only dies on a specific combination, H3 confirmed.

### S4. D435i firmware / librealsense update (20 min, verifying H2)
- **Goal**: Whether an SDK/FW mismatch is the cause.
- **Precondition**: Only if S1–S3 were inconclusive.
- **Run**:
  ```bash
  rs-fw-update -l               # print current FW and recommended FW
  # If needed: rs-fw-update -f <signed.bin>
  realsense-viewer --version    # SDK version
  ```
  Recommended D435i FW = 5.13.0.50 or higher (2023+). SDK ≥ 2.54.
- **Decision**: After FW/SDK update, confirm by re-running S1.

### S5. Standalone ORB-SLAM3 binary run + repeat (30 min, verifying H4)
- **Goal**: Bypass the Python wrapper. Compare the librealsense control transfer sequences of dead vs. surviving attempts → pinpoint the race window location.
- **Handling the stochastic pattern**: A single run gives insufficient info. **Run 5 times and preserve the stderr of every attempt**, then compare.
- **Run (single attempt)**:
  ```bash
  cd /home/team1/ORB_SLAM3
  export ORBSLAM_NO_VIEWER=1
  export LD_LIBRARY_PATH=$PWD/lib:$PWD/Thirdparty/DBoW2/lib:$PWD/Thirdparty/g2o/lib
  export LRS_LOG_LEVEL=DEBUG    # librealsense verbose log
  ./Examples/RGB-D/rgbd_realsense_D435i \
      Vocabulary/ORBvoc.txt \
      Examples/RGB-D/RealSense_D435i_pi.yaml \
      > /tmp/s5_${ATTEMPT}.stdout 2> /tmp/s5_${ATTEMPT}.stderr &
  PID=$!
  sleep 45                      # vocab ~25s + streaming ~20s
  kill -TERM $PID 2>/dev/null
  ```
- **Repeat-run script**: 5 attempts → automatically classify dead vs. surviving → diff the last 50 lines.
- **Decision**:
  - If the last stderr line of the dead attempts all stop at the same librealsense call → that call is the race trigger.
  - If the timing of the uvcvideo error differs between dead/surviving attempts → uvcvideo detach race confirmed.

### S6. Strengthening the code-side defenses (worth it regardless of the S1 result, 15 min)
- **E1**: Change the default of `LocalizerConfig.skip_flush_first_attempt` to `False`. Apply Python-side flush + (optional) hardware_reset from the very first attempt.
- **E2**: Separate the dead-detect side-effect of `is_alive()` (automatically setting `_restarting=True`) so it only fires while the watchdog is running. So that the startup-stage retry works as intended.
- **E3**: Always `_flush_realsense(hardware_reset=True)` on the first attempt. Force the unplug-and-replug effect via code.

## 5. Results log

| Step | Date/time | Environment | Result | Next |
|------|------|------|------|------|
| S0   | 2026-05-07 20:30 | Pi 5 Rev1.1, kernel 6.8.0-1053-raspi, D435i FW 5.15.1.55, librealsense 2.55.1.0, MaxPower 720mA, D435i currently on Bus 5-1 | Even just calling `rs-enumerate-devices` immediately produces a burst of `uvcvideo 5-1:1.2: Failed to set UVC probe control : -32`. The uvcvideo module is loaded but lsusb shows Driver=[none] (libusb holds it). FW/SDK/Power are all normal. | Strongly supports H1 (uvcvideo conflict) → proceed to S1 |
| S1   | 2026-05-07 20:32 | Bus 5-1, 640×480 RGB-D @6fps, Python pyrealsense2 standalone 60s | **PASS**: 354 frames / 0 failures / 0 disconnects. The UVC probe error only bursts at start/stop, silent during streaming. | Determined that the camera/USB/uvcvideo is not fatal — H1/H3/H5 weakened, **H4 (ORB-SLAM3-side cause) strongly supported**. S2/S3/S4 deferred, **go straight to S5**. |
| S5a  | 2026-05-07 20:39 | raw `rgbd_realsense_D435i` 5× × 45s, default yaml, static desk | None of the 5 subprocesses died (forced termination by timeout). All looped infinitely at STATE=1 (INIT), POSE 0. Cannot init tracking because of the yaml default intrinsics (`fx=308`). 0 disconnects. | **The user-reported "dies after 10-15 frames" was not reproduced**. The raw binary does not die. Track down the difference on the wrapper side → S5b. |
| S5b  | 2026-05-07 20:55 | wrapper 1× × 180s, archive enabled, static desk | First attempt entered INIT→OK (~11s), **1046 POSE over 180s / did not die / 0 disconnects**. Filled in the yaml with the cached calibration to pass the correct intrinsics (`fx=605`). | The current wrapper runs stably — likely the "stabilized at some point" state the user mentioned. Force a cold-start → S6. |
| S6   | 2026-05-07 21:01 | wrapper 5× × 30s, **hardware_reset (USB power-cycle) right before each attempt**, static desk | None of the 5 subprocesses died. Only Run 1 had tracking OK (incidental slight motion), Runs 2-5 only INIT (init failure in a static environment is normal). The disconnect events were only the normal resets caused by hardware_reset. | A cold-start right after hw_reset is safe. Only the first attempt without hw_reset can trigger the race → E3 |
| E1   | 2026-05-07 21:15 | wrapper 1× after changing default to `skip_flush_first_attempt=False` (hw_reset_on_first_attempt left as False) | **Reproduced the user-reported pattern**: first attempt vocab load → camera open → dies after 3 POSE lines. watchdog respawns after hardware_reset → second attempt tracking OK. | Confirmed that **flush only, no hw_reset, is the narrowest race window**. → set E3 default to True |
| E3   | 2026-05-07 21:18 | wrapper 1× after changing default to `hw_reset_on_first_attempt=True` for verification | start() 32.3s (including 5s hw_reset), did not die after the first attempt entered INIT, **0 respawns**. Tracking OK entry is separate since it's a static environment (needs motion). | Confirmed the first-attempt race window is passed. The wrapper's default behavior is stable. |
| E2   | 2026-05-07 21:18 | Apply the dead-detect side-effect of `is_alive()` only while the watchdog is running. | Code change complete, no regression (passed together during E3 verification). | The startup retry path can now work as intended. |

## 7. Conclusion + follow-up actions

### 7.1 root cause (confirmed as of 2026-05-07)
1. **D435i cold-start race**: When `_flush_realsense` briefly opens/closes the USB pipeline, the camera's internal state enters a race-prone mode. If ORB-SLAM3 starts streaming right after, the subprocess dies within the first ~5–15 frames. USB power-cycling with `hardware_reset` guarantees a clean cold state.
2. **Stochastic pattern**: Even with the same race window, if it happens to pass, that session is stable. → the user's "once it stabilizes at some point, it keeps working well".
3. (Separate) **The startup race of `is_alive()`**: When the subprocess dies, the `_restarting=True` side-effect is set even while the watchdog is not running, turning the `start()` retry loop into dead code. Functionally the watchdog covers it, but a 20s stale wait occurs.

### 7.2 Applied fixes
- `skip_flush_first_attempt`: True → **False** (default for the stability test environment).
- `hw_reset_on_first_attempt`: new option, **True** default. Avoids the cold-start race at a +5s startup-time cost.
- `is_alive()`: apply the dead-detect side-effect only when the watchdog thread is actually running.
- `archive_dir`: new debug option. When set, the dead tmp_dir is preserved instead of deleted on watchdog respawn → enables per-attempt comparison analysis.

### 7.3 User-side verification needed
Confirm that the wrapper with these fixes runs stably from the first attempt in the user's usual environment. If it still dies, preserve all attempts with `LocalizerConfig(archive_dir="/tmp/orbslam_archive")`, then compare `attempt_*/stdout.log` and `attempt_*/stderr.log`.
| S2a  | | | | |
| S2b  | | | | |
| S3   | | | | |
| S4   | | | | |
| S5   | | | | |
| E1   | | | | |
| E2   | | | | |
| E3   | | | | |

## 6. Progress rules
- At the end of each step, write the result in one line in §5, and decide the next step based on that result.
- If a step takes more than 30 minutes, stop and break it into smaller steps.
- Record all commands together with the result of `journalctl -k --since "2 min ago" | grep -E "uvc|usb"`.


## 8. Follow-up finding — mid-run undervoltage failure (2026-05-15)

### 8.1 New symptom
Observed a case where the wrapper with the cold-start fix from 1b209c0 applied **passes the first attempt's INIT normally**, but **dies with a SIGSEGV (rc=-11) mid-run after entering tracking**. A different mode from the cold-start pattern in §1 (dying in the INIT stage within 5–15 frames).

### 8.2 Decisive evidence — time-aligning dmesg / stdout / stderr
Ran `drive_to.py --x 1.0 --y 0.0 --swap-lr --archive-slam` to preserve `/tmp/orbslam_archive/attempt_01`. Cross-reference with the dmesg from the same window:

```
[3632.95]  hwmon hwmon3: Undervoltage detected!       ← undervoltage occurs
[3635.00]  hwmon hwmon3: Voltage normalised           ← recovers after lasting 2 seconds
[3635.66]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32 (exp. 48)   ← 0.66s after recovery
[3635.69]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32             ← STALL ×5
[3635.76]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32
[3635.79]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32
[3635.83]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32
[3636.33]  usb 5-1: USB disconnect, device number 33  ← watchdog cleans up
```

ORB-SLAM3 side (attempt_01):
- `stdout.log`: 9 POSE lines, with `1 dropped frs` 4 times in between → tracking OK state
- `stderr.log`: `STATE: 2` × 8 (TRACKING_OK), **silent SIGSEGV with no error message**

### 8.3 Corrected root cause
Confirmed that H5 (insufficient power) and H1 (UVC STALL) are **not separate causes but two stages of the same sequence**.

```
Battery power → temporary 5V line sag (≥1 second)
   ↓
Pi5 USB host controller renegotiates the control endpoint with the D435i
   ↓
SET_CUR PROBE STALL ×N (errno -32 = EPIPE)
   ↓
NULL deref somewhere in the librealsense → ORB-SLAM3 data path
   ↓
SIGSEGV (rc=-11), silent exit with no error in stderr
```

The race window closed by the fix in §7 is the cold-start window between `_flush_realsense` and the C++ binary starting streaming. **The mid-run renegotiation caused by undervoltage is a new race that opens outside that window**, so the 1b209c0 fix does not cover it.

### 8.4 The test matrix that was uncovered at the time
The area that §7.3's "User-side verification needed" was pointing at:

| Variable | During §5 verification | Actual operation |
|---|---|---|
| Environment | static desk | while the robot is driving (vibration, IR scene change) |
| Power | AC adapter | **battery (undervoltage occurs)** |
| Motor | OFF | 15Hz serial traffic + PWM noise |
| Run count | 1 isolated run | continuous (first run → STOP → second run) |

Among these, the **battery environment** is the decisive variable. On AC, this mode is never triggered (consistent with the user's usual experience).

### 8.5 Mitigation options
| Option | Cost | Effect |
|---|---|---|
| A. 5V/5A PD battery or a powered USB hub dedicated to the D435i | hardware | Fundamental fix. Eliminates the undervoltage itself |
| B. try/catch on the librealsense STALL response on the ORB-SLAM3 C++ side | ORB-SLAM3 rebuild | Graceful exit on a mid-run STALL → watchdog respawns cleanly |
| C. Extend `SafetyConfig.lost_warn_sec` on the wrapper side + grace ABORT during respawn | drive_to.py change (~10 min) | Temporary. Doesn't prevent the SIGSEGV itself, but buys the watchdog time to recover |

A is essential. B is low priority because the ORB-SLAM3 rebuild cost is high. C is a temporary measure to reinforce A.

### 8.6 Added archive tooling
Added the `--archive-slam [DIR]` flag to `drive_to.py` — exposes `LocalizerConfig(archive_dir=...)`. Use it as the first command when reproducing the same mode going forward:
```bash
python3 ./Driving/drive_to.py --x 1.0 --y 0.0 --swap-lr --archive-slam
# On crash, stdout/stderr/yaml are preserved in /tmp/orbslam_archive/attempt_NN/
```
