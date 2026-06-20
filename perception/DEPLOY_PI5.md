# Raspberry Pi 5 Deployment Guide

Guide for deploying the Perception project (ORB-SLAM3 + Python VIO) to a Raspberry Pi 5.

## Overall structure

There are two main things that need to be deployed to the Pi:

1. **ORB-SLAM3** (C++ binary) - must be built directly on the Pi
2. **perception** (Python project) - copy as-is, then only fix paths

```
~/perception/          # Python project
~/ORB_SLAM3/           # ORB-SLAM3 (built on the Pi)
```

---

## Step 1: Pi base environment setup

### OS
- **Ubuntu 24.04 LTS (Noble Numbat) 64-bit** for Raspberry Pi 5
- Must use 64-bit (aarch64)

### Installing system packages

```bash
sudo apt update && sudo apt upgrade -y

# Build tools
sudo apt install -y build-essential cmake git pkg-config

# OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv

# Eigen3
sudo apt install -y libeigen3-dev

# Pangolin dependencies
sudo apt install -y libgl1-mesa-dev libglew-dev libwayland-dev \
    libxkbcommon-dev wayland-protocols

# Boost (for ORB-SLAM3 serialization)
sudo apt install -y libboost-serialization-dev libssl-dev

# Python
sudo apt install -y python3-pip python3-venv python3-numpy

# RealSense SDK
sudo apt install -y libusb-1.0-0-dev
```

### Expanding swap (preventing build OOM)

The Pi5 4GB model may run out of memory during the build, so increase the swap (can be skipped on the 8GB model):

```bash
# Ubuntu 24.04 manages the swapfile directly instead of dphys-swapfile
# Disable existing swap
sudo swapoff -a

# Create a 2GB swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Persist across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

free -h  # verify
```

### Installing the RealSense SDK (source build for the Pi)

There is no apt package on the Pi, so a source build is required:

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.55.1  # stable version

mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=false \
    -DBUILD_GRAPHICAL_EXAMPLES=false \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DFORCE_RSUSB_BACKEND=true
make -j4   # Pi5 can use -j4
sudo make install
sudo ldconfig
```

Install the udev rules (camera access permissions):
```bash
cd ~/librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Step 2: Transferring and building the modified ORB-SLAM3 files

### 2-1. List of modified files

**Files modified directly** in the desktop ORB-SLAM3 (`/home/sim2real1/WALJU/deps/ORB_SLAM3`):

| File | Description |
|------|------|
| `Examples/RGB-D/rgbd_realsense_D435i.cc` | RGB-D mode - frame output, calibration override, reads resolution/FPS from YAML |
| `Examples/RGB-D/RealSense_D435i.yaml` | RGB-D config (640x480@30fps) |
| `Examples/RGB-D/RealSense_D435i_pi.yaml` | **Pi-optimized config** (424x240@15fps, 500 features, 4 levels) |
| `Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc` | RGB-D+IMU mode - same modifications applied |
| `Examples/RGB-D-Inertial/RealSense_D435i.yaml` | RGB-D-Inertial config |

### 2-2. Transferring the full source

Copy the entire ORB-SLAM3 to the Pi (excluding build artifacts, must be rebuilt on the Pi):

```bash
# Run on the desktop
rsync -avz --progress \
    --exclude='build/' \
    --exclude='lib/*.so' \
    --exclude='Examples/*/rgbd_realsense_D435i' \
    --exclude='Examples/*/rgbd_inertial_realsense_D435i' \
    --exclude='Examples/*/rgbd_tum' \
    --exclude='Examples/*/stereo_*' \
    --exclude='Examples/*/mono_*' \
    --exclude='Thirdparty/DBoW2/build/' \
    --exclude='Thirdparty/DBoW2/lib/' \
    --exclude='Thirdparty/g2o/build/' \
    --exclude='Thirdparty/g2o/lib/' \
    --exclude='Thirdparty/Sophus/build/' \
    /home/sim2real1/WALJU/deps/ORB_SLAM3/ \
    pi@<PI_IP>:~/ORB_SLAM3/
```

Or, to send **only the modified files** separately (when the original ORB-SLAM3 is already on the Pi):
```bash
scp /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D/rgbd_realsense_D435i.cc \
    /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D/RealSense_D435i.yaml \
    /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D/RealSense_D435i_pi.yaml \
    /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc \
    /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D-Inertial/RealSense_D435i.yaml \
    pi@<PI_IP>:~/ORB_SLAM3_modified/
```

### 2-3. Building Pangolin (Pi)

```bash
# Run on the Pi
cd ~
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
git checkout v0.8
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install
sudo ldconfig
```

### 2-4. Building ORB-SLAM3 (Pi)

```bash
# Run on the Pi
cd ~/ORB_SLAM3

# 1) Decompress the Vocabulary
cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

# 2) Build the third-party libraries
cd Thirdparty/DBoW2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../g2o
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../Sophus
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../../

# 3) Main ORB-SLAM3 build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j3   # On the Pi5 you can use -j3~-j4 (8GB: -j4, 4GB: -j3)
```

> **Build time**: A full build on the Pi5 takes about 15~30 minutes (roughly 2x faster than Pi4).
> Since `-march=native` is in CMakeLists.txt, it is automatically optimized for the Pi5's ARM Cortex-A76.

### 2-5. Verifying the build

```bash
ls -la ~/ORB_SLAM3/Examples/RGB-D/rgbd_realsense_D435i
ls -la ~/ORB_SLAM3/Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i
ls -la ~/ORB_SLAM3/lib/libORB_SLAM3.so
```

---

## Step 3: Transferring the Python project

### 3-1. Copying the perception project

```bash
# Run on the desktop
rsync -avz --progress \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='dataset/' \
    --exclude='models/' \
    --exclude='.claude/' \
    --exclude='Pangolin/' \
    --exclude='librealsense/' \
    /home/sim2real1/CapstoneDesign2026/perception/ \
    pi@<PI_IP>:~/perception/
```

### 3-2. Installing Python dependencies

```bash
# Run on the Pi
cd ~/perception

# Create a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy Pillow psutil
pip install opencv-python  # can be skipped when using the system opencv
pip install ultralytics    # YOLO (only when using detect mode)
```

> **pyrealsense2 note**: `pip install pyrealsense2` may not be supported on Pi ARM.
> If you built librealsense with `-DBUILD_PYTHON_BINDINGS=true` in Step 1,
> it is installed automatically into the system Python. When using a venv, a symbolic link is required:
> ```bash
> # Find the pyrealsense2.so location
> find /usr/local/lib -name "pyrealsense2*" 2>/dev/null
>
> # Link it into the venv (adjust for your Python version)
> ln -s /usr/local/lib/python3.12/dist-packages/pyrealsense2* \
>       ~/perception/venv/lib/python3.12/site-packages/
> ```

### 3-3. Fixing paths (required)

`vio/orbslam_runner.py` line 23 — ORB-SLAM3 path:

```python
# Before
ORBSLAM3_DIR = "/home/sim2real1/WALJU/deps/ORB_SLAM3"

# After
ORBSLAM3_DIR = os.path.expanduser("~/ORB_SLAM3")
```

`vio/orbslam_runner.py` ~line 365 — library path:

```python
# Before
rs_lib = "/usr/lib/x86_64-linux-gnu"

# After (Pi 64-bit)
rs_lib = "/usr/lib/aarch64-linux-gnu"
```

---

## Step 4: Running

### Running in Pi-optimized mode (recommended)

```bash
cd ~/perception

# ORB-SLAM3 RGB-D mode (no IMU, Pi-optimized)
python3 main.py orbslam --pi --no-imu

# ORB-SLAM3 RGB-D-Inertial mode (with IMU, Pi-optimized)
python3 main.py orbslam --pi

# Custom VIO mode (without ORB-SLAM3)
python3 main.py vio --no-imu
```

### Pi-optimized config comparison

| Parameter | Default | Pi value | Effect |
|----------|--------|-------|------|
| Resolution | 640x480 | 424x240 | ~3x fewer pixels |
| FPS | 30 | 15 | 50% less CPU load |
| ORB Features | 1250 | 500 | 60% fewer feature extractions |
| Pyramid Levels | 8 | 4 | 50% less memory/compute |

> **Note**: The Pi5 is roughly 2~3x faster than the Pi4, so if you have headroom you can also run
> with the default config (640x480@30fps) without Pi optimization. Try testing without the `--pi` flag.

### Headless mode (over SSH without a monitor)

```bash
# Disable the Pangolin viewer
export ORBSLAM_NO_VIEWER=1

# SSH with X forwarding (to see OpenCV windows)
ssh -X pi@<PI_IP>

# Or, to run fully headless, you need to comment out the
# cv2.imshow-related code in orbslam_runner.py
```

---

## Step 5: Troubleshooting

### Out of memory during build (OOM Killed)

```bash
# Check swap
free -h

# If it still fails, build with -j1
make -j1
```

### RealSense camera not recognized

```bash
# Check USB devices
lsusb | grep Intel

# Check with Python
python3 -c "import pyrealsense2 as rs; print(rs.context().query_devices())"

# In case of a permission issue
sudo usermod -aG video $USER
# Log out and log back in
```

### libORB_SLAM3.so not found

```bash
# Add the shared library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/ORB_SLAM3/lib:~/ORB_SLAM3/Thirdparty/DBoW2/lib:~/ORB_SLAM3/Thirdparty/g2o/lib

# Apply permanently
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/ORB_SLAM3/lib:~/ORB_SLAM3/Thirdparty/DBoW2/lib:~/ORB_SLAM3/Thirdparty/g2o/lib' >> ~/.bashrc
```

### OpenCV version issue

```bash
# Must be 4.4 or higher to build ORB-SLAM3
pkg-config --modversion opencv4
# Ubuntu 24.04 includes OpenCV 4.6+, so usually no problem
```

---

## Summary checklist

- [ ] Install Ubuntu 24.04 64-bit
- [ ] Install system packages (cmake, opencv, eigen3, boost, etc.)
- [ ] Expand swap to 2GB (4GB model only)
- [ ] Source-build librealsense + Python bindings
- [ ] Install udev rules
- [ ] Build Pangolin
- [ ] Transfer the full ORB-SLAM3 source (excluding build artifacts)
- [ ] Build ORB-SLAM3 third-party → main in order (`-j3~-j4`)
- [ ] Transfer the perception Python project
- [ ] Install Python dependencies
- [ ] Fix `orbslam_runner.py` paths (ORBSLAM3_DIR, rs_lib)
- [ ] Test with `python3 main.py orbslam --pi --no-imu`
