# Raspberry Pi 5 Deployment Guide

Perception 프로젝트(ORB-SLAM3 + Python VIO)를 Raspberry Pi 5에 배포하는 가이드.

## 전체 구조

Pi에 배포해야 하는 것은 크게 두 가지:

1. **ORB-SLAM3** (C++ 바이너리) - Pi에서 직접 빌드 필요
2. **perception** (Python 프로젝트) - 그대로 복사 후 경로만 수정

```
~/perception/          # Python 프로젝트
~/ORB_SLAM3/           # ORB-SLAM3 (Pi에서 빌드)
```

---

## 1단계: Pi 기본 환경 설정

### OS
- **Ubuntu 24.04 LTS (Noble Numbat) 64-bit** for Raspberry Pi 5
- 반드시 64-bit (aarch64) 사용

### 시스템 패키지 설치

```bash
sudo apt update && sudo apt upgrade -y

# 빌드 도구
sudo apt install -y build-essential cmake git pkg-config

# OpenCV 의존성
sudo apt install -y libopencv-dev python3-opencv

# Eigen3
sudo apt install -y libeigen3-dev

# Pangolin 의존성
sudo apt install -y libgl1-mesa-dev libglew-dev libwayland-dev \
    libxkbcommon-dev wayland-protocols

# Boost (ORB-SLAM3 직렬화용)
sudo apt install -y libboost-serialization-dev libssl-dev

# Python
sudo apt install -y python3-pip python3-venv python3-numpy

# RealSense SDK
sudo apt install -y libusb-1.0-0-dev
```

### 스왑 확장 (빌드 OOM 방지)

Pi5 4GB 모델은 빌드 중 메모리 부족할 수 있으므로 스왑을 늘린다 (8GB 모델은 생략 가능):

```bash
# Ubuntu 24.04는 dphys-swapfile 대신 swapfile 직접 관리
# 기존 스왑 비활성화
sudo swapoff -a

# 2GB 스왑 파일 생성
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 재부팅 후에도 유지
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

free -h  # 확인
```

### RealSense SDK 설치 (Pi용 소스 빌드)

Pi에서는 apt 패키지가 없으므로 소스 빌드 필요:

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.55.1  # 안정 버전

mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=false \
    -DBUILD_GRAPHICAL_EXAMPLES=false \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DFORCE_RSUSB_BACKEND=true
make -j4   # Pi5는 -j4 사용 가능
sudo make install
sudo ldconfig
```

udev 규칙 설치 (카메라 접근 권한):
```bash
cd ~/librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 2단계: ORB-SLAM3 수정 파일 전송 및 빌드

### 2-1. 수정된 파일 목록

데스크탑 ORB-SLAM3(`/home/sim2real1/WALJU/deps/ORB_SLAM3`)에서 **직접 수정한 파일들**:

| 파일 | 설명 |
|------|------|
| `Examples/RGB-D/rgbd_realsense_D435i.cc` | RGB-D 모드 - 프레임 출력, 캘리브 오버라이드, YAML 해상도/FPS 읽기 |
| `Examples/RGB-D/RealSense_D435i.yaml` | RGB-D 설정 (640x480@30fps) |
| `Examples/RGB-D/RealSense_D435i_pi.yaml` | **Pi 최적화 설정** (424x240@15fps, 500 features, 4 levels) |
| `Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc` | RGB-D+IMU 모드 - 동일한 수정 적용 |
| `Examples/RGB-D-Inertial/RealSense_D435i.yaml` | RGB-D-Inertial 설정 |

### 2-2. 전체 소스 전송

ORB-SLAM3 전체를 Pi로 복사 (빌드 산출물 제외, Pi에서 다시 빌드해야 함):

```bash
# 데스크탑에서 실행
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

또는 **수정 파일만** 따로 보내려면 (ORB-SLAM3 원본이 Pi에 이미 있을 때):
```bash
scp /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D/rgbd_realsense_D435i.cc \
    /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D/RealSense_D435i.yaml \
    /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D/RealSense_D435i_pi.yaml \
    /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc \
    /home/sim2real1/WALJU/deps/ORB_SLAM3/Examples/RGB-D-Inertial/RealSense_D435i.yaml \
    pi@<PI_IP>:~/ORB_SLAM3_modified/
```

### 2-3. Pangolin 빌드 (Pi)

```bash
# Pi에서 실행
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

### 2-4. ORB-SLAM3 빌드 (Pi)

```bash
# Pi에서 실행
cd ~/ORB_SLAM3

# 1) Vocabulary 압축 해제
cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

# 2) Third-party 라이브러리 빌드
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

# 3) ORB-SLAM3 메인 빌드
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j3   # Pi5에서는 -j3~-j4 사용 가능 (8GB: -j4, 4GB: -j3)
```

> **빌드 시간**: Pi5에서 전체 빌드 약 15~30분 소요 (Pi4 대비 약 2배 빠름).
> `-march=native`가 CMakeLists.txt에 있으므로 Pi5의 ARM Cortex-A76에 맞게 자동 최적화됨.

### 2-5. 빌드 확인

```bash
ls -la ~/ORB_SLAM3/Examples/RGB-D/rgbd_realsense_D435i
ls -la ~/ORB_SLAM3/Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i
ls -la ~/ORB_SLAM3/lib/libORB_SLAM3.so
```

---

## 3단계: Python 프로젝트 전송

### 3-1. perception 프로젝트 복사

```bash
# 데스크탑에서 실행
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

### 3-2. Python 의존성 설치

```bash
# Pi에서 실행
cd ~/perception

# 가상환경 생성 (선택)
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install numpy scipy Pillow psutil
pip install opencv-python  # 시스템 opencv 사용 시 생략 가능
pip install ultralytics    # YOLO (detect 모드 사용 시만)
```

> **pyrealsense2 주의**: `pip install pyrealsense2`는 Pi ARM에서 지원 안 될 수 있음.
> 1단계에서 librealsense를 `-DBUILD_PYTHON_BINDINGS=true`로 빌드했으면
> 시스템 Python에 자동 설치됨. venv 사용 시 심볼릭 링크 필요:
> ```bash
> # pyrealsense2.so 위치 확인
> find /usr/local/lib -name "pyrealsense2*" 2>/dev/null
>
> # venv에 링크 (Python 버전에 맞게 수정)
> ln -s /usr/local/lib/python3.12/dist-packages/pyrealsense2* \
>       ~/perception/venv/lib/python3.12/site-packages/
> ```

### 3-3. 경로 수정 (필수)

`vio/orbslam_runner.py` 23번째 줄 — ORB-SLAM3 경로:

```python
# 변경 전
ORBSLAM3_DIR = "/home/sim2real1/WALJU/deps/ORB_SLAM3"

# 변경 후
ORBSLAM3_DIR = os.path.expanduser("~/ORB_SLAM3")
```

`vio/orbslam_runner.py` ~365번째 줄 — 라이브러리 경로:

```python
# 변경 전
rs_lib = "/usr/lib/x86_64-linux-gnu"

# 변경 후 (Pi 64-bit)
rs_lib = "/usr/lib/aarch64-linux-gnu"
```

---

## 4단계: 실행

### Pi 최적화 모드로 실행 (권장)

```bash
cd ~/perception

# ORB-SLAM3 RGB-D 모드 (IMU 없음, Pi 최적화)
python3 main.py orbslam --pi --no-imu

# ORB-SLAM3 RGB-D-Inertial 모드 (IMU 사용, Pi 최적화)
python3 main.py orbslam --pi

# 커스텀 VIO 모드 (ORB-SLAM3 없이)
python3 main.py vio --no-imu
```

### Pi 최적화 설정 비교

| 파라미터 | 기본값 | Pi 값 | 효과 |
|----------|--------|-------|------|
| 해상도 | 640x480 | 424x240 | 픽셀 수 ~3배 감소 |
| FPS | 30 | 15 | CPU 부하 50% 감소 |
| ORB Features | 1250 | 500 | 특징점 추출 60% 감소 |
| Pyramid Levels | 8 | 4 | 메모리/연산 50% 감소 |

> **참고**: Pi5는 Pi4 대비 약 2~3배 성능이므로, 여유가 있다면 Pi 최적화 없이
> 기본 설정(640x480@30fps)으로도 실행 가능. `--pi` 플래그 없이 테스트해 볼 것.

### Headless 모드 (모니터 없이 SSH)

```bash
# Pangolin viewer 비활성화
export ORBSLAM_NO_VIEWER=1

# X forwarding으로 SSH 접속 (OpenCV 창 보려면)
ssh -X pi@<PI_IP>

# 또는 완전 headless로 돌리려면 orbslam_runner.py에서
# cv2.imshow 관련 코드 주석 처리 필요
```

---

## 5단계: 트러블슈팅

### 빌드 중 메모리 부족 (OOM Killed)

```bash
# 스왑 확인
free -h

# 그래도 안 되면 -j1로 빌드
make -j1
```

### RealSense 카메라 인식 안 됨

```bash
# USB 장치 확인
lsusb | grep Intel

# Python으로 확인
python3 -c "import pyrealsense2 as rs; print(rs.context().query_devices())"

# 권한 문제 시
sudo usermod -aG video $USER
# 로그아웃 후 재로그인
```

### libORB_SLAM3.so 찾을 수 없음

```bash
# shared library 경로 추가
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/ORB_SLAM3/lib:~/ORB_SLAM3/Thirdparty/DBoW2/lib:~/ORB_SLAM3/Thirdparty/g2o/lib

# 영구 적용
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/ORB_SLAM3/lib:~/ORB_SLAM3/Thirdparty/DBoW2/lib:~/ORB_SLAM3/Thirdparty/g2o/lib' >> ~/.bashrc
```

### OpenCV 버전 문제

```bash
# 4.4 이상이어야 ORB-SLAM3 빌드 가능
pkg-config --modversion opencv4
# Ubuntu 24.04는 OpenCV 4.6+ 포함되어 있으므로 보통 문제 없음
```

---

## 요약 체크리스트

- [ ] Ubuntu 24.04 64-bit 설치
- [ ] 시스템 패키지 설치 (cmake, opencv, eigen3, boost 등)
- [ ] 스왑 2GB로 확장 (4GB 모델만)
- [ ] librealsense 소스 빌드 + Python 바인딩
- [ ] udev 규칙 설치
- [ ] Pangolin 빌드
- [ ] ORB-SLAM3 전체 소스 전송 (빌드 산출물 제외)
- [ ] ORB-SLAM3 Third-party → 메인 순서로 빌드 (`-j3~-j4`)
- [ ] perception Python 프로젝트 전송
- [ ] Python 의존성 설치
- [ ] `orbslam_runner.py` 경로 수정 (ORBSLAM3_DIR, rs_lib)
- [ ] `python3 main.py orbslam --pi --no-imu`로 테스트
