# indoor.pt → indoor.hef 변환 가이드 (Linux 전용)

`perception/detection/indoor.pt` (YOLO26n, 1-class)를 Hailo-8L용 `.hef`로 컴파일하는 절차.
컴파일은 **x86_64 Linux**에서만 가능. 실행은 Pi5 + AI Hat+(Hailo-8L)에서.

[DanielDubinsky/yolo26_hailo](https://github.com/DanielDubinsky/yolo26_hailo) 파이프라인 사용.

---

## 0. 사전 준비

### Mac에서 미리 한 것 (이미 완료)
- `perception/detection/indoor.onnx` (9.4MB) — `opset=11, imgsz=640, nms=False`로 export 완료
- 6개 end_nodes 존재 검증 완료:
  ```
  /model.23/one2one_cv3.{0,1,2}/one2one_cv3.{0,1,2}.2/Conv  → (1, 1, S, S)   [cls, 1 class]
  /model.23/one2one_cv2.{0,1,2}/one2one_cv2.{0,1,2}.2/Conv  → (1, 4, S, S)   [reg, l/t/r/b]
  ```
  S ∈ {80, 40, 20} (stride 8/16/32)

### Linux 머신 요구사항
- **OS**: Ubuntu 22.04 LTS (또는 20.04). 24.04는 Hailo DFC 공식 지원 매트릭스 밖이라 비추.
- **CPU**: x86_64
- **RAM**: 16GB 이상 권장 (양자화 단계가 메모리 무거움)
- **디스크**: 30GB 여유 (SDK + 의존성)
- **GPU**: 불필요 (DFC는 CPU만 사용)

### 전송할 파일 (Mac → Linux)
```bash
# Mac에서
scp perception/detection/indoor.onnx user@LINUX:~/yolo26_hailo_work/
rsync -avz perception/dataset/images/ user@LINUX:~/yolo26_hailo_work/calib_images/
```
`perception/dataset/images/` 안에 학습 데이터 이미지 532장이 있음 — 그대로 calibration set으로 재사용.

---

## 1. Hailo SDK 설치 (Linux)

### 1-1. Hailo Developer Zone 계정 + 파일 다운로드

[hailo.ai/developer-zone](https://hailo.ai/developer-zone/) 계정 필요 (무료). 로그인 후 Software Downloads에서 다음 2개를 받음:

| 파일 | 용도 | 대략 크기 |
|---|---|---|
| `hailo_dataflow_compiler-X.X.X-py3-none-linux_x86_64.whl` | ONNX → HEF 컴파일러 | ~500MB |
| `hailort-X.X.X.deb` 또는 `hailo-all` apt 저장소 | (선택) Linux에서도 추론 테스트하려면 | — |

> Pi5 측 HailoRT는 별도. 여기선 컴파일러(DFC)만 필수.

### 1-2. Python 가상환경 + DFC 설치

DFC는 **Python 3.10 또는 3.8** 지원 (버전 본인 SDK 릴리스 노트 재확인).

```bash
# 시스템 의존성
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev \
    build-essential graphviz libgraphviz-dev

# venv 생성
python3.10 -m venv ~/hailo_dfc_venv
source ~/hailo_dfc_venv/bin/activate
pip install --upgrade pip

# DFC 설치 (다운로드한 .whl 경로)
pip install ~/Downloads/hailo_dataflow_compiler-*.whl

# 검증
python -c "from hailo_sdk_client import ClientRunner; print('DFC OK')"
hailo --version
```

만약 `Could not load shared library` 같은 에러가 나면 SDK 노트의 `libgraphviz`, `libdrm-dev` 등 시스템 의존성을 추가 설치.

---

## 2. yolo26_hailo 레포 클론 + 의존성

```bash
cd ~
git clone https://github.com/DanielDubinsky/yolo26_hailo.git
cd yolo26_hailo

# 같은 venv에 의존성 추가
pip install -r requirements.txt
# 핵심: ultralytics==8.4.7 (이미 export는 Mac에서 8.4.35로 완료해도, end_nodes 매칭은 검증됨)
# 추가로 tensorflow 필요 (quantize 스텝이 calibration loader로 사용):
pip install tensorflow
```

ONNX와 calibration을 레포 안으로 배치:
```bash
mkdir -p models data
cp ~/yolo26_hailo_work/indoor.onnx models/
cp -r ~/yolo26_hailo_work/calib_images data/
```

---

## 3. 1-class 모델용 코드 패치

`yolo26_hailo`의 [export/config.py](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/export/config.py)와 `.alls`는 클래스 수와 무관 — **그대로 사용 가능**. 변환 자체엔 패치 불필요.

추론 후처리(`python/common.py`)는 COCO 80-class 가정이라 추론 시점에 패치 필요. 컴파일 후에 손봐도 되지만, 어차피 한 번에 끝내려면 지금:

```bash
# python/common.py 백업
cp python/common.py python/common.py.bak
```

[python/common.py:165-172](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/python/common.py#L165) 의 `shape_to_name` 을 1-class용으로 교체:

```python
# 변경 전 (80-class COCO)
self.shape_to_name = {
    (1, 80, 80, 80): 'cls_80',
    (1, 40, 40, 80): 'cls_40',
    (1, 20, 20, 80): 'cls_20',
    (1, 80, 80, 4):  'reg_80',
    (1, 40, 40, 4):  'reg_40',
    (1, 20, 20, 4):  'reg_20',
}

# 변경 후 (1-class indoor.pt)
self.shape_to_name = {
    (1, 80, 80, 1): 'cls_80',
    (1, 40, 40, 1): 'cls_40',
    (1, 20, 20, 1): 'cls_20',
    (1, 80, 80, 4): 'reg_80',
    (1, 40, 40, 4): 'reg_40',
    (1, 20, 20, 4): 'reg_20',
}
```

[python/common.py:227](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/python/common.py#L227) 의 `cls_flat = cls_data.reshape(-1, 80)` → `reshape(-1, 1)`.

[python/common.py:333-342](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/python/common.py#L333) 의 `COCO_CLASSES` 사전을 교체:
```python
cls.COCO_CLASSES = {0: 'item'}   # indoor.pt의 names = {0: 'item'}
```

---

## 4. 컴파일 실행

```bash
cd ~/yolo26_hailo
source ~/hailo_dfc_venv/bin/activate

python -m export.cli \
    --variant yolo26n \
    --target hailo8l \
    --onnx models/indoor.onnx \
    --calib_dir data/calib_images \
    --tag indoor
```

### 파이프라인 진행 (각 스텝, 예상 소요시간 — i7급 기준)

| 스텝 | 내용 | 시간 |
|---|---|---|
| 1. extract | ONNX를 backbone(`images→6개 conv`)과 head(`6개 conv→output0`)로 분할. `onnx.utils.extract_model` 사용. | ~10s |
| 2. convert | Backbone ONNX → HAR. `ClientRunner.translate_onnx_model()`. | ~30s |
| 3. quantize | calibration 1024장(없으면 가용 전부)으로 fp32→int8 양자화. `runner.optimize(images, data_type="dataset")`. **여기서 가장 오래 걸림.** | 10~30분 |
| 4. compile | quantized HAR → HEF. `runner.compile()`. multi-context 분할 자동. | 5~20분 |

### 산출물 위치

```
~/yolo26_hailo/experiments/yolo26n_hailo8l_indoor_YYYYMMDD_HHMMSS/
├── run.log                              # 전체 로그
├── model_script.alls                    # 사용된 .alls (재현용)
└── artifacts/
    ├── 0_subgraphs/
    │   ├── yolo26n_backbone.onnx
    │   └── yolo26n_head.onnx
    ├── 1_har/model.har
    ├── 2_quantized/model_quantized.har
    └── 3_compiled/model.hef             # ← 최종 산출물
```

### 검증

```bash
RUN_DIR=~/yolo26_hailo/experiments/yolo26n_hailo8l_indoor_*
HEF=$(ls $RUN_DIR/artifacts/3_compiled/model.hef | tail -1)
ls -lh "$HEF"

# 출력 vstream 구조 확인 (1-class면 cls 채널이 1이어야 함)
hailortcli parse-hef "$HEF"
# 기대 출력:
#   Input:  images  uint8  1x640x640x3
#   Output: ...one2one_cv3.0...  float32  1x80x80x1   ← 1-class
#   Output: ...one2one_cv2.0...  float32  1x80x80x4
#   ... (총 6개 output)
```

`cls` 출력 마지막 차원이 `80`이면 → 학습 모델 클래스 수가 80인 상태. 본인 모델 학습 단계의 nc 확인.

---

## 5. Pi5로 배포 및 추론 테스트

```bash
# Linux에서 Pi로 전송
scp "$HEF" pi@<PI_IP>:~/CapstoneDesign2026/hailo_models/indoor.hef

# yolo26_hailo의 patched common.py도 Pi의 hailo venv에 두면 좋음
scp python/common.py pi@<PI_IP>:~/yolo26_hailo/python/
scp python/detect_image.py pi@<PI_IP>:~/yolo26_hailo/python/
```

Pi5에서:
```bash
source ~/CapstoneDesign2026/.venv311_hailo/bin/activate   # 기존 venv
cd ~/yolo26_hailo
python python/detect_image.py /path/to/test_indoor.jpg \
    --hef ~/CapstoneDesign2026/hailo_models/indoor.hef \
    --conf-threshold 0.25 \
    --output /tmp/det.jpg
```

기대 FPS (단일 이미지 latency 환산): YOLO26n @ Hailo-8L ≈ **10~12ms** (Hailo 단독), Python head 포함 end-to-end **~50~80 FPS**.

벤치마크:
```bash
python python/benchmark_inference.py --hef ~/CapstoneDesign2026/hailo_models/indoor.hef --iterations 500
```

---

## 6. 우리 코드베이스에 통합

현재 `perception/detection/realtime_infer_hailo.py`는 `HAILO_NMS_BY_CLASS` 출력 가정([:80-99](realtime_infer_hailo.py#L80))이라 YOLO26 HEF와 **호환 안 됨**. 통합은 별도 작업:

1. `parse_hailo_nms_by_class` 를 yolo26_hailo의 `_run_python_head` 로직(약 80줄)으로 교체
2. `realtime_infer_hailo.py` 의 `find_latest_hef()` 가 `hailo_models/indoor.hef` 를 찾도록 그대로 둠
3. Phase 1 통합은 `RealRobot.get_visual_servo_detection()` 안에서 같은 Python head 디코딩 수행

이 통합은 HEF가 검증된 후 진행 권장.

---

## 7. 트러블슈팅

| 증상 | 원인/대응 |
|---|---|
| `extract_model` 에서 `head input not found` | ONNX export 시 ultralytics 버전이 yolo26_hailo와 너무 다름. 8.4.7로 export 재시도 |
| Quantize OOM | calib 이미지 수 줄이기. `.alls`의 `calibset_size=1024` → 256으로 |
| `Agent infeasible` 컴파일 에러 | `.alls`에 `performance_param(compiler_optimization_level=max)` 이미 포함 — 그래도 실패 시 `context_switch_param(mode=allowed)` 확인 |
| Pi에서 `Failed to load HEF` | Pi의 HailoRT 메이저 버전 ≥ Linux의 DFC 메이저 버전이어야 함. `hailortcli fw-control identify` 로 버전 확인 |
| 추론 결과가 전부 conf 낮음 | 양자화 정확도 손실. calibration set 다양성 부족이 흔한 원인. `perception/dataset/images/` 외에 검증 split도 섞어서 재컴파일 |

---

## 참고

- `.alls` 스크립트 의미: [export/yolo26n.alls](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/export/yolo26n.alls)
  - `normalization([0,0,0], [255,255,255])` — Hailo가 입력 uint8을 fp/255 정규화. 모델 학습 시 ToTensor()와 동일
  - `optimization_level=2` — 표준 양자화. 0(빠름) ~ 2(정확) 사이
  - `clipping_values=[0.01, 99.99]` — 상하 0.01% percentile clipping (양자화 안정성)
- HAR 파일 들여다보기: `hailo har-info path/to/model.har`
- 컴파일된 HEF 들여다보기: `hailo profiler path/to/model.hef`
