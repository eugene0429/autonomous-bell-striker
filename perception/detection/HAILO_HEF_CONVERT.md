# indoor.pt → indoor.hef conversion guide (Linux only)

Procedure for compiling `perception/detection/indoor.pt` (YOLO26n, 1-class) into a `.hef` for the Hailo-8L.
Compilation is only possible on **x86_64 Linux**. Execution is on Pi5 + AI Hat+ (Hailo-8L).

Uses the [DanielDubinsky/yolo26_hailo](https://github.com/DanielDubinsky/yolo26_hailo) pipeline.

---

## 0. Prerequisites

### Done on the Mac in advance (already complete)
- `perception/detection/indoor.onnx` (9.4MB) — exported with `opset=11, imgsz=640, nms=False`
- Verified the 6 end_nodes exist:
  ```
  /model.23/one2one_cv3.{0,1,2}/one2one_cv3.{0,1,2}.2/Conv  → (1, 1, S, S)   [cls, 1 class]
  /model.23/one2one_cv2.{0,1,2}/one2one_cv2.{0,1,2}.2/Conv  → (1, 4, S, S)   [reg, l/t/r/b]
  ```
  S ∈ {80, 40, 20} (stride 8/16/32)

### Linux machine requirements
- **OS**: Ubuntu 22.04 LTS (or 20.04). 24.04 is outside the official Hailo DFC support matrix, so not recommended.
- **CPU**: x86_64
- **RAM**: 16GB or more recommended (the quantization step is memory-heavy)
- **Disk**: 30GB free (SDK + dependencies)
- **GPU**: not needed (the DFC uses CPU only)

### Files to transfer (Mac → Linux)
```bash
# On the Mac
scp perception/detection/indoor.onnx user@LINUX:~/yolo26_hailo_work/
rsync -avz perception/dataset/images/ user@LINUX:~/yolo26_hailo_work/calib_images/
```
`perception/dataset/images/` contains 532 training data images — reuse them as-is for the calibration set.

---

## 1. Installing the Hailo SDK (Linux)

### 1-1. Hailo Developer Zone account + file download

A [hailo.ai/developer-zone](https://hailo.ai/developer-zone/) account is required (free). After logging in, download the following 2 items from Software Downloads:

| File | Purpose | Approx. size |
|---|---|---|
| `hailo_dataflow_compiler-X.X.X-py3-none-linux_x86_64.whl` | ONNX → HEF compiler | ~500MB |
| `hailort-X.X.X.deb` or the `hailo-all` apt repository | (optional) if you also want to run inference tests on Linux | — |

> The Pi5-side HailoRT is separate. Here only the compiler (DFC) is required.

### 1-2. Python virtual environment + DFC installation

The DFC supports **Python 3.10 or 3.8** (re-check the version in your own SDK release notes).

```bash
# System dependencies
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev \
    build-essential graphviz libgraphviz-dev

# Create venv
python3.10 -m venv ~/hailo_dfc_venv
source ~/hailo_dfc_venv/bin/activate
pip install --upgrade pip

# Install the DFC (path to the downloaded .whl)
pip install ~/Downloads/hailo_dataflow_compiler-*.whl

# Verify
python -c "from hailo_sdk_client import ClientRunner; print('DFC OK')"
hailo --version
```

If you get an error like `Could not load shared library`, install additional system dependencies from the SDK notes such as `libgraphviz`, `libdrm-dev`, etc.

---

## 2. Cloning the yolo26_hailo repo + dependencies

```bash
cd ~
git clone https://github.com/DanielDubinsky/yolo26_hailo.git
cd yolo26_hailo

# Add dependencies into the same venv
pip install -r requirements.txt
# Key: ultralytics==8.4.7 (even though the export was done on the Mac with 8.4.35, the end_nodes matching is verified)
# tensorflow is also needed (the quantize step uses it as the calibration loader):
pip install tensorflow
```

Place the ONNX and calibration set inside the repo:
```bash
mkdir -p models data
cp ~/yolo26_hailo_work/indoor.onnx models/
cp -r ~/yolo26_hailo_work/calib_images data/
```

---

## 3. Code patch for the 1-class model

The [export/config.py](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/export/config.py) and `.alls` in `yolo26_hailo` are independent of the number of classes — **usable as-is**. No patch is needed for the conversion itself.

The inference post-processing (`python/common.py`) assumes COCO 80-class, so a patch is needed at inference time. You can fix it after compilation, but to get it all done in one pass, do it now:

```bash
# Back up python/common.py
cp python/common.py python/common.py.bak
```

Replace the `shape_to_name` in [python/common.py:165-172](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/python/common.py#L165) with the 1-class version:

```python
# Before (80-class COCO)
self.shape_to_name = {
    (1, 80, 80, 80): 'cls_80',
    (1, 40, 40, 80): 'cls_40',
    (1, 20, 20, 80): 'cls_20',
    (1, 80, 80, 4):  'reg_80',
    (1, 40, 40, 4):  'reg_40',
    (1, 20, 20, 4):  'reg_20',
}

# After (1-class indoor.pt)
self.shape_to_name = {
    (1, 80, 80, 1): 'cls_80',
    (1, 40, 40, 1): 'cls_40',
    (1, 20, 20, 1): 'cls_20',
    (1, 80, 80, 4): 'reg_80',
    (1, 40, 40, 4): 'reg_40',
    (1, 20, 20, 4): 'reg_20',
}
```

In [python/common.py:227](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/python/common.py#L227): `cls_flat = cls_data.reshape(-1, 80)` → `reshape(-1, 1)`.

Replace the `COCO_CLASSES` dictionary in [python/common.py:333-342](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/python/common.py#L333):
```python
cls.COCO_CLASSES = {0: 'item'}   # indoor.pt's names = {0: 'item'}
```

---

## 4. Running the compilation

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

### Pipeline progress (each step, estimated duration — i7-class baseline)

| Step | Description | Time |
|---|---|---|
| 1. extract | Split the ONNX into backbone (`images→6 convs`) and head (`6 convs→output0`). Uses `onnx.utils.extract_model`. | ~10s |
| 2. convert | Backbone ONNX → HAR. `ClientRunner.translate_onnx_model()`. | ~30s |
| 3. quantize | fp32→int8 quantization with 1024 calibration images (or all available if fewer). `runner.optimize(images, data_type="dataset")`. **This takes the longest.** | 10~30 min |
| 4. compile | quantized HAR → HEF. `runner.compile()`. Multi-context splitting is automatic. | 5~20 min |

### Output location

```
~/yolo26_hailo/experiments/yolo26n_hailo8l_indoor_YYYYMMDD_HHMMSS/
├── run.log                              # full log
├── model_script.alls                    # the .alls used (for reproduction)
└── artifacts/
    ├── 0_subgraphs/
    │   ├── yolo26n_backbone.onnx
    │   └── yolo26n_head.onnx
    ├── 1_har/model.har
    ├── 2_quantized/model_quantized.har
    └── 3_compiled/model.hef             # ← final output
```

### Verification

```bash
RUN_DIR=~/yolo26_hailo/experiments/yolo26n_hailo8l_indoor_*
HEF=$(ls $RUN_DIR/artifacts/3_compiled/model.hef | tail -1)
ls -lh "$HEF"

# Check the output vstream structure (for 1-class, the cls channel should be 1)
hailortcli parse-hef "$HEF"
# Expected output:
#   Input:  images  uint8  1x640x640x3
#   Output: ...one2one_cv3.0...  float32  1x80x80x1   ← 1-class
#   Output: ...one2one_cv2.0...  float32  1x80x80x4
#   ... (6 outputs total)
```

If the last dimension of the `cls` output is `80` → the trained model has 80 classes. Check the nc in your model's training step.

---

## 5. Deploying to the Pi5 and inference testing

```bash
# Transfer from Linux to the Pi
scp "$HEF" pi@<PI_IP>:~/CapstoneDesign2026/hailo_models/indoor.hef

# It's good to also place yolo26_hailo's patched common.py into the Pi's hailo venv
scp python/common.py pi@<PI_IP>:~/yolo26_hailo/python/
scp python/detect_image.py pi@<PI_IP>:~/yolo26_hailo/python/
```

On the Pi5:
```bash
source ~/CapstoneDesign2026/.venv311_hailo/bin/activate   # existing venv
cd ~/yolo26_hailo
python python/detect_image.py /path/to/test_indoor.jpg \
    --hef ~/CapstoneDesign2026/hailo_models/indoor.hef \
    --conf-threshold 0.25 \
    --output /tmp/det.jpg
```

Expected FPS (single-image latency equivalent): YOLO26n @ Hailo-8L ≈ **10~12ms** (Hailo alone), end-to-end including the Python head **~50~80 FPS**.

Benchmark:
```bash
python python/benchmark_inference.py --hef ~/CapstoneDesign2026/hailo_models/indoor.hef --iterations 500
```

---

## 6. Integrating into our codebase

The current `perception/detection/realtime_infer_hailo.py` assumes `HAILO_NMS_BY_CLASS` output ([:80-99](realtime_infer_hailo.py#L80)), so it is **incompatible** with the YOLO26 HEF. Integration is a separate task:

1. Replace `parse_hailo_nms_by_class` with yolo26_hailo's `_run_python_head` logic (about 80 lines)
2. Leave `realtime_infer_hailo.py`'s `find_latest_hef()` as-is so it finds `hailo_models/indoor.hef`
3. For Phase 1 integration, perform the same Python head decoding inside `RealRobot.get_visual_servo_detection()`

It is recommended to do this integration after the HEF is verified.

---

## 7. Troubleshooting

| Symptom | Cause / response |
|---|---|
| `head input not found` in `extract_model` | The ultralytics version at ONNX export time is too different from yolo26_hailo. Re-export with 8.4.7 |
| Quantize OOM | Reduce the number of calib images. In `.alls`, `calibset_size=1024` → 256 |
| `Agent infeasible` compile error | `.alls` already includes `performance_param(compiler_optimization_level=max)` — if it still fails, check `context_switch_param(mode=allowed)` |
| `Failed to load HEF` on the Pi | The Pi's HailoRT major version must be ≥ the Linux DFC major version. Check versions with `hailortcli fw-control identify` |
| All inference results have low conf | Quantization accuracy loss. Insufficient calibration set diversity is a common cause. Re-compile mixing in the validation split in addition to `perception/dataset/images/` |

---

## References

- `.alls` script meanings: [export/yolo26n.alls](https://github.com/DanielDubinsky/yolo26_hailo/blob/main/export/yolo26n.alls)
  - `normalization([0,0,0], [255,255,255])` — Hailo normalizes the uint8 input to fp/255. Same as ToTensor() during model training
  - `optimization_level=2` — standard quantization. Between 0 (fast) and 2 (accurate)
  - `clipping_values=[0.01, 99.99]` — top/bottom 0.01% percentile clipping (quantization stability)
- Inspecting a HAR file: `hailo har-info path/to/model.har`
- Inspecting a compiled HEF: `hailo profiler path/to/model.hef`
