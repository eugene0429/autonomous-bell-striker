"""YOLO26 1-class Hailo-8/8L inference (Python head decode).

Ports the dual-head decode from DanielDubinsky/yolo26_hailo (`python/common.py
::HailoPythonInferenceEngine._run_python_head`), specialized for:

  - 1-class HEF (cls channel dim = 1; compiled per perception/detection/HAILO_HEF_CONVERT.md)
  - single-frame BGR ndarray input (no file I/O)
  - single highest-conf bbox return (Phase 1 visual-servo: spec §9 single-object assumption)

NMS is intentionally skipped — the caller picks one bbox per frame, and overlapping
anchors of the same object all decode to ~the same box.

hailo_platform import is at module level, so import this module only when the
hailo backend is actually selected (Pi5 venv has the package; Mac does not).
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    InferVStreams,
)

BBox = Tuple[int, int, int, int]

IMGSZ = 640
STRIDES = (8, 16, 32)
GRID_SIZES = (80, 40, 20)


def _letterbox_rgb(img_bgr: np.ndarray, target: int = IMGSZ) -> Tuple[np.ndarray, float, int, int]:
    """BGR ndarray → uint8 RGB letterboxed (target, target, 3); returns (canvas, scale, px, py)."""
    h, w = img_bgr.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    px = (target - new_w) // 2
    py = (target - new_h) // 2
    canvas[py:py + new_h, px:px + new_w] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), scale, px, py


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class HailoYolo26Detector:
    """Hailo-8/8L YOLO26 1-class detector with Python head decode.

    Must be used as a context manager — VDevice, InferVStreams, and network_group
    activation each have their own lifecycle and are composed via ExitStack.

    Usage:
        with HailoYolo26Detector(hef_path, conf=0.25) as det:
            result = det.predict(color_bgr)   # Optional[(bbox_xyxy, conf)]
    """

    # NHWC last-3 dims → tensor role. 1-class indoor.hef shape verified via
    # ClientRunner.get_hn_dict() (see HAILO_HEF_CONVERT.md §4 verification).
    _SHAPE_TO_ROLE = {
        (80, 80, 1): "cls_80",
        (40, 40, 1): "cls_40",
        (20, 20, 1): "cls_20",
        (80, 80, 4): "reg_80",
        (40, 40, 4): "reg_40",
        (20, 20, 4): "reg_20",
    }

    def __init__(self, hef_path: Path, conf: float):
        self.hef_path = Path(hef_path)
        if not self.hef_path.is_file():
            raise FileNotFoundError(f"HEF not found: {self.hef_path}")
        if not (0.0 < conf < 1.0):
            raise ValueError(f"conf must be in (0, 1), got {conf}")
        self.conf = float(conf)
        # Pre-sigmoid logit threshold: sigmoid(logit) > conf  ⇔  logit > -ln(1/conf - 1)
        self._logit_thr = float(-np.log(1.0 / self.conf - 1.0))
        self._stack: Optional[ExitStack] = None
        self._pipeline: Optional[InferVStreams] = None
        self._in_name: Optional[str] = None

    def __enter__(self) -> "HailoYolo26Detector":
        stack = ExitStack()
        stack.__enter__()
        try:
            hef = HEF(str(self.hef_path))
            target = stack.enter_context(VDevice())
            cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            ng = target.configure(hef, cfg)[0]
            ng_params = ng.create_params()
            in_p = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
            out_p = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
            self._in_name = hef.get_input_vstream_infos()[0].name
            self._pipeline = stack.enter_context(InferVStreams(ng, in_p, out_p))
            stack.enter_context(ng.activate(ng_params))
            self._stack = stack
            return self
        except Exception:
            stack.close()
            raise

    def __exit__(self, *exc) -> None:
        if self._stack is not None:
            self._stack.__exit__(*exc)
        self._stack = None
        self._pipeline = None
        self._in_name = None

    def predict(self, color_bgr: np.ndarray) -> Optional[Tuple[BBox, float]]:
        """Run inference + decode + return highest-conf bbox in original-image coords."""
        if self._pipeline is None or self._in_name is None:
            raise RuntimeError("HailoYolo26Detector must be used as a context manager")

        h0, w0 = color_bgr.shape[:2]
        canvas_rgb, scale, px, py = _letterbox_rgb(color_bgr, IMGSZ)
        inp = canvas_rgb[None, ...]   # (1, 640, 640, 3) uint8

        out = self._pipeline.infer({self._in_name: inp})

        tensors = {}
        for arr in out.values():
            shape = arr.shape
            if len(shape) == 4 and shape[0] == 1:
                role = self._SHAPE_TO_ROLE.get(shape[1:])
                if role is not None:
                    tensors[role] = arr[0]

        best_conf = -1.0
        best_box: Optional[BBox] = None
        for stride, grid_dim in zip(STRIDES, GRID_SIZES):
            cls = tensors.get(f"cls_{grid_dim}")     # (H, W, 1)
            reg = tensors.get(f"reg_{grid_dim}")     # (H, W, 4)
            if cls is None or reg is None:
                continue

            cls_flat = cls.reshape(-1)                # (H*W,)
            mask = cls_flat > self._logit_thr
            if not mask.any():
                continue

            indices = np.where(mask)[0]
            scores = _sigmoid(cls_flat[indices])
            j = int(np.argmax(scores))
            score = float(scores[j])
            if score <= best_conf:
                continue

            i = int(indices[j])
            row, col = divmod(i, grid_dim)
            l, t, r, b = reg.reshape(-1, 4)[i]
            x1m = (col + 0.5 - float(l)) * stride
            y1m = (row + 0.5 - float(t)) * stride
            x2m = (col + 0.5 + float(r)) * stride
            y2m = (row + 0.5 + float(b)) * stride
            # undo letterbox: model-space px → original-image px
            x1 = int(round((x1m - px) / scale))
            y1 = int(round((y1m - py) / scale))
            x2 = int(round((x2m - px) / scale))
            y2 = int(round((y2m - py) / scale))
            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0 - 1, x2))
            y2 = max(0, min(h0 - 1, y2))

            best_conf = score
            best_box = (x1, y1, x2, y2)

        if best_box is None:
            return None
        return best_box, best_conf
