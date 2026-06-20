# Offline Dataset Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `perception/training/augment_dataset.py` that uses albumentations to write 5 augmented copies per train image (default `--multiplier 5`) to `data/images/train/` and matching label files, leaving val/test untouched. Idempotent with `--rebuild`. No changes to `train.py`.

**Architecture:** A standalone script that runs between `prepare_dataset.py` and `train.py`. Reads originals from `data/images/train/` (excluding any pre-existing `_aug*` files), applies an albumentations pipeline (HSV / rotation ±5° / scale ±10% / brightness-contrast / hflip / Gauss noise), writes results as `<stem>_aug{i}.jpg` + `.txt` alongside originals.

**Tech Stack:** Python 3 stdlib + albumentations 1.4+, opencv-python (existing), numpy (existing). Tests use stdlib `unittest`.

**Spec:** [docs/superpowers/specs/2026-04-27-offline-augmentation-design.md](../specs/2026-04-27-offline-augmentation-design.md)

---

## File Structure

```
perception/
├── requirements.txt              # MODIFY — add albumentations>=1.4.0
└── training/
    ├── augment_dataset.py        # NEW — pipeline + IO + CLI
    └── tests/
        └── test_augment_dataset.py  # NEW — unittest with synthetic fixtures
```

`train.py`, `prepare_dataset.py`, and `dataset.yaml` are untouched.

---

## Task 1: Add albumentations dependency

**Files:**
- Modify: `perception/requirements.txt`

- [ ] **Step 1: Append albumentations to requirements**

Append to [perception/requirements.txt](perception/requirements.txt) (after `Pillow>=10.0.0` and before `pyrealsense2>=2.50.0`):

```
albumentations>=1.4.0
```

- [ ] **Step 2: Install into the venv**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/pip install 'albumentations>=1.4.0'`
Expected: install completes, no conflicts with installed torch/ultralytics.

- [ ] **Step 3: Verify import works**

Run: `.venv/bin/python -c "import albumentations as A; import cv2; print('A', A.__version__, 'cv2', cv2.__version__)"`
Expected: prints two version strings, exit 0.

- [ ] **Step 4: Commit**

```bash
git add perception/requirements.txt
git commit -m "chore: add albumentations>=1.4.0 dep for offline augmentation"
```

---

## Task 2: Discovery and wipe helpers (TDD)

**Files:**
- Create: `perception/training/augment_dataset.py`
- Create: `perception/training/tests/test_augment_dataset.py`

Behaviors:
1. `list_original_train_pairs(training_root)` — returns `[(img_path, label_path), ...]` for originals only (filenames *without* `_aug` in stem).
2. `wipe_augmented(training_root)` — deletes every `*_aug*.jpg` and `*_aug*.txt` in `data/images/train/` and `data/labels/train/`.

- [ ] **Step 1: Write failing tests**

Create [perception/training/tests/test_augment_dataset.py](perception/training/tests/test_augment_dataset.py):

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from perception.training.augment_dataset import (
    list_original_train_pairs,
    wipe_augmented,
)


def _seed_train_dir(training_root: Path,
                    originals: list[str],
                    augmented: list[str] = ()) -> None:
    """Create the data/{images,labels}/train/ skeleton with given stems."""
    img_dir = training_root / "data" / "images" / "train"
    lab_dir = training_root / "data" / "labels" / "train"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    for stem in originals:
        (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (lab_dir / f"{stem}.txt").write_text("0 .5 .5 .1 .1")
    for stem in augmented:
        (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (lab_dir / f"{stem}.txt").write_text("0 .5 .5 .1 .1")


class TestListOriginalTrainPairs(unittest.TestCase):
    def test_returns_only_pairs_without_aug_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_train_dir(tr,
                originals=["img_001", "img_002"],
                augmented=["img_001_aug0", "img_002_aug3"])
            pairs = list_original_train_pairs(tr)
            stems = sorted(p[0].stem for p in pairs)
            self.assertEqual(stems, ["img_001", "img_002"])

    def test_pairs_are_image_label_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_train_dir(tr, originals=["img_a"])
            pairs = list_original_train_pairs(tr)
            self.assertEqual(len(pairs), 1)
            img, lab = pairs[0]
            self.assertEqual(img.suffix, ".jpg")
            self.assertEqual(lab.suffix, ".txt")
            self.assertEqual(img.stem, lab.stem)

    def test_raises_when_train_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                list_original_train_pairs(Path(tmp))


class TestWipeAugmented(unittest.TestCase):
    def test_deletes_aug_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_train_dir(tr,
                originals=["img_001"],
                augmented=["img_001_aug0", "img_001_aug1"])
            wipe_augmented(tr)
            img_dir = tr / "data" / "images" / "train"
            lab_dir = tr / "data" / "labels" / "train"
            remaining_imgs = sorted(p.name for p in img_dir.iterdir())
            remaining_labs = sorted(p.name for p in lab_dir.iterdir())
            self.assertEqual(remaining_imgs, ["img_001.jpg"])
            self.assertEqual(remaining_labs, ["img_001.txt"])

    def test_idempotent_when_no_aug_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_train_dir(tr, originals=["img_001"])
            wipe_augmented(tr)  # should not raise
            img_dir = tr / "data" / "images" / "train"
            self.assertTrue((img_dir / "img_001.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest perception.training.tests.test_augment_dataset -v`
Expected: ImportError on `list_original_train_pairs` and `wipe_augmented`.

- [ ] **Step 3: Create `augment_dataset.py` with the two helpers**

Create [perception/training/augment_dataset.py](perception/training/augment_dataset.py):

```python
"""
Offline dataset augmentation for YOLO26n bell detection.

Reads originals from `data/images/train/` (after `prepare_dataset.py` has
materialised the symlink tree) and writes `multiplier` albumentations-augmented
copies per original to the same dirs. Files are named `<stem>_aug{i}.jpg` /
`<stem>_aug{i}.txt`. Val/test directories are not touched.

Run directly:
    python -m perception.training.augment_dataset                  # multiplier=5, idempotent
    python -m perception.training.augment_dataset --multiplier 10
    python -m perception.training.augment_dataset --rebuild        # wipe *_aug* and regenerate
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def _train_dirs(training_root: Path) -> Tuple[Path, Path]:
    return (
        training_root / "data" / "images" / "train",
        training_root / "data" / "labels" / "train",
    )


def list_original_train_pairs(training_root: Path) -> List[Tuple[Path, Path]]:
    """Return [(img, lab), ...] for originals only (no `_aug` in stem).

    Raises FileNotFoundError if the train image dir doesn't exist
    (i.e. prepare_dataset.py hasn't been run yet).
    """
    img_dir, lab_dir = _train_dirs(training_root)
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"train image dir not found: {img_dir} "
            "(run prepare_dataset.py first)"
        )
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "_aug" in img.stem:
            continue
        lab = lab_dir / f"{img.stem}.txt"
        if not lab.is_file():
            continue
        if lab.stat().st_size == 0:
            continue
        pairs.append((img, lab))
    return pairs


def wipe_augmented(training_root: Path) -> None:
    """Delete every `*_aug*.jpg` / `*_aug*.txt` from train/ image and label dirs."""
    img_dir, lab_dir = _train_dirs(training_root)
    for d in (img_dir, lab_dir):
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if "_aug" in p.stem:
                p.unlink()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m unittest perception.training.tests.test_augment_dataset -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perception/training/augment_dataset.py perception/training/tests/test_augment_dataset.py
git commit -m "feat(augment): add original-pair discovery and aug-file wipe helpers"
```

---

## Task 3: Albumentations transform builder (TDD)

**Files:**
- Modify: `perception/training/augment_dataset.py`
- Modify: `perception/training/tests/test_augment_dataset.py`

Behavior: `build_transform()` returns an `albumentations.Compose` configured with the spec's six transforms and YOLO bbox params (`min_visibility=0.3`).

The test asserts structural properties — that the Compose is callable, accepts a YOLO bbox + class_labels, and returns valid output of the right shapes. We do NOT lock in random output (albumentations randomness is a black box) but we DO assert `random_seed` argument makes it deterministic for the same seed.

- [ ] **Step 1: Append failing tests**

Append to [perception/training/tests/test_augment_dataset.py](perception/training/tests/test_augment_dataset.py) above the `if __name__` block:

```python
import numpy as np

from perception.training.augment_dataset import build_transform


class TestBuildTransform(unittest.TestCase):
    def _dummy_inputs(self):
        # 480x640x3 uint8, one bbox covering the centre 20% of the frame
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        bboxes = [(0.5, 0.5, 0.2, 0.2)]
        class_labels = [0]
        return img, bboxes, class_labels

    def test_returns_callable_compose(self):
        t = build_transform()
        # albumentations Compose is callable as a function
        self.assertTrue(callable(t))

    def test_output_shape_matches_input(self):
        t = build_transform()
        img, bboxes, class_labels = self._dummy_inputs()
        out = t(image=img, bboxes=bboxes, class_labels=class_labels)
        self.assertEqual(out["image"].shape[2], 3)  # still RGB
        self.assertTrue(0 < out["image"].shape[0] <= 720)  # within reason
        self.assertTrue(0 < out["image"].shape[1] <= 960)

    def test_bboxes_stay_in_unit_range(self):
        t = build_transform()
        img, bboxes, class_labels = self._dummy_inputs()
        out = t(image=img, bboxes=bboxes, class_labels=class_labels)
        for cx, cy, w, h in out["bboxes"]:
            self.assertGreaterEqual(cx, 0.0); self.assertLessEqual(cx, 1.0)
            self.assertGreaterEqual(cy, 0.0); self.assertLessEqual(cy, 1.0)
            self.assertGreater(w, 0.0); self.assertLessEqual(w, 1.0)
            self.assertGreater(h, 0.0); self.assertLessEqual(h, 1.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m unittest perception.training.tests.test_augment_dataset -v`
Expected: ImportError on `build_transform`.

- [ ] **Step 3: Implement `build_transform`**

Add to top-of-file imports of [perception/training/augment_dataset.py](perception/training/augment_dataset.py):

```python
import albumentations as A
import cv2
```

Append the function (above existing helpers or below — placement doesn't matter, but keep helpers grouped):

```python
def build_transform() -> A.Compose:
    """Build the per-sample albumentations pipeline.

    Each albumentations call samples its own random state internally; deterministic
    seeding for the whole augmentation run is handled in `augment()` by calling
    `np.random.seed(...)` before each transform invocation.
    """
    return A.Compose(
        [
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=70,
                val_shift_limit=40,
                p=1.0,
            ),
            A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomScale(scale_limit=0.1, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(std_range=(0.0, 0.02), p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m unittest perception.training.tests.test_augment_dataset -v`
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perception/training/augment_dataset.py perception/training/tests/test_augment_dataset.py
git commit -m "feat(augment): build albumentations transform pipeline"
```

---

## Task 4: Generate one augmented sample (TDD)

**Files:**
- Modify: `perception/training/augment_dataset.py`
- Modify: `perception/training/tests/test_augment_dataset.py`

Behavior: `generate_one(img_path, lab_path, out_img, out_lab, transform, seed)` reads the source image+label, applies the transform with a deterministic seed, and writes the result. Empty-bbox case (transform dropped all boxes) writes a zero-byte label file.

- [ ] **Step 1: Append failing tests**

Append to [perception/training/tests/test_augment_dataset.py](perception/training/tests/test_augment_dataset.py):

```python
import cv2 as _cv2

from perception.training.augment_dataset import generate_one


class TestGenerateOne(unittest.TestCase):
    def _write_real_jpg(self, path: Path, h=480, w=640):
        img = np.full((h, w, 3), 128, dtype=np.uint8)
        _cv2.imwrite(str(path), img)

    def test_writes_image_and_label_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_img = root / "src.jpg"; self._write_real_jpg(src_img)
            src_lab = root / "src.txt"; src_lab.write_text("0 0.5 0.5 0.2 0.2\n")
            out_img = root / "out.jpg"; out_lab = root / "out.txt"
            t = build_transform()
            generate_one(src_img, src_lab, out_img, out_lab, t, seed=42)
            self.assertTrue(out_img.is_file())
            self.assertTrue(out_lab.is_file())
            written = _cv2.imread(str(out_img))
            self.assertIsNotNone(written)
            self.assertEqual(written.shape[2], 3)

    def test_label_format_is_yolo_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_img = root / "src.jpg"; self._write_real_jpg(src_img)
            src_lab = root / "src.txt"; src_lab.write_text("0 0.5 0.5 0.2 0.2\n")
            out_img = root / "out.jpg"; out_lab = root / "out.txt"
            t = build_transform()
            generate_one(src_img, src_lab, out_img, out_lab, t, seed=42)
            for line in out_lab.read_text().splitlines():
                parts = line.split()
                self.assertEqual(len(parts), 5)
                self.assertEqual(parts[0], "0")
                for v in parts[1:]:
                    f = float(v)
                    self.assertGreaterEqual(f, 0.0)
                    self.assertLessEqual(f, 1.0)

    def test_deterministic_with_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_img = root / "src.jpg"; self._write_real_jpg(src_img)
            src_lab = root / "src.txt"; src_lab.write_text("0 0.5 0.5 0.2 0.2\n")
            t = build_transform()
            out_a = root / "a.jpg"; lab_a = root / "a.txt"
            out_b = root / "b.jpg"; lab_b = root / "b.txt"
            generate_one(src_img, src_lab, out_a, lab_a, t, seed=42)
            generate_one(src_img, src_lab, out_b, lab_b, t, seed=42)
            self.assertEqual(out_a.read_bytes(), out_b.read_bytes())
            self.assertEqual(lab_a.read_text(), lab_b.read_text())

    def test_empty_label_when_all_bboxes_dropped(self):
        # Bbox at the extreme right edge with min_visibility=0.3 + heavy synthetic
        # dropout via mocked transform that returns no bboxes.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_img = root / "src.jpg"; self._write_real_jpg(src_img)
            src_lab = root / "src.txt"; src_lab.write_text("0 0.5 0.5 0.2 0.2\n")
            out_img = root / "out.jpg"; out_lab = root / "out.txt"

            class _DropAllTransform:
                def __call__(self, image, bboxes, class_labels):
                    return {"image": image, "bboxes": [], "class_labels": []}

            generate_one(src_img, src_lab, out_img, out_lab,
                         _DropAllTransform(), seed=42)
            self.assertTrue(out_lab.is_file())
            self.assertEqual(out_lab.stat().st_size, 0)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m unittest perception.training.tests.test_augment_dataset -v`
Expected: ImportError on `generate_one`.

- [ ] **Step 3: Implement `generate_one`**

Add `import numpy as np` and `import random` to [perception/training/augment_dataset.py](perception/training/augment_dataset.py) imports, then append:

```python
def _read_label_yolo(lab_path: Path) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
    """Parse a YOLO label file into (bboxes, class_labels)."""
    bboxes: list[tuple[float, float, float, float]] = []
    classes: list[int] = []
    for line in lab_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        bboxes.append((cx, cy, w, h))
        classes.append(cls)
    return bboxes, classes


def _write_label_yolo(
    lab_path: Path,
    bboxes: List[Tuple[float, float, float, float]],
    classes: List[int],
) -> None:
    if not bboxes:
        lab_path.write_text("")
        return
    lines = [
        f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cls, (cx, cy, w, h) in zip(classes, bboxes)
    ]
    lab_path.write_text("\n".join(lines) + "\n")


def generate_one(
    img_path: Path,
    lab_path: Path,
    out_img: Path,
    out_lab: Path,
    transform,
    seed: int,
) -> None:
    """Apply `transform` to (img, label) and write to (out_img, out_lab).

    Sets numpy + python random seeds before the transform call so that
    subsequent runs with the same seed produce identical output.
    """
    np.random.seed(seed)
    random.seed(seed)

    image = cv2.imread(str(img_path))
    if image is None:
        raise IOError(f"could not read image: {img_path}")
    bboxes, class_labels = _read_label_yolo(lab_path)

    out = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    cv2.imwrite(str(out_img), out["image"])
    _write_label_yolo(out_lab, list(out["bboxes"]), list(out["class_labels"]))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m unittest perception.training.tests.test_augment_dataset -v`
Expected: 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perception/training/augment_dataset.py perception/training/tests/test_augment_dataset.py
git commit -m "feat(augment): generate one augmented sample with bbox-aware IO"
```

---

## Task 5: `augment()` orchestrator (TDD)

**Files:**
- Modify: `perception/training/augment_dataset.py`
- Modify: `perception/training/tests/test_augment_dataset.py`

Behavior: `augment(training_root, multiplier, rebuild, seed)` discovers originals, generates `multiplier` augmented copies per original, and warns at the end with the count of empty-label augmented files. Idempotent: returns early if any `*_aug*` file already exists in train/ unless `rebuild=True`.

- [ ] **Step 1: Append failing tests**

Append to [perception/training/tests/test_augment_dataset.py](perception/training/tests/test_augment_dataset.py):

```python
import io
import contextlib

from perception.training.augment_dataset import augment


def _seed_real_train_dir(training_root: Path, n_originals: int = 3) -> None:
    img_dir = training_root / "data" / "images" / "train"
    lab_dir = training_root / "data" / "labels" / "train"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    for i in range(n_originals):
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        _cv2.imwrite(str(img_dir / f"img_{i:03d}.jpg"), img)
        (lab_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    # also seed val/test as empty dirs (real prepare creates these)
    (training_root / "data" / "images" / "val").mkdir(parents=True)
    (training_root / "data" / "labels" / "val").mkdir(parents=True)


class TestAugment(unittest.TestCase):
    def test_creates_multiplier_copies_per_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_real_train_dir(tr, n_originals=3)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                augment(tr, multiplier=4, rebuild=False, seed=42)
            img_dir = tr / "data" / "images" / "train"
            lab_dir = tr / "data" / "labels" / "train"
            n_imgs = sum(1 for _ in img_dir.iterdir())
            n_labs = sum(1 for _ in lab_dir.iterdir())
            self.assertEqual(n_imgs, 3 * (1 + 4))  # 3 originals + 12 aug
            self.assertEqual(n_labs, 3 * (1 + 4))

    def test_idempotent_skips_when_aug_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_real_train_dir(tr, n_originals=2)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                augment(tr, multiplier=3, rebuild=False, seed=42)
            n_after_first = sum(1 for _ in (tr / "data" / "images" / "train").iterdir())
            with contextlib.redirect_stdout(io.StringIO()):
                augment(tr, multiplier=3, rebuild=False, seed=42)
            n_after_second = sum(1 for _ in (tr / "data" / "images" / "train").iterdir())
            self.assertEqual(n_after_first, n_after_second)

    def test_rebuild_wipes_then_regenerates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_real_train_dir(tr, n_originals=2)
            with contextlib.redirect_stdout(io.StringIO()):
                augment(tr, multiplier=3, rebuild=False, seed=42)
            stale = tr / "data" / "images" / "train" / "img_000_aug99.jpg"
            stale.write_bytes(b"\xff\xd8\xff\xd9")  # not part of multiplier=3
            with contextlib.redirect_stdout(io.StringIO()):
                augment(tr, multiplier=3, rebuild=True, seed=42)
            # stale file should be gone, exactly multiplier copies present
            self.assertFalse(stale.exists())
            n_imgs = sum(1 for _ in (tr / "data" / "images" / "train").iterdir())
            self.assertEqual(n_imgs, 2 * (1 + 3))

    def test_multiplier_zero_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_real_train_dir(tr, n_originals=2)
            with contextlib.redirect_stdout(io.StringIO()):
                augment(tr, multiplier=0, rebuild=False, seed=42)
            n_imgs = sum(1 for _ in (tr / "data" / "images" / "train").iterdir())
            self.assertEqual(n_imgs, 2)  # originals only

    def test_does_not_touch_val_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_real_train_dir(tr, n_originals=2)
            (tr / "data" / "images" / "val" / "v.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            with contextlib.redirect_stdout(io.StringIO()):
                augment(tr, multiplier=2, rebuild=False, seed=42)
            n_val = sum(1 for _ in (tr / "data" / "images" / "val").iterdir())
            self.assertEqual(n_val, 1)  # untouched
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m unittest perception.training.tests.test_augment_dataset -v`
Expected: ImportError on `augment`.

- [ ] **Step 3: Implement `augment()`**

Append to [perception/training/augment_dataset.py](perception/training/augment_dataset.py):

```python
def augment(
    training_root: Path,
    multiplier: int = 5,
    rebuild: bool = False,
    seed: int = 42,
) -> None:
    """Generate `multiplier` augmented copies per original train pair.

    Idempotent: skips when any `*_aug*` file already exists in train/ unless
    `rebuild=True`. With `rebuild=True`, all `*_aug*` files are deleted first.
    `multiplier=0` is a no-op (returns immediately after wipe if rebuild).
    """
    img_dir, _lab_dir = _train_dirs(training_root)

    has_aug = any("_aug" in p.stem for p in img_dir.iterdir())
    if has_aug and not rebuild:
        print(f"[augment] augmented files already present in {img_dir}, skipping "
              f"(use --rebuild to regenerate)")
        return
    if rebuild:
        wipe_augmented(training_root)

    if multiplier <= 0:
        print("[augment] multiplier=0, no augmentation generated")
        return

    pairs = list_original_train_pairs(training_root)
    transform = build_transform()
    empty_labels = 0

    for src_img, src_lab in pairs:
        for i in range(multiplier):
            out_img = src_img.with_name(f"{src_img.stem}_aug{i}{src_img.suffix}")
            out_lab = src_lab.with_name(f"{src_lab.stem}_aug{i}{src_lab.suffix}")
            sample_seed = seed + hash((src_img.stem, i)) % (2**32)
            generate_one(src_img, src_lab, out_img, out_lab, transform, sample_seed)
            if out_lab.stat().st_size == 0:
                empty_labels += 1

    total_aug = len(pairs) * multiplier
    print(f"[augment] generated {total_aug} augmented copies "
          f"({len(pairs)} originals × {multiplier})")
    if empty_labels:
        pct = 100.0 * empty_labels / total_aug
        print(f"[augment] WARNING: {empty_labels}/{total_aug} ({pct:.1f}%) "
              f"augmented samples have empty labels (bboxes dropped). "
              f"If excessive, re-run with tighter rotation/scale params.")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m unittest perception.training.tests.test_augment_dataset -v`
Expected: 17 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perception/training/augment_dataset.py perception/training/tests/test_augment_dataset.py
git commit -m "feat(augment): orchestrator with idempotent rebuild and empty-label warning"
```

---

## Task 6: CLI entrypoint

**Files:**
- Modify: `perception/training/augment_dataset.py`

- [ ] **Step 1: Append CLI**

Add `import argparse` to the top imports. Append to [perception/training/augment_dataset.py](perception/training/augment_dataset.py):

```python
def main():
    here = Path(__file__).resolve().parent
    default_training = here

    ap = argparse.ArgumentParser(
        description="Offline augmentation for YOLO26n bell-detection train split.",
    )
    ap.add_argument("--training-root", type=Path, default=default_training,
                    help=f"training root containing data/ (default: {default_training})")
    ap.add_argument("--multiplier", type=int, default=5,
                    help="augmented copies per original (default: 5)")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe existing *_aug* files before regenerating")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    augment(args.training_root,
            multiplier=args.multiplier,
            rebuild=args.rebuild,
            seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m perception.training.augment_dataset --help`
Expected: argparse usage text showing `--training-root`, `--multiplier`, `--rebuild`, `--seed`. Exit 0.

- [ ] **Step 3: Confirm tests still pass**

Run: `.venv/bin/python -m unittest perception.training.tests.test_augment_dataset -v`
Expected: 17/17 PASS.

- [ ] **Step 4: Commit**

```bash
git add perception/training/augment_dataset.py
git commit -m "feat(augment): add augment_dataset CLI"
```

---

## Task 7: Run augmentation against the real prepared dataset (verification only)

**Files:** none modified.

This task assumes `perception/training/data/` already exists from earlier prep runs (it does — Task 6 of the previous plan ran prepare against the real dataset and the symlink tree is gitignored but on-disk).

- [ ] **Step 1: Confirm prep state**

Run:
```bash
ls perception/training/data/images/train | wc -l
ls perception/training/data/images/val   | wc -l
ls perception/training/data/images/test  | wc -l
```
Expected: 710 / 87 / 92.

If these don't match, run `.venv/bin/python -m perception.training.prepare_dataset` first (it's idempotent).

- [ ] **Step 2: Run augmentation with default multiplier=5**

Run: `.venv/bin/python -m perception.training.augment_dataset`
Expected output (warning count may vary slightly):
```
[augment] generated 3550 augmented copies (710 originals × 5)
```
And possibly a low-percentage warning about empty-label augmented files (should be < 1%).

- [ ] **Step 3: Verify file counts**

Run:
```bash
ls perception/training/data/images/train | wc -l   # 4260
ls perception/training/data/labels/train | wc -l   # 4260
ls perception/training/data/images/val   | wc -l   # 87 (unchanged)
ls perception/training/data/images/test  | wc -l   # 92 (unchanged)
```

- [ ] **Step 4: Spot-check one augmented file is a valid image**

Run:
```bash
.venv/bin/python -c "
import cv2
from pathlib import Path
p = next(p for p in Path('perception/training/data/images/train').iterdir() if '_aug' in p.stem)
img = cv2.imread(str(p))
print(p.name, 'shape=', img.shape if img is not None else 'FAILED')
"
```
Expected: prints something like `img_NNNNNNNNNNNN_aug2.jpg shape= (480, 640, 3)` (height/width may shift slightly due to RandomScale).

- [ ] **Step 5: Verify idempotency**

Run: `.venv/bin/python -m perception.training.augment_dataset`
Expected: prints `[augment] augmented files already present ... skipping`. File counts unchanged.

- [ ] **Step 6: Verify rebuild works**

Run: `.venv/bin/python -m perception.training.augment_dataset --rebuild`
Expected: regenerates the same total count (4260 train images), prints generation message again.

No commit — this is a verification-only task, and the augmented files are gitignored along with the rest of `data/`.

---

## Task 8: End-to-end smoke train on the augmented set (verification only)

**Files:** none modified.

- [ ] **Step 1: Run a 1-epoch smoke training**

Run:
```bash
.venv/bin/python -m perception.training.train --epochs 1 --name smoke_aug --batch 16
```
Expected:
- ultralytics prints train dataset stats showing **~4260 train images** (vs the previous 710 baseline)
- 1 epoch completes (slower than baseline — roughly 6× the iterations)
- best.pt + last.pt produced under `perception/training/runs/smoke_aug/weights/`
- Test eval section prints `[train] test mAP@0.5 = 0.xxxx`
- exit 0

- [ ] **Step 2: Confirm the train data scan log shows the expanded count**

In the captured stdout from Step 1, look for the ultralytics `train: Scanning ...` line. It should say `4260 images` (or very close) instead of `710`.

- [ ] **Step 3: Cleanup smoke artifacts**

Run: `rm -rf perception/training/runs/smoke_aug perception/training/runs/smoke_aug_test`
Also: `rm -f /home/sim2real/CapstoneDesign2026/yolo26n.pt` (ultralytics auto-downloads this to CWD during AMP check).

No commit — verification only.

---

## Task 9: Full training run with augmented data (manual — user decides when)

**Not part of automated execution.** Once Task 8 passes, run the full schedule on the user's schedule:

```bash
.venv/bin/python -m perception.training.augment_dataset    # idempotent, only runs once
.venv/bin/python -m perception.training.train               # 150 epochs on 4260-image train set
```

Expect ~6× longer wall-clock per epoch versus the un-augmented baseline. Final weights at `perception/training/runs/bell_yolo26n/weights/best.pt`.
