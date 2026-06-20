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

    def test_clips_bbox_with_floating_point_underflow(self):
        # Real-data regression: some labels in the dataset compute y_min slightly
        # below 0 due to rounding (e.g. cy=0.0146, h=0.0417 -> y_min=-0.006).
        # With clip=True, BboxParams should clamp instead of raising.
        t = build_transform()
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        bboxes = [(0.5242, 0.0146, 0.0484, 0.0417)]  # y_min ~= -0.006
        class_labels = [0]
        # Should not raise
        out = t(image=img, bboxes=bboxes, class_labels=class_labels)
        # Surviving bbox (if any) must be within unit range
        for cx, cy, w, h in out["bboxes"]:
            self.assertGreaterEqual(cy - h / 2, 0.0)
            self.assertLessEqual(cy + h / 2, 1.0)


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


if __name__ == "__main__":
    unittest.main()
