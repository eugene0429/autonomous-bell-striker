from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from perception.training.visualize_dataset import (
    draw_bbox_on_image,
    iter_pairs_for_scenario,
    iter_pairs_for_split,
    list_scenarios,
    read_yolo_bboxes,
)


def _seed_split(training_root: Path, split: str,
                originals: list[str], augmented: list[str] = ()) -> None:
    img_dir = training_root / "data" / "images" / split
    lab_dir = training_root / "data" / "labels" / split
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    for stem in (*originals, *augmented):
        cv2.imwrite(str(img_dir / f"{stem}.jpg"),
                    np.full((100, 200, 3), 128, dtype=np.uint8))
        (lab_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2")


class TestIterPairsForSplit(unittest.TestCase):
    def test_train_default_returns_aug_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_split(tr, "train",
                        originals=["img_a", "img_b"],
                        augmented=["img_a_aug0", "img_b_aug3"])
            pairs = iter_pairs_for_split(tr, "train", include_originals=False)
            stems = sorted(p[0].stem for p in pairs)
            self.assertEqual(stems, ["img_a_aug0", "img_b_aug3"])

    def test_train_with_originals_returns_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_split(tr, "train",
                        originals=["img_a"],
                        augmented=["img_a_aug0"])
            pairs = iter_pairs_for_split(tr, "train", include_originals=True)
            stems = sorted(p[0].stem for p in pairs)
            self.assertEqual(stems, ["img_a", "img_a_aug0"])

    def test_val_returns_all_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_split(tr, "val", originals=["v1", "v2"])
            pairs = iter_pairs_for_split(tr, "val")
            stems = sorted(p[0].stem for p in pairs)
            self.assertEqual(stems, ["v1", "v2"])

    def test_raises_when_split_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                iter_pairs_for_split(Path(tmp), "train")

    def test_skips_image_without_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp)
            _seed_split(tr, "val", originals=["v1"])
            # add an extra image with no label
            cv2.imwrite(
                str(tr / "data" / "images" / "val" / "orphan.jpg"),
                np.full((10, 10, 3), 0, dtype=np.uint8),
            )
            pairs = iter_pairs_for_split(tr, "val")
            stems = sorted(p[0].stem for p in pairs)
            self.assertEqual(stems, ["v1"])


class TestReadYoloBboxes(unittest.TestCase):
    def test_parses_single_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.txt"
            p.write_text("0 0.5 0.5 0.2 0.2\n")
            self.assertEqual(read_yolo_bboxes(p), [(0.5, 0.5, 0.2, 0.2)])

    def test_parses_multiple_bboxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.txt"
            p.write_text("0 0.1 0.1 0.05 0.05\n0 0.9 0.9 0.05 0.05\n")
            self.assertEqual(len(read_yolo_bboxes(p)), 2)

    def test_empty_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.txt"
            p.write_text("")
            self.assertEqual(read_yolo_bboxes(p), [])

    def test_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.txt"
            p.write_text("0 0.5 0.5 0.2 0.2\n\n   \n")
            self.assertEqual(len(read_yolo_bboxes(p)), 1)


def _seed_raw_dataset(dataset_root: Path,
                      scenarios: dict[str, list[str]]) -> None:
    """Build a synthetic raw dataset matching the prepare_dataset layout.

    `scenarios` maps directory name (e.g. 'scenario_01_4m_left') to a list of
    image stems. Each image gets a single dummy YOLO label.
    """
    for scen_dir, stems in scenarios.items():
        scen_id = scen_dir.split("_")[1]
        imgs = dataset_root / "imgs" / scen_dir
        labs = dataset_root / "labels" / f"{scen_id}_labels"
        imgs.mkdir(parents=True)
        labs.mkdir(parents=True)
        for stem in stems:
            cv2.imwrite(str(imgs / f"{stem}.jpg"),
                        np.full((100, 200, 3), 128, dtype=np.uint8))
            (labs / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2")


class TestIterPairsForScenario(unittest.TestCase):
    def test_match_by_zero_padded_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp)
            _seed_raw_dataset(ds, {
                "scenario_01_4m_left":   ["a", "b"],
                "scenario_02_4m_middle": ["c"],
            })
            pairs = iter_pairs_for_scenario(ds, "01")
            self.assertEqual(sorted(p[0].stem for p in pairs), ["a", "b"])

    def test_match_by_unpadded_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp)
            _seed_raw_dataset(ds, {"scenario_01_4m_left": ["a"]})
            pairs = iter_pairs_for_scenario(ds, "1")  # unpadded
            self.assertEqual(len(pairs), 1)

    def test_match_by_name_substring(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp)
            _seed_raw_dataset(ds, {
                "scenario_01_4m_left":   ["a"],
                "scenario_06_2m_right":  ["z"],
            })
            pairs = iter_pairs_for_scenario(ds, "2m_right")
            self.assertEqual([p[0].stem for p in pairs], ["z"])

    def test_raises_on_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp)
            _seed_raw_dataset(ds, {"scenario_01_4m_left": ["a"]})
            with self.assertRaises(ValueError) as ctx:
                iter_pairs_for_scenario(ds, "nope")
            self.assertIn("nope", str(ctx.exception))


class TestListScenarios(unittest.TestCase):
    def test_returns_sorted_dir_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp)
            _seed_raw_dataset(ds, {
                "scenario_02_4m_middle": ["b"],
                "scenario_01_4m_left":   ["a"],
            })
            names = list_scenarios(ds)
            self.assertEqual(names,
                             ["scenario_01_4m_left", "scenario_02_4m_middle"])


class TestDrawBboxOnImage(unittest.TestCase):
    def test_returns_copy_does_not_mutate_input(self):
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        before = img.copy()
        out = draw_bbox_on_image(img, [(0.5, 0.5, 0.2, 0.2)])
        # original untouched
        self.assertTrue(np.array_equal(img, before))
        # output differs (rectangle drawn)
        self.assertFalse(np.array_equal(out, before))

    def test_draws_at_expected_pixel_coordinates(self):
        img = np.full((100, 200, 3), 0, dtype=np.uint8)  # black
        # bbox: cx=0.5, cy=0.5, w=0.5, h=0.5 → x1=50, y1=25, x2=150, y2=75
        out = draw_bbox_on_image(
            img, [(0.5, 0.5, 0.5, 0.5)], color=(0, 255, 0), thickness=1,
        )
        # Top edge of the rectangle should be green at y=25, somewhere in [50, 150]
        self.assertTrue(np.array_equal(out[25, 100], np.array([0, 255, 0])))
        # Outside the bbox should still be black
        self.assertTrue(np.array_equal(out[10, 10], np.array([0, 0, 0])))

    def test_handles_empty_bbox_list(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        out = draw_bbox_on_image(img, [])
        self.assertTrue(np.array_equal(out, img))


if __name__ == "__main__":
    unittest.main()
