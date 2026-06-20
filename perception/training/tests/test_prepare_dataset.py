from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from perception.training.prepare_dataset import (
    discover_scenarios,
    pair_images_with_labels,
)


def _make_synthetic_dataset(root: Path, scenarios: dict[str, list[tuple[str, str]]]):
    """
    scenarios: { "scenario_01_4m_left": [("img_a", "0 .5 .5 .1 .1"), ("img_b", None), ...] }
    None label means no .txt file. Empty-string label means zero-byte file.
    """
    for scen_dir, items in scenarios.items():
        scen_id = scen_dir.split("_")[1]
        imgs = root / "imgs" / scen_dir
        labs = root / "labels" / f"{scen_id}_labels"
        imgs.mkdir(parents=True)
        labs.mkdir(parents=True)
        for stem, label in items:
            (imgs / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # tiny valid-ish jpg
            if label is not None:
                (labs / f"{stem}.txt").write_text(label)


class TestDiscoverScenarios(unittest.TestCase):
    def test_pairs_image_dirs_with_label_dirs_by_nn_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_synthetic_dataset(root, {
                "scenario_01_4m_left":   [("a", "0 .5 .5 .1 .1")],
                "scenario_02_4m_middle": [("b", "0 .5 .5 .1 .1")],
            })
            result = discover_scenarios(root)
            ids = sorted(r[0] for r in result)
            self.assertEqual(ids, ["01", "02"])
            for sid, imgs, labs in result:
                self.assertTrue(imgs.is_dir())
                self.assertTrue(labs.is_dir())
                self.assertIn(f"scenario_{sid}_", imgs.name)
                self.assertEqual(labs.name, f"{sid}_labels")

    def test_raises_when_label_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "imgs" / "scenario_99_2m_left").mkdir(parents=True)
            # no labels/99_labels
            with self.assertRaises(FileNotFoundError):
                discover_scenarios(root)


class TestPairing(unittest.TestCase):
    def test_skips_images_without_label_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_synthetic_dataset(root, {
                "scenario_01_4m_left": [
                    ("a", "0 .5 .5 .1 .1"),
                    ("b", None),                  # no .txt
                    ("c", "0 .3 .3 .2 .2"),
                ],
            })
            imgs = root / "imgs" / "scenario_01_4m_left"
            labs = root / "labels" / "01_labels"
            pairs = pair_images_with_labels(imgs, labs)
            stems = sorted(p[0].stem for p in pairs)
            self.assertEqual(stems, ["a", "c"])

    def test_skips_zero_byte_label_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_synthetic_dataset(root, {
                "scenario_01_4m_left": [
                    ("a", "0 .5 .5 .1 .1"),
                    ("b", ""),                    # zero-byte .txt
                ],
            })
            imgs = root / "imgs" / "scenario_01_4m_left"
            labs = root / "labels" / "01_labels"
            pairs = pair_images_with_labels(imgs, labs)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0].stem, "a")


from perception.training.prepare_dataset import stratified_split


class TestStratifiedSplit(unittest.TestCase):
    def _fake_pairs(self, n: int, prefix: str) -> list:
        return [(Path(f"/tmp/{prefix}_{i}.jpg"), Path(f"/tmp/{prefix}_{i}.txt"))
                for i in range(n)]

    def test_split_sizes_per_scenario_match_ratios(self):
        pairs_by_scenario = {
            "01": self._fake_pairs(100, "s01"),
            "02": self._fake_pairs(100, "s02"),
        }
        splits = stratified_split(pairs_by_scenario, ratios=(0.8, 0.1, 0.1), seed=42)
        # 100 * (0.8, 0.1, 0.1) = (80, 10, 10) per scenario, 200 total split as 160/20/20
        self.assertEqual(len(splits["train"]), 160)
        self.assertEqual(len(splits["val"]), 20)
        self.assertEqual(len(splits["test"]), 20)

    def test_split_handles_uneven_counts_without_dropping(self):
        # 7 items, 0.8/0.1/0.1 -> 5/1/1 (with last bucket taking remainder)
        pairs_by_scenario = {"01": self._fake_pairs(7, "s01")}
        splits = stratified_split(pairs_by_scenario, ratios=(0.8, 0.1, 0.1), seed=42)
        total = sum(len(splits[k]) for k in ("train", "val", "test"))
        self.assertEqual(total, 7)
        self.assertGreaterEqual(len(splits["train"]), 5)
        self.assertGreaterEqual(len(splits["val"]), 1)
        self.assertGreaterEqual(len(splits["test"]), 1)

    def test_split_is_deterministic_with_seed(self):
        pairs_by_scenario = {"01": self._fake_pairs(50, "s01")}
        a = stratified_split(pairs_by_scenario, seed=42)
        b = stratified_split(pairs_by_scenario, seed=42)
        self.assertEqual([p[0].name for p in a["train"]],
                         [p[0].name for p in b["train"]])

    def test_split_changes_with_different_seed(self):
        pairs_by_scenario = {"01": self._fake_pairs(50, "s01")}
        a = stratified_split(pairs_by_scenario, seed=42)
        b = stratified_split(pairs_by_scenario, seed=7)
        self.assertNotEqual([p[0].name for p in a["train"]],
                            [p[0].name for p in b["train"]])

    def test_no_overlap_between_splits(self):
        pairs_by_scenario = {"01": self._fake_pairs(100, "s01")}
        splits = stratified_split(pairs_by_scenario, seed=42)
        train_set = {p[0].name for p in splits["train"]}
        val_set   = {p[0].name for p in splits["val"]}
        test_set  = {p[0].name for p in splits["test"]}
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set  & test_set), 0)


from perception.training.prepare_dataset import (
    build_symlink_tree,
    write_dataset_yaml,
    prepare,
)


class TestBuildSymlinkTree(unittest.TestCase):
    def test_creates_expected_directory_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_imgs = root / "src_imgs"; src_imgs.mkdir()
            src_labs = root / "src_labs"; src_labs.mkdir()
            (src_imgs / "a.jpg").write_bytes(b"\xff")
            (src_labs / "a.txt").write_text("0 .5 .5 .1 .1")
            splits = {
                "train": [(src_imgs / "a.jpg", src_labs / "a.txt")],
                "val": [], "test": [],
            }
            out = root / "data"
            build_symlink_tree(splits, out)
            self.assertTrue((out / "images" / "train" / "a.jpg").is_symlink())
            self.assertTrue((out / "labels" / "train" / "a.txt").is_symlink())
            for split in ("val", "test"):
                self.assertTrue((out / "images" / split).is_dir())
                self.assertTrue((out / "labels" / split).is_dir())

    def test_raises_on_filename_collision_within_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d1 = root / "d1"; d1.mkdir()
            d2 = root / "d2"; d2.mkdir()
            (d1 / "x.jpg").write_bytes(b"\xff")
            (d2 / "x.jpg").write_bytes(b"\xff")
            (d1 / "x.txt").write_text("0 .5 .5 .1 .1")
            (d2 / "x.txt").write_text("0 .5 .5 .1 .1")
            splits = {
                "train": [(d1 / "x.jpg", d1 / "x.txt"),
                          (d2 / "x.jpg", d2 / "x.txt")],
                "val": [], "test": [],
            }
            with self.assertRaises(ValueError) as ctx:
                build_symlink_tree(splits, root / "data")
            self.assertIn("collision", str(ctx.exception).lower())


class TestWriteDatasetYaml(unittest.TestCase):
    def test_writes_yaml_with_correct_keys_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = root / "dataset.yaml"
            write_dataset_yaml(root / "data", yaml_path, ["bell"])
            content = yaml_path.read_text()
            self.assertIn(f"path: {(root / 'data').resolve()}", content)
            self.assertIn("train: images/train", content)
            self.assertIn("val: images/val", content)
            self.assertIn("test: images/test", content)
            self.assertIn("0: bell", content)


class TestPrepareOrchestrator(unittest.TestCase):
    def _build_dataset(self, root: Path, n_per_scenario: int = 30):
        scenarios = {
            "scenario_01_4m_left":   [(f"img_{i:03d}", "0 .5 .5 .1 .1")
                                      for i in range(n_per_scenario)],
            "scenario_02_2m_middle": [(f"img_{n_per_scenario + i:03d}", "0 .5 .5 .1 .1")
                                      for i in range(n_per_scenario)],
        }
        _make_synthetic_dataset(root, scenarios)

    def test_prepare_creates_yaml_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "ds"; ds.mkdir()
            self._build_dataset(ds)
            train_root = root / "training"
            yaml_path = prepare(ds, train_root)
            self.assertTrue(yaml_path.is_file())
            self.assertTrue((train_root / "data" / "images" / "train").is_dir())
            n_imgs = sum(1 for _ in (train_root / "data" / "images" / "train").iterdir())
            self.assertGreater(n_imgs, 0)

    def test_prepare_is_idempotent_skips_when_yaml_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "ds"; ds.mkdir()
            self._build_dataset(ds)
            train_root = root / "training"
            prepare(ds, train_root)
            mtime_before = (train_root / "dataset.yaml").stat().st_mtime
            # second call should be a no-op
            prepare(ds, train_root)
            mtime_after = (train_root / "dataset.yaml").stat().st_mtime
            self.assertEqual(mtime_before, mtime_after)

    def test_prepare_rebuild_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "ds"; ds.mkdir()
            self._build_dataset(ds)
            train_root = root / "training"
            prepare(ds, train_root)
            # touch a stale file inside data/ to verify rebuild wipes it
            stale = train_root / "data" / "images" / "train" / "stale.jpg"
            stale.write_bytes(b"x")
            prepare(ds, train_root, rebuild=True)
            self.assertFalse(stale.exists())


import io
import contextlib

from perception.training.prepare_dataset import warn_on_nonzero_classes


class TestWarnOnNonzeroClasses(unittest.TestCase):
    def test_warns_when_label_has_non_zero_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok = root / "ok.txt"; ok.write_text("0 .5 .5 .1 .1")
            bad = root / "bad.txt"; bad.write_text("2 .5 .5 .1 .1\n0 .3 .3 .1 .1")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                warn_on_nonzero_classes([(Path("dummy.jpg"), ok),
                                         (Path("dummy.jpg"), bad)])
            out = buf.getvalue()
            self.assertIn("warning", out.lower())
            self.assertIn("bad.txt", out)

    def test_silent_when_all_labels_are_class_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok = root / "ok.txt"; ok.write_text("0 .5 .5 .1 .1\n0 .3 .3 .1 .1")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                warn_on_nonzero_classes([(Path("dummy.jpg"), ok)])
            self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
