# YOLO26n Bell Detection — Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-script YOLO26n training pipeline (`prepare_dataset.py` + `train.py`) that takes the existing 6-scenario bell dataset and produces a deployable single-class detector.

**Architecture:** Stratified 80/10/10 split per scenario, symlink tree for ultralytics, pretrained YOLO26n fine-tuning on RTX 3090, automatic test-set evaluation after training. TDD on the prep script's pure logic (pairing, split, symlinks); smoke test the train script end-to-end.

**Tech Stack:** Python 3, ultralytics 8.4.41 (already installed), torch 2.11+cu130, stdlib `unittest` for tests (no new deps).

**Spec:** [docs/superpowers/specs/2026-04-27-yolo26n-bell-detection-training-design.md](../specs/2026-04-27-yolo26n-bell-detection-training-design.md)

---

## File Structure

```
perception/training/
├── __init__.py             # makes it a package
├── prepare_dataset.py      # NEW — pairing / split / symlinks / yaml
├── train.py                # NEW — ultralytics training + test eval
├── weights/
│   └── yolo26n.pt          # pretrained checkpoint (moved from repo root)
├── tests/
│   ├── __init__.py
│   └── test_prepare_dataset.py   # NEW — unittest, synthetic fixtures
├── dataset.yaml            # generated, gitignored
├── data/                   # generated symlink tree, gitignored
└── runs/                   # ultralytics output, gitignored
```

`.gitignore` additions:

```
perception/training/data/
perception/training/runs/
perception/training/dataset.yaml
perception/training/weights/
```

---

## Task 1: Scaffolding

**Files:**
- Create: `perception/training/__init__.py`
- Create: `perception/training/tests/__init__.py`
- Create: `perception/training/weights/` (dir for pretrained pt)
- Modify: `.gitignore`
- Move: `yolo26n.pt` (repo root) → `perception/training/weights/yolo26n.pt`

- [ ] **Step 1: Create package directories and empty `__init__.py` files**

```bash
mkdir -p perception/training/tests perception/training/weights
touch perception/training/__init__.py perception/training/tests/__init__.py
```

- [ ] **Step 2: Move stray pretrained checkpoint into the weights dir**

```bash
mv yolo26n.pt perception/training/weights/yolo26n.pt
```

- [ ] **Step 3: Append training-local paths to `.gitignore`**

Append to end of [.gitignore](.gitignore):

```
# Training (generated artifacts)
perception/training/data/
perception/training/runs/
perception/training/dataset.yaml
perception/training/weights/
```

- [ ] **Step 4: Verify git sees the right things**

Run: `git status --short`
Expected: shows the new `perception/training/__init__.py`, `perception/training/tests/__init__.py`, and modified `.gitignore`. Does NOT show `yolo26n.pt` (covered by `*.pt` and now `weights/`). Does NOT show `data/` or `runs/` (don't exist yet, but won't show even when they do).

- [ ] **Step 5: Commit**

```bash
git add perception/training/__init__.py perception/training/tests/__init__.py .gitignore
git commit -m "chore: scaffold perception/training package and gitignore generated artifacts"
```

---

## Task 2: `prepare_dataset.py` — scenario discovery and pairing logic (TDD)

**Files:**
- Create: `perception/training/prepare_dataset.py`
- Create: `perception/training/tests/test_prepare_dataset.py`
- Test: `python -m unittest perception.training.tests.test_prepare_dataset -v`

The behaviors to lock in here:
1. `discover_scenarios(dataset_root)` returns a list of `(scenario_id, imgs_dir, labels_dir)` tuples by matching `imgs/scenario_NN_*/` against `labels/NN_labels/`.
2. `pair_images_with_labels(imgs_dir, labels_dir)` returns `[(img_path, label_path), ...]` skipping any image whose `<stem>.txt` is missing or zero-byte.

- [ ] **Step 1: Write failing tests for discovery + pairing**

Create [perception/training/tests/test_prepare_dataset.py](perception/training/tests/test_prepare_dataset.py):

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest perception.training.tests.test_prepare_dataset -v`
Expected: `ImportError` / `ModuleNotFoundError` because `prepare_dataset.py` doesn't exist yet.

- [ ] **Step 3: Create [perception/training/prepare_dataset.py](perception/training/prepare_dataset.py) with discovery + pairing**

```python
"""
Prepare the YOLO26n bell-detection dataset.

Pairs images in `perception/dataset/imgs/scenario_NN_*/` with labels in
`perception/dataset/labels/NN_labels/`, performs a deterministic stratified
80/10/10 split per scenario, materialises a symlink tree under
`perception/training/data/`, and writes `perception/training/dataset.yaml`
for ultralytics.

Run directly:
    python -m perception.training.prepare_dataset           # idempotent
    python -m perception.training.prepare_dataset --rebuild
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

SCENARIO_RE = re.compile(r"^scenario_(\d+)_")


def discover_scenarios(dataset_root: Path) -> List[Tuple[str, Path, Path]]:
    """Return [(scenario_id, imgs_dir, labels_dir), ...] sorted by scenario_id.

    Raises FileNotFoundError if any image dir's matching label dir is missing.
    """
    imgs_root = dataset_root / "imgs"
    labels_root = dataset_root / "labels"
    out: list[tuple[str, Path, Path]] = []
    for imgs_dir in sorted(p for p in imgs_root.iterdir() if p.is_dir()):
        m = SCENARIO_RE.match(imgs_dir.name)
        if not m:
            continue
        sid = m.group(1)
        labels_dir = labels_root / f"{sid}_labels"
        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"label dir missing for {imgs_dir.name}: expected {labels_dir}"
            )
        out.append((sid, imgs_dir, labels_dir))
    return out


def pair_images_with_labels(
    imgs_dir: Path, labels_dir: Path
) -> List[Tuple[Path, Path]]:
    """Return [(img_path, label_path), ...] for images with non-empty label txt.

    Images without a `<stem>.txt` or with a zero-byte `<stem>.txt` are dropped.
    """
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(imgs_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lab = labels_dir / f"{img.stem}.txt"
        if not lab.is_file():
            continue
        if lab.stat().st_size == 0:
            continue
        pairs.append((img, lab))
    return pairs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest perception.training.tests.test_prepare_dataset -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perception/training/prepare_dataset.py perception/training/tests/test_prepare_dataset.py
git commit -m "feat(training): add scenario discovery and image-label pairing"
```

---

## Task 3: `prepare_dataset.py` — stratified split (TDD)

**Files:**
- Modify: `perception/training/prepare_dataset.py`
- Modify: `perception/training/tests/test_prepare_dataset.py`

Behavior to add: `stratified_split(pairs_by_scenario, ratios=(0.8, 0.1, 0.1), seed=42)` returns a dict `{"train": [...], "val": [...], "test": [...]}` where each list contains `(img, lab)` tuples drawn proportionally from each scenario.

- [ ] **Step 1: Append failing tests**

Append to [perception/training/tests/test_prepare_dataset.py](perception/training/tests/test_prepare_dataset.py) (above the `if __name__` block):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest perception.training.tests.test_prepare_dataset -v`
Expected: 5 new tests fail with `ImportError: cannot import name 'stratified_split'`.

- [ ] **Step 3: Implement `stratified_split` in `prepare_dataset.py`**

Add to [perception/training/prepare_dataset.py](perception/training/prepare_dataset.py):

```python
import random
from typing import Dict, Sequence

SplitDict = Dict[str, List[Tuple[Path, Path]]]


def stratified_split(
    pairs_by_scenario: Dict[str, List[Tuple[Path, Path]]],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> SplitDict:
    """Per-scenario shuffle then split into train/val/test by `ratios`.

    The last bucket absorbs any rounding remainder so no pair is dropped.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    rng = random.Random(seed)
    out: SplitDict = {"train": [], "val": [], "test": []}
    keys = ("train", "val", "test")
    for sid in sorted(pairs_by_scenario):
        items = list(pairs_by_scenario[sid])
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        # test takes the remainder so we never drop pairs
        cuts = [0, n_train, n_train + n_val, n]
        for i, key in enumerate(keys):
            out[key].extend(items[cuts[i]: cuts[i + 1]])
    return out
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `.venv/bin/python -m unittest perception.training.tests.test_prepare_dataset -v`
Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perception/training/prepare_dataset.py perception/training/tests/test_prepare_dataset.py
git commit -m "feat(training): add deterministic stratified per-scenario split"
```

---

## Task 4: `prepare_dataset.py` — symlink tree, dataset.yaml, idempotency (TDD)

**Files:**
- Modify: `perception/training/prepare_dataset.py`
- Modify: `perception/training/tests/test_prepare_dataset.py`

Behaviors:
- `build_symlink_tree(splits, out_root)` — creates `out_root/{images,labels}/{train,val,test}/` and symlinks files in. Raises if filenames collide across scenarios within the same split.
- `write_dataset_yaml(out_root, yaml_path, class_names)` — writes the ultralytics yaml.
- A top-level `prepare(dataset_root, training_root, rebuild=False)` orchestrator. Idempotent: if `dataset.yaml` exists and `rebuild=False`, return early.

- [ ] **Step 1: Append failing tests**

Append to [perception/training/tests/test_prepare_dataset.py](perception/training/tests/test_prepare_dataset.py):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest perception.training.tests.test_prepare_dataset -v`
Expected: 6 new tests fail (ImportError on the new symbols).

- [ ] **Step 3: Implement symlink tree, yaml writer, and orchestrator**

Append to [perception/training/prepare_dataset.py](perception/training/prepare_dataset.py):

```python
import shutil


def build_symlink_tree(splits: SplitDict, out_root: Path) -> None:
    """Materialise out_root/{images,labels}/{train,val,test}/ as symlinks.

    Raises ValueError if two source files would land at the same destination.
    """
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, pairs in splits.items():
        seen: set[str] = set()
        for img, lab in pairs:
            if img.name in seen:
                raise ValueError(
                    f"filename collision in split={split}: {img.name} "
                    f"appears twice (rename source images to make them unique)"
                )
            seen.add(img.name)
            dst_img = out_root / "images" / split / img.name
            dst_lab = out_root / "labels" / split / lab.name
            dst_img.symlink_to(img.resolve())
            dst_lab.symlink_to(lab.resolve())


def write_dataset_yaml(
    data_root: Path, yaml_path: Path, class_names: List[str]
) -> None:
    """Emit the minimal ultralytics dataset yaml."""
    lines = [
        f"path: {data_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n")


def prepare(
    dataset_root: Path,
    training_root: Path,
    rebuild: bool = False,
    seed: int = 42,
    class_names: Sequence[str] = ("bell",),
) -> Path:
    """Top-level: pair → split → symlink → yaml. Returns yaml path.

    Idempotent: if dataset.yaml already exists and rebuild=False, returns
    immediately without touching anything.
    """
    yaml_path = training_root / "dataset.yaml"
    data_root = training_root / "data"

    if yaml_path.is_file() and not rebuild:
        return yaml_path

    if rebuild and data_root.exists():
        shutil.rmtree(data_root)
    if rebuild and yaml_path.exists():
        yaml_path.unlink()

    training_root.mkdir(parents=True, exist_ok=True)

    pairs_by_scenario: Dict[str, List[Tuple[Path, Path]]] = {}
    for sid, imgs_dir, labels_dir in discover_scenarios(dataset_root):
        pairs_by_scenario[sid] = pair_images_with_labels(imgs_dir, labels_dir)

    splits = stratified_split(pairs_by_scenario, seed=seed)
    build_symlink_tree(splits, data_root)
    write_dataset_yaml(data_root, yaml_path, list(class_names))

    print(f"[prepare] scenarios: {len(pairs_by_scenario)}  "
          f"train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test={len(splits['test'])}")
    print(f"[prepare] dataset.yaml -> {yaml_path}")
    return yaml_path
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m unittest perception.training.tests.test_prepare_dataset -v`
Expected: 15 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perception/training/prepare_dataset.py perception/training/tests/test_prepare_dataset.py
git commit -m "feat(training): build symlink tree, dataset.yaml, idempotent prepare()"
```

---

## Task 4b: Warn on label files with non-zero class IDs (TDD)

Spec §9 mitigation: `single_cls=True` would silently mask multi-class label corruption. Add a defensive sweep that *warns* (does not raise) so a typo like `2 .5 .5 .1 .1` is visible.

**Files:**
- Modify: `perception/training/prepare_dataset.py`
- Modify: `perception/training/tests/test_prepare_dataset.py`

- [ ] **Step 1: Append failing test**

Append to [perception/training/tests/test_prepare_dataset.py](perception/training/tests/test_prepare_dataset.py):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest perception.training.tests.test_prepare_dataset -v`
Expected: 2 new tests fail (ImportError on `warn_on_nonzero_classes`).

- [ ] **Step 3: Implement `warn_on_nonzero_classes` and call it from `prepare()`**

Append to [perception/training/prepare_dataset.py](perception/training/prepare_dataset.py):

```python
def warn_on_nonzero_classes(pairs: List[Tuple[Path, Path]]) -> None:
    """Print a warning for any label file containing a class id != 0.

    Defensive check: train.py runs with single_cls=True, which would otherwise
    silently treat e.g. a stray `2 ...` line as class 0. This makes typos
    visible without aborting the run.
    """
    offenders: list[tuple[Path, set[str]]] = []
    for _img, lab in pairs:
        bad: set[str] = set()
        for line in lab.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls = line.split(maxsplit=1)[0]
            if cls != "0":
                bad.add(cls)
        if bad:
            offenders.append((lab, bad))
    if offenders:
        print(f"[prepare] WARNING: {len(offenders)} label file(s) contain "
              f"non-zero class ids (single_cls=True will collapse them):")
        for lab, bad in offenders[:10]:
            print(f"  - {lab} (classes: {sorted(bad)})")
        if len(offenders) > 10:
            print(f"  ... and {len(offenders) - 10} more")
```

Modify the `prepare()` body in the same file: after the `pairs_by_scenario` dict is built (right before `splits = stratified_split(...)`), insert:

```python
    all_pairs = [p for ps in pairs_by_scenario.values() for p in ps]
    warn_on_nonzero_classes(all_pairs)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m unittest perception.training.tests.test_prepare_dataset -v`
Expected: 17 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add perception/training/prepare_dataset.py perception/training/tests/test_prepare_dataset.py
git commit -m "feat(training): warn on non-zero class ids to surface label typos"
```

---

## Task 5: `prepare_dataset.py` — CLI entrypoint

**Files:**
- Modify: `perception/training/prepare_dataset.py`

- [ ] **Step 1: Append CLI to `prepare_dataset.py`**

Append to [perception/training/prepare_dataset.py](perception/training/prepare_dataset.py):

```python
import argparse


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    default_dataset = repo_root / "perception" / "dataset"
    default_training = here

    ap = argparse.ArgumentParser(
        description="Prepare YOLO26n bell-detection dataset (split + symlink + yaml)."
    )
    ap.add_argument("--dataset-root", type=Path, default=default_dataset,
                    help=f"raw dataset root (default: {default_dataset})")
    ap.add_argument("--training-root", type=Path, default=default_training,
                    help=f"training output root (default: {default_training})")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe data/ and dataset.yaml before regenerating")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prepare(args.dataset_root, args.training_root,
            rebuild=args.rebuild, seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI works**

Run: `.venv/bin/python -m perception.training.prepare_dataset --help`
Expected: argparse usage text printed, exit 0.

- [ ] **Step 3: Commit**

```bash
git add perception/training/prepare_dataset.py
git commit -m "feat(training): add prepare_dataset CLI"
```

---

## Task 6: Run prep against the real dataset (verification only)

**Files:** none modified.

- [ ] **Step 1: Run prep against the actual dataset**

Run: `.venv/bin/python -m perception.training.prepare_dataset`
Expected output (counts may vary by ±1):
```
[prepare] scenarios: 6  train=711  val=88  test=90
[prepare] dataset.yaml -> /home/sim2real/CapstoneDesign2026/perception/training/dataset.yaml
```

- [ ] **Step 2: Sanity-check the generated tree**

Run:
```bash
ls perception/training/data/images/train | wc -l
ls perception/training/data/images/val   | wc -l
ls perception/training/data/images/test  | wc -l
cat perception/training/dataset.yaml
```
Expected: counts match the prep output (711 / 88 / 90 ± 1). yaml has `path:`, `train:`, `val:`, `test:`, `0: bell`.

- [ ] **Step 3: Verify a symlink resolves to the real image**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path
p = next(Path('perception/training/data/images/train').iterdir())
print(p, '->', p.resolve(), 'exists:', p.resolve().is_file())
"
```
Expected: prints something like `.../scenario_NN_*/img_*.jpg exists: True`.

- [ ] **Step 4: Verify idempotency**

Run: `.venv/bin/python -m perception.training.prepare_dataset`
Expected: no output (or near-instant return); `dataset.yaml` mtime unchanged.

No commit — verification only. The generated `data/` and `dataset.yaml` are gitignored.

---

## Task 7: `train.py` — implementation

**Files:**
- Create: `perception/training/train.py`

- [ ] **Step 1: Create [perception/training/train.py](perception/training/train.py)**

```python
"""
Train YOLO26n on the bell-detection dataset.

Lazy-prepares the dataset (calls prepare() if dataset.yaml is missing), then
fine-tunes from the pretrained yolo26n.pt and runs a final test-set eval.

Usage:
    python -m perception.training.train
    python -m perception.training.train --epochs 50 --batch 16 --name run2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from perception.training.prepare_dataset import prepare

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_DATASET = REPO_ROOT / "perception" / "dataset"
DEFAULT_WEIGHTS = HERE / "weights" / "yolo26n.pt"
DEFAULT_PROJECT = HERE / "runs"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train YOLO26n bell detector."
    )
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1,
                    help="-1 = ultralytics auto batch (default)")
    ap.add_argument("--device", default="0",
                    help="cuda device id, 'cpu', or comma list e.g. '0,1'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="bell_yolo26n")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--no-test", action="store_true",
                    help="skip post-training test-set eval")
    return ap.parse_args()


def main():
    args = parse_args()

    yaml_path = prepare(DEFAULT_DATASET, HERE, rebuild=False, seed=args.seed)

    if not args.weights.is_file():
        raise FileNotFoundError(
            f"pretrained weights not found at {args.weights}; "
            "ultralytics will auto-download yolo26n.pt if you run "
            f"`python -c 'from ultralytics import YOLO; YOLO(\"yolo26n.pt\")'` "
            f"and then move the file there."
        )

    model = YOLO(str(args.weights))

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=str(DEFAULT_PROJECT),
        name=args.name,
        single_cls=True,
        cos_lr=True,
        amp=True,
        close_mosaic=10,
        exist_ok=False,
    )

    if args.no_test:
        return

    run_dir = DEFAULT_PROJECT / args.name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        print(f"[train] best.pt not found at {best_pt}, skipping test eval")
        return

    print(f"\n[train] evaluating {best_pt} on test split ...")
    best = YOLO(str(best_pt))
    metrics = best.val(
        data=str(yaml_path),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        project=str(DEFAULT_PROJECT),
        name=f"{args.name}_test",
    )
    print(f"[train] test mAP@0.5      = {metrics.box.map50:.4f}")
    print(f"[train] test mAP@0.5:0.95 = {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses**

Run: `.venv/bin/python -m perception.training.train --help`
Expected: argparse help text, exit 0.

- [ ] **Step 3: Commit**

```bash
git add perception/training/train.py
git commit -m "feat(training): add YOLO26n training script with auto test eval"
```

---

## Task 8: End-to-end smoke run (verification only)

**Files:** none modified.

- [ ] **Step 1: Run a 1-epoch smoke training to verify wiring**

Run:
```bash
.venv/bin/python -m perception.training.train --epochs 1 --name smoke --batch 16
```
Expected:
- ultralytics prints model summary, dataset stats (train ~711 / val ~88)
- 1 epoch completes (a few minutes on 3090)
- `[train] evaluating .../weights/best.pt on test split ...` appears
- `[train] test mAP@0.5      = 0.xxxx` printed (low but non-zero)
- exit 0

- [ ] **Step 2: Confirm artifacts**

Run: `ls perception/training/runs/smoke/weights/ && ls perception/training/runs/smoke_test/`
Expected: `best.pt`, `last.pt` in `smoke/weights/`; eval plots/csv in `smoke_test/`.

- [ ] **Step 3: Clean up smoke run (optional)**

Run: `rm -rf perception/training/runs/smoke perception/training/runs/smoke_test`

No commit — verification only. `runs/` is gitignored.

---

## Task 9: Full training run (manual — user decides when)

**Not part of automated execution.** Once Task 8 passes, run the full schedule on the user's schedule:

```bash
.venv/bin/python -m perception.training.train
# default: 150 epochs, patience 30, auto batch, device 0, name bell_yolo26n
```

Best weights land at `perception/training/runs/bell_yolo26n/weights/best.pt`.
