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

import argparse
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SCENARIO_RE = re.compile(r"^scenario_(\d+)_")
SplitDict = Dict[str, List[Tuple[Path, Path]]]


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
        # ensure val/test each get at least 1 when enough items exist,
        # so small scenarios still contribute to every split
        if n >= 3 and n_val == 0:
            n_val = 1
        if n >= 3 and n - n_train - n_val < 1:
            n_train = max(1, n - n_val - 1)
        # test takes the remainder so we never drop pairs
        cuts = [0, n_train, n_train + n_val, n]
        for i, key in enumerate(keys):
            out[key].extend(items[cuts[i]: cuts[i + 1]])
    return out


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


def prepare(
    dataset_root: Path,
    training_root: Path,
    rebuild: bool = False,
    seed: int = 42,
    class_names: Sequence[str] = ("bell",),
) -> Path:
    """Top-level: pair -> split -> symlink -> yaml. Returns yaml path.

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

    all_pairs = [p for ps in pairs_by_scenario.values() for p in ps]
    warn_on_nonzero_classes(all_pairs)

    splits = stratified_split(pairs_by_scenario, seed=seed)
    build_symlink_tree(splits, data_root)
    write_dataset_yaml(data_root, yaml_path, list(class_names))

    print(f"[prepare] scenarios: {len(pairs_by_scenario)}  "
          f"train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test={len(splits['test'])}")
    print(f"[prepare] dataset.yaml -> {yaml_path}")
    return yaml_path


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
