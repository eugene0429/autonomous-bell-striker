"""
Interactive bbox visualizer for the YOLO26n training dataset.

Default: cycle through augmented train images (`*_aug*.jpg`) showing each
with its bbox overlaid. Useful to spot-check that albumentations bbox
transforms are tracking the bell correctly after rotation/scale/flip.

Keys (cv2 window in focus):
    n, space   next
    p          previous
    s          save current frame as viz_<stem>.jpg in CWD
    q, Esc     quit

Usage:
    python -m perception.training.visualize_dataset                  # aug train
    python -m perception.training.visualize_dataset --all-train      # orig + aug
    python -m perception.training.visualize_dataset --split val
    python -m perception.training.visualize_dataset --scenario 01    # raw scenario_01_*
    python -m perception.training.visualize_dataset --scenario 4m_left
    python -m perception.training.visualize_dataset --list-scenarios
    python -m perception.training.visualize_dataset --shuffle --seed 7
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from perception.training.prepare_dataset import (
    discover_scenarios,
    pair_images_with_labels,
)

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = HERE.parent / "dataset"


def iter_pairs_for_scenario(
    dataset_root: Path,
    query: str,
) -> List[Tuple[Path, Path]]:
    """Return (img, lab) pairs for a scenario from the raw dataset.

    `query` matches against the scenario id (e.g. "01", "1") OR a substring
    of the directory name (e.g. "4m_left", "scenario_01"). The first match
    wins, by id first then by name substring.

    Raises ValueError if no scenario matches.
    """
    scenarios = discover_scenarios(dataset_root)
    padded = query.zfill(2) if query.isdigit() else None
    if padded is not None:
        for sid, imgs_dir, labels_dir in scenarios:
            if sid == padded:
                return pair_images_with_labels(imgs_dir, labels_dir)
    q_lower = query.lower()
    for sid, imgs_dir, labels_dir in scenarios:
        if q_lower in imgs_dir.name.lower():
            return pair_images_with_labels(imgs_dir, labels_dir)
    available = ", ".join(s[1].name for s in scenarios)
    raise ValueError(f"scenario not found: {query!r}. available: {available}")


def list_scenarios(dataset_root: Path) -> List[str]:
    """Return human-readable scenario names sorted by id."""
    return [imgs_dir.name for _sid, imgs_dir, _lab in discover_scenarios(dataset_root)]


def iter_pairs_for_split(
    training_root: Path,
    split: str,
    include_originals: bool = False,
) -> List[Tuple[Path, Path]]:
    """List (img, lab) pairs for `split`.

    For split == "train":
        include_originals=False → augmented files only (`_aug` in stem)
        include_originals=True  → originals AND augmented
    For split == "val" / "test":
        always returns all pairs (no `_aug` files exist there)
    """
    img_dir = training_root / "data" / "images" / split
    lab_dir = training_root / "data" / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"image dir not found: {img_dir} "
            "(run prepare_dataset.py and possibly augment_dataset.py first)"
        )
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        is_aug = "_aug" in img.stem
        if split == "train" and not include_originals and not is_aug:
            continue
        lab = lab_dir / f"{img.stem}.txt"
        if not lab.is_file():
            continue
        pairs.append((img, lab))
    return pairs


def read_yolo_bboxes(lab_path: Path) -> List[Tuple[float, float, float, float]]:
    """Parse a YOLO label file into (cx, cy, w, h) tuples (normalised)."""
    if lab_path.stat().st_size == 0:
        return []
    bboxes: list[tuple[float, float, float, float]] = []
    for line in lab_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        bboxes.append(
            (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        )
    return bboxes


def draw_bbox_on_image(
    image: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of `image` with YOLO-normalised bboxes drawn in BGR `color`."""
    out = image.copy()
    h, w = image.shape[:2]
    for cx, cy, bw, bh in bboxes:
        x1 = int(round((cx - bw / 2) * w))
        y1 = int(round((cy - bh / 2) * h))
        x2 = int(round((cx + bw / 2) * w))
        y2 = int(round((cy + bh / 2) * h))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def _annotate_for_display(
    image: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]],
    filename: str,
    index: int,
    total: int,
) -> np.ndarray:
    annotated = draw_bbox_on_image(image, bboxes)
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, filename, (10, 25), font, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated, f"{index + 1}/{total}",
                (w - 110, 25), font, 0.6, (0, 255, 255), 2)
    if not bboxes:
        cv2.putText(annotated, "no bbox", (10, h - 15),
                    font, 0.7, (0, 0, 255), 2)
    return annotated


def run_viewer(pairs: List[Tuple[Path, Path]]) -> None:
    if not pairs:
        print("[viz] no images to show")
        return

    n = len(pairs)
    print(f"[viz] {n} images. Keys: n/space=next, p=prev, s=save, q/Esc=quit")
    cv2.namedWindow("dataset", cv2.WINDOW_NORMAL)
    i = 0
    while True:
        img_path, lab_path = pairs[i]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[viz] could not read {img_path}, skipping")
            i = (i + 1) % n
            continue
        bboxes = read_yolo_bboxes(lab_path)
        annotated = _annotate_for_display(image, bboxes, img_path.name, i, n)
        cv2.imshow("dataset", annotated)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):  # 27 = Esc
            break
        if key in (ord("n"), ord(" ")):
            i = (i + 1) % n
        elif key == ord("p"):
            i = (i - 1) % n
        elif key == ord("s"):
            out = Path.cwd() / f"viz_{img_path.stem}.jpg"
            cv2.imwrite(str(out), annotated)
            print(f"[viz] saved {out}")
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="Interactive bbox visualizer for the YOLO26n training dataset.",
    )
    ap.add_argument("--training-root", type=Path, default=HERE,
                    help=f"training root containing data/ (default: {HERE})")
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                    help=f"raw dataset root with imgs/scenario_NN_*/ "
                         f"(default: {DEFAULT_DATASET_ROOT})")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--all-train", action="store_true",
                    help="include originals along with augmented (split=train only)")
    ap.add_argument("--scenario", type=str, default=None,
                    help="show originals from a specific raw scenario "
                         "(id like '01' or substring like '4m_left'). "
                         "When set, --split / --all-train are ignored.")
    ap.add_argument("--list-scenarios", action="store_true",
                    help="print available raw scenarios and exit")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomise traversal order")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed used when --shuffle is set")
    args = ap.parse_args()

    if args.list_scenarios:
        for name in list_scenarios(args.dataset_root):
            print(name)
        return

    if args.scenario is not None:
        pairs = iter_pairs_for_scenario(args.dataset_root, args.scenario)
    else:
        pairs = iter_pairs_for_split(
            args.training_root, args.split, include_originals=args.all_train,
        )

    if args.shuffle:
        random.Random(args.seed).shuffle(pairs)

    run_viewer(pairs)


if __name__ == "__main__":
    main()
