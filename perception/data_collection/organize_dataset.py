"""
Organize YOLO training dataset structure
Converts captured images into the folder structure required for YOLO training.

YOLO dataset structure:
  dataset/
  ├── data.yaml          ← Dataset configuration file
  ├── train/
  │   ├── images/
  │   └── labels/
  └── val/
      ├── images/
      └── labels/

Usage:
  python organize_dataset.py
  python organize_dataset.py --ratio 0.8
  python organize_dataset.py --classes person car dog
"""

import argparse
import os
import random
import shutil
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PATHS, YOLO


def organize(train_ratio=0.8, classes=None, seed=42):
    """
    Organize captured images into YOLO dataset structure

    Args:
        train_ratio: Training data ratio (0.0~1.0)
        classes: Class name list
        seed: Random seed
    """
    random.seed(seed)

    source_images = PATHS["images"]
    source_labels = PATHS["labels"]
    dataset_dir = PATHS["dataset"]

    # List image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        f for f in os.listdir(source_images)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not images:
        print("[ERROR] No images found. Capture images with capture.py first.")
        return

    print(f"[INFO] Found {len(images)} image(s).")

    # Shuffle and split
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"[INFO] Train: {len(train_images)} | Val: {len(val_images)}")

    # Create YOLO directory structure
    dirs = {
        "train_images": os.path.join(dataset_dir, "train", "images"),
        "train_labels": os.path.join(dataset_dir, "train", "labels"),
        "val_images": os.path.join(dataset_dir, "val", "images"),
        "val_labels": os.path.join(dataset_dir, "val", "labels"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Copy files
    def copy_files(file_list, img_dst, lbl_dst):
        copied = 0
        for img_file in file_list:
            # Copy image
            src = os.path.join(source_images, img_file)
            dst = os.path.join(img_dst, img_file)
            shutil.copy2(src, dst)

            # Copy label file if it exists
            label_name = os.path.splitext(img_file)[0] + ".txt"
            label_src = os.path.join(source_labels, label_name)
            if os.path.exists(label_src):
                label_dst = os.path.join(lbl_dst, label_name)
                shutil.copy2(label_src, label_dst)
                copied += 1

        return copied

    print("\n[COPY] Copying files...")
    train_labels = copy_files(train_images, dirs["train_images"], dirs["train_labels"])
    val_labels = copy_files(val_images, dirs["val_images"], dirs["val_labels"])

    print(f"  → Train: {len(train_images)} images, {train_labels} labels")
    print(f"  → Val: {len(val_images)} images, {val_labels} labels")

    # Generate data.yaml
    if classes is None:
        classes = YOLO.get("classes", [])

    if not classes:
        print("\n[WARNING] No classes specified.")
        print("  → Edit the 'names' field in data.yaml manually.")
        print("  → Or specify classes with the --classes option.")
        classes = ["class_0"]

    data_yaml = {
        "path": os.path.abspath(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": len(classes),
        "names": classes,
    }

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[YAML] Dataset config file created: {yaml_path}")
    print(f"  → Number of classes: {len(classes)}")
    print(f"  → Class list: {classes}")

    # Result summary
    print("\n" + "=" * 50)
    print("  Dataset organization complete!")
    print("=" * 50)
    print(f"\nDataset structure:")
    print(f"  {dataset_dir}/")
    print(f"  ├── data.yaml")
    print(f"  ├── train/")
    print(f"  │   ├── images/ ({len(train_images)} files)")
    print(f"  │   └── labels/ ({train_labels} files)")
    print(f"  └── val/")
    print(f"      ├── images/ ({len(val_images)} files)")
    print(f"      └── labels/ ({val_labels} files)")

    print(f"\n[NEXT] Use a labeling tool to create annotations:")
    print(f"  → CVAT: https://www.cvat.ai/")
    print(f"  → Roboflow: https://roboflow.com/")
    print(f"  → LabelImg: pip install labelImg")

    if train_labels == 0 and val_labels == 0:
        print(f"\n[TIP] No label files found yet.")
        print(f"  → Add YOLO-format .txt files to the dataset/labels/ folder,")
        print(f"  → then re-run this script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize YOLO dataset structure")
    parser.add_argument(
        "--ratio", type=float, default=0.8,
        help="Train split ratio (0.0~1.0). Default: 0.8"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Class name list. Example: --classes person car dog"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42"
    )

    args = parser.parse_args()
    organize(args.ratio, args.classes, args.seed)
