# Offline Dataset Augmentation — Design

**Date:** 2026-04-27
**Owner:** eugene (sim2real)
**Status:** Approved (brainstorming) → ready for plan
**Related spec:** [2026-04-27-yolo26n-bell-detection-training-design.md](./2026-04-27-yolo26n-bell-detection-training-design.md)

---

## 1. Goal

Expand the training split on disk by generating 5 augmented copies per
original image (`--multiplier 5`, default), preserving the originals → train/
goes from 710 to 4260 image files (710 originals + 3550 augmented). Val/test
remain untouched for honest evaluation.

Multiplier semantics: `--multiplier N` writes N augmented copies per original.
N=0 disables augmentation; N=1 doubles the train set; N=5 makes it 6×.

Non-goals:
- Augmenting val/test (would inflate metrics)
- Replacing ultralytics' online augmentation (kept ON; compounds with offline)
- Domain-specific tricks beyond standard photometric/geometric (motion blur
  was considered and excluded by user choice)
- Multi-class concerns (single class only)

---

## 2. Architecture

Inserted between dataset prep and training:

```
prepare_dataset.py  →  augment_dataset.py (NEW)  →  train.py
                          ↓
   data/images/train/      data/images/train/
   710 originals           710 originals + 710×5 augmented = 4260 files
                           (suffix: <stem>_aug{0..4}.jpg)
```

`val/` and `test/` are untouched. `dataset.yaml` does not need to change —
the augmented files live alongside originals in `train/`, so ultralytics
just sees a larger train set automatically.

---

## 3. New File: `perception/training/augment_dataset.py`

### Responsibilities

1. Read every `(img, label)` pair under `data/images/train/` +
   `data/labels/train/`, **excluding files that already have an `_aug`
   suffix** (so re-running doesn't recursively augment its own output).
2. For each pair, generate `multiplier` augmented copies via
   albumentations. Bboxes are auto-transformed.
3. Write each copy as `<stem>_aug{i}.jpg` and `<stem>_aug{i}.txt` to the
   same `train/` directories.
4. Idempotent: if any `*_aug*.jpg` already exists in `train/` and `--rebuild`
   is not set, return early. With `--rebuild`, delete every `*_aug*` file
   first, then regenerate.

### Augmentation pipeline (albumentations)

| Transform | Parameters | Probability |
|---|---|---|
| `HueSaturationValue` | `hue_shift_limit=10, sat_shift_limit=70, val_shift_limit=40` | 1.0 |
| `Rotate` | `limit=5°, border_mode=cv2.BORDER_REFLECT` | 0.5 |
| `RandomScale` | `scale_limit=0.1` (±10%) | 0.5 |
| `RandomBrightnessContrast` | `brightness_limit=0.15, contrast_limit=0.15` | 0.5 |
| `HorizontalFlip` | — | 0.5 |
| `GaussNoise` | `var_limit=(0, 25)` (σ ≤ 5) | 0.3 |

`bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)`
— a bbox is dropped if less than 30% remains visible after transform. With
±5° rotation and ±10% scale this should rarely trigger.

**Excluded by user choice:** motion blur. **Not relevant for this task:**
vertical flip, perspective, MixUp, CutMix.

Each augmented sample uses an independent RNG state derived from
`(seed, source_stem, aug_idx)` for reproducibility.

### CLI

```bash
python -m perception.training.augment_dataset                    # idempotent, multiplier=5
python -m perception.training.augment_dataset --multiplier 10
python -m perception.training.augment_dataset --rebuild          # wipe *_aug* and regenerate
python -m perception.training.augment_dataset --seed 7
```

Flags:
- `--training-root` (default `perception/training/`) — where `data/` lives
- `--multiplier` (default 5)
- `--rebuild` — wipe and regenerate
- `--seed` (default 42)

---

## 4. Edge Cases

| Case | Behavior |
|---|---|
| Augment script run twice without `--rebuild` | second run is a no-op; print "augmented files already present, skipping" |
| `data/images/train/` doesn't exist | error: "run prepare_dataset.py first" |
| Original image has zero-byte label (shouldn't happen — prep skips these) | filter out at start, same logic as `pair_images_with_labels` |
| Bbox transformed entirely out of frame (`min_visibility < 0.3`) | albumentations drops it; resulting label may be empty → still written as zero-byte file; **NOT** re-fed to prepare. The augmented file with no bbox is effectively a background image, which ultralytics handles fine |
| Filename collision (e.g., source already named `..._aug0.jpg`) | abort with clear error before writing |

---

## 5. Dependency Addition

Add to `perception/requirements.txt`:

```
albumentations>=1.4.0
```

`opencv-python` is already required (>=4.8.0) — albumentations uses it.
Install: `pip install albumentations`. No CUDA/torch coupling.

---

## 6. Train.py Changes

**None required.** ultralytics reads from the same `train/` dir; the larger
file count is invisible to the script. Online augmentation defaults stay as
specified in the original training spec.

Optional follow-on (not in this spec): expose `--mosaic` CLI flag in
`train.py` so user can dial back online mosaic if 5× offline already
provides enough variety. Out of scope here.

---

## 7. Validation & Acceptance

- After running `augment_dataset.py` with default multiplier=5, `train/`
  contains exactly 710 originals + 3550 augmented = 4260 image files (and
  matching label files), assuming the prep step ran on the actual dataset
  with 710 train pairs.
- Re-running without `--rebuild` is a no-op (file count unchanged).
- Re-running with `--rebuild` produces the same set of augmented filenames
  given the same seed (deterministic).
- Spot-check: visually inspect 5 random augmented images — bboxes should
  still tightly bound the bell after rotation/scale.
- After augmentation, `train.py` runs as before with no code change.

---

## 8. Out of Scope

- Augmenting val/test
- Mixing online and offline augmentation knobs in a single config (kept
  decoupled — online via `train.py`, offline via `augment_dataset.py`)
- Visual debug tool for augmented samples (manual spot-check is enough)
- Augmentation strategies beyond the 6 transforms above (e.g., albumentations'
  `OneOf` weighted compositions)

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Bbox transforms incorrect at edges (e.g., rotation pushing bell off-frame) | `min_visibility=0.3` keeps mostly-visible bboxes; ±5° rotation is small enough that this rarely fires |
| `albumentations` install conflicts with existing torch/CUDA | none observed; albumentations is pure Python + NumPy + OpenCV. Pin `>=1.4.0` for stable bbox API |
| Filename collisions if user re-runs prepare with different seed → orphaned aug files | `--rebuild` wipes everything matching `*_aug*` before generating, regardless of upstream |
| Augmented files with empty label (bbox dropped) silently degrade training as background-only | mitigation: log a warning at the end with count of empty-label augmented files (no abort — user inspects log and re-runs with `--rebuild` and tighter params if the count is excessive) |
