
"""
split_dataset.py

Split YOLO-formatted images and labels into train/validation sets.

This script:
- collects image files from the raw image directory
- shuffles them with a fixed random seed
- splits them according to TRAIN_RATIO
- copies images into YOLO-style train/val folders
- copies matching label files when present
- creates empty label files for background-only images
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from config import RAW_IMAGE_DIR, RAW_LABEL_DIR, SPLIT_OUT_DIR, TRAIN_RATIO, RANDOM_SEED

IMAGE_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def reset_output_directory(output_dir: Path) -> None:
    """Remove any existing split directory and recreate YOLO folder structure."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for subdir in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def collect_image_files(image_dir: Path) -> list[Path]:
    """Collect image files from the raw image directory."""
    image_files: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        image_files.extend(image_dir.glob(pattern))
    return sorted(image_files)


def split_files(image_files: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Shuffle and split image files into train and validation sets."""
    shuffled_files = image_files.copy()
    random.seed(seed)
    random.shuffle(shuffled_files)

    split_idx = int(len(shuffled_files) * train_ratio)
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    return train_files, val_files


def copy_image_label_pair(img_path: Path, split: str) -> None:
    """
    Copy an image and its corresponding YOLO label file into the split directory.

    If no label file exists, create an empty label file to represent a background image.
    """
    out_img = SPLIT_OUT_DIR / "images" / split / img_path.name
    shutil.copy2(img_path, out_img)

    label_path = RAW_LABEL_DIR / f"{img_path.stem}.txt"
    out_label = SPLIT_OUT_DIR / "labels" / split / f"{img_path.stem}.txt"

    if label_path.exists():
        shutil.copy2(label_path, out_label)
    else:
        out_label.write_text("", encoding="utf-8")


def main() -> None:
    """Run dataset splitting pipeline."""
    reset_output_directory(SPLIT_OUT_DIR)

    image_files = collect_image_files(RAW_IMAGE_DIR)
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {RAW_IMAGE_DIR}")

    train_files, val_files = split_files(image_files, TRAIN_RATIO, RANDOM_SEED)

    for img_path in train_files:
        copy_image_label_pair(img_path, "train")

    for img_path in val_files:
        copy_image_label_pair(img_path, "val")

    print(f"Done: created {len(train_files)} training samples and {len(val_files)} validation samples.")


if __name__ == "__main__":
    main()
