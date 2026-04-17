
"""
analyze_predictions.py

Run YOLO inference over a set of image folders and generate a per-image
summary table containing:
- actual object count from YOLO label files
- predicted object count from model inference
- average confidence score

Also creates a bubble scatter plot comparing actual vs predicted counts.
"""

import gc
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import config

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp", "*.webp")

DEVICE = "cpu"      # "cpu", "cuda", or 0
CHUNK_SIZE = 25
IMGSZ = 416
CONF = 0.25

OUTPUT_CSV = "prediction_summary.csv"
OUTPUT_PLOT = "actual_vs_predicted_count.png"


def collect_image_paths(*folders):
    """Collect image paths from one or more directories."""
    image_paths = []
    for folder in folders:
        for pattern in IMAGE_EXTENSIONS:
            image_paths.extend(folder.glob(pattern))
    return sorted(image_paths)


def count_actual_objects(image_path):
    """
    Count ground-truth objects from the matching YOLO label file.
    Assumes dataset structure:
    DATASET_DIR/images/<split>/file.ext
    DATASET_DIR/labels/<split>/file.txt
    """
    split = image_path.parent.name
    label_path = config.DATASET_DIR / "labels" / split / f"{image_path.stem}.txt"

    if not label_path.exists():
        return 0

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    return len(lines)


def main():
    train_dir = config.DATASET_DIR / "images" / "train"
    val_dir = config.DATASET_DIR / "images" / "val"

    image_paths = collect_image_paths(train_dir, val_dir)

    print(f"Total images found: {len(image_paths)}")
    if not image_paths:
        raise FileNotFoundError("No image files found in train/val folders.")

    model = YOLO(str(config.INIT_WEIGHTS))
    rows = []

    num_chunks = math.ceil(len(image_paths) / CHUNK_SIZE)

    for i in range(num_chunks):
        chunk = image_paths[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        print(f"Processing chunk {i + 1}/{num_chunks} ({len(chunk)} images)")

        results = model.predict(
            source=[str(path) for path in chunk],
            conf=CONF,
            imgsz=IMGSZ,
            batch=1,
            stream=True,
            device=DEVICE,
            save=False,
            verbose=False,
        )

        for image_path, result in zip(chunk, results):
            actual_count = count_actual_objects(image_path)

            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.detach().cpu().numpy()
                predicted_count = len(confidences)
                avg_confidence = float(np.mean(confidences))
            else:
                predicted_count = 0
                avg_confidence = np.nan

            rows.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "actual_count": actual_count,
                    "predicted_count": predicted_count,
                    "avg_confidence": round(avg_confidence, 4) if not np.isnan(avg_confidence) else np.nan,
                }
            )

        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved prediction summary to: {Path(OUTPUT_CSV).resolve()}")

    plot_counts = (
        df.groupby(["actual_count", "predicted_count"])
        .size()
        .reset_index(name="n_images")
    )

    max_val = max(df["actual_count"].max(), df["predicted_count"].max())

    plt.figure(figsize=(8, 8))
    plt.scatter(
        plot_counts["actual_count"],
        plot_counts["predicted_count"],
        s=plot_counts["n_images"] * 8,
        alpha=0.5,
    )
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.xlabel("Actual Count")
    plt.ylabel("Predicted Count")
    plt.title("Actual vs Predicted Count per Image (Bubble Size = Number of Images)")
    plt.grid(True)
    plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved scatter plot to: {Path(OUTPUT_PLOT).resolve()}")
    print("\nMost common count combinations:")
    print(plot_counts.sort_values("n_images", ascending=False).head(15))


if __name__ == "__main__":
    main()
