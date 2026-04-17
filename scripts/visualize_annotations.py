
"""
visualize_annotations.py

Visualize YOLO-annotated images with bounding boxes for quick spot-checking
of annotation quality and dataset integrity.
"""

import random
import cv2
import matplotlib.pyplot as plt
import config

# Choose dataset split
SUBSET = "val"   # change to "train", "val", or "test"
NUM_SAMPLES = 5
RANDOM_SEED = 42

img_dir = config.DATASET_DIR / "images" / SUBSET
label_dir = config.DATASET_DIR / "labels" / SUBSET


def draw_boxes(img_path, label_path):
    """Draw YOLO bounding boxes on one image and display it."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: could not load {img_path}")
        return

    h, w, _ = img.shape

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines:
        if not line.strip():
            continue

        cls, cx, cy, bw, bh = map(float, line.strip().split())

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        class_idx = int(cls)
        class_name = (
            config.CLASS_NAMES[class_idx]
            if 0 <= class_idx < len(config.CLASS_NAMES)
            else f"class_{class_idx}"
        )

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            class_name,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.title(img_path.name)
    plt.show()


def main():
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    img_paths = list(img_dir.glob("*.tif"))
    if not img_paths:
        raise FileNotFoundError(f"No .tif images found in: {img_dir}")

    random.seed(RANDOM_SEED)
    samples = random.sample(img_paths, min(NUM_SAMPLES, len(img_paths)))

    for img_path in samples:
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"Warning: no label found for {img_path.name}")
            continue
        draw_boxes(img_path, lbl_path)


if __name__ == "__main__":
    main()
