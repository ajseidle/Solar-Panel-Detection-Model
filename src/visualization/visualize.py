
"""
Purpose:
Visualizes YOLO-annotated images with bounding boxes for spot-checking
annotation quality and dataset integrity.
"""

import cv2
import matplotlib.pyplot as plt
from config import SPLIT_OUT_DIR
import random

# === Choose 'train' or 'val' subset ===
subset = "val"  # or "val"

img_dir = SPLIT_OUT_DIR / f"images/{subset}"
label_dir = SPLIT_OUT_DIR / f"labels/{subset}"

def draw_boxes(img_path, label_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Could not load {img_path}")
        return
    
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        cls, cx, cy, bw, bh = map(float, line.strip().split())
        # Convert to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Panel", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.title(img_path.name)
    plt.show()

# === Sample Random Images ===
img_paths = list(img_dir.glob("*.tif"))
random.seed(42)
samples = random.sample(img_paths, 5)

for img_path in samples:
    lbl_path = label_dir / img_path.with_suffix(".txt").name
    if not lbl_path.exists():
        print(f"⚠️ No label found for {img_path.name}")
        continue
    draw_boxes(img_path, lbl_path)

