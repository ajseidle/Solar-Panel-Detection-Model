

"""
tile_dataset.py

Tile large YOLO-format images into smaller patches while adjusting bounding boxes.

Includes optional visualization to verify tiles and labels.
"""

import os
import cv2
import math
import random
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS 
# =========================
INPUT_ROOT = "data/lebanon_test_dataset"
OUTPUT_ROOT = "data/lebanon_test_dataset_tiled"

SPLITS = ["test"]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

TILE_SIZE = 512
STRIDE = 384

MIN_AREA_RATIO = 0.30
MIN_BOX_SIZE_PX = 6
SAVE_EMPTY_TILES = True

VISUALIZE = True
NUM_SAMPLES = 6


# =========================
# HELPERS
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_image_files(folder):
    return sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    return xc - w/2, yc - h/2, xc + w/2, yc + h/2


def xyxy_to_yolo(x1, y1, x2, y2, tile_w, tile_h):
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return xc / tile_w, yc / tile_h, bw / tile_w, bh / tile_h


def generate_positions(full_size, tile_size, stride):
    positions = []
    pos = 0
    while pos + tile_size < full_size:
        positions.append(pos)
        pos += stride
    positions.append(max(0, full_size - tile_size))
    return sorted(set(positions))


def box_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)


# =========================
# TILING
# =========================
def tile_one_split(split):
    img_in = os.path.join(INPUT_ROOT, "images", split)
    lbl_in = os.path.join(INPUT_ROOT, "labels", split)

    img_out = os.path.join(OUTPUT_ROOT, "images", split)
    lbl_out = os.path.join(OUTPUT_ROOT, "labels", split)

    ensure_dir(img_out)
    ensure_dir(lbl_out)

    image_files = find_image_files(img_in)
    print(f"\nProcessing {split} ({len(image_files)} images)")

    total_tiles = 0
    empty_tiles = 0

    for img_file in image_files:
        img_path = os.path.join(img_in, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        name, ext = os.path.splitext(img_file)

        label_path = os.path.join(lbl_in, f"{name}.txt")

        boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    c, xc, yc, bw, bh = map(float, parts)
                    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
                    boxes.append((int(c), x1, y1, x2, y2))

        xs = generate_positions(w, TILE_SIZE, STRIDE)
        ys = generate_positions(h, TILE_SIZE, STRIDE)

        tile_idx = 0

        for y in ys:
            for x in xs:
                tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                tile_labels = []

                for c, bx1, by1, bx2, by2 in boxes:
                    original_area = box_area(bx1, by1, bx2, by2)

                    inter_x1 = max(bx1, x)
                    inter_y1 = max(by1, y)
                    inter_x2 = min(bx2, x + TILE_SIZE)
                    inter_y2 = min(by2, y + TILE_SIZE)

                    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                        continue

                    clipped_area = box_area(inter_x1, inter_y1, inter_x2, inter_y2)
                    if clipped_area / original_area < MIN_AREA_RATIO:
                        continue

                    tx1 = inter_x1 - x
                    ty1 = inter_y1 - y
                    tx2 = inter_x2 - x
                    ty2 = inter_y2 - y

                    if (tx2 - tx1) < MIN_BOX_SIZE_PX or (ty2 - ty1) < MIN_BOX_SIZE_PX:
                        continue

                    xc, yc, bw, bh = xyxy_to_yolo(tx1, ty1, tx2, ty2, TILE_SIZE, TILE_SIZE)
                    tile_labels.append(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                if not SAVE_EMPTY_TILES and not tile_labels:
                    continue

                out_img = os.path.join(img_out, f"{name}_tile_{tile_idx}{ext}")
                out_lbl = os.path.join(lbl_out, f"{name}_tile_{tile_idx}.txt")

                cv2.imwrite(out_img, tile)

                with open(out_lbl, "w") as f:
                    f.write("\n".join(tile_labels))

                total_tiles += 1
                if not tile_labels:
                    empty_tiles += 1

                tile_idx += 1

    print(f"Saved tiles: {total_tiles}")
    print(f"Empty tiles: {empty_tiles}")


# =========================
# VISUALIZATION
# =========================
def visualize_tiles(split="train"):
    img_dir = os.path.join(OUTPUT_ROOT, "images", split)
    lbl_dir = os.path.join(OUTPUT_ROOT, "labels", split)

    files = sorted(os.listdir(img_dir))[:NUM_SAMPLES]

    plt.figure(figsize=(14, 10))

    for i, img_file in enumerate(files, 1):
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + ".txt")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    _, xc, yc, bw, bh = map(float, parts)

                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)

                    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

        plt.subplot(2, 3, i)
        plt.imshow(img)
        plt.title(img_file)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# RUN
# =========================
def main():
    for split in SPLITS:
        tile_one_split(split)

    print("\nTiling complete.")

    if VISUALIZE:
        visualize_tiles(SPLITS[0])


if __name__ == "__main__":
    main()
