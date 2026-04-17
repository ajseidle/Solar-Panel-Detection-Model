
"""
run_object_diagnostics.py

Runs YOLO object diagnostics on images from the active dataset split and computes
per-object features relative to the full image:
- relative brightness
- relative edge density
- shape features
- texture features
"""

import sys
from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import config

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


# =========================
# Config helpers
# =========================

def get_feature_config():
    """Return diagnostics feature config from config.py."""
    return getattr(config, "FEATURE_CONFIG", {})


def get_prediction_config():
    """Return prediction config from config.py."""
    return getattr(config, "PREDICT_CONFIG_DIAGNOSTIC", {"conf": 0.05, "iou": 0.7})


# =========================
# Utility helpers
# =========================

def safe_div(a, b, eps):
    return a / (b + eps)


def get_diagnostics_images_dir() -> Path:
    split = getattr(config, "DIAGNOSTICS_SPLIT", "val")
    dataset_dir = Path(getattr(config, "DATASET_DIR", config.SPLIT_OUT_DIR))
    return dataset_dir / "images" / split


def list_image_paths(images_dir: Path, max_images=None):
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    ])
    if max_images is not None:
        image_paths = image_paths[: int(max_images)]
    return image_paths


def bbox_crop(img, x1, y1, x2, y2):
    """Safe crop from image using bounding box coordinates."""
    h, w = img.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return img[y1:y2, x1:x2]


def compute_entropy(gray_crop, bins, eps):
    """Simple grayscale entropy."""
    hist = cv2.calcHist([gray_crop], [0], None, [bins], [0, 256]).flatten()
    p = hist / (hist.sum() + eps)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + eps)))


def compute_edge_density(gray_img, t1, t2):
    """Edge density = fraction of pixels marked as edges by Canny."""
    edges = cv2.Canny(gray_img, t1, t2)
    density = float(np.count_nonzero(edges)) / float(edges.size) if edges.size > 0 else np.nan
    return density, edges


def compute_shape_features_from_box(x1, y1, x2, y2, eps=1e-6):
    """
    Shape features from the box itself.
    For a raw YOLO box, rectangularity is always 1.0.
    """
    w = max(float(x2 - x1), eps)
    h = max(float(y2 - y1), eps)
    area = w * h
    perimeter = 2.0 * (w + h)

    aspect_ratio = w / h
    compactness = (4.0 * math.pi * area) / ((perimeter ** 2) + eps)

    return {
        "box_w": w,
        "box_h": h,
        "box_area": area,
        "aspect_ratio": aspect_ratio,
        "compactness": compactness,
    }


def compute_texture_features(gray_crop, feature_cfg, eps):
    """Compute texture features if enabled."""
    if not feature_cfg.get("compute_texture_features", True):
        return {
            "gray_std": np.nan,
            "gray_entropy": np.nan,
            "glcm_contrast": np.nan,
            "glcm_homogeneity": np.nan,
            "glcm_energy": np.nan,
        }

    bins = int(feature_cfg.get("texture_entropy_bins", 32))

    feats = {
        "gray_std": float(np.std(gray_crop)) if gray_crop.size > 0 else np.nan,
        "gray_entropy": compute_entropy(gray_crop, bins=bins, eps=eps) if gray_crop.size > 0 else np.nan,
    }

    use_glcm = feature_cfg.get("compute_glcm_if_available", True)

    if SKIMAGE_AVAILABLE and use_glcm and gray_crop.size > 0:
        gray_q = (gray_crop / 32).astype(np.uint8)  # 0..7
        glcm = graycomatrix(
            gray_q,
            distances=[1],
            angles=[0],
            levels=8,
            symmetric=True,
            normed=True
        )
        feats["glcm_contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
        feats["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
        feats["glcm_energy"] = float(graycoprops(glcm, "energy")[0, 0])
    else:
        feats["glcm_contrast"] = np.nan
        feats["glcm_homogeneity"] = np.nan
        feats["glcm_energy"] = np.nan

    return feats


def compute_relative_object_metrics(image_bgr, x1, y1, x2, y2, feature_cfg):
    eps = float(feature_cfg.get("eps", 1e-6))

    crop_bgr = bbox_crop(image_bgr, x1, y1, x2, y2)
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    brightness_space = feature_cfg.get("brightness_space", "HSV_V")
    if brightness_space != "HSV_V":
        raise ValueError(f"Unsupported brightness_space: {brightness_space}")

    t1 = int(feature_cfg.get("canny_threshold1", 50))
    t2 = int(feature_cfg.get("canny_threshold2", 150))

    # Full image representations
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_v = image_hsv[:, :, 2]

    image_mean_v = float(np.mean(image_v))
    image_std_v = float(np.std(image_v))
    image_edge_density, _ = compute_edge_density(image_gray, t1, t2)

    # Object crop representations
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    crop_v = crop_hsv[:, :, 2]

    obj_mean_v = float(np.mean(crop_v))
    obj_std_v = float(np.std(crop_v))
    obj_edge_density, _ = compute_edge_density(crop_gray, t1, t2)

    # Relative metrics
    relative_brightness = safe_div(obj_mean_v, image_mean_v, eps)
    relative_brightness_std = safe_div(obj_std_v, image_std_v, eps)
    relative_edge_density = safe_div(obj_edge_density, image_edge_density, eps)

    result = {
        "mean_v": obj_mean_v,
        "std_v": obj_std_v,
        "edge_density": obj_edge_density,
        "image_mean_v": image_mean_v,
        "image_std_v": image_std_v,
        "image_edge_density": image_edge_density,
        "relative_brightness": relative_brightness,
        "relative_brightness_std": relative_brightness_std,
        "relative_edge_density": relative_edge_density,
    }

    if feature_cfg.get("compute_shape_features", True):
        result.update(compute_shape_features_from_box(x1, y1, x2, y2, eps=eps))
    else:
        result.update({
            "box_w": np.nan,
            "box_h": np.nan,
            "box_area": np.nan,
            "aspect_ratio": np.nan,
            "compactness": np.nan,
        })

    result.update(compute_texture_features(crop_gray, feature_cfg, eps))

    return result

def run_diagnostics_on_images(model, image_paths, pred_cfg, imgsz, feature_cfg):
    rows = []

    conf = float(pred_cfg.get("conf", 0.05))
    iou = float(pred_cfg.get("iou", 0.7))
    save_flag = bool(pred_cfg.get("save", False))
    verbose_flag = bool(pred_cfg.get("verbose", False))

    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save_flag,
            verbose=verbose_flag,
        )

        if not results:
            continue

        r = results[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box)

            m = compute_relative_object_metrics(img, x1, y1, x2, y2, feature_cfg)
            if not m:
                continue

            m["image"] = str(img_path.name)
            m["conf"] = float(confs[i])
            m["cls"] = int(clss[i])
            m["x1"], m["y1"], m["x2"], m["y2"] = x1, y1, x2, y2
            rows.append(m)

    return pd.DataFrame(rows)


def apply_object_filters(df, filt_cfg):
    if filt_cfg is None or len(df) == 0:
        return df.copy()

    df_f = df.copy()

    if "min_relative_edge_density" in filt_cfg and "relative_edge_density" in df_f.columns:
        df_f = df_f[df_f["relative_edge_density"] >= float(filt_cfg["min_relative_edge_density"])]

    if "max_relative_brightness" in filt_cfg and "relative_brightness" in df_f.columns:
        df_f = df_f[df_f["relative_brightness"] <= float(filt_cfg["max_relative_brightness"])]

    if "max_brightness_std_ratio" in filt_cfg and "relative_brightness_std" in df_f.columns:
        df_f = df_f[df_f["relative_brightness_std"] <= float(filt_cfg["max_brightness_std_ratio"])]

    if "min_conf" in filt_cfg and "conf" in df_f.columns:
        df_f = df_f[df_f["conf"] >= float(filt_cfg["min_conf"])]

    return df_f


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python scripts/run_object_diagnostics.py <best_pt_path> <data_yaml_path>")

    best_pt = Path(sys.argv[1])
    data_yaml = Path(sys.argv[2])

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    feature_cfg = get_feature_config()
    pred_cfg = get_prediction_config()

    images_dir = get_diagnostics_images_dir()
    if not images_dir.exists():
        raise FileNotFoundError(f"Diagnostics images folder not found: {images_dir}")

    max_images = getattr(config, "DIAGNOSTICS_MAX_IMAGES", None)
    image_paths = list_image_paths(images_dir, max_images=max_images)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    print(f"[Diagnostics] Using model: {best_pt}")
    print(f"[Diagnostics] Images: {len(image_paths)} from {images_dir}")
    print(f"[Diagnostics] Prediction config: {pred_cfg}")

    imgsz = int(getattr(config, "IMGSZ", 640))
    print(f"[Diagnostics] Using imgsz={imgsz}")

    model = YOLO(str(best_pt))

    df = run_diagnostics_on_images(
        model=model,
        image_paths=image_paths,
        pred_cfg=pred_cfg,
        imgsz=imgsz,
        feature_cfg=feature_cfg,
    )

    print(f"[Diagnostics] Objects processed: {len(df)}")

    # Save raw metrics
    out_csv = Path(config.OBJECT_METRICS_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[Diagnostics] Saved metrics CSV: {out_csv}")

    # Save filtered metrics
    filt_cfg = getattr(config, "OBJECT_FILTER_THRESHOLDS", None)
    df_f = apply_object_filters(df, filt_cfg)

    out_f_csv = Path(config.FILTERED_OBJECT_METRICS_CSV)
    df_f.to_csv(out_f_csv, index=False)
    print(f"[Diagnostics] Saved filtered CSV: {out_f_csv}")
    print(f"[Diagnostics] Filtered objects: {len(df_f)}")



if __name__ == "__main__":
    main()
