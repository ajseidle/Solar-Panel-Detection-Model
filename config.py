
"""
config.py

Central configuration file for the solar panel detection project.

Edit this file to control dataset selection, training parameters,
diagnostic behavior, and output locations without modifying pipeline logic.
Paths are resolved relative to this file so the project remains portable
across environments.
"""

from pathlib import Path

# === Project Root (folder containing this config.py) ===
BASE_DIR = Path(__file__).resolve().parent

# === Raw Input Data ===
RAW_IMAGE_DIR = BASE_DIR / "data" / "raw" / "images"
RAW_LABEL_DIR = BASE_DIR / "data" / "raw" / "labels"

# === Output Dataset ===
SPLIT_OUT_DIR = BASE_DIR / "data" / "final_split_validation"
TILED_OUT_DIR = BASE_DIR / "data" / "final_tiled_dataset"

DATASET_DIR = BASE_DIR / "data" / "final_tiled_dataset"
YOLO_DATA_YAML_PATH = DATASET_DIR / "data.yaml"

# === Splitting Parameters ===
TRAIN_RATIO = 0.8
RANDOM_SEED = 123

# === Class Labels ===
CLASS_NAMES = ["solar_panel"]

# === Training parameters ===
INIT_WEIGHTS = BASE_DIR / "runs" / "solar_v4_final_resplitretiled_100epochs" / "weights" / "best.pt"
EPOCHS = 50
IMGSZ = 640
BATCH = 4
DEVICE = "cuda"  

# === Experiment tracking ===
RUNS_DIR = BASE_DIR / "runs"
RUN_NAME = "change_runname_here"
EXIST_OK = True

# === Pipeline toggles ===
USE_AUGMENTATION = True
RUN_OBJECT_DIAGNOSTICS = True

# Diagnostics settings
DIAGNOSTICS_SPLIT = "test"
DIAGNOSTICS_MAX_IMAGES = None

# === Analysis / Diagnostics Output ===
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

OBJECT_METRICS_CSV = ANALYSIS_DIR / "object_metrics.csv"
FILTERED_OBJECT_METRICS_CSV = ANALYSIS_DIR / "object_metrics_filtered.csv"
GALLERY_DIR = ANALYSIS_DIR / "galleries"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# === Inference / Prediction settings ===
PREDICT_CONFIG_DIAGNOSTIC = {
    "conf": 0.02,
    "iou": 0.65,
    "save": False,
    "verbose": True,
}

PREDICT_CONFIG_CLEAN = {
    "conf": 0.25,
    "iou": 0.65,
    "save": True,
    "verbose": False,
}

# === Augmentation settings ===
AUGMENT_CONFIG = {
    "augment": True,

    # Lighting / color robustness
    "hsv_h": 0.015,
    "hsv_s": 0.40,
    "hsv_v": 0.40,

    # Geometry perturbations
    "degrees": 5.0,
    "translate": 0.08,
    "scale": 0.20,
    "shear": 0.0,
    "perspective": 0.0015,

    # Context augmentations
    "mosaic": 0.2,
    "close_mosaic": 10,
    "mixup": 0.0,
}

# === Feature config for diagnostics ===
FEATURE_CONFIG = {
    # Relative-to-image normalization
    "use_relative_brightness": True,
    "use_relative_edge_density": True,

    # Brightness channel
    "brightness_space": "HSV_V",

    # Edge extraction
    "edge_method": "canny",
    "canny_threshold1": 50,
    "canny_threshold2": 150,

    # Shape features
    "compute_shape_features": True,
    "compute_compactness": True,
    "compute_rectangularity": True,
    "compute_aspect_ratio": True,

    # Texture features
    "compute_texture_features": True,
    "texture_entropy_bins": 32,
    "compute_glcm_if_available": True,

    # Numerical safety
    "eps": 1e-6,
}

# === Training config bundle ===
TRAIN_CONFIG = {
    "data": str(YOLO_DATA_YAML_PATH),
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "batch": BATCH,
    "device": DEVICE,
    "project": str(RUNS_DIR),
    "name": RUN_NAME,
    "exist_ok": EXIST_OK,
    "multi_scale": False,
    "workers": 4,
    "patience": 10,
    **(AUGMENT_CONFIG if USE_AUGMENTATION else {}),
}

# === Object-level post-filter thresholds ===
OBJECT_FILTER_THRESHOLDS = {
    "min_relative_edge_density": 0.80,
    "max_relative_brightness": 1.05,
    "max_brightness_std_ratio": 1.10,
    "min_conf": 0.02,
}
