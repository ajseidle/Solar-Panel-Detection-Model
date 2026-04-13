
"""
general_control.py

Primary orchestration script for the solar panel object detection pipeline.

Handles dataset loading, model initialization, evaluation on external datasets,
and reporting of key metrics (precision, recall, mAP, F1). Also controls
optional post-processing steps such as object-level diagnostics.

All configurable parameters (paths, hyperparameters, toggles) are defined in
config.py to support reproducible experiments and consistent pipeline behavior.
"""

from ultralytics import YOLO
from pathlib import Path
import subprocess
import os
import sys

# --- Resolve project root safely ---
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

sys.path.append(str(ROOT))

import config


def main():

    # --- Environment so subprocess scripts can import config.py ---
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    # --- 1) Prepare dataset ---
    if not getattr(config, "USE_TILED_DATASET", False):

        print("[Dataset] Splitting raw dataset...")

        subprocess.run(
            ["python", "scripts/split_dataset.py"],
            check=True,
            env=env
        )

        print("[Dataset] Creating data.yaml...")

        subprocess.run(
            ["python", "scripts/create_data_yaml.py"],
            check=True,
            env=env
        )

    else:
        print("[Dataset] Using tiled dataset — skipping split and yaml creation.")

    # --- 2) Locate dataset YAML ---

    print("[Dataset] Using existing Lebanon tiled test dataset.")
    data_yaml = Path(config.YOLO_DATA_YAML_PATH)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {data_yaml}")

    print("\n[Eval] YAML contents:")
    print(data_yaml.read_text())


    # --- 3) Train model ---
    model = YOLO(config.MODEL_WEIGHTS)

    train_kwargs = dict(getattr(config, "TRAIN_CONFIG", {}))
    train_kwargs["data"] = str(data_yaml)

    print("\n[Train] Training config:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")
    print(f"  class_names: {getattr(config, 'CLASS_NAMES', 'Not set in config')}")

    model.train(**train_kwargs)

    weights_path = Path(config.MODEL_WEIGHTS)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    print(f"\n[Eval] Loading weights from: {weights_path}")
    model = YOLO(str(weights_path))


    # --- 4) Evaluate best weights ---
    best_pt = Path(config.RUNS_DIR) / config.RUN_NAME / "weights" / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found at: {best_pt}")

    model = YOLO(str(best_pt))
    
    print(f"[Dataset] Using dataset YAML at: {data_yaml}")

    metrics = model.val(
        data=str(data_yaml),
        imgsz=config.IMGSZ,
        device=config.DEVICE
    )

    # --- 5) Print key metrics ---
    results = metrics.mean_results()

    precision = results[0]
    recall = results[1]
    map50 = results[2]
    map5095 = results[3]

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n=== RESULTS ===")
    print(f"Run name:      {config.RUN_NAME}")
    print(f"data.yaml:     {data_yaml}")
    print(f"best.pt:       {best_pt}")
    print(f"Precision:     {precision:.3f}")
    print(f"Recall:        {recall:.3f}")
    print(f"mAP@0.5:       {map50:.3f}")
    print(f"mAP@0.5:0.95:  {map5095:.3f}")
    print(f"F1 Score:      {f1:.3f}")


    # --- 6) Diagnostics ---
    if getattr(config, "RUN_OBJECT_DIAGNOSTICS", True):

        print("\n[Diagnostics] Running object-level diagnostics...")

        subprocess.run(
            ["python", "scripts/run_object_diagnostics.py", str(weights_path), str(data_yaml)],
            check=True,
            env=env
        )

        print(f"[Diagnostics] Done. Outputs in: {getattr(config, 'ANALYSIS_DIR', 'analysis/')}")


if __name__ == "__main__":
    main()





