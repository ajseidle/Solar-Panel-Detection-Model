
"""
general_control.py

Primary orchestration script for the solar panel object detection pipeline.

What this script does:
1. Validates the dataset and data.yaml
2. Trains YOLO from the configured starting checkpoint
3. Evaluates the trained best.pt checkpoint
4. Optionally runs object-level diagnostics

All configurable parameters (paths, hyperparameters, toggles) are defined in
config.py to support reproducible experiments and consistent pipeline behavior.
"""

from pathlib import Path
import subprocess
import os
import sys

from ultralytics import YOLO


# =========================
# Resolve project root safely
# =========================
def resolve_project_root() -> Path:
    
    here = Path(__file__).resolve().parent

    if (here / "config.py").exists():
        return here

    if (here.parent / "config.py").exists():
        return here.parent

    raise FileNotFoundError(
        "Could not locate project root. Expected config.py either next to "
        "general_control.py or one folder above it."
    )


ROOT = resolve_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


# =========================
# Environment helpers
# =========================
def build_subprocess_env() -> dict:
    """
    Create an environment so subprocess scripts can import project modules.
    """
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    return env


def get_python_executable() -> str:
    return sys.executable


def get_diagnostics_script_path() -> Path:
   
    candidate_paths = [
        ROOT / "scripts" / "run_object_diagnostics.py",
        ROOT / "run_object_diagnostics.py",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find run_object_diagnostics.py in either "
        f"{ROOT / 'scripts'} or {ROOT}"
    )


# =========================
# Dataset resolution
# =========================
def resolve_data_yaml() -> Path:
    """
    Resolve and validate the configured dataset directory and YAML path.
    """
    dataset_dir = Path(config.DATASET_DIR)
    data_yaml = Path(config.YOLO_DATA_YAML_PATH)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {data_yaml}")

    print(f"[Dataset] Using dataset directory: {dataset_dir}")
    print(f"[Dataset] Using dataset YAML: {data_yaml}")

    print("\n[Dataset] YAML contents:")
    print(data_yaml.read_text(encoding="utf-8"))

    return data_yaml


# =========================
# Training
# =========================
def train_model(data_yaml: Path) -> Path:
   
    init_weights = Path(config.INIT_WEIGHTS)
    if not init_weights.exists():
        raise FileNotFoundError(f"Initial weights not found at: {init_weights}")

    print(f"\n[Train] Initializing model from: {init_weights}")
    model = YOLO(str(init_weights))

    train_kwargs = dict(getattr(config, "TRAIN_CONFIG", {}))
    train_kwargs["data"] = str(data_yaml)

    print("\n[Train] Training config:")
    for key, value in train_kwargs.items():
        print(f"  {key}: {value}")
    print(f"  class_names: {getattr(config, 'CLASS_NAMES', 'Not set in config')}")

    model.train(**train_kwargs)

    best_pt = Path(config.RUNS_DIR) / config.RUN_NAME / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(
            f"Training appears to have finished, but best.pt was not found at: {best_pt}"
        )

    return best_pt


# =========================
# Evaluation
# =========================
def evaluate_model(best_pt: Path, data_yaml: Path) -> None:
  
    print(f"\n[Eval] Loading best checkpoint from: {best_pt}")
    model = YOLO(str(best_pt))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=config.IMGSZ,
        device=config.DEVICE,
    )

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


# =========================
# Diagnostics
# =========================
def run_object_diagnostics(best_pt: Path, data_yaml: Path) -> None:
   
    if not getattr(config, "RUN_OBJECT_DIAGNOSTICS", True):
        print("\n[Diagnostics] Skipped because RUN_OBJECT_DIAGNOSTICS=False")
        return

    diagnostics_script = get_diagnostics_script_path()
    env = build_subprocess_env()

    print("\n[Diagnostics] Running object-level diagnostics...")
    print(f"[Diagnostics] Script: {diagnostics_script}")

    subprocess.run(
        [
            get_python_executable(),
            str(diagnostics_script),
            str(best_pt),
            str(data_yaml),
        ],
        check=True,
        cwd=str(ROOT),
        env=env,
    )

    print(f"[Diagnostics] Done. Outputs in: {getattr(config, 'ANALYSIS_DIR', ROOT / 'analysis')}")


# =========================
# Main pipeline
# =========================
def main() -> None:
    
    print(f"[Project] Root: {ROOT}")

    data_yaml = resolve_data_yaml()
    best_pt = train_model(data_yaml)
    evaluate_model(best_pt, data_yaml)
    run_object_diagnostics(best_pt, data_yaml)


if __name__ == "__main__":
    main()
