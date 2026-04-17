
"""
create_data_yaml.py

Simple utility to generate YOLO data.yaml from config settings.
Use only when dataset structure changes.
"""

import yaml
from pathlib import Path
import config


def main():
    dataset_dir = Path(config.DATASET_DIR)
    yaml_path = Path(config.YOLO_DATA_YAML_PATH)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    data_yaml = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(config.CLASS_NAMES),
        "names": config.CLASS_NAMES,
    }

    # Add test split if it exists
    if (dataset_dir / "images" / "test").exists():
        data_yaml["test"] = "images/test"

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"data.yaml written to: {yaml_path}")


if __name__ == "__main__":
    main()
