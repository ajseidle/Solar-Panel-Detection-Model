# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:52:08 2025

@author: alonzo
"""

"""
create_data_yaml.py

Purpose:
Generates YOLOv10-compatible data.yaml config file for training,
based on dataset split and shared config settings.
"""

import yaml
from config import SPLIT_OUT_DIR, CLASS_NAMES, YOLO_DATA_YAML_PATH

data_yaml = {
    "path": str(SPLIT_OUT_DIR),
    "train": "images/train",
    "val": "images/val",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES
}

with open(YOLO_DATA_YAML_PATH, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"✅ Wrote data.yaml to: {YOLO_DATA_YAML_PATH}")
