# Solar Panel Detection from Satellite Imagery using YOLOv10

## Overview

This project develops a deep learning pipeline to detect rooftop solar panels in satellite imagery using a fine-tuned YOLOv10 model.

Training was performed on imagery from Jordan, with evaluation conducted on unseen data from refugee camp regions in Lebanon.

This work supports analysis of solar energy adoption in resource-constrained environments.

---

## Dataset

- **Training Region:** Jordan (Mafraq, Irbid, Amman)  
- **Testing Region:** Lebanon (refugee camp regions near Tripoli, Tyre, Beirut/Sidon)  
- **Annotation Format:** YOLO  

### Preprocessing

- Images tiled into **512×512 patches** with overlap  
- Empty tiles retained to provide negative examples  
- Bounding boxes adjusted for tiled images  

### Key Challenges

- Small object size relative to image resolution  
- Dense and cluttered urban environments  
- Visual similarity to non-solar structures  
- Domain shift between training and test regions  

---

## Objectives

- Build an object detection model for solar panel identification  
- Improve detection performance on small and dense rooftop installations  
- Analyze model errors using object-level diagnostics  

---

## Model

- **Base architecture:** `yolov10s.pt` (pretrained, fine-tuned on custom dataset)  
- **Task:** Single-class object detection (`solar_panel`)  
- **Framework:** Ultralytics YOLO  

### Key Design Choices

- **Single-class training** to improve precision/recall stability  
- **Image tiling (512px with overlap)** to better detect small objects  
- **Custom augmentations** for satellite imagery robustness  
- **Post-hoc diagnostics** using solar panel features such as shape and edge density  

---

## Pipeline Overview

1. **Dataset Preparation**
   - Split raw satellite images into train/validation sets  
   - Optionally apply tiling to increase effective resolution  

2. **Model Training**
   - Initialize from pretrained `yolov10s.pt`  
   - Train using custom dataset and augmentation settings  

3. **Evaluation**
   - Evaluate performance using:
     - Precision  
     - Recall  
     - mAP@0.5  
     - mAP@0.5:0.95  
     - F1 Score  

4. **Diagnostics**
   - Analyze detection quality using:
     - Relative brightness  
     - Edge density  
     - Shape features  
     - Object-level filtering heuristics  

---

## How to Run

pip install -r requirements.txt  
python src/pipelines/general_control.py  

> Note: Update file paths in `config.py` before running.

---

## Results

### Model Performance

| Dataset        | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | F1 Score |
|----------------|----------|--------|--------|--------------|----------|
| Jordan (Val)   | 0.765    | 0.704  | 0.799  | 0.422        | 0.733    |
| Lebanon (Test) | 0.486    | 0.222  | 0.284  | 0.110        | 0.305    |


### Key Findings

- The model achieves strong performance on the validation set (Jordan)  
- Performance drops on the Lebanon test set due to domain shift  
- Recall decreases significantly on smaller and densely packed solar installations  


### Interpretation

- **High precision, lower recall** indicates the model is conservative in detections  
- Reduced performance on Lebanon highlights challenges in generalization:
  - smaller object size 
  - higher object density
  - different spatial context
- These results suggest the model is learning region-specific visual patterns rather than fully generalizable solar panel features
- This demonstrates the importance of evaluating object detection models under domain shift, particularly for applications involving geographically diverse satellite imagery.

---

