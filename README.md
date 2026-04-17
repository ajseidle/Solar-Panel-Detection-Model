# Solar Panel Detection from Satellite Imagery using YOLOv10

## Overview

This project develops a deep learning pipeline to detect rooftop solar panels in satellite imagery using a fine-tuned YOLOv10 model.

Training was performed on imagery from Jordan, with evaluation conducted on unseen data from refugee camp regions in Lebanon.

This work supports analysis of solar energy adoption in resource-constrained environments.

---

## Dataset

- Created custom dataset
  
- **Training Region:** Jordan (Mafraq, Irbid, Amman)  
- **Testing Region:** Lebanon (refugee camp regions near Tripoli, Tyre, Beirut/Sidon)  
- **Annotation Format:** YOLO  

### Preprocessing

- Images/bounding boxes tiled into **512×512 patches** with overlap  
- Empty tiles retained to provide negative examples  

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
   - Files: split_dataset.py, tile_dataset.py, create_data_yaml.py
   - Split raw satellite images into train/validation sets  
   - Optionally apply tiling to increase effective resolution  

3. **Model Training**
   - Files: config.py, general_control.py
   - Initialize from pretrained `yolov10s.pt`  
   - Train using custom dataset and augmentation settings  

5. **Evaluation**
   - Files: general_control.py
   - Evaluate performance using:
     - Precision  
     - Recall  
     - mAP@0.5  
     - mAP@0.5:0.95  
     - F1 Score  

7. **Diagnostics**
   - Files: run_object_diagnostics.py, analyze_predictions.py, visualize_annotations.py
   - Analyze detection quality using (run_object_diagnostics.py):
     - Edge density  
     - Shape features  
     - Object-level filtering heuristics
   - Compare actual vs predicted object counts and confidence per image (analyze_predictions.py)
   - Visually inspect annotations and labeled images for quality control (visualize_annotations.py)

---

## How to Run

pip install -r requirements.txt  
python general_control.py  

> Note: Update dataset and file paths in `config.py` before running.

> Note: This project was initially developed and executed in Google Colab. 

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

