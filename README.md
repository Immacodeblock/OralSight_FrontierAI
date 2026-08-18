# OralSight Frontier AI

Experimental AI-assisted oral image screening prototype.

## What it does
- Checks whether an uploaded image is a suitable oral/teeth image.
- Checks image quality.
- Highlights suspicious caries-like regions.
- Provides non-diagnostic guidance.

## What it does not do
- Does not diagnose dental caries.
- Does not replace a dentist.
- Does not provide clinical advice.

## Current model
- YOLOv8s augmented detector for one-class caries-like region detection.
- YOLOv8n classification model for oral/non-oral image validity.

## How to run
1. Create environment.
2. Install requirements.
3. Place model files in `app/models/`.
4. Run Streamlit.

## Model files
Expected local paths:
app/models/oral_validity_yolov8n_cls_best.pt
app/models/caries_yolov8s_aug30_best.pt

## Run app
streamlit run app/app.py