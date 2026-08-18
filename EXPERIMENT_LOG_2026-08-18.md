# OralSight Frontier AI — Experiment Log

**Project:** OralSight Frontier AI  
**Document purpose:** Technical experiment log for the OralSight V1 prototype  
**Last updated:** 2026-08-18  
**Current V1 status:** Working Streamlit prototype completed  
**Selected caries model:** YOLOv8s augmented, 30 epochs  
**Selected oral-validity model:** YOLOv8n classification model  

---

## 1. Project Overview

OralSight is an experimental AI project exploring oral health risk assessment from intraoral images and, later, patient-reported symptoms.

The current V1 technical question is:

> Can we build an end-to-end prototype that accepts a user-uploaded oral image, checks whether it is suitable, assesses image quality, and highlights suspicious caries-like regions using an object detection model?

This work is **technical feasibility only**. It is **not a clinical diagnostic tool**.

The correct language for outputs is:

- suspicious caries-like region
- possible area of concern
- AI-highlighted region
- model-detected region
- non-diagnostic screening support

Avoid claiming:

- confirmed decay
- diagnosis
- disease detection
- treatment recommendation
- replacement for dental review

---

## 2. Dataset Source

Dataset: public annotated intraoral image dataset for dental caries detection.

The associated paper describes the dataset as intraoral JPG images annotated using LabelMe, with annotations converted into YOLO, Pascal VOC, and COCO formats.

The paper explicitly defines:

| Label | Meaning |
|---|---|
| `D` | permanent tooth decay |
| `d` | primary tooth decay |

Additional labels found during local audit:

| Label | Local interpretation / handling |
|---|---|
| `M` | missing / other defect, ignored for V1 |
| `m` | missing / other defect, ignored for V1 |

For the first caries detector:

```text
D + d → caries
M + m → ignored
```

Final V1 caries detector class mapping:

```text
0 = caries
```

Important dataset limitation:

> The dataset images are curated intraoral images, not fully representative of uncontrolled patient-generated smartphone photos.

This limitation motivated the degradation, robustness, quality-gate, and oral-validity experiments.

---

## 3. Dataset Structure Understood

Original dataset structure:

```text
Dataset/
  Images/
    no_retractors/
      frontal/
      left_lateral/
      mandibular/
      maxillary_occlusal/
      right_lateral/

    pilot/
      frontal/
      left_lateral/
      mandibular/
      maxillary_occlusal/
      right_lateral/

    retractors/
      frontal/
      left_lateral/
      mandibular/
      maxillary_occlusal/
      right_lateral/

  Annotations/
    Darknet_YOLO/
    Labelme/
    MS_coco/
    pascal-voc/
```

Meaning:

| Folder / term | Meaning |
|---|---|
| `Images` | intraoral dental photos |
| `Annotations` | label files describing where defects/caries are located |
| `Darknet_YOLO` | YOLO `.txt` annotation files |
| `Labelme` | JSON files with labels such as `D`, `d`, `M`, `m` |
| `pascal-voc` | XML annotation format |
| `MS_coco` | COCO JSON annotation format |
| `retractors` | images captured with cheek retractors |
| `no_retractors` | images captured without cheek retractors |
| `pilot` | smaller pilot subset |
| `frontal`, `left_lateral`, `right_lateral`, `mandibular`, `maxillary_occlusal` | different intraoral views |

---

## 4. Dataset Pair Audit

A script was run to count images and matching YOLO annotation files across all folders.

### Summary

| Count type | Count |
|---|---:|
| Total images | 6,265 |
| Total YOLO label files | 2,245 |
| Total matching image-label pairs | 2,227 |

### Strongest labelled subsets

| Subset | Matching pairs |
|---|---:|
| `no_retractors/mandibular` | 495 |
| `no_retractors/maxillary_occlusal` | 372 |
| `retractors/mandibular` | 495 |
| `retractors/maxillary_occlusal` | 378 |

These four folders contained 1,740 labelled image pairs.

### Important interpretation

Many images did not have YOLO annotation files. These should **not** automatically be treated as negative/no-caries examples, because they may simply be unannotated.

---

## 5. LabelMe Class Audit

A script was run to count label names inside all LabelMe JSON files.

### Result

| Label | Count |
|---|---:|
| `D` | 6,228 |
| `d` | 554 |
| `M` | 44 |
| `m` | 1 |
| Bad/unreadable JSON files | 0 |

### Interpretation

For the first model:

```text
D + d → caries
M + m → ignored
```

The first caries model is a one-class detector:

```text
0 = caries
```

---

## 6. Clean YOLO Dataset Creation

A cleaned YOLO dataset was created at:

```text
D:\OralSight_FrontierAI\oralsight_caries_v1
```

Structure:

```text
oralsight_caries_v1/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

Class mapping:

```text
0 = caries
```

Conversion process:

1. Read LabelMe JSON files.
2. Kept boxes labelled `D` and `d`.
3. Converted those boxes into YOLO format.
4. Rewrote all kept labels as class `0`.
5. Ignored `M` and `m`.
6. Split the result into train, validation, and test folders.

---

## 7. Annotation Sanity Check

Random samples from the cleaned dataset were visually checked by drawing YOLO boxes back onto the images.

Observation:

- Most boxes appeared correctly placed.
- Boxes were the right size.
- Boxes generally appeared over teeth that looked discoloured, decayed, or caries-like.
- It was not possible to determine whether all decayed regions were labelled.

Important caveat:

> Visual inspection can confirm that existing boxes look sensible, but it cannot confirm that there are no missing annotations. A dental expert review would be needed to audit false negatives.

---

# Experiment 001 — Baseline YOLOv8n Caries Detection

## Date

2026-07-04

## Goal

Train a first object detection baseline for visible dental caries using the cleaned one-class YOLO dataset.

## Dataset

```text
D:\OralSight_FrontierAI\oralsight_caries_v1
```

Class mapping:

```text
0 = caries
```

## Model

```text
Model: YOLOv8n
Task: one-class object detection
Epochs: 50
Image size: 640
Batch size: 8
Device: CPU
Training time: 7.995 hours
```

## Training command

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=8
```

## Validation result

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 427 | 1,334 | 0.847 | 0.922 | 0.942 | 0.749 |

## Test result

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 217 | 735 | 0.843 | 0.900 | 0.939 | 0.740 |

One test image was ignored because of out-of-bounds YOLO label coordinates:

```text
test_00165_anonymous_003-008-1187-01_1732714576631_Right_Lateral_View.jpg
```

Issue:

```text
non-normalized or out of bounds coordinates
```

## Visual prediction check

Predictions were run on the test set and visually inspected.

Observation:

- Predicted boxes looked visually sensible.
- Boxes generally appeared around discoloured or decay-like tooth regions.
- There was no obvious widespread random detection behaviour.

## Conclusion

Baseline 1 was successful.

A small YOLOv8n model can learn a one-class caries detector from the cleaned dataset and generalises well from validation to test.

However, this only proves performance on curated dataset-style intraoral images. It does not yet prove robustness on messy patient-generated smartphone photos.

---

# Experiment 002 — Basic Synthetic Degradation Test

## Date

2026-07-04

## Goal

Test whether the trained YOLOv8n caries detector degrades when test images are synthetically modified to resemble poorer-quality patient-style images.

## Model used

```text
D:\OralSight_FrontierAI\oralsight_caries_v1uns\detect	rain\weightsest.pt
```

## Original test baseline

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.843 | 0.900 | 0.939 | 0.740 |

## Degradation conditions

Basic degraded test sets were created:

```text
D:\OralSight_FrontierAI\oralsight_degradation_tests  original  blur  dark  bright  compressed  rotated  occluded```

Each folder contained:

```text
images/test/
labels/test/
data.yaml
```

## Important setup issue fixed

Initial degradation results were identical across all conditions, which indicated YOLO was probably reading the original test set instead of the degraded folders.

Cause:

```yaml
path: .
```

in each degraded `data.yaml` could be interpreted relative to the current working directory.

Fix:

Each `data.yaml` was rewritten with an absolute path, for example:

```yaml
path: D:/OralSight_FrontierAI/oralsight_degradation_tests/blur
train: images/test
val: images/test
test: images/test

names:
  0: caries
```

## Results

| Test condition | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Original | 0.843 | 0.900 | 0.939 | 0.740 |
| Blur | 0.857 | 0.854 | 0.924 | 0.722 |
| Dark | 0.833 | 0.913 | 0.940 | 0.732 |
| Bright | 0.863 | 0.838 | 0.913 | 0.689 |
| Compressed | 0.837 | 0.898 | 0.938 | 0.737 |
| Occluded | 0.834 | 0.868 | 0.911 | 0.707 |
| Rotated | 0.085 | 0.103 | 0.011 | 0.002 |

## Rotated test caveat

The rotated result is not valid as a formal metric because the images were rotated but the bounding box labels were not geometrically transformed. The model was therefore evaluated against incorrect box positions.

## Conclusion

Basic degradation testing did not break the model.

The model was more robust than expected to simple blur, darkening, compression, brightness change, and mild occlusion. However, these degradations were probably too mild and artificial compared with true patient-generated smartphone photos.

A stronger patient-photo-style degradation test was needed next.

---

# Experiment 003 — Stronger Patient-Photo-Style Degradation

## Date

2026-07-05

## Goal

Create a more realistic degradation benchmark to test whether the baseline model survives conditions closer to patient-generated smartphone photos.

## Model used

```text
YOLOv8n baseline from Experiment 001
```

## Degradation conditions

Stronger degraded test sets were created:

```text
D:\OralSight_FrontierAI\oralsight_degradation_tests_v2  original  severe_blur  very_dark  flash_glare  low_resolution  heavy_compression  combined_bad_photo```

`combined_bad_photo` simulated multiple issues together, such as low resolution, blur, brightness shift, glare, and compression.

## Results — baseline YOLOv8n v1

| Test condition | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Original | 217 | 735 | 0.841 | 0.900 | 0.939 | 0.739 |
| Severe blur | 217 | 735 | 0.868 | 0.707 | 0.855 | 0.647 |
| Very dark | 217 | 735 | 0.822 | 0.812 | 0.874 | 0.658 |
| Flash glare | 217 | 735 | 0.831 | 0.872 | 0.920 | 0.717 |
| Low resolution | 217 | 735 | 0.858 | 0.877 | 0.936 | 0.731 |
| Heavy compression | 217 | 735 | 0.826 | 0.865 | 0.922 | 0.695 |
| Combined bad photo | 217 | 735 | 0.797 | 0.649 | 0.753 | 0.511 |

## Key finding

The strongest failure mode was `combined_bad_photo`.

| Metric | Original | Combined bad photo | Drop |
|---|---:|---:|---:|
| Recall | 0.900 | 0.649 | -0.251 |
| mAP50 | 0.939 | 0.753 | -0.186 |
| mAP50-95 | 0.739 | 0.511 | -0.228 |

## Interpretation

The model remained fairly robust to isolated degradation such as low resolution, compression, and glare. However, combined poor image quality caused a major performance drop.

## Conclusion

This experiment strongly supports the OralSight thesis:

> A model trained on curated intraoral images can perform well on clean images but degrade under realistic compound patient-photo-style image-quality issues.

---

# Experiment 004 — Robustness Training with Augmented Dataset

## Date

2026-07-05

## Goal

Train a second caries detector using degraded training images to improve robustness.

## Augmented dataset

New dataset:

```text
D:\OralSight_FrontierAI\oralsight_caries_v2_augmented
```

Design:

- Clean train images were copied.
- One randomly selected degraded version was added for each training image.
- Validation and test sets remained clean.
- Degradations were non-geometric, so bounding boxes stayed valid.

Augmentations used:

- severe blur
- very dark
- flash glare
- low resolution
- heavy compression
- combined bad photo

## Training environment

Training moved from CPU to Google Colab GPU.

```text
GPU: Tesla T4
VRAM: ~15 GB
```

---

## 4A. YOLOv8n Augmented, 30 Epochs

### Model

```text
YOLOv8n
Epochs: 30
Image size: 640
Batch size: 16
```

### Training command

```bash
yolo detect train model=yolov8n.pt data=/content/oralsight_caries_v2_augmented/data.yaml epochs=30 imgsz=640 batch=16 name=train_aug_v2
```

### Results

| Condition | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Clean/original | 217 | 735 | 0.859 | 0.901 | 0.939 | 0.738 |
| Severe blur | 217 | 735 | 0.817 | 0.858 | 0.919 | 0.714 |
| Very dark | 217 | 735 | 0.844 | 0.894 | 0.929 | 0.723 |
| Flash glare | 217 | 735 | 0.840 | 0.895 | 0.931 | 0.728 |
| Low resolution | 217 | 735 | 0.827 | 0.925 | 0.942 | 0.742 |
| Heavy compression | 217 | 735 | 0.846 | 0.902 | 0.934 | 0.731 |
| Combined bad photo | 217 | 735 | 0.793 | 0.807 | 0.880 | 0.645 |

### Key comparison against v1 baseline

| Condition | Metric | v1 baseline | v2 YOLOv8n aug 30 | Change |
|---|---|---:|---:|---:|
| Clean | mAP50 | 0.939 | 0.939 | 0.000 |
| Combined bad photo | Recall | 0.649 | 0.807 | +0.158 |
| Combined bad photo | mAP50 | 0.753 | 0.880 | +0.127 |
| Combined bad photo | mAP50-95 | 0.511 | 0.645 | +0.134 |

### Conclusion

Augmentation-based robustness training worked. It substantially improved degraded-image performance while preserving clean-image performance.

YOLOv8n augmented 30 became the first strong robustness baseline.

---

## 4B. YOLOv8n Augmented, 50 Epochs

## Goal

Check whether longer training improves the YOLOv8n augmented model.

### Results

| Condition | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Clean/original | 217 | 735 | 0.855 | 0.902 | 0.938 | 0.741 |
| Severe blur | 217 | 735 | 0.817 | 0.858 | 0.919 | 0.714 |
| Very dark | 217 | 735 | 0.842 | 0.887 | 0.930 | 0.734 |
| Flash glare | 217 | 735 | 0.842 | 0.888 | 0.930 | 0.728 |
| Low resolution | 217 | 735 | 0.862 | 0.879 | 0.937 | 0.740 |
| Heavy compression | 217 | 735 | 0.845 | 0.881 | 0.928 | 0.720 |
| Combined bad photo | 217 | 735 | 0.789 | 0.841 | 0.879 | 0.645 |

### Comparison with YOLOv8n augmented 30

| Condition | Metric | 30 epochs | 50 epochs | Change |
|---|---|---:|---:|---:|
| Clean | mAP50 | 0.939 | 0.938 | -0.001 |
| Combined bad photo | Recall | 0.807 | 0.841 | +0.034 |
| Combined bad photo | mAP50 | 0.880 | 0.879 | -0.001 |
| Low resolution | Recall | 0.925 | 0.879 | -0.046 |
| Heavy compression | Recall | 0.902 | 0.881 | -0.021 |

### Conclusion

Longer training preserved clean performance and slightly improved combined bad-photo recall. However, it did not produce consistent robustness gains across all degraded conditions.

YOLOv8n augmented 50 was logged but not selected as the main model.

---

## 4C. YOLOv8s Augmented, 30 Epochs

## Goal

Check whether a larger YOLO model improves detection quality.

### Model

```text
YOLOv8s
Epochs: 30
Image size: 640
```

YOLOv8s is larger than YOLOv8n:

| Model | Parameters | GFLOPs |
|---|---:|---:|
| YOLOv8n | ~3.0M | 8.1 |
| YOLOv8s | ~11.1M | 28.4 |

### Results

| Condition | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Clean/original | 217 | 735 | 0.855 | 0.919 | 0.946 | 0.749 |
| Severe blur | 217 | 735 | 0.822 | 0.874 | 0.923 | 0.720 |
| Very dark | 217 | 735 | 0.849 | 0.913 | 0.941 | 0.742 |
| Flash glare | 217 | 735 | 0.867 | 0.888 | 0.938 | 0.738 |
| Low resolution | 217 | 735 | 0.861 | 0.894 | 0.944 | 0.746 |
| Heavy compression | 217 | 735 | 0.874 | 0.857 | 0.935 | 0.734 |
| Combined bad photo | 217 | 735 | 0.837 | 0.776 | 0.882 | 0.656 |

### Comparison with YOLOv8n augmented 30

| Condition | Metric | YOLOv8n aug 30 | YOLOv8s aug 30 | Change |
|---|---|---:|---:|---:|
| Clean | Recall | 0.901 | 0.919 | +0.018 |
| Clean | mAP50 | 0.939 | 0.946 | +0.007 |
| Clean | mAP50-95 | 0.738 | 0.749 | +0.011 |
| Severe blur | mAP50-95 | 0.714 | 0.720 | +0.006 |
| Very dark | mAP50-95 | 0.723 | 0.742 | +0.019 |
| Flash glare | mAP50-95 | 0.728 | 0.738 | +0.010 |
| Low resolution | mAP50-95 | 0.742 | 0.746 | +0.004 |
| Heavy compression | mAP50-95 | 0.731 | 0.734 | +0.003 |
| Combined bad photo | mAP50-95 | 0.645 | 0.656 | +0.011 |

### Important trade-off

YOLOv8s improved most mAP and strict localisation metrics. However, YOLOv8n retained better recall in some degraded conditions:

| Condition | YOLOv8n recall | YOLOv8s recall |
|---|---:|---:|
| Combined bad photo | 0.807 | 0.776 |
| Low resolution | 0.925 | 0.894 |
| Heavy compression | 0.902 | 0.857 |

### Conclusion

YOLOv8s augmented 30 was selected as the main caries detector going forward because it achieved the strongest overall localisation quality and clean-image performance.

YOLOv8n augmented 30 was retained as a lightweight high-recall comparison model.

---

# 8. Final Caries Model Selection

## Selected main model

```text
YOLOv8s augmented, 30 epochs
```

Use this for:

- main OralSight demo
- main experiment narrative
- future image-quality gate integration
- final reported V1 detector

## Secondary comparison model

```text
YOLOv8n augmented, 30 epochs
```

Use this for:

- lightweight comparison
- robustness comparison
- recall-focused analysis under degraded conditions

## Final ranking

| Rank | Model | Decision |
|---:|---|---|
| 1 | YOLOv8s augmented 30 | Selected main model |
| 2 | YOLOv8n augmented 30 | Lightweight robustness baseline |
| 3 | YOLOv8n augmented 50 | Logged but not selected |
| 4 | YOLOv8n v1 baseline | Original baseline only |

Final wording:

> YOLOv8s augmented 30 was selected as the main OralSight model because it delivered the best overall localisation quality and strongest average performance across clean and degraded test conditions. YOLOv8n augmented 30 is retained as a lightweight high-recall comparison model.

---

# Experiment 005 — Rule-Based Image Quality Gate

## Date

2026-07-05

## Goal

Build a pre-detection image quality gate that classifies uploaded oral images as:

```text
usable
borderline
unusable
```

This is needed because degradation experiments showed that poor image quality can reduce detector reliability.

## Features used

The quality gate was rule-based and used:

| Feature | Purpose |
|---|---|
| Blur score | Detect out-of-focus images |
| Brightness | Detect too-dark or too-bright images |
| Contrast | Detect flat/low-information images |
| Resolution | Detect very small images |
| Glare percentage | Detect overexposed reflective areas |

## Initial quality gate result

The first thresholds were too strict.

| Condition | Usable | Borderline | Unusable |
|---|---:|---:|---:|
| Original | 49 (22.5%) | 112 (51.4%) | 57 (26.1%) |
| Severe blur | 0 (0.0%) | 94 (43.1%) | 124 (56.9%) |
| Very dark | 0 (0.0%) | 0 (0.0%) | 218 (100.0%) |
| Flash glare | 80 (36.7%) | 102 (46.8%) | 36 (16.5%) |
| Low resolution | 0 (0.0%) | 105 (48.2%) | 113 (51.8%) |
| Heavy compression | 103 (47.2%) | 99 (45.4%) | 16 (7.3%) |
| Combined bad photo | 0 (0.0%) | 4 (1.8%) | 214 (98.2%) |

Problem:

> 26.1% of original curated images were being rejected as unusable, which was too harsh.

## Tuned quality gate result

Thresholds were adjusted to reduce unfair rejection of normal images while still flagging poor-quality images.

| Condition | Usable | Borderline | Unusable |
|---|---:|---:|---:|
| Original | 147 (67.4%) | 71 (32.6%) | 0 (0.0%) |
| Severe blur | 0 (0.0%) | 217 (99.5%) | 1 (0.5%) |
| Very dark | 0 (0.0%) | 0 (0.0%) | 218 (100.0%) |
| Flash glare | 176 (80.7%) | 42 (19.3%) | 0 (0.0%) |
| Low resolution | 0 (0.0%) | 217 (99.5%) | 1 (0.5%) |
| Heavy compression | 207 (95.0%) | 11 (5.0%) | 0 (0.0%) |
| Combined bad photo | 1 (0.5%) | 83 (38.1%) | 134 (61.5%) |

## Interpretation

The tuned quality gate behaved sensibly:

- Original curated images were mostly usable or borderline, with no unusable classifications.
- Very dark images were correctly rejected as unusable.
- Severe blur and low-resolution images were mostly marked borderline.
- Combined bad-photo images were mostly marked unusable or borderline.
- Heavy compression was mostly accepted, matching detector results that showed compression was not a severe failure mode.

## Conclusion

Experiment 005 produced a usable rule-based quality gate.

The gate is suitable for V1 as a pre-detection safety and reliability check.

---

# Experiment 006 — Quality Gate Validation Against Detector Performance

## Date

2026-07-05

## Goal

Validate whether the quality gate labels correspond to actual differences in YOLO detector performance.

The key question was:

> Do images classified as usable produce better caries-like region detection performance than borderline or unusable images?

## Method

The Experiment 005 quality gate CSV was used to create three YOLO-compatible test subsets:

```text
D:\OralSight_FrontierAI\experiment_006_quality_buckets  usable    images	est    labels	est    data.yaml

  borderline    images	est    labels	est    data.yaml

  unusable    images	est    labels	est    data.yaml
```

Images from the degradation test sets were grouped according to their quality-gate label.

## Model used

```text
YOLOv8s augmented, 30 epochs
```

## Bucket creation counts

| Quality bucket | Images copied |
|---|---:|
| Usable | 531 |
| Borderline | 641 |
| Unusable | 354 |
| Missing labels | 0 |
| Missing images | 0 |

During YOLO validation, a small number of images were ignored because the same known invalid-label issue appeared in the bucketed datasets.

## Validation results

| Quality bucket | Images evaluated | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Usable | 528 | 1,831 | 0.871 | 0.883 | 0.942 | 0.747 |
| Borderline | 639 | 2,115 | 0.846 | 0.866 | 0.926 | 0.718 |
| Unusable | 352 | 1,199 | 0.840 | 0.867 | 0.918 | 0.709 |

## Key finding

Detector performance was highest on images classified as `usable`.

| Metric | Usable | Borderline | Unusable |
|---|---:|---:|---:|
| mAP50 | 0.942 | 0.926 | 0.918 |
| mAP50-95 | 0.747 | 0.718 | 0.709 |

This showed the desired trend:

```text
usable > borderline > unusable
```

## Interpretation

The quality gate is not just producing arbitrary labels. It separates images into groups where the detector performs differently.

The drop from usable to unusable was:

```text
mAP50:     0.942 → 0.918  (-0.024)
mAP50-95:  0.747 → 0.709  (-0.038)
```

The drop is not catastrophic, suggesting that the selected YOLOv8s detector is fairly robust. However, the consistent decrease supports the use of quality gating and retake guidance.

## Conclusion

Experiment 006 validated the usefulness of the image quality gate.

The detector performed best on images classified as usable and lower on borderline and unusable images. This supports the V1 workflow where image quality is checked before detection.

Recommended system behaviour:

| Quality label | System action |
|---|---|
| Usable | Run detector normally |
| Borderline | Run detector with a reliability warning |
| Unusable | Ask user to retake the image, with optional research-mode detection |

---

# Experiment 007 — Oral Image Validity Classifier

## Date

2026-07-15 to 2026-07-16

## Goal

Build a classifier that checks whether an uploaded image is actually suitable for oral/teeth analysis before running the caries detector.

This protects the app from running the caries detector on random non-dental images such as food, rooms, animals, documents, or general face photos.

## Task

Binary image classification:

```text
oral
non_oral
```

Definition used:

| Class | Meaning |
|---|---|
| `oral` | close-up oral/teeth image suitable for OralSight analysis |
| `non_oral` | anything else, including general scenes, objects, people, smiling face photos, and images where teeth do not dominate the frame |

Important distinction:

> The classifier is not asking whether teeth exist somewhere in the image. It is asking whether the image is suitable for OralSight dental-image analysis.

## Dataset sources

### Oral class

All raw oral images were used from:

```text
D:\OralSight_FrontierAI\Dataset\Images
```

Total raw oral images available:

```text
6,265
```

### Non-oral class

A general non-oral image folder was used:

```text
D:\OralSight_FrontierAI
on_oral_source_Images
```

This folder contained 8,000+ images, including general Flickr-style scenes and many people/smiling-face images.

The non-oral class was **not randomly downsampled**, because preserving variety and tricky negatives was more useful than strict balancing.

## Created dataset

```text
D:\OralSight_FrontierAI\oralsight_validity_dataset
```

Structure:

```text
oralsight_validity_dataset/
  train/
    oral/
    non_oral/
  val/
    oral/
    non_oral/
  test/
    oral/
    non_oral/
```

Dataset size reported by Ultralytics:

| Split | Images | Classes |
|---|---:|---:|
| Train | 10,048 | 2 |
| Val | 2,152 | 2 |
| Test | 2,156 | 2 |

## Model

```text
YOLOv8n-cls
Task: binary image classification
Image size: 224
```

## Validation result

| Split | top1 accuracy | top5 accuracy |
|---|---:|---:|
| Validation | 1.000 | 1.000 |

For binary classification, `top1_acc` is the important metric. `top5_acc` is not meaningful because there are only two classes.

## Test set confusion matrix

Test set size:

```text
oral: 941 images
non_oral: 1215 images
```

Confusion matrix:

| True label | Predicted non_oral | Predicted oral | Total |
|---|---:|---:|---:|
| non_oral | 1,215 | 0 | 1,215 |
| oral | 0 | 941 | 941 |
| Total | 1,215 | 941 | 2,156 |

Overall test accuracy:

```text
1.000
```

Accuracy by class:

| Class | Accuracy |
|---|---:|
| non_oral | 1.000 |
| oral | 1.000 |

Most important safety result:

```text
Non-oral images wrongly accepted as oral: 0 / 1215
False accept rate: 0.0%
```

## Threshold analysis

Thresholds from 0.50 to 0.95 all produced:

```text
Oral accept rate: 1.0
Non-oral false accept rate: 0.0
```

A conservative app threshold was selected:

```text
Oral acceptance threshold: 0.80
Non-oral rejection threshold: 0.85
```

App logic:

```python
if pred_label == "oral" and confidence >= 0.80:
    decision = "ACCEPT"
elif pred_label == "non_oral" and confidence >= 0.85:
    decision = "REJECT"
else:
    decision = "UNCERTAIN"
```

## Phone-captured image sanity test

A small manual test was performed using eight phone-captured images.

| File | Prediction | Confidence | Decision |
|---|---|---:|---|
| `IMG_20260715_162753810.jpg` | non_oral | 0.6314 | UNCERTAIN |
| `IMG_20260716_114028629.jpg` | non_oral | 0.9991 | REJECT |
| `IMG_20260716_114039733.jpg` | oral | 0.9709 | ACCEPT |
| `IMG_20260716_114120389.jpg` | oral | 1.0000 | ACCEPT |
| `IMG_20260716_114143513.jpg` | oral | 1.0000 | ACCEPT |
| `IMG_20260716_114214398_MP.jpg` | oral | 1.0000 | ACCEPT |
| `IMG_20260716_114234049_HDR.jpg` | oral | 0.9998 | ACCEPT |
| `IMG_20260716_114307503.jpg` | oral | 0.9981 | ACCEPT |

Manual interpretation:

- Close-up teeth/oral photos were accepted with high confidence.
- Face-heavy or smile-style photos were rejected or marked uncertain.
- The classifier appears to have learned a useful suitability boundary: images where teeth dominate the frame are accepted, while general face/smile images are rejected or marked uncertain.

## Conclusion

Experiment 007 successfully produced an oral-validity classifier for the V1 pipeline.

The classifier achieved perfect validation and test accuracy on the prepared dataset and performed sensibly on a small phone-image sanity check.

This enables the app to reject or flag unsuitable images before caries-like region detection.

---

# Experiment 008 — OralSight Streamlit Demo App

## Date

2026-07-16 onward

## Goal

Build a working end-to-end OralSight V1 demo app that combines:

1. image upload,
2. oral image validity classification,
3. image quality gating,
4. caries-like region detection,
5. annotated image output,
6. safe non-diagnostic guidance.

## App framework

```text
Streamlit
```

The app runs locally on the laptop and can be accessed from an Android phone browser over the same Wi-Fi network.

## App folder

```text
D:\OralSight_FrontierAI\oralsight_demo_app
```

Expected structure:

```text
oralsight_demo_app/
  app.py
  models/
    oral_validity_yolov8n_cls_best.pt
    caries_yolov8s_aug30_best.pt
```

## Models used

### Oral-validity classifier

```text
oral_validity_yolov8n_cls_best.pt
```

Purpose:

```text
Check whether the uploaded image is a suitable close-up oral/teeth image.
```

### Caries-like region detector

```text
caries_yolov8s_aug30_best.pt
```

Purpose:

```text
Highlight suspicious caries-like regions in accepted oral images.
```

## V1 app pipeline

```text
User uploads image
↓
Step 1: Oral-validity classifier
  ACCEPT / UNCERTAIN / REJECT
↓
Step 2: Image quality gate
  usable / borderline / unusable
↓
Step 3: YOLOv8s caries-like detector
↓
Display annotated image + detection confidences + safe guidance
```

## Oral-validity app behaviour

| Decision | Meaning | App behaviour |
|---|---|---|
| ACCEPT | image appears suitable for oral analysis | continue |
| UNCERTAIN | may not show teeth clearly enough | warn; allow demo continuation |
| REJECT | not a suitable close-up oral/teeth image | stop and ask user to retake |

## Image-quality app behaviour

| Quality label | Meaning | App behaviour |
|---|---|---|
| usable | acceptable image quality | run detector normally |
| borderline | possible reliability issue | run detector with warning |
| unusable | too poor for reliable detection | ask for retake; optional research/demo mode |

## Safety wording

The app includes warnings that:

- the prototype is experimental,
- it does not provide a dental diagnosis,
- highlighted regions are AI-generated suspicious caries-like areas,
- outputs should be reviewed by a dental professional.

## Local laptop test

The app was successfully run locally using:

```bash
streamlit run app.py
```

The app worked in the laptop browser.

## Android phone test

The app was also tested from an Android phone over the same Wi-Fi network.

Laptop Wi-Fi IPv4 used:

```text
192.168.0.180
```

Streamlit was run using:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Phone browser URL:

```text
http://192.168.0.180:8501
```

Result:

```text
The app worked from the Android phone browser.
```

## Conclusion

Experiment 008 completed the first working OralSight V1 application prototype.

The app demonstrates the complete V1 workflow:

```text
oral-validity check
+
image-quality gate
+
YOLOv8s caries-like region detector
+
annotated output
+
non-diagnostic guidance
```

This turns the project from a model experiment into an end-to-end AI-assisted oral image screening prototype.

---

# Current V1 Status

OralSight V1 is complete as a working technical prototype.

## Completed components

| Component | Status |
|---|---|
| Dataset audit | Done |
| Label policy definition | Done |
| Clean YOLO dataset creation | Done |
| Baseline YOLOv8n caries detector | Done |
| Basic degradation test | Done |
| Stronger patient-photo-style degradation test | Done |
| Augmented robustness training | Done |
| YOLOv8n vs YOLOv8s comparison | Done |
| Main detector selection | Done |
| Rule-based image quality gate | Done |
| Quality gate validation | Done |
| Oral/non-oral validity classifier | Done |
| Phone-image sanity check | Done |
| Streamlit demo app | Done |
| Android phone browser access | Done |

## V1 selected models

| Role | Model |
|---|---|
| Main caries detector | YOLOv8s augmented, 30 epochs |
| Lightweight comparison caries detector | YOLOv8n augmented, 30 epochs |
| Oral image validity classifier | YOLOv8n-cls |

## V1 claim

OralSight V1 can be described as:

> A working AI-assisted oral image screening prototype that checks whether an uploaded image is a suitable close-up oral/teeth image, assesses image quality, highlights suspicious caries-like regions using a robustness-trained YOLOv8s detector, and provides safe non-diagnostic guidance.

## V1 should not claim

OralSight V1 should not claim:

- clinical diagnosis,
- confirmed caries detection,
- overall dental disease assessment,
- dentist replacement,
- validated performance on real patient-generated images,
- gum disease / ulcer / plaque / oral cancer detection.

---

# Current Caveats

## 1. This is not a diagnostic tool

The model detects caries-like labelled regions in images. It does not diagnose a patient.

## 2. Dataset annotations may have false negatives

Visual inspection confirmed that many labelled boxes look sensible, but it cannot confirm whether all caries regions were labelled.

## 3. Dataset images are curated

The original dataset images were captured under more controlled conditions than real patient-at-home smartphone photos.

## 4. Synthetic degradation is not the same as real phone photography

The degradation tests are useful but still artificial.

Real patient images may include:

```text
wrong mouth angle
teeth not centred
partial mouth visible
tongue/finger blocking view
flash reflection
motion blur
saliva glare
low resolution
bad focus
too close / too far
wrong view entirely
face-heavy smiling photo
background objects
```

## 5. Current caries detector is one-class only

The V1 detector uses:

```text
D + d → caries
```

It does not distinguish:

- primary vs permanent decay,
- plaque,
- gum disease,
- ulcers,
- missing teeth,
- fillings,
- crowns,
- tooth fracture,
- oral cancer signs.

## 6. Real-world clinical accuracy is not yet validated

The model has benchmark results on labelled dataset images and synthetic degradation sets. It does not yet have dentist-labelled real phone image validation.

---

# Recommended Next Experiments

## Experiment 009 — End-to-End Demo Pipeline Evaluation

Goal:

> Test the full app pipeline on a controlled set of images and record expected vs actual behaviour.

Create a folder:

```text
D:\OralSight_FrontierAI\oralsight_v1_test_suite
```

Suggested test set:

| Category | Count |
|---|---:|
| Random non-oral images | 20 |
| Smiling face / face-heavy images | 10 |
| Close-up phone teeth images | 10–20 |
| Clean dataset oral images | 20 |
| Degraded oral images | 20 |
| Poor-quality teeth images | 10 |

Expected checks:

| Input type | Expected behaviour |
|---|---|
| Random object/scene | Reject as non-oral |
| Smiling face | Reject or uncertain |
| Close-up teeth | Accept |
| Dark/blurry teeth | Accept oral, but warn quality |
| Clean oral dataset image | Run detector |
| Bad oral image | Ask for retake or warn |

## Experiment 010 — App Logging

Add CSV logging to the Streamlit app.

Log fields:

```text
timestamp
filename
oral_validity_label
oral_validity_confidence
oral_validity_decision
quality_label
quality_reasons
number_of_caries_detections
detection_confidences
final_app_message
```

This will make app testing measurable instead of memory-based.

## Experiment 011 — Symptom Questionnaire and Risk Guidance

Add a short symptom form:

```text
tooth pain
sensitivity to cold/sweet
bleeding gums
swelling
fever
duration
severe/worsening pain
trauma
difficulty swallowing/breathing
```

Then combine:

```text
image findings + symptoms + quality result
```

into non-diagnostic risk guidance:

```text
low immediate concern
routine dental check advised
dental appointment advised
urgent dental care advised
retake image required
```

## Experiment 012 — Real Phone Image Validation

Longer-term validation would require:

```text
real phone oral photos
+
dentist/expert annotations
+
comparison against YOLO predictions
```

This is required before making any stronger claim about real-world clinical usefulness.

---

# Useful Commands

## Train baseline YOLOv8n

```bash
cd D:\OralSight_FrontierAI\oralsight_caries_v1
yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=8
```

## Validate baseline on test split

```bash
cd D:\OralSight_FrontierAI\oralsight_caries_v1
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml split=test
```

## Validate selected YOLOv8s model on clean test

```bash
yolo detect val model=/content/runs/detect/train_aug_v2_yolov8s_30/weights/best.pt data=/content/oralsight_caries_v2_augmented/data.yaml split=test
```

## Validate selected YOLOv8s model on combined bad-photo test

```bash
yolo detect val model=/content/runs/detect/train_aug_v2_yolov8s_30/weights/best.pt data=/content/oralsight_degradation_tests_v2/combined_bad_photo/data.yaml split=test
```

## Train oral-validity classifier

```bash
yolo classify train model=yolov8n-cls.pt data=/content/oralsight_validity_dataset epochs=20 imgsz=224 batch=64 name=oral_validity_yolov8n_cls
```

## Validate oral-validity classifier

```bash
yolo classify val model=/content/runs/classify/oral_validity_yolov8n_cls/weights/best.pt data=/content/oralsight_validity_dataset
```

## Run Streamlit demo locally

```bash
D:\OralSight_FrontierAI\oralsight_env\Scriptsctivate
cd D:\OralSight_FrontierAI\oralsight_demo_app
streamlit run app.py
```

## Run Streamlit demo for Android phone testing

```bash
D:\OralSight_FrontierAI\oralsight_env\Scriptsctivate
cd D:\OralSight_FrontierAI\oralsight_demo_app
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Example phone URL:

```text
http://192.168.0.180:8501
```

---

# Final V1 Summary

OralSight V1 has progressed from dataset audit to a working demo application.

The completed V1 pipeline is:

```text
Upload image
↓
Oral-validity classifier
↓
Image quality gate
↓
YOLOv8s caries-like region detector
↓
Annotated output + confidence values
↓
Non-diagnostic guidance / retake advice
```

The strongest technical result is that robustness training improved degraded-image performance while preserving clean-image performance, and the quality gate was validated against detector performance.

The strongest product result is that the Streamlit app now runs locally and from an Android phone browser, demonstrating the full OralSight V1 workflow.

Next priority:

```text
Experiment 009 — End-to-End Demo Pipeline Evaluation
```
