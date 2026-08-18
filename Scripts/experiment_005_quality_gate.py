from pathlib import Path
import csv
import cv2
import numpy as np


# ------------------------------------------------------------
# Experiment 005 — OralSight Image Quality Gate
# ------------------------------------------------------------
# This script checks dental image quality before running YOLO.
#
# It gives each image one of three labels:
#   usable
#   borderline
#   unusable
#
# It checks:
#   - blur
#   - brightness
#   - contrast
#   - resolution
#   - glare / overexposure
#
# This is rule-based, not machine learning.
# ------------------------------------------------------------


# Change this if your folder is somewhere else
INPUT_ROOT = Path(r"D:\OralSight_FrontierAI\oralsight_degradation_tests_v2")

OUTPUT_CSV = Path(r"D:\OralSight_FrontierAI\experiment_005_quality_gate_results.csv")


CONDITIONS = [
    "original",
    "severe_blur",
    "very_dark",
    "flash_glare",
    "low_resolution",
    "heavy_compression",
    "combined_bad_photo",
]


# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------
# These are starting values.
# We will tune them after seeing results.

MIN_WIDTH_UNUSABLE = 320
MIN_HEIGHT_UNUSABLE = 320

MIN_WIDTH_BORDERLINE = 500
MIN_HEIGHT_BORDERLINE = 500

BLUR_UNUSABLE = 25
BLUR_BORDERLINE = 80

BRIGHTNESS_DARK_UNUSABLE = 40
BRIGHTNESS_DARK_BORDERLINE = 65

BRIGHTNESS_BRIGHT_BORDERLINE = 215
BRIGHTNESS_BRIGHT_UNUSABLE = 240

CONTRAST_UNUSABLE = 18
CONTRAST_BORDERLINE = 32

GLARE_BORDERLINE_PERCENT = 5.0
GLARE_UNUSABLE_PERCENT = 12.0


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calculate_quality_metrics(image_path: Path):
    """
    Reads an image and calculates quality metrics.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur score
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness
    brightness = float(np.mean(gray))

    # Contrast
    contrast = float(np.std(gray))

    # Glare / overexposure
    # Count pixels that are almost white in all channels.
    b, g, r = cv2.split(image)
    glare_mask = (r > 245) & (g > 245) & (b > 245)
    glare_percent = float(np.sum(glare_mask) / glare_mask.size * 100)

    return {
        "width": width,
        "height": height,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "glare_percent": glare_percent,
    }


def classify_image(metrics):
    """
    Applies simple rules to classify image quality.
    """

    reasons = []
    severe_issues = 0
    borderline_issues = 0

    width = metrics["width"]
    height = metrics["height"]
    blur_score = metrics["blur_score"]
    brightness = metrics["brightness"]
    contrast = metrics["contrast"]
    glare_percent = metrics["glare_percent"]

    # Resolution check
    # Resolution check
    if width < MIN_WIDTH_UNUSABLE or height < MIN_HEIGHT_UNUSABLE:
        severe_issues += 1
        reasons.append("very low resolution")
    elif width < MIN_WIDTH_BORDERLINE or height < MIN_HEIGHT_BORDERLINE:
        borderline_issues += 1
        reasons.append("low resolution")

    # Blur check
    if blur_score < BLUR_UNUSABLE:
        severe_issues += 1
        reasons.append("too blurry")
    elif blur_score < BLUR_BORDERLINE:
        borderline_issues += 1
        reasons.append("slightly blurry")

    # Brightness check: too dark
    if brightness < BRIGHTNESS_DARK_UNUSABLE:
        severe_issues += 1
        reasons.append("too dark")
    elif brightness < BRIGHTNESS_DARK_BORDERLINE:
        borderline_issues += 1
        reasons.append("dark")

    # Brightness check: too bright
    if brightness > BRIGHTNESS_BRIGHT_UNUSABLE:
        severe_issues += 1
        reasons.append("too bright")
    elif brightness > BRIGHTNESS_BRIGHT_BORDERLINE:
        borderline_issues += 1
        reasons.append("bright")

    # Contrast check
    if contrast < CONTRAST_UNUSABLE:
        severe_issues += 1
        reasons.append("low contrast")
    elif contrast < CONTRAST_BORDERLINE:
        borderline_issues += 1
        reasons.append("borderline contrast")

    # Glare check
    if glare_percent > GLARE_UNUSABLE_PERCENT:
        severe_issues += 1
        reasons.append("heavy glare")
    elif glare_percent > GLARE_BORDERLINE_PERCENT:
        borderline_issues += 1
        reasons.append("some glare")

    # Final quality label
    if severe_issues >= 2:
        quality_label = "unusable"
    elif severe_issues == 1 and borderline_issues >= 2:
        quality_label = "unusable"
    elif severe_issues == 1:
        quality_label = "borderline"
    elif borderline_issues >= 2:
        quality_label = "borderline"
    else:
        quality_label = "usable"

    return quality_label, reasons


def find_images(folder: Path):
    """
    Finds image files inside a folder.
    """

    image_files = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(path)

    return sorted(image_files)


def main():
    rows = []

    print("Experiment 005 — Image Quality Gate")
    print("-----------------------------------")
    print(f"Input root: {INPUT_ROOT}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print()

    for condition in CONDITIONS:
        condition_folder = INPUT_ROOT / condition / "images" / "test"

        if not condition_folder.exists():
            print(f"Skipping missing folder: {condition_folder}")
            continue

        image_files = find_images(condition_folder)

        print(f"{condition}: found {len(image_files)} images")

        for image_path in image_files:
            metrics = calculate_quality_metrics(image_path)

            if metrics is None:
                rows.append({
                    "condition": condition,
                    "filename": image_path.name,
                    "quality_label": "unusable",
                    "width": "",
                    "height": "",
                    "blur_score": "",
                    "brightness": "",
                    "contrast": "",
                    "glare_percent": "",
                    "reasons": "unreadable image",
                })
                continue

            quality_label, reasons = classify_image(metrics)

            rows.append({
                "condition": condition,
                "filename": image_path.name,
                "quality_label": quality_label,
                "width": metrics["width"],
                "height": metrics["height"],
                "blur_score": round(metrics["blur_score"], 2),
                "brightness": round(metrics["brightness"], 2),
                "contrast": round(metrics["contrast"], 2),
                "glare_percent": round(metrics["glare_percent"], 2),
                "reasons": "; ".join(reasons),
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "condition",
            "filename",
            "quality_label",
            "width",
            "height",
            "blur_score",
            "brightness",
            "contrast",
            "glare_percent",
            "reasons",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Done.")
    print(f"Saved results to: {OUTPUT_CSV}")

    print()
    print("Summary:")
    print("--------")

    summary = {}

    for row in rows:
        condition = row["condition"]
        quality_label = row["quality_label"]

        if condition not in summary:
            summary[condition] = {
                "usable": 0,
                "borderline": 0,
                "unusable": 0,
                "total": 0,
            }

        summary[condition][quality_label] += 1
        summary[condition]["total"] += 1

    for condition, counts in summary.items():
        total = counts["total"]

        usable_pct = counts["usable"] / total * 100 if total else 0
        borderline_pct = counts["borderline"] / total * 100 if total else 0
        unusable_pct = counts["unusable"] / total * 100 if total else 0

        print()
        print(condition)
        print(f"  Total:      {total}")
        print(f"  Usable:     {counts['usable']} ({usable_pct:.1f}%)")
        print(f"  Borderline: {counts['borderline']} ({borderline_pct:.1f}%)")
        print(f"  Unusable:   {counts['unusable']} ({unusable_pct:.1f}%)")


if __name__ == "__main__":
    main()