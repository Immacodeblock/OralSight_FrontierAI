from pathlib import Path
import json
import random
import shutil
from PIL import Image

# CHANGE THESE TWO PATHS
DATASET_ROOT = Path(r"D:\OralSight_FrontierAI\Dataset")
OUTPUT_ROOT = Path(r"D:\OralSight_FrontierAI\oralsight_caries_v1")

LABELME_ROOT = DATASET_ROOT / "Annotations" / "Labelme"
IMAGES_ROOT = DATASET_ROOT / "Images"

CARIES_LABELS = {"D", "d"}   # keep these
IGNORE_LABELS = {"M", "m"}   # ignore these

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

random.seed(42)


def find_matching_image(json_file, data):
    """
    Try to find the image that belongs to this LabelMe JSON file.
    We use the JSON filename first, then LabelMe's imagePath if needed.
    """

    image_stem = json_file.stem

    # Work out relative path after Labelme folder
    # Example:
    # Labelme/no_retractors/frontal/file.json
    # Images/no_retractors/frontal/file.jpg
    relative_parent = json_file.parent.relative_to(LABELME_ROOT)

    candidate_folder = IMAGES_ROOT / relative_parent

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = candidate_folder / f"{image_stem}{ext}"
        if candidate.exists():
            return candidate

    # fallback: try imagePath inside JSON
    image_path_from_json = data.get("imagePath")
    if image_path_from_json:
        candidate = candidate_folder / Path(image_path_from_json).name
        if candidate.exists():
            return candidate

    return None


def labelme_box_to_yolo(points, image_width, image_height):
    """
    Convert LabelMe rectangle points to YOLO format:
    x_center y_center width height
    all normalized between 0 and 1.
    """

    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]

    x_min = min(x_values)
    y_min = min(y_values)
    x_max = max(x_values)
    y_max = max(y_values)

    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    box_width = (x_max - x_min) / image_width
    box_height = (y_max - y_min) / image_height

    return x_center, y_center, box_width, box_height


# Clear old output if it exists
if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

for split in ["train", "val", "test"]:
    (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


items = []
skipped_no_image = 0
skipped_no_caries = 0
bad_json = 0

for json_file in LABELME_ROOT.rglob("*.json"):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        bad_json += 1
        continue

    image_file = find_matching_image(json_file, data)

    if image_file is None:
        skipped_no_image += 1
        continue

    try:
        with Image.open(image_file) as img:
            image_width, image_height = img.size
    except Exception:
        skipped_no_image += 1
        continue

    yolo_lines = []

    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points", [])

        if label not in CARIES_LABELS:
            continue

        if len(points) < 2:
            continue

        x_center, y_center, box_width, box_height = labelme_box_to_yolo(
            points, image_width, image_height
        )

        # class 0 = caries
        yolo_lines.append(
            f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    if not yolo_lines:
        skipped_no_caries += 1
        continue

    items.append((image_file, yolo_lines))


random.shuffle(items)

total = len(items)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": items[:train_end],
    "val": items[train_end:val_end],
    "test": items[val_end:],
}


for split, split_items in splits.items():
    for idx, (image_file, yolo_lines) in enumerate(split_items):
        # Make filename unique, because different folders may contain same filename
        new_stem = f"{split}_{idx:05d}_{image_file.stem}"
        new_image_name = new_stem + image_file.suffix.lower()
        new_label_name = new_stem + ".txt"

        shutil.copy2(image_file, OUTPUT_ROOT / "images" / split / new_image_name)

        with open(OUTPUT_ROOT / "labels" / split / new_label_name, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))


data_yaml = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: caries
"""

with open(OUTPUT_ROOT / "data.yaml", "w", encoding="utf-8") as f:
    f.write(data_yaml)


print("Created OralSight caries YOLO dataset")
print("------------------------------------")
print(f"Output folder: {OUTPUT_ROOT}")
print(f"Total usable images with D/d caries labels: {total}")
print(f"Train images: {len(splits['train'])}")
print(f"Val images: {len(splits['val'])}")
print(f"Test images: {len(splits['test'])}")
print()
print(f"Skipped JSON files with no matching image: {skipped_no_image}")
print(f"Skipped JSON files with no D/d caries labels: {skipped_no_caries}")
print(f"Bad JSON files: {bad_json}")
print()
print("Class mapping:")
print("  0 = caries")