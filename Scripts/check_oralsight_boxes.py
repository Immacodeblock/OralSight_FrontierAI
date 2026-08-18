from pathlib import Path
import random
from PIL import Image, ImageDraw

# Change this to your actual output folder
DATASET_ROOT = Path(r"D:\OralSight_FrontierAI\oralsight_caries_v1")

# Choose which split to inspect: train, val, or test
SPLIT = "train"

images_dir = DATASET_ROOT / "images" / SPLIT
labels_dir = DATASET_ROOT / "labels" / SPLIT
output_dir = DATASET_ROOT / "box_check_outputs"
output_dir.mkdir(exist_ok=True)

image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))

print(f"Found {len(image_files)} images in {images_dir}")

sample_images = random.sample(image_files, min(10, len(image_files)))

for image_path in sample_images:
    label_path = labels_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"No label for {image_path.name}")
        continue

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    draw = ImageDraw.Draw(image)

    lines = label_path.read_text().strip().splitlines()

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            print(f"Bad label line in {label_path.name}: {line}")
            continue

        class_id, x_center, y_center, box_width, box_height = parts

        x_center = float(x_center)
        y_center = float(y_center)
        box_width = float(box_width)
        box_height = float(box_height)

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, max(0, y1 - 20)), "caries", fill="red")

    output_path = output_dir / f"checked_{image_path.name}"
    image.save(output_path)

    print(f"Saved: {output_path}")

print()
print("Done. Open the images inside:")
print(output_dir)