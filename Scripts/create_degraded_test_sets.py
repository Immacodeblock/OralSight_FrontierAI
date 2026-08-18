from pathlib import Path
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

SOURCE_DATASET = Path(r"D:\OralSight_FrontierAI\oralsight_caries_v1")
OUTPUT_ROOT = Path(r"D:\OralSight_FrontierAI\oralsight_degradation_tests")

SOURCE_IMAGES = SOURCE_DATASET / "images" / "test"
SOURCE_LABELS = SOURCE_DATASET / "labels" / "test"

random.seed(42)


def prepare_dataset_folder(condition_name):
    condition_root = OUTPUT_ROOT / condition_name
    images_out = condition_root / "images" / "test"
    labels_out = condition_root / "labels" / "test"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    data_yaml = """path: .
train: images/test
val: images/test
test: images/test

names:
  0: caries
"""
    with open(condition_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml)

    return condition_root, images_out, labels_out


def copy_label(image_path, labels_out):
    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
    if label_path.exists():
        shutil.copy2(label_path, labels_out / label_path.name)


def save_image(img, path):
    img.save(path, quality=95)


def degrade_original(img):
    return img


def degrade_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=3))


def degrade_dark(img):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(0.45)


def degrade_bright(img):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1.65)


def degrade_compressed(img):
    # Save with low JPEG quality later
    return img


def degrade_rotated(img):
    angle = random.choice([-12, -8, 8, 12])
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0))


def degrade_occluded(img):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Add a semi-random dark rectangle covering part of image
    occ_w = int(width * random.uniform(0.18, 0.30))
    occ_h = int(height * random.uniform(0.12, 0.22))
    x1 = random.randint(0, max(0, width - occ_w))
    y1 = random.randint(0, max(0, height - occ_h))
    x2 = x1 + occ_w
    y2 = y1 + occ_h

    draw.rectangle([x1, y1, x2, y2], fill=(20, 20, 20))
    return img


CONDITIONS = {
    "original": degrade_original,
    "blur": degrade_blur,
    "dark": degrade_dark,
    "bright": degrade_bright,
    "compressed": degrade_compressed,
    "rotated": degrade_rotated,
    "occluded": degrade_occluded,
}


def main():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    image_files = (
        list(SOURCE_IMAGES.glob("*.jpg"))
        + list(SOURCE_IMAGES.glob("*.jpeg"))
        + list(SOURCE_IMAGES.glob("*.png"))
    )

    print(f"Found {len(image_files)} test images")

    for condition_name, transform_func in CONDITIONS.items():
        condition_root, images_out, labels_out = prepare_dataset_folder(condition_name)

        for image_path in image_files:
            label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

            # Use only images that have labels
            if not label_path.exists():
                continue

            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Skipping unreadable image: {image_path} | {e}")
                continue

            degraded = transform_func(img)

            out_image_path = images_out / image_path.name

            if condition_name == "compressed":
                degraded.save(out_image_path, quality=25)
            else:
                save_image(degraded, out_image_path)

            shutil.copy2(label_path, labels_out / label_path.name)

        count = len(list(images_out.glob("*.*")))
        print(f"{condition_name}: created {count} images")
        print(f"  Folder: {condition_root}")

    print("\nDone.")
    print(f"Created degraded datasets in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()