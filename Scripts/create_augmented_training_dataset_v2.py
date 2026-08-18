from pathlib import Path
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

SOURCE_DATASET = Path(r"D:\OralSight_FrontierAI\oralsight_caries_v1")
OUTPUT_DATASET = Path(r"D:\OralSight_FrontierAI\oralsight_caries_v2_augmented")

random.seed(42)

TRAIN_IMAGES = SOURCE_DATASET / "images" / "train"
TRAIN_LABELS = SOURCE_DATASET / "labels" / "train"

VAL_IMAGES = SOURCE_DATASET / "images" / "val"
VAL_LABELS = SOURCE_DATASET / "labels" / "val"

TEST_IMAGES = SOURCE_DATASET / "images" / "test"
TEST_LABELS = SOURCE_DATASET / "labels" / "test"


def make_dirs():
    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)

    for split in ["train", "val", "test"]:
        (OUTPUT_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_clean_split(split_name, src_images, src_labels):
    out_images = OUTPUT_DATASET / "images" / split_name
    out_labels = OUTPUT_DATASET / "labels" / split_name

    image_files = list(src_images.glob("*.jpg")) + list(src_images.glob("*.jpeg")) + list(src_images.glob("*.png"))

    copied = 0

    for image_path in image_files:
        label_path = src_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        shutil.copy2(image_path, out_images / image_path.name)
        shutil.copy2(label_path, out_labels / label_path.name)
        copied += 1

    return copied


def severe_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=5))


def very_dark(img):
    img = ImageEnhance.Brightness(img).enhance(0.30)
    img = ImageEnhance.Contrast(img).enhance(0.85)
    return img


def flash_glare(img):
    img = img.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    for _ in range(random.randint(1, 2)):
        radius = random.randint(int(min(width, height) * 0.05), int(min(width, height) * 0.13))
        x = random.randint(radius, max(radius, width - radius))
        y = random.randint(radius, max(radius, height - radius))

        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(255, 255, 255, 100),
        )

        inner = int(radius * 0.45)
        draw.ellipse(
            [x - inner, y - inner, x + inner, y + inner],
            fill=(255, 255, 255, 180),
        )

    return img.convert("RGB")


def low_resolution(img):
    width, height = img.size
    small_w = max(64, int(width * 0.30))
    small_h = max(64, int(height * 0.30))

    img_small = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
    img_up = img_small.resize((width, height), Image.Resampling.BILINEAR)
    return img_up


def heavy_compression(img):
    return img


def combined_bad_photo(img):
    width, height = img.size

    small_w = max(64, int(width * 0.32))
    small_h = max(64, int(height * 0.32))

    img = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
    img = img.resize((width, height), Image.Resampling.BILINEAR)

    img = img.filter(ImageFilter.GaussianBlur(radius=2.8))

    brightness = random.choice([0.40, 0.50, 1.45, 1.65])
    img = ImageEnhance.Brightness(img).enhance(brightness)

    contrast = random.choice([0.70, 0.80, 1.20])
    img = ImageEnhance.Contrast(img).enhance(contrast)

    if random.random() < 0.50:
        img = flash_glare(img)

    return img


AUGMENTATIONS = {
    "severe_blur": severe_blur,
    "very_dark": very_dark,
    "flash_glare": flash_glare,
    "low_resolution": low_resolution,
    "heavy_compression": heavy_compression,
    "combined_bad_photo": combined_bad_photo,
}


def create_augmented_train():
    out_images = OUTPUT_DATASET / "images" / "train"
    out_labels = OUTPUT_DATASET / "labels" / "train"

    image_files = list(TRAIN_IMAGES.glob("*.jpg")) + list(TRAIN_IMAGES.glob("*.jpeg")) + list(TRAIN_IMAGES.glob("*.png"))

    original_count = 0
    augmented_count = 0

    for image_path in image_files:
        label_path = TRAIN_LABELS / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        # 1. Copy original image and label
        shutil.copy2(image_path, out_images / image_path.name)
        shutil.copy2(label_path, out_labels / label_path.name)
        original_count += 1

        # 2. Create one randomly selected degraded copy per image
        # This doubles the training set without making CPU training unbearable.
        aug_name, aug_func = random.choice(list(AUGMENTATIONS.items()))

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping unreadable image: {image_path} | {e}")
            continue

        aug_img = aug_func(img)

        new_stem = f"{image_path.stem}__aug_{aug_name}"
        new_image_path = out_images / f"{new_stem}.jpg"
        new_label_path = out_labels / f"{new_stem}.txt"

        if aug_name == "heavy_compression":
            aug_img.save(new_image_path, quality=10)
        elif aug_name == "combined_bad_photo":
            aug_img.save(new_image_path, quality=20)
        else:
            aug_img.save(new_image_path, quality=95)

        shutil.copy2(label_path, new_label_path)
        augmented_count += 1

    return original_count, augmented_count


def write_data_yaml():
    data_yaml = f"""path: {OUTPUT_DATASET.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: caries
"""
    (OUTPUT_DATASET / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main():
    make_dirs()

    original_train, augmented_train = create_augmented_train()
    val_count = copy_clean_split("val", VAL_IMAGES, VAL_LABELS)
    test_count = copy_clean_split("test", TEST_IMAGES, TEST_LABELS)

    write_data_yaml()

    print("Created augmented YOLO dataset")
    print("--------------------------------")
    print(f"Output: {OUTPUT_DATASET}")
    print(f"Original train images copied: {original_train}")
    print(f"Augmented train images created: {augmented_train}")
    print(f"Total train images: {original_train + augmented_train}")
    print(f"Val images copied: {val_count}")
    print(f"Test images copied: {test_count}")
    print()
    print("Class mapping:")
    print("  0 = caries")


if __name__ == "__main__":
    main()