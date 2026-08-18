from pathlib import Path
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

SOURCE_DATASET = Path(r"D:\OralSight_FrontierAI\oralsight_caries_v1")
OUTPUT_ROOT = Path(r"D:\OralSight_FrontierAI\oralsight_degradation_tests_v2")

SOURCE_IMAGES = SOURCE_DATASET / "images" / "test"
SOURCE_LABELS = SOURCE_DATASET / "labels" / "test"

random.seed(42)


def prepare_dataset_folder(condition_name):
    condition_root = OUTPUT_ROOT / condition_name
    images_out = condition_root / "images" / "test"
    labels_out = condition_root / "labels" / "test"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Important: absolute path so YOLO reads the correct folder
    data_yaml = f"""path: {condition_root.as_posix()}
train: images/test
val: images/test
test: images/test

names:
  0: caries
"""
    with open(condition_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml)

    return condition_root, images_out, labels_out


def save_jpeg(img, path, quality=95):
    img.save(path, quality=quality)


def severe_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=7))


def very_dark(img):
    img = ImageEnhance.Brightness(img).enhance(0.22)
    img = ImageEnhance.Contrast(img).enhance(0.80)
    return img


def flash_glare(img):
    img = img.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # Add 1–3 white/transparent glare circles
    for _ in range(random.randint(1, 3)):
        radius = random.randint(int(min(width, height) * 0.06), int(min(width, height) * 0.16))
        x = random.randint(radius, max(radius, width - radius))
        y = random.randint(radius, max(radius, height - radius))

        # Outer translucent white circle
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(255, 255, 255, 120),
        )

        # Inner stronger glare
        inner = int(radius * 0.45)
        draw.ellipse(
            [x - inner, y - inner, x + inner, y + inner],
            fill=(255, 255, 255, 200),
        )

    return img.convert("RGB")


def low_resolution(img):
    width, height = img.size

    # Downscale heavily, then upscale back to original size
    small_w = max(64, int(width * 0.25))
    small_h = max(64, int(height * 0.25))

    img_small = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
    img_up = img_small.resize((width, height), Image.Resampling.BILINEAR)
    return img_up


def heavy_compression(img):
    return img


def combined_bad_photo(img):
    width, height = img.size

    # 1. Low resolution
    small_w = max(64, int(width * 0.28))
    small_h = max(64, int(height * 0.28))
    img = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
    img = img.resize((width, height), Image.Resampling.BILINEAR)

    # 2. Blur
    img = img.filter(ImageFilter.GaussianBlur(radius=3.5))

    # 3. Random brightness shift
    brightness = random.choice([0.35, 0.45, 1.55, 1.75])
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # 4. Lower contrast sometimes
    img = ImageEnhance.Contrast(img).enhance(random.choice([0.65, 0.75, 1.25]))

    # 5. Add glare sometimes
    if random.random() < 0.65:
        img = flash_glare(img)

    return img


CONDITIONS = {
    "original": lambda img: img,
    "severe_blur": severe_blur,
    "very_dark": very_dark,
    "flash_glare": flash_glare,
    "low_resolution": low_resolution,
    "heavy_compression": heavy_compression,
    "combined_bad_photo": combined_bad_photo,
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

        created = 0

        for image_path in image_files:
            label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Skipping unreadable image: {image_path} | {e}")
                continue

            degraded = transform_func(img)

            out_image_path = images_out / image_path.name

            if condition_name == "heavy_compression":
                save_jpeg(degraded, out_image_path, quality=8)
            elif condition_name == "combined_bad_photo":
                save_jpeg(degraded, out_image_path, quality=18)
            else:
                save_jpeg(degraded, out_image_path, quality=95)

            shutil.copy2(label_path, labels_out / label_path.name)
            created += 1

        print(f"{condition_name}: created {created} images")
        print(f"  Folder: {condition_root}")

    print("\nDone.")
    print(f"Created degraded datasets in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()