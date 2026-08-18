from pathlib import Path
import shutil
import random
from PIL import Image


# ------------------------------------------------------------
# Experiment 007 — Oral Image Validity Dataset Builder
# ------------------------------------------------------------
# Creates a binary image classification dataset:
#
#   train/oral
#   train/non_oral
#   val/oral
#   val/non_oral
#   test/oral
#   test/non_oral
#
# oral:
#   all intraoral/teeth images from the raw dental dataset
#
# non_oral:
#   all images from the Flickr/general non-oral source folder
#
# Important:
#   This script does NOT balance the classes.
#   It uses all available oral and non-oral images.
# ------------------------------------------------------------


ORAL_SOURCE_ROOT = Path(r"D:\OralSight_FrontierAI\Dataset\Images")

NON_ORAL_SOURCE_ROOT = Path(r"D:\OralSight_FrontierAI\non_oral_source_Images")

OUTPUT_ROOT = Path(r"D:\OralSight_FrontierAI\oralsight_validity_dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def find_images(folder: Path):
    images = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)


def is_readable_image(image_path: Path):
    """
    Checks whether PIL can open the image.
    This avoids copying broken/corrupt files into the classifier dataset.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def clean_output():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val", "test"]:
        for class_name in ["oral", "non_oral"]:
            (OUTPUT_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)


def split_images(images):
    images = images.copy()
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    return {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }


def copy_images(split_dict, class_name):
    copied_counts = {}
    skipped_counts = {}

    for split, image_paths in split_dict.items():
        copied = 0
        skipped = 0

        for index, image_path in enumerate(image_paths):
            if not is_readable_image(image_path):
                skipped += 1
                continue

            new_name = f"{class_name}_{split}_{index:05d}{image_path.suffix.lower()}"
            output_path = OUTPUT_ROOT / split / class_name / new_name

            try:
                shutil.copy2(image_path, output_path)
                copied += 1
            except Exception as e:
                print(f"Could not copy {image_path}: {e}")
                skipped += 1

        copied_counts[split] = copied
        skipped_counts[split] = skipped

    return copied_counts, skipped_counts


def main():
    random.seed(RANDOM_SEED)

    print("Experiment 007 — Oral Image Validity Dataset Builder")
    print("---------------------------------------------------")
    print(f"Oral source:     {ORAL_SOURCE_ROOT}")
    print(f"Non-oral source: {NON_ORAL_SOURCE_ROOT}")
    print(f"Output:          {OUTPUT_ROOT}")
    print()

    if not ORAL_SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Oral source folder not found: {ORAL_SOURCE_ROOT}")

    if not NON_ORAL_SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Non-oral source folder not found: {NON_ORAL_SOURCE_ROOT}")

    oral_images = find_images(ORAL_SOURCE_ROOT)
    non_oral_images = find_images(NON_ORAL_SOURCE_ROOT)

    print(f"Found oral images:     {len(oral_images)}")
    print(f"Found non-oral images: {len(non_oral_images)}")
    print()

    if len(oral_images) == 0:
        raise ValueError("No oral images found. Check ORAL_SOURCE_ROOT.")

    if len(non_oral_images) == 0:
        raise ValueError("No non-oral images found. Check NON_ORAL_SOURCE_ROOT.")

    print("Using all available images.")
    print("No class balancing applied.")
    print()

    clean_output()

    oral_splits = split_images(oral_images)
    non_oral_splits = split_images(non_oral_images)

    oral_counts, oral_skipped = copy_images(oral_splits, "oral")
    non_oral_counts, non_oral_skipped = copy_images(non_oral_splits, "non_oral")

    print("Created dataset:")
    print("----------------")

    total_oral = 0
    total_non_oral = 0

    for split in ["train", "val", "test"]:
        oral_count = oral_counts[split]
        non_oral_count = non_oral_counts[split]

        total_oral += oral_count
        total_non_oral += non_oral_count

        print(split)
        print(f"  oral:     {oral_count}")
        print(f"  non_oral: {non_oral_count}")
        print(f"  skipped oral:     {oral_skipped[split]}")
        print(f"  skipped non_oral: {non_oral_skipped[split]}")
        print()

    print("Final totals:")
    print(f"  oral:     {total_oral}")
    print(f"  non_oral: {total_non_oral}")
    print(f"  total:    {total_oral + total_non_oral}")
    print()

    print("Class distribution:")
    total = total_oral + total_non_oral
    print(f"  oral:     {total_oral / total * 100:.1f}%")
    print(f"  non_oral: {total_non_oral / total * 100:.1f}%")
    print()

    print("Done.")
    print(f"Dataset created at: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()