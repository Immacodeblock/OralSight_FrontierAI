from pathlib import Path

# Change this to wherever your Dataset folder is
DATASET_ROOT = Path(r"D:\OralSight_FrontierAI\Dataset")

#image_folder = DATASET_ROOT / "Images" / "no_retractors" / "frontal"
#label_folder = DATASET_ROOT / "Annotations" / "Darknet_YOLO" / "no_retractors" / "frontal"

groups = ["no_retractors", "pilot", "retractors"]
views = [
    "frontal",
    "left_lateral",
    "mandibular",
    "maxillary_occlusal",
    "right_lateral",
]

# Find image files
image_extensions = [".jpg", ".jpeg", ".png"]
total_images = 0
total_labels = 0
total_pairs = 0

print("Dataset pair check across all folders")
print("------------------------------------")

for group in groups:
    for view in views:
        image_folder = DATASET_ROOT / "Images" / group / view
        label_folder = DATASET_ROOT / "Annotations" / "Darknet_YOLO" / group / view

        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(f"*{ext}"))

        label_files = list(label_folder.glob("*.txt"))

        image_stems = {img.stem for img in image_files}
        label_stems = {lbl.stem for lbl in label_files}

        matching_pairs = image_stems & label_stems
        images_without_labels = image_stems - label_stems
        labels_without_images = label_stems - image_stems

        total_images += len(image_files)
        total_labels += len(label_files)
        total_pairs += len(matching_pairs)

        print(f"\n{group}/{view}")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        print(f"  Matching pairs: {len(matching_pairs)}")
        print(f"  Images without labels: {len(images_without_labels)}")
        print(f"  Labels without images: {len(labels_without_images)}")

print("\nTOTAL")
print("-----")
print(f"Total images: {total_images}")
print(f"Total labels: {total_labels}")
print(f"Total matching pairs: {total_pairs}")