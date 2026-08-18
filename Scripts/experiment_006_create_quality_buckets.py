from pathlib import Path
import csv
import shutil


QUALITY_CSV = Path(r"D:\OralSight_FrontierAI\experiment_005_quality_gate_results.csv")

DEGRADATION_ROOT = Path(r"D:\OralSight_FrontierAI\oralsight_degradation_tests_v2")

OUTPUT_ROOT = Path(r"D:\OralSight_FrontierAI\experiment_006_quality_buckets")

QUALITY_BUCKETS = ["usable", "borderline", "unusable"]


def make_clean_output_dirs():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for bucket in QUALITY_BUCKETS:
        (OUTPUT_ROOT / bucket / "images" / "test").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / bucket / "labels" / "test").mkdir(parents=True, exist_ok=True)


def write_data_yaml(bucket: str):
    bucket_root = OUTPUT_ROOT / bucket

    yaml_text = f"""path: {bucket_root.as_posix()}
train: images/test
val: images/test
test: images/test

names:
  0: caries
"""

    (bucket_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    make_clean_output_dirs()

    counts = {
        "usable": 0,
        "borderline": 0,
        "unusable": 0,
        "missing_label": 0,
        "missing_image": 0,
    }

    with open(QUALITY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            condition = row["condition"]
            filename = row["filename"]
            quality_label = row["quality_label"]

            if quality_label not in QUALITY_BUCKETS:
                continue

            image_path = DEGRADATION_ROOT / condition / "images" / "test" / filename
            label_path = DEGRADATION_ROOT / condition / "labels" / "test" / f"{Path(filename).stem}.txt"

            if not image_path.exists():
                counts["missing_image"] += 1
                continue

            if not label_path.exists():
                counts["missing_label"] += 1
                continue

            # Prefix condition to avoid filename collisions between folders
            new_stem = f"{condition}__{Path(filename).stem}"
            new_image_name = f"{new_stem}{Path(filename).suffix}"
            new_label_name = f"{new_stem}.txt"

            out_image = OUTPUT_ROOT / quality_label / "images" / "test" / new_image_name
            out_label = OUTPUT_ROOT / quality_label / "labels" / "test" / new_label_name

            shutil.copy2(image_path, out_image)
            shutil.copy2(label_path, out_label)

            counts[quality_label] += 1

    for bucket in QUALITY_BUCKETS:
        write_data_yaml(bucket)

    print("Experiment 006 — Quality bucket datasets created")
    print("------------------------------------------------")
    print(f"Output root: {OUTPUT_ROOT}")
    print()
    print("Counts:")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    print()
    print("Created:")
    for bucket in QUALITY_BUCKETS:
        print(f"  {OUTPUT_ROOT / bucket}")


if __name__ == "__main__":
    main()