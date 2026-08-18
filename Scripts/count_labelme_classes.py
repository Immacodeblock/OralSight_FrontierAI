from pathlib import Path
import json
from collections import Counter

# Change this to your actual Dataset folder
DATASET_ROOT = Path(r"D:\OralSight_FrontierAI\Dataset")
labelme_root = DATASET_ROOT / "Annotations" / "Labelme"

label_counter = Counter()
file_counter = 0
bad_files = []

for json_file in labelme_root.rglob("*.json"):
    file_counter += 1

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])

        for shape in shapes:
            label = shape.get("label")
            if label:
                label_counter[label] += 1

    except Exception as e:
        bad_files.append((json_file, str(e)))

print("LabelMe audit")
print("-------------")
print(f"Total LabelMe JSON files checked: {file_counter}")
print()

print("Labels found:")
for label, count in label_counter.most_common():
    print(f"  {label}: {count}")

print()
print(f"Bad/unreadable JSON files: {len(bad_files)}")

if bad_files:
    print("\nFirst 10 bad files:")
    for file, error in bad_files[:10]:
        print(file, error)