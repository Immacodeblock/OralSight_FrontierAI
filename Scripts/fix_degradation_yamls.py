from pathlib import Path

ROOT = Path(r"D:\OralSight_FrontierAI\oralsight_degradation_tests")

conditions = [
    "original",
    "blur",
    "dark",
    "bright",
    "compressed",
    "rotated",
    "occluded",
]

for condition in conditions:
    condition_root = ROOT / condition
    yaml_path = condition_root / "data.yaml"

    yaml_text = f"""path: {condition_root.as_posix()}
train: images/test
val: images/test
test: images/test

names:
  0: caries
"""

    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Updated {yaml_path}")