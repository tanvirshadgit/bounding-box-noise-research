from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import math
import sys

import torch
import ultralytics
from ultralytics import YOLO

print("STEP 32.95 - INDEPENDENT TEST EVALUATION")
print("=" * 78)

root = Path("results/coco2017_noise_multiseed")
data_root = Path(
    "datasets/coco2017_expansion/noise_experiments"
)
validation_root = Path("results/standardized_validation")
output_root = Path("results/independent_test")

output_csv = output_root / "independent_test_results.csv"
protocol_json = output_root / "independent_test_protocol.json"

noise_levels = [0, 5, 10, 20]
seeds = [42, 123, 456, 789, 2026]

experiments = [
    (noise, seed)
    for noise in noise_levels
    for seed in seeds
]

# Prevent accidental duplicate independent-test evaluation.
if output_csv.exists() or protocol_json.exists():
    print("STOP: Independent test output already exists.")
    print("CSV:", output_csv)
    print("Protocol:", protocol_json)
    print("Existing results have not been modified.")
    sys.exit(1)

# Verify standardized validation was completed.
validation_csv = (
    validation_root / "standardized_validation_results.csv"
)

if not validation_csv.is_file():
    print("STOP: Standardized validation CSV missing.")
    sys.exit(1)

with validation_csv.open(newline="") as f:
    validation_rows = list(csv.DictReader(f))

expected = set(experiments)

actual = {
    (int(r["noise_level"]), int(r["training_seed"]))
    for r in validation_rows
    if r.get("status") == "PASS"
}

if actual != expected or len(validation_rows) != 20:
    print("STOP: Standardized validation coverage invalid.")
    sys.exit(1)

# Verify all required checkpoints and dataset files.
missing = []

for noise, seed in experiments:
    checkpoint = (
        root / f"noise_{noise}_seed_{seed}"
        / "weights/best.pt"
    )
    dataset = (
        data_root / f"noise_{noise}/dataset.yaml"
    )

    if not checkpoint.is_file():
        missing.append(str(checkpoint))

    if not dataset.is_file():
        missing.append(str(dataset))

if missing:
    print("STOP: Missing required files.")
    for path in missing:
        print("MISSING:", path)
    sys.exit(1)

if not torch.backends.mps.is_available():
    print("STOP: MPS device unavailable.")
    sys.exit(1)

print("PREFLIGHT: PASS")
print("Models:", len(experiments))
print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)

output_root.mkdir(parents=True, exist_ok=True)

protocol = {
    "protocol": "independent_test_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "ultralytics_version": ultralytics.__version__,
    "torch_version": torch.__version__,
    "noise_levels": noise_levels,
    "training_seeds": seeds,
    "split": "test",
    "imgsz": 640,
    "batch": 8,
    "device": "mps",
    "workers": 2,
    "rect": False,
    "half": False,
    "augment": False,
    "checkpoint": "best.pt",
    "expected_test_images": 300,
    "expected_test_boxes": 2333,
    "note": (
        "Independent held-out test evaluation. "
        "Checkpoints selected using validation only. "
        "No model selection or hyperparameter tuning "
        "based on test results."
    )
}

fields = [
    "noise_level",
    "training_seed",
    "checkpoint",
    "test_precision",
    "test_recall",
    "test_map50",
    "test_map50_95",
    "status"
]

with output_csv.open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

protocol_json.write_text(
    json.dumps(protocol, indent=2),
    encoding="utf-8"
)

completed = 0
errors = []

for index, (noise, seed) in enumerate(experiments, 1):
    name = f"noise_{noise}_seed_{seed}"

    print("\n" + "=" * 78)
    print(f"TEST {index}/20: {name}")
    print("=" * 78)

    checkpoint = (
        root / name / "weights/best.pt"
    )
    dataset = (
        data_root / f"noise_{noise}/dataset.yaml"
    )

    try:
        model = YOLO(str(checkpoint.resolve()))

        metrics = model.val(
            data=str(dataset.resolve()),
            split="test",
            imgsz=640,
            batch=8,
            device="mps",
            workers=2,
            rect=False,
            half=False,
            augment=False,
            plots=False,
            save_json=False,
            project=str(output_root.resolve()),
            name=name,
            exist_ok=False,
            verbose=False
        )

        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)

        values = [
            precision, recall, map50, map50_95
        ]

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite test metric")

        row = {
            "noise_level": noise,
            "training_seed": seed,
            "checkpoint": str(checkpoint),
            "test_precision": round(precision, 6),
            "test_recall": round(recall, 6),
            "test_map50": round(map50, 6),
            "test_map50_95": round(map50_95, 6),
            "status": "PASS"
        }

        with output_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(row)

        completed += 1

        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print(f"mAP50: {map50:.6f}")
        print(f"mAP50-95: {map50_95:.6f}")
        print("STATUS: PASS")

        del model

    except Exception as exc:
        errors.append((name, str(exc)))
        print("STATUS: ERROR")
        print("ERROR:", exc)
        print("Stopping to preserve the partial report.")
        break

print("\n" + "=" * 78)
print("FINAL SUMMARY")
print("=" * 78)

print("Expected models:", len(experiments))
print("Completed tests:", completed)
print("Errors:", len(errors))
print("CSV report:", output_csv)
print("Protocol:", protocol_json)

for name, error in errors:
    print(f"ERROR {name}: {error}")

if completed == 20 and not errors:
    print("STATUS: INDEPENDENT TEST PASS - 20/20")
else:
    print("STATUS: INCOMPLETE - REVIEW REQUIRED")
