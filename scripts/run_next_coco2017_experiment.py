from pathlib import Path
import argparse
import csv
import json
import time
from datetime import datetime, timezone

import torch
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "yolo11n.pt"
DATASETS = ROOT / "datasets/coco2017_expansion/noise_experiments"
OUTPUT = ROOT / "results/coco2017_noise_multiseed"

LEVELS = [0, 5, 10, 20]
SEEDS = [42, 123, 456, 789, 2026]

def inspect_run(destination, level, seed):
    if not destination.exists():
        return "pending"

    required = [
        destination / "weights/best.pt",
        destination / "weights/last.pt",
        destination / "results.csv",
        destination / "experiment_metadata.json",
    ]

    if not all(p.is_file() for p in required):
        return "incomplete"

    try:
        info = json.loads(
            (destination / "experiment_metadata.json").read_text()
        )

        with (destination / "results.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))

        if (
            info.get("noise_level") == level
            and info.get("training_seed") == seed
            and info.get("epochs") == 20
            and len(rows) == 20
        ):
            return "completed"

    except (OSError, ValueError, KeyError):
        pass

    return "incomplete"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="Train next pending experiment"
    )
    args = parser.parse_args()

    experiments = [
        (level, seed)
        for level in LEVELS
        for seed in SEEDS
    ]

    completed = []
    pending = []
    incomplete = []

    for level, seed in experiments:
        name = f"noise_{level}_seed_{seed}"
        destination = OUTPUT / name
        status = inspect_run(destination, level, seed)

        if status == "completed":
            completed.append(name)
        elif status == "pending":
            pending.append((level, seed, name))
        else:
            incomplete.append(name)

    print("COCO2017 SAFE SINGLE-EXPERIMENT RUNNER")
    print("=" * 65)
    print("Completed:", len(completed))
    print("Pending:", len(pending))
    print("Incomplete:", len(incomplete))

    if incomplete:
        print("Incomplete runs:", incomplete)
        raise SystemExit("STOP: Manual inspection required")

    if not pending:
        print("STATUS: ALL EXPERIMENTS COMPLETE")
        return

    level, seed, name = pending[0]
    destination = OUTPUT / name
    data = DATASETS / f"noise_{level}/dataset.yaml"

    print("Next experiment:", name)
    print("Noise:", level)
    print("Seed:", seed)
    print("Dataset:", data)
    print("Epochs: 20")
    print("Device: mps")

    if not args.run:
        print("STATUS: NEXT EXPERIMENT READY")
        return

    if destination.exists():
        raise SystemExit("STOP: Destination already exists")

    if not MODEL.is_file() or not data.is_file():
        raise SystemExit("STOP: Model or dataset missing")

    if not torch.backends.mps.is_available():
        raise SystemExit("STOP: MPS unavailable")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    model = YOLO(str(MODEL))

    model.train(
        data=str(data),
        epochs=20,
        imgsz=640,
        batch=8,
        device="mps",
        seed=seed,
        deterministic=True,
        workers=2,
        project=str(OUTPUT),
        name=name,
        exist_ok=False,
        pretrained=True,
        plots=False,
        save=True,
        verbose=True,
    )

    elapsed = time.perf_counter() - start

    best = destination / "weights/best.pt"
    last = destination / "weights/last.pt"
    results = destination / "results.csv"

    if not all(p.is_file() for p in [best, last, results]):
        raise SystemExit("STOP: Training output incomplete")

    with results.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 20:
        raise SystemExit("STOP: Unexpected epoch count")

    metadata = {
        "noise_level": level,
        "noise_generation_seed": 42,
        "training_seed": seed,
        "epochs": 20,
        "imgsz": 640,
        "batch": 8,
        "device": "mps",
        "dataset": str(data),
        "model": str(MODEL),
        "ultralytics_version": ultralytics_version,
        "torch_version": torch.__version__,
        "elapsed_minutes": round(elapsed / 60, 3),
        "completed_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with (destination / "experiment_metadata.json").open("x") as f:
        json.dump(metadata, f, indent=2)

    print("COMPLETE:", name)
    print("Duration:", round(elapsed / 60, 2), "minutes")
    print("STATUS: SINGLE EXPERIMENT TRAINING PASS")

if __name__ == "__main__":
    main()
