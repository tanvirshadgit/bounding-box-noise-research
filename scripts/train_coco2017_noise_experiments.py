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
EPOCHS = 20
IMGSZ = 640
BATCH = 8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute training; otherwise perform a dry run"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of experiments to execute"
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= 20:
        raise SystemExit("STOP: --limit must be between 1 and 20")

    if not MODEL.is_file():
        raise SystemExit("STOP: Pretrained model missing")

    if not torch.backends.mps.is_available():
        raise SystemExit("STOP: Apple MPS unavailable")

    experiments = [
        (level, seed)
        for level in LEVELS
        for seed in SEEDS
    ]

    print("COCO2017 NOISE MULTI-SEED EXPERIMENTS")
    print("=" * 65)
    print("Ultralytics:", ultralytics_version)
    print("PyTorch:", torch.__version__)
    print("Device: mps")
    print("Epochs:", EPOCHS)
    print("Image size:", IMGSZ)
    print("Batch:", BATCH)
    print("Noise levels:", LEVELS)
    print("Training seeds:", SEEDS)
    print("Planned experiments:", len(experiments))
    print("Execution mode:", "TRAIN" if args.run else "DRY RUN")

    for index, (level, seed) in enumerate(
        experiments[:args.limit], 1
    ):
        data = DATASETS / f"noise_{level}/dataset.yaml"
        name = f"noise_{level}_seed_{seed}"
        destination = OUTPUT / name

        if not data.is_file():
            raise SystemExit(f"STOP: Missing dataset: {data}")

        print(
            f"[{index}/{args.limit}] "
            f"{name} | epochs={EPOCHS}"
        )

        if not args.run:
            continue

        if destination.exists():
            raise SystemExit(
                f"STOP: Existing run directory: {destination}"
            )

        start = time.perf_counter()

        model = YOLO(str(MODEL))

        model.train(
            data=str(data),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
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
        results = destination / "results.csv"

        if not best.is_file() or not results.is_file():
            raise SystemExit(
                f"STOP: Missing training outputs for {name}"
            )

        metadata = {
            "noise_level": level,
            "noise_generation_seed": 42,
            "training_seed": seed,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
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

        print(
            f"COMPLETE: {name} | "
            f"{elapsed / 60:.2f} minutes"
        )

    if args.run:
        print("STATUS: REQUESTED TRAINING RUNS COMPLETE")
    else:
        print("STATUS: TRAINING DRY RUN PASS")

if __name__ == "__main__":
    main()
