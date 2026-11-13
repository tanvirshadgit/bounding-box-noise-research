from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts/train_coco2017_noise_experiments.py"
OUTPUT = ROOT / "results/coco2017_noise_multiseed"

LEVELS = [0, 5, 10, 20]
SEEDS = [42, 123, 456, 789, 2026]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    if not 1 <= args.limit <= 20:
        raise SystemExit("STOP: Invalid limit")

    experiments = [
        (level, seed)
        for level in LEVELS
        for seed in SEEDS
    ]

    completed = []
    pending = []
    incomplete = []

    for index, (level, seed) in enumerate(experiments, 1):
        name = f"noise_{level}_seed_{seed}"
        destination = OUTPUT / name

        if not destination.exists():
            pending.append((index, name))
            continue

        required = [
            destination / "weights/best.pt",
            destination / "weights/last.pt",
            destination / "results.csv",
            destination / "experiment_metadata.json",
        ]

        if all(p.is_file() for p in required):
            try:
                metadata = json.loads(
                    (destination / "experiment_metadata.json").read_text()
                )
                if (
                    metadata.get("noise_level") == level
                    and metadata.get("training_seed") == seed
                    and metadata.get("epochs") == 20
                ):
                    completed.append((index, name))
                else:
                    incomplete.append((index, name))
            except (OSError, ValueError):
                incomplete.append((index, name))
        else:
            incomplete.append((index, name))

    print("COCO2017 SAFE RESUME STATUS")
    print("=" * 65)

    print("Completed:", len(completed))
    print("Pending:", len(pending))
    print("Incomplete:", len(incomplete))

    for index, name in completed:
        print(f"SKIP [{index}/20]: {name}")

    for index, name in pending:
        print(f"READY [{index}/20]: {name}")

    for index, name in incomplete:
        print(f"BLOCKED [{index}/20]: {name}")

    if incomplete:
        raise SystemExit(
            "STOP: Incomplete runs require manual inspection"
        )

    if not args.run:
        print("STATUS: SAFE RESUME DRY RUN PASS")
        return

    raise SystemExit(
        "STOP: Execution not yet enabled. "
        "Safe resume plan requires verification first."
    )

if __name__ == "__main__":
    main()
