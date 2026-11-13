from pathlib import Path
import random
import math
import csv
import json
import hashlib

ROOT = Path.cwd()
SOURCE = ROOT / "datasets/coco2017_expansion"
OUTPUT = SOURCE / "noise_experiments"

LEVELS = [0, 5, 10, 20]
SEED = 42
SHIFT = 0.10

def main():
    print("STEP 32.17 - COCO2017 CONTROLLED NOISE GENERATION")
    print("=" * 65)

    source_labels = SOURCE / "labels/train"
    files = sorted(source_labels.glob("*.txt"))

    if len(files) != 1000:
        raise SystemExit("STOP: Expected 1000 training label files.")

    if OUTPUT.exists():
        raise SystemExit(
            "STOP: Noise output directory already exists. "
            "No files overwritten."
        )

    annotations = []
    original = {}

    for file in files:
        lines = file.read_text().splitlines()
        original[file.name] = lines

        for index, line in enumerate(lines):
            if line.strip():
                annotations.append((file.name, index))

    total = len(annotations)

    if total != 7249:
        raise SystemExit(
            f"STOP: Expected 7249 annotations, found {total}."
        )

    rng = random.Random(SEED)
    shuffled = annotations.copy()
    rng.shuffle(shuffled)

    report = []
    modification_records = []

    # Prepare all data in memory before writing anything.
    generated = {}

    for level in LEVELS:
        count = round(total * level / 100)
        selected = set(shuffled[:count])

        modified = 0
        generated[level] = {}

        for file in files:
            lines = original[file.name]
            new_lines = []

            for index, line in enumerate(lines):
                key = (file.name, index)

                if key not in selected:
                    new_lines.append(line)
                    continue

                values = line.split()

                if len(values) != 5:
                    raise SystemExit(
                        f"STOP: Invalid label in {file.name}"
                    )

                cls = values[0]
                x, y, w, h = map(float, values[1:])

                dx = SHIFT * w * rng.choice([-1, 1])
                dy = SHIFT * h * rng.choice([-1, 1])

                nx = min(max(x + dx, w / 2), 1 - w / 2)
                ny = min(max(y + dy, h / 2), 1 - h / 2)

                if math.isclose(nx, x) and math.isclose(ny, y):
                    new_lines.append(line)
                    continue

                updated = (
                    f"{cls} {nx:.8f} {ny:.8f} "
                    f"{w:.8f} {h:.8f}"
                )

                new_lines.append(updated)
                modified += 1

                modification_records.append({
                    "noise_level": level,
                    "file": file.name,
                    "line_index": index,
                    "original": line,
                    "modified": updated
                })

            generated[level][file.name] = new_lines

        actual = modified / total * 100

        report.append({
            "target_noise": level,
            "selected": count,
            "modified": modified,
            "actual_noise_percent": round(actual, 6)
        })

        print(
            f"Noise {level}%: selected={count}, "
            f"modified={modified}/{total} "
            f"({actual:.4f}%)"
        )

    # Save new datasets without modifying source labels.
    OUTPUT.mkdir(parents=True, exist_ok=False)

    for level in LEVELS:
        destination = OUTPUT / f"noise_{level}" / "labels/train"
        destination.mkdir(parents=True, exist_ok=False)

        for filename, lines in generated[level].items():
            content = "\n".join(lines)
            if lines:
                content += "\n"

            (destination / filename).write_text(
                content,
                encoding="utf-8"
            )

    report_path = OUTPUT / "noise_report.csv"

    with report_path.open("x", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_noise",
                "selected",
                "modified",
                "actual_noise_percent"
            ]
        )
        writer.writeheader()
        writer.writerows(report)

    audit_path = OUTPUT / "modification_audit.json"

    audit_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "shift": SHIFT,
                "total_annotations": total,
                "selection_method": "nested_random_shuffle",
                "modifications": modification_records
            },
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nFINAL SUMMARY")
    print("-" * 65)
    print("Training images:", len(files))
    print("Training annotations:", total)
    print("Noise levels:", LEVELS)
    print("Noise generation seed:", SEED)
    print("Shift magnitude:", SHIFT)
    print("Output:", OUTPUT)
    print("Source training labels modified: NONE")
    print("Validation/test labels modified: NONE")
    print("Original research files modified: NONE")
    print("STATUS: NOISE GENERATION COMPLETE")

if __name__ == "__main__":
    main()
