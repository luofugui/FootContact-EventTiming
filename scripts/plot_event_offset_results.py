import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "subject": str(row["subject"]),
                    "mae_ms": float(row["mae_ms"]),
                    "median_ms": float(row["median_ms"]),
                    "p90_ms": float(row["p90_ms"]),
                    "count": int(float(row["count"])),
                }
            )
    return rows


def save_metric_plot(rows, out_dir):
    labels = [f"S{row['subject']}" for row in rows] + ["AVG"]
    mae = [row["mae_ms"] for row in rows]
    median = [row["median_ms"] for row in rows]
    p90 = [row["p90_ms"] for row in rows]

    mae.append(float(np.mean(mae)))
    median.append(float(np.mean(median)))
    p90.append(float(np.mean(p90)))

    x = np.arange(len(labels))
    width = 0.26

    plt.figure(figsize=(13, 5))
    plt.bar(x - width, mae, width, label="MAE")
    plt.bar(x, median, width, label="Median")
    plt.bar(x + width, p90, width, label="P90")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Milliseconds")
    plt.title("Event Offset Timing Error")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "event_offset_timing_error.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def save_count_plot(rows, out_dir):
    labels = [f"S{row['subject']}" for row in rows]
    counts = [row["count"] for row in rows]

    plt.figure(figsize=(12, 4))
    plt.bar(labels, counts)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Event windows")
    plt.title("Test Event Window Count")
    plt.tight_layout()
    path = out_dir / "event_offset_test_counts.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot event-offset LOSO CSV results.")
    parser.add_argument("--csv", required=True, help="Path to event_offset_results.csv.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to CSV directory.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    metric_path = save_metric_plot(rows, out_dir)
    count_path = save_count_plot(rows, out_dir)
    print(f"Saved {metric_path}")
    print(f"Saved {count_path}")


if __name__ == "__main__":
    main()
