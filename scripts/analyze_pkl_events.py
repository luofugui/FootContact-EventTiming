import argparse
import csv
import glob
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from footcontact_event_timing.data.contact_events import get_event_indices


def load_contact_pair(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    gt = np.asarray(data["targets"]["contact"])
    pred = np.asarray(data["predictions"]["contact"])
    return gt, pred


def binarize_pred(pred, threshold):
    return (np.asarray(pred) > threshold).astype(int)


def nearest_event_errors(gt_events, pred_events, max_dist_frames):
    gt_events = list(gt_events)
    pred_events = list(pred_events)
    unused_pred = set(range(len(pred_events)))
    errors = []
    missed = 0

    for gt in gt_events:
        candidates = [
            (abs(pred_events[i] - gt), i)
            for i in unused_pred
            if abs(pred_events[i] - gt) <= max_dist_frames
        ]
        if not candidates:
            missed += 1
            continue
        _, pred_idx = min(candidates)
        errors.append(pred_events[pred_idx] - gt)
        unused_pred.remove(pred_idx)

    return np.asarray(errors), missed, len(unused_pred)


def collect_event_windows(gt_binary, pred_binary, event_kind, event_frames, half_window):
    gt_windows = []
    pred_windows = []
    for frame in event_frames:
        start = frame - half_window
        end = frame + half_window + 1
        if start < 0 or end > len(gt_binary):
            continue
        gt_windows.append(gt_binary[start:end])
        pred_windows.append(pred_binary[start:end])
    return gt_windows, pred_windows


def evaluate_file(path, fps, threshold, half_window_sec):
    gt, pred_raw = load_contact_pair(path)
    pred = binarize_pred(pred_raw, threshold)
    gt_binary = (gt >= 0.5).astype(int)

    half_window = int(round(half_window_sec * fps))
    max_dist_frames = half_window
    file_rows = []
    windows = {"onset": {"gt": [], "pred": []}, "departure": {"gt": [], "pred": []}}

    for channel in range(gt_binary.shape[1]):
        gt_onsets, gt_departures = get_event_indices(gt_binary[:, channel])
        pred_onsets, pred_departures = get_event_indices(pred[:, channel])

        for kind, gt_events, pred_events in [
            ("onset", gt_onsets, pred_onsets),
            ("departure", gt_departures, pred_departures),
        ]:
            errors, missed, false = nearest_event_errors(gt_events, pred_events, max_dist_frames)
            mae_ms = float(np.mean(np.abs(errors)) * 1000.0 / fps) if errors.size else 0.0
            median_ms = float(np.median(np.abs(errors)) * 1000.0 / fps) if errors.size else 0.0
            file_rows.append(
                {
                    "file": os.path.basename(path),
                    "channel": channel,
                    "event": kind,
                    "mae_ms": mae_ms,
                    "median_ms": median_ms,
                    "matched": int(errors.size),
                    "missed": int(missed),
                    "false": int(false),
                }
            )

            gt_win, pred_win = collect_event_windows(
                gt_binary[:, channel],
                pred[:, channel],
                kind,
                gt_events,
                half_window,
            )
            windows[kind]["gt"].extend(gt_win)
            windows[kind]["pred"].extend(pred_win)

    return file_rows, windows


def plot_windows(all_windows, fps, out_prefix):
    for kind, series in all_windows.items():
        if not series["gt"]:
            continue
        gt_mean = np.mean(np.stack(series["gt"]), axis=0)
        pred_mean = np.mean(np.stack(series["pred"]), axis=0)
        half = len(gt_mean) // 2
        xs = (np.arange(len(gt_mean)) - half) / fps

        plt.figure(figsize=(8, 4))
        plt.plot(xs, gt_mean, label="GT contact probability")
        plt.plot(xs, pred_mean, label="Pred contact probability")
        plt.axvline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Time around GT event (s)")
        plt.ylabel("Contact rate")
        plt.title(f"{kind.title()} event window")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{kind}_event_window.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze event timing from output pkl files.")
    parser.add_argument("--dir", required=True, help="Directory containing subject*_output.pkl files.")
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--half-window-sec", type=float, default=0.5)
    parser.add_argument("--out-prefix", default="event_timing")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "subject*_output.pkl")))
    if not files:
        raise FileNotFoundError(f"No subject*_output.pkl files found in {args.dir}")

    rows = []
    all_windows = {
        "onset": {"gt": [], "pred": []},
        "departure": {"gt": [], "pred": []},
    }
    for path in files:
        file_rows, windows = evaluate_file(path, args.fps, args.threshold, args.half_window_sec)
        rows.extend(file_rows)
        for kind in all_windows:
            all_windows[kind]["gt"].extend(windows[kind]["gt"])
            all_windows[kind]["pred"].extend(windows[kind]["pred"])

    csv_path = f"{args.out_prefix}_event_timing_results_{args.fps}fps.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "channel", "event", "mae_ms", "median_ms", "matched", "missed", "false"],
        )
        writer.writeheader()
        writer.writerows(rows)

    for event in ("onset", "departure"):
        event_rows = [row for row in rows if row["event"] == event]
        weighted_error = np.average(
            [row["mae_ms"] for row in event_rows],
            weights=[max(row["matched"], 1) for row in event_rows],
        )
        print(f"{event}: weighted MAE={weighted_error:.2f} ms")

    plot_windows(all_windows, args.fps, args.out_prefix)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
