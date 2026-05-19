import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from footcontact_event_timing.data.contact_events import get_event_indices
from footcontact_event_timing.data.underpressure_event_dataset import list_underpressure_files, subject_from_path


EVENT_NAMES = ["left_contact", "left_departure", "right_contact", "right_departure"]


def extract_foot_contacts(raw, contact_key="contacts"):
    for key in [contact_key, "contacts", "contact"]:
        if key in raw:
            contacts = torch.as_tensor(raw[key])
            break
    else:
        raise KeyError(f"Could not find contact key in {raw.keys()}")

    contacts = torch.nan_to_num(contacts.float()) >= 0.5
    if contacts.ndim == 3:
        if contacts.shape[1] < 2:
            raise ValueError(f"Expected contacts with at least two feet, got {tuple(contacts.shape)}")
        return contacts[:, :2, :].any(dim=2).cpu().numpy().astype(bool)
    if contacts.ndim == 2:
        if contacts.shape[1] == 2:
            return contacts.cpu().numpy().astype(bool)
        if contacts.shape[1] >= 4:
            left = contacts[:, :2].any(dim=1)
            right = contacts[:, 2:4].any(dim=1)
            return torch.stack([left, right], dim=1).cpu().numpy().astype(bool)
    raise ValueError(f"Expected contacts [T, 2, R], [T, 2], or [T, >=4], got {tuple(contacts.shape)}")


def event_times_from_contacts(contacts, fps):
    events = {name: [] for name in EVENT_NAMES}
    for foot_idx, foot_name in enumerate(["left", "right"]):
        onsets, departures = get_event_indices(contacts[:, foot_idx])
        events[f"{foot_name}_contact"] = (np.asarray(onsets, dtype=np.float64) / fps).tolist()
        events[f"{foot_name}_departure"] = (np.asarray(departures, dtype=np.float64) / fps).tolist()
    return events


def resample_contacts_nearest(contacts, src_fps, dst_fps):
    if dst_fps <= 0:
        raise ValueError("dst_fps must be positive.")
    duration = len(contacts) / float(src_fps)
    sample_times = np.arange(0.0, duration, 1.0 / float(dst_fps), dtype=np.float64)
    src_indices = np.rint(sample_times * float(src_fps)).astype(np.int64)
    src_indices = np.clip(src_indices, 0, len(contacts) - 1)
    return contacts[src_indices]


def match_event_times(gt_times, pred_times, max_match_sec):
    gt_times = list(float(t) for t in gt_times)
    pred_times = list(float(t) for t in pred_times)
    if not gt_times or not pred_times:
        return [], len(gt_times), len(pred_times)

    pairs = []
    for gt_idx, gt_t in enumerate(gt_times):
        for pred_idx, pred_t in enumerate(pred_times):
            err = abs(pred_t - gt_t)
            if err <= max_match_sec:
                pairs.append((err, gt_idx, pred_idx))
    pairs.sort(key=lambda x: x[0])

    used_gt = set()
    used_pred = set()
    errors = []
    for err, gt_idx, pred_idx in pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        errors.append(err)

    return errors, len(gt_times) - len(used_gt), len(pred_times) - len(used_pred)


def summarize_errors(errors_sec):
    if not errors_sec:
        return {"mae_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0}
    errors_ms = np.asarray(errors_sec, dtype=np.float64) * 1000.0
    return {
        "mae_ms": float(np.mean(errors_ms)),
        "median_ms": float(np.median(errors_ms)),
        "p90_ms": float(np.percentile(errors_ms, 90)),
    }


def evaluate_fps(files, target_fps, label_fps, contact_key, max_match_sec):
    all_errors = []
    per_event = {
        name: {"errors": [], "missed": 0, "false": 0, "gt": 0, "pred": 0}
        for name in EVENT_NAMES
    }

    for path in files:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        contacts = extract_foot_contacts(raw, contact_key=contact_key)
        gt_events = event_times_from_contacts(contacts, label_fps)
        sampled = resample_contacts_nearest(contacts, label_fps, target_fps)
        pred_events = event_times_from_contacts(sampled, target_fps)

        for name in EVENT_NAMES:
            errors, missed, false = match_event_times(
                gt_events[name],
                pred_events[name],
                max_match_sec=max_match_sec,
            )
            all_errors.extend(errors)
            per_event[name]["errors"].extend(errors)
            per_event[name]["missed"] += missed
            per_event[name]["false"] += false
            per_event[name]["gt"] += len(gt_events[name])
            per_event[name]["pred"] += len(pred_events[name])

    summary = summarize_errors(all_errors)
    summary.update(
        {
            "fps": target_fps,
            "files": len(files),
            "matched": len(all_errors),
            "missed": sum(v["missed"] for v in per_event.values()),
            "false": sum(v["false"] for v in per_event.values()),
            "gt_events": sum(v["gt"] for v in per_event.values()),
            "pred_events": sum(v["pred"] for v in per_event.values()),
        }
    )

    event_rows = []
    for name, values in per_event.items():
        row = summarize_errors(values["errors"])
        row.update(
            {
                "fps": target_fps,
                "event": name,
                "matched": len(values["errors"]),
                "missed": values["missed"],
                "false": values["false"],
                "gt_events": values["gt"],
                "pred_events": values["pred"],
            }
        )
        event_rows.append(row)

    return summary, event_rows


def filter_files(files, subjects, limit_files):
    if subjects:
        keep = set(subjects)
        files = [p for p in files if subject_from_path(p) in keep]
    if limit_files is not None:
        files = files[: int(limit_files)]
    return files


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Oracle event-timing sanity check from downsampled UnderPressure GT contacts."
    )
    parser.add_argument("--data-root", required=True, help="UnderPressure dataset root containing S*/preprocessed/*.pth")
    parser.add_argument("--fps-list", nargs="+", type=float, default=[30, 60, 100])
    parser.add_argument("--label-fps", type=float, default=100.0)
    parser.add_argument("--contact-key", default="contacts")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subjects, e.g. S1 S2")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--max-match-sec", type=float, default=0.5)
    parser.add_argument("--out-prefix", default="underpressure_oracle_event_timing")
    args = parser.parse_args()

    files = filter_files(
        list_underpressure_files(args.data_root),
        subjects=args.subjects,
        limit_files=args.limit_files,
    )
    if not files:
        raise FileNotFoundError(f"No UnderPressure .pth files found under {args.data_root}")

    print("\n" + "=" * 78)
    print("UnderPressure GT-contact FPS oracle")
    print("=" * 78)
    print(f"files: {len(files)}")
    print(f"label_fps: {args.label_fps:g}")
    print(f"fps_list: {', '.join(str(f) for f in args.fps_list)}")
    print(f"max_match_sec: {args.max_match_sec:g}")

    summary_rows = []
    event_rows = []
    for fps in args.fps_list:
        summary, rows = evaluate_fps(
            files=files,
            target_fps=float(fps),
            label_fps=float(args.label_fps),
            contact_key=args.contact_key,
            max_match_sec=float(args.max_match_sec),
        )
        summary_rows.append(summary)
        event_rows.extend(rows)
        print("\n" + "-" * 78)
        print(
            f"fps={fps:g} | MAE={summary['mae_ms']:.2f} ms | "
            f"median={summary['median_ms']:.2f} ms | p90={summary['p90_ms']:.2f} ms"
        )
        print(
            f"matched={summary['matched']} missed={summary['missed']} false={summary['false']} "
            f"gt={summary['gt_events']} pred={summary['pred_events']}"
        )
        for row in rows:
            print(
                f"  {row['event']:<15} MAE={row['mae_ms']:7.2f} ms "
                f"matched={row['matched']:6d} missed={row['missed']:5d} false={row['false']:5d}"
            )

    summary_csv = f"{args.out_prefix}_summary.csv"
    event_csv = f"{args.out_prefix}_by_event.csv"
    write_csv(
        summary_csv,
        summary_rows,
        ["fps", "files", "mae_ms", "median_ms", "p90_ms", "matched", "missed", "false", "gt_events", "pred_events"],
    )
    write_csv(
        event_csv,
        event_rows,
        ["fps", "event", "mae_ms", "median_ms", "p90_ms", "matched", "missed", "false", "gt_events", "pred_events"],
    )
    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {event_csv}")


if __name__ == "__main__":
    main()
