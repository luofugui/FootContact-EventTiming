import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from footcontact_event_timing.data.underpressure_event_dataset import (
    UnderPressureEventWindowDataset,
    list_underpressure_files,
    subject_from_path,
)
from footcontact_event_timing.utils.config import load_config


def normalize_subject(subject):
    subject = str(subject)
    if subject.startswith("S"):
        return subject
    return f"S{int(subject)}"


def make_dataset(cfg, files):
    return UnderPressureEventWindowDataset(
        files=files,
        split="inspect",
        input_fps=cfg.data.input_fps,
        label_fps=cfg.data.label_fps,
        window_sec=cfg.data.window_sec,
        stride_sec=cfg.data.window_stride_sec,
        pose_key=cfg.data.pose_key,
        contact_key=cfg.data.contact_key,
        joint_dim=cfg.model.joint_dim,
        event_names=cfg.data.event_names,
        max_cached_sequences=cfg.data.max_cached_sequences,
        preload=True,
        share_memory=False,
        seed=getattr(cfg.default, "seed", 0),
        verbose=True,
    )


def select_files(cfg, subject, limit_files):
    subject = normalize_subject(subject)
    files = [p for p in list_underpressure_files(cfg.data.data_root) if subject_from_path(p) == subject]
    if not files:
        raise FileNotFoundError(f"No UnderPressure files found for {subject} under {cfg.data.data_root}")
    if limit_files is not None:
        files = files[: int(limit_files)]
    return files


def choose_indices(dataset, count, seed, mode):
    if mode == "first":
        return list(range(min(count, len(dataset))))
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return indices[: min(count, len(indices))]


def window_contact_events(contact_window):
    events = []
    for foot_idx, foot_name in enumerate(["left", "right"]):
        signal = contact_window[:, foot_idx].int()
        padded = torch.cat([torch.zeros(1, dtype=signal.dtype), signal, torch.zeros(1, dtype=signal.dtype)])
        diff = padded[1:] - padded[:-1]
        for idx in torch.where(diff == 1)[0].tolist():
            events.append((foot_name, "contact", int(idx)))
        for idx in torch.where(diff == -1)[0].tolist():
            events.append((foot_name, "departure", int(idx)))
    return sorted(events, key=lambda x: x[2])


def inspect_window(dataset, idx, cfg, out_dir):
    item = dataset.windows[idx]
    pose, contacts = dataset._get_pose_contacts(item["seq_idx"])
    start = int(item["start"])
    end = int(item["end"])
    target_time = item["target_time"]
    event_valid = item["event_valid"]
    target_frame = item["target_frame"]
    sample_positions = torch.linspace(start, end - 1, dataset.input_frames)
    sample_indices = sample_positions.round().long().clamp(0, pose.shape[0] - 1)
    contact_window = contacts[start:end]
    rel_frames = torch.arange(start, end) - start
    rel_sec = rel_frames.float() / float(cfg.data.label_fps)
    sample_rel_sec = (sample_indices.float() - start) / float(cfg.data.label_fps)

    summary = {
        "dataset_index": int(idx),
        "file": str(dataset.files[item["seq_idx"]]),
        "seq_idx": int(item["seq_idx"]),
        "window_start_frame": start,
        "window_end_frame_exclusive": end,
        "window_frames": end - start,
        "input_sample_indices": sample_indices.tolist(),
        "input_sample_offsets_sec": [float(x) for x in sample_rel_sec.tolist()],
        "event_names": dataset.event_names,
        "event_valid": [float(x) for x in event_valid.tolist()],
        "target_frame": [int(x) for x in target_frame.tolist()],
        "target_time_normalized": [float(x) for x in target_time.tolist()],
        "target_offset_sec": [
            None if int(frame) < 0 else (int(frame) - start) / float(cfg.data.label_fps)
            for frame in target_frame.tolist()
        ],
        "all_contact_edges_in_window": [
            {"foot": foot, "event": event, "frame": frame, "offset_sec": frame / float(cfg.data.label_fps)}
            for foot, event, frame in window_contact_events(contact_window)
        ],
    }

    print("\n" + "=" * 88)
    print(f"Window index {idx}")
    print(f"file: {summary['file']}")
    print(f"window: [{start}, {end}) frames={end - start}")
    print(f"sample_indices: {sample_indices.tolist()}")
    print("targets:")
    for event_idx, name in enumerate(dataset.event_names):
        valid = bool(event_valid[event_idx].item() > 0.5)
        frame = int(target_frame[event_idx].item())
        offset = None if frame < 0 else (frame - start) / float(cfg.data.label_fps)
        print(
            f"  {name:<16} valid={valid:<5} frame={frame:<5} "
            f"target_time={float(target_time[event_idx]):.3f} offset_sec={offset}"
        )
    print("all contact edges in window:")
    for edge in summary["all_contact_edges_in_window"]:
        print(f"  {edge['foot']:<5} {edge['event']:<9} frame={edge['frame']:<4} offset_sec={edge['offset_sec']:.3f}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(rel_sec.numpy(), contact_window[:, 0].numpy(), where="post", label="left contact")
    ax.step(rel_sec.numpy(), contact_window[:, 1].numpy() + 1.2, where="post", label="right contact + 1.2")
    for sec in sample_rel_sec.tolist():
        ax.axvline(sec, color="gray", alpha=0.12, linewidth=0.8)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for event_idx, name in enumerate(dataset.event_names):
        if event_valid[event_idx] <= 0.5:
            continue
        frame = int(target_frame[event_idx].item())
        offset_sec = (frame - start) / float(cfg.data.label_fps)
        ax.axvline(offset_sec, color=colors[event_idx % len(colors)], linestyle="--", linewidth=2, label=name)
    ax.set_title(f"UnderPressure event window {idx}")
    ax.set_xlabel("Seconds from window start")
    ax.set_ylabel("Contact")
    ax.set_ylim(-0.2, 2.4)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    png_path = out_dir / f"window_{idx}.png"
    json_path = out_dir / f"window_{idx}.json"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return png_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Inspect UnderPressure event-time window targets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--limit-files", type=int, default=1)
    parser.add_argument("--num-windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["random", "first"], default="random")
    parser.add_argument("--out-dir", default="inspection_underpressure_event_windows")
    args = parser.parse_args()

    cfg = load_config(args.config)
    files = select_files(cfg, args.subject, args.limit_files)
    dataset = make_dataset(cfg, files)
    indices = choose_indices(dataset, args.num_windows, args.seed, args.mode)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 88)
    print("UnderPressure event window inspection")
    print("#" * 88)
    print(f"subject={normalize_subject(args.subject)} files={len(files)} windows={len(dataset)}")
    print(f"selected_indices={indices}")
    print(f"out_dir={out_dir}")

    saved = []
    for idx in indices:
        saved.append(inspect_window(dataset, idx, cfg, out_dir))

    print("\nSaved inspection files:")
    for png_path, json_path in saved:
        print(f"  {png_path}")
        print(f"  {json_path}")


if __name__ == "__main__":
    main()
