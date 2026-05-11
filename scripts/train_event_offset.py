import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from footcontact_event_timing.data.psu_event_dataset import (
    PSUEventOffsetDataset,
    split_chunk_paths,
)
from footcontact_event_timing.models.pose_event_regressor import PoseEventRegressor
from footcontact_event_timing.utils.config import load_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg):
    requested = getattr(cfg.default, "device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def make_dataset(cfg, split, files, verbose=True):
    return PSUEventOffsetDataset(
        chunk_files=files,
        foot_mask_path=cfg.data.foot_mask_path,
        split=split,
        window_frames=cfg.data.window_frames,
        centered_window=getattr(cfg.data, "centered_window", False),
        samples_per_event=cfg.data.samples_per_event,
        eval_offsets_per_event=getattr(cfg.data, "eval_offsets_per_event", 5),
        min_event_offset=cfg.data.min_event_offset,
        max_event_offset=cfg.data.max_event_offset,
        num_regions=cfg.data.num_regions,
        contact_threshold=cfg.data.contact_threshold,
        active_only=cfg.data.active_only,
        max_cached_chunks=cfg.data.max_cached_chunks,
        preload_joints=getattr(cfg.data, "preload_joints", True),
        share_memory=getattr(cfg.data, "share_memory", True),
        seed=getattr(cfg.default, "seed", 0),
        verbose=verbose,
    )


def make_loaders(cfg, subject, limit_files=None):
    train_files, val_files, test_files = split_chunk_paths(
        cfg.data.chunk_dir,
        subject,
        train_val_split=cfg.data.train_val_split,
        seed=getattr(cfg.default, "seed", 0),
    )
    if limit_files:
        train_files = train_files[:limit_files]
        val_files = val_files[: max(1, limit_files // 10)]
        test_files = test_files[: max(1, limit_files // 10)]

    train_dataset = make_dataset(cfg, "train", train_files)
    val_dataset = make_dataset(cfg, "val", val_files)
    test_dataset = make_dataset(cfg, "test", test_files)

    kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.training.dataloader_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if cfg.training.dataloader_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = getattr(cfg.training, "prefetch_factor", 4)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader, test_loader


def save_dataset_event_window_plots(dataset, cfg, out_dir, subject, split):
    half_window_sec = getattr(cfg.data, "plot_half_window_sec", 0.5)
    half_window_frames = int(round(half_window_sec * cfg.data.fps))
    windows = dataset.collect_centered_event_windows(
        half_window_frames=half_window_frames,
        max_events_per_kind=getattr(cfg.data, "plot_max_events_per_kind", 2000),
    )

    xs = np.arange(-half_window_frames, half_window_frames + 1) / cfg.data.fps
    saved = []
    for kind, values in windows.items():
        if values.size == 0:
            continue
        mean_signal = values.mean(axis=0)
        plt.figure(figsize=(8, 4))
        plt.plot(xs, mean_signal, label="GT contact rate")
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1, label="event time")
        plt.xlabel("Time around event (s)")
        plt.ylabel("Contact rate")
        plt.title(f"Subject {subject} {split} {kind} window (+/- {half_window_sec:.1f}s)")
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"subject{subject}_{split}_{kind}_centered_window.png"
        plt.savefig(path, dpi=300)
        plt.close()
        saved.append(path)
    for path in saved:
        print(f"Saved event window plot: {path}", flush=True)


def make_model(cfg):
    return PoseEventRegressor(
        num_joints=cfg.model.num_joints,
        joint_dim=cfg.model.joint_dim,
        window_frames=cfg.data.window_frames,
        num_channels=cfg.model.num_channels,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    )


def batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch["joint"], batch["event_type"], batch["channel"])
        loss = loss_fn(pred, batch["offset"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pred.shape[0]
        total_count += pred.shape[0]
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg):
    model.eval()
    total_loss = 0.0
    total_count = 0
    errors_ms = []
    scale_ms = (cfg.data.window_frames - 1) * 1000.0 / cfg.data.fps

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = batch_to_device(batch, device)
        pred = model(batch["joint"], batch["event_type"], batch["channel"])
        loss = loss_fn(pred, batch["offset"])
        error = (pred - batch["offset"]).abs() * scale_ms

        total_loss += loss.item() * pred.shape[0]
        total_count += pred.shape[0]
        errors_ms.append(error.detach().cpu().numpy())

    if errors_ms:
        errors_ms = np.concatenate(errors_ms)
    else:
        errors_ms = np.array([], dtype=np.float32)

    return {
        "loss": total_loss / max(total_count, 1),
        "mae_ms": float(errors_ms.mean()) if errors_ms.size else 0.0,
        "median_ms": float(np.median(errors_ms)) if errors_ms.size else 0.0,
        "p90_ms": float(np.percentile(errors_ms, 90)) if errors_ms.size else 0.0,
        "count": int(total_count),
    }


def train_subject(cfg, subject, out_dir, limit_files=None):
    device = resolve_device(cfg)
    print(f"\n=== Subject {subject} LOSO | device={device} ===", flush=True)
    train_loader, val_loader, test_loader = make_loaders(cfg, subject, limit_files)
    print(
        f"windows: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}",
        flush=True,
    )
    if getattr(cfg.data, "plot_event_windows", True):
        save_dataset_event_window_plots(train_loader.dataset, cfg, out_dir, subject, "train")

    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_path = out_dir / f"subject{subject}_best.pt"
    stale_epochs = 0

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, cfg)
        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_mae={val_metrics['mae_ms']:.2f} ms",
            flush=True,
        )

        if val_metrics["mae_ms"] < best_val:
            best_val = val_metrics["mae_ms"]
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "subject": subject,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= cfg.training.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, loss_fn, device, cfg)
    print(
        f"subject {subject} test: MAE={test_metrics['mae_ms']:.2f} ms "
        f"median={test_metrics['median_ms']:.2f} ms p90={test_metrics['p90_ms']:.2f} ms",
        flush=True,
    )
    return test_metrics


def write_results(out_dir, rows):
    csv_path = out_dir / "event_offset_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "loss", "mae_ms", "median_ms", "p90_ms", "count"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "subjects": len(rows),
        "mean_mae_ms": float(np.mean([r["mae_ms"] for r in rows])) if rows else 0.0,
        "mean_median_ms": float(np.mean([r["median_ms"] for r in rows])) if rows else 0.0,
        "mean_p90_ms": float(np.mean([r["p90_ms"] for r in rows])) if rows else 0.0,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {csv_path}")
    print(json.dumps(summary, indent=2))


def parse_subjects(value):
    if value == "all":
        return list(range(1, 11))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train pose-window event offset regression.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", default="all", help="'all', one subject, or comma list.")
    parser.add_argument("--limit-files", type=int, default=None, help="Debug with fewer chunk files.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(getattr(cfg.default, "seed", 0))

    out_dir = Path(cfg.default.output_dir) / Path(args.config).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subject in parse_subjects(args.subject):
        metrics = train_subject(cfg, subject, out_dir, limit_files=args.limit_files)
        rows.append({"subject": subject, **metrics})
        write_results(out_dir, rows)


if __name__ == "__main__":
    main()
