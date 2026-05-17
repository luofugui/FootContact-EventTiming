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

from footcontact_event_timing.data.underpressure_event_dataset import (
    UnderPressureEventWindowDataset,
    split_underpressure_files,
)
from footcontact_event_timing.models.no_pooling_footformer import (
    NoPoolingFootFormerEventDetector,
)
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


def make_dataset(cfg, files, split, verbose=True):
    return UnderPressureEventWindowDataset(
        files=files,
        split=split,
        fps=cfg.data.fps,
        half_window_sec=cfg.data.half_window_sec,
        pose_key=cfg.data.pose_key,
        contact_key=cfg.data.contact_key,
        joint_dim=cfg.model.joint_dim,
        event_sigma_frames=cfg.data.event_sigma_frames,
        contact_channel_names=cfg.data.contact_channel_names,
        max_cached_sequences=cfg.data.max_cached_sequences,
        preload=cfg.data.preload,
        share_memory=cfg.data.share_memory,
        seed=getattr(cfg.default, "seed", 0),
        verbose=verbose,
    )


def make_loaders(cfg, limit_files=None):
    train_files, val_files, test_files = split_underpressure_files(
        cfg.data.data_root,
        cfg.data.test_subjects,
        train_val_split=cfg.data.train_val_split,
        seed=getattr(cfg.default, "seed", 0),
    )
    if limit_files:
        train_files = train_files[:limit_files]
        val_files = val_files[: max(1, limit_files // 10)]
        test_files = test_files[: max(1, limit_files // 10)]

    train_dataset = make_dataset(cfg, train_files, "train")
    val_dataset = make_dataset(cfg, val_files, "val")
    test_dataset = make_dataset(cfg, test_files, "test")

    kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.training.dataloader_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if cfg.training.dataloader_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = getattr(cfg.training, "prefetch_factor", 4)
    return (
        DataLoader(train_dataset, shuffle=True, drop_last=False, **kwargs),
        DataLoader(val_dataset, shuffle=False, drop_last=False, **kwargs),
        DataLoader(test_dataset, shuffle=False, drop_last=False, **kwargs),
    )


def make_model(cfg, num_event_classes, window_frames):
    return NoPoolingFootFormerEventDetector(
        num_joints=cfg.model.num_joints,
        joint_dim=cfg.model.joint_dim,
        window_frames=window_frames,
        num_event_classes=num_event_classes,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        pose_embedder=getattr(cfg.model, "pose_embedder", "gcn"),
    )


def batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def event_timing_metrics(logits, targets, fps):
    probs = torch.sigmoid(logits)
    pred_idx = probs.argmax(dim=1)
    gt_idx = targets.argmax(dim=1)
    has_event = targets.amax(dim=1) > 0.5
    if not has_event.any():
        return []
    errors = (pred_idx[has_event] - gt_idx[has_event]).abs().float()
    return (errors * (1000.0 / fps)).detach().cpu().numpy().tolist()


def run_epoch(model, loader, loss_fn, optimizer, device, cfg, train):
    model.train(train)
    total_loss = 0.0
    total_count = 0
    errors_ms = []
    desc = "train" if train else "eval"

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(batch["joint"])
        loss = loss_fn(logits, batch["event_heatmap"])

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * logits.shape[0]
        total_count += logits.shape[0]
        errors_ms.extend(event_timing_metrics(logits, batch["event_heatmap"], cfg.data.fps))

    errors = np.asarray(errors_ms, dtype=np.float32)
    return {
        "loss": total_loss / max(total_count, 1),
        "mae_ms": float(errors.mean()) if errors.size else 0.0,
        "median_ms": float(np.median(errors)) if errors.size else 0.0,
        "p90_ms": float(np.percentile(errors, 90)) if errors.size else 0.0,
        "event_count": int(errors.size),
        "window_count": int(total_count),
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg):
    return run_epoch(model, loader, loss_fn, None, device, cfg, train=False)


def train(cfg, args):
    set_seed(getattr(cfg.default, "seed", 0))
    out_dir = Path(cfg.default.output_dir) / Path(args.config).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(cfg, args.limit_files)
    train_dataset = train_loader.dataset
    print(
        f"windows: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}",
        flush=True,
    )
    print(f"event output classes: {train_dataset.event_names}", flush=True)

    device = resolve_device(cfg)
    model = make_model(cfg, train_dataset.num_event_classes, train_dataset.window_frames).to(device)
    pos_weight = torch.full(
        (train_dataset.num_event_classes,),
        float(getattr(cfg.training, "pos_weight", 1.0)),
        device=device,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    best_val = float("inf")
    stale_epochs = 0
    best_path = out_dir / "underpressure_event_heatmap_best.pt"
    history = []

    for epoch in range(cfg.training.epochs):
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, cfg, train=True)
        val_metrics = evaluate(model, val_loader, loss_fn, device, cfg)
        history.append({"epoch": epoch, "split": "train", **train_metrics})
        history.append({"epoch": epoch, "split": "val", **val_metrics})
        print(
            f"epoch {epoch:03d} | train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_mae={val_metrics['mae_ms']:.2f} ms",
            flush=True,
        )
        if val_metrics["mae_ms"] < best_val:
            best_val = val_metrics["mae_ms"]
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "event_names": train_dataset.event_names,
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
        f"test: MAE={test_metrics['mae_ms']:.2f} ms "
        f"median={test_metrics['median_ms']:.2f} ms p90={test_metrics['p90_ms']:.2f} ms",
        flush=True,
    )

    with open(out_dir / "underpressure_event_heatmap_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "split", "loss", "mae_ms", "median_ms", "p90_ms", "event_count", "window_count"],
        )
        writer.writeheader()
        writer.writerows(history)
    with open(out_dir / "underpressure_event_heatmap_results.json", "w", encoding="utf-8") as f:
        json.dump({"test": test_metrics, "event_names": train_dataset.event_names}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train UnderPressure no-pooling event heatmap detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args)


if __name__ == "__main__":
    main()
