import argparse
import csv
import json
import pickle
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


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


def normalize_subject(subject):
    subject = str(subject)
    if subject.lower() == "all":
        return "all"
    if subject.startswith("S"):
        return subject
    return f"S{int(subject)}"


def parse_subjects(value):
    value = str(value)
    if value.lower() == "all":
        return [f"S{i}" for i in range(1, 11)]
    return [normalize_subject(item.strip()) for item in value.split(",") if item.strip()]


def make_loaders(cfg, test_subjects, limit_files=None):
    train_files, val_files, test_files = split_underpressure_files(
        cfg.data.data_root,
        test_subjects,
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
        transformer=getattr(cfg.model, "transformer", "multi"),
        pos=getattr(cfg.model, "pos", "learnable"),
        mlp_dim=getattr(cfg.model, "mlp_dim", cfg.model.hidden_dim * 2),
        temporal_window=getattr(cfg.model, "temporal_window", 2),
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


def run_epoch(model, loader, loss_fn, optimizer, device, cfg, train, writer=None, global_step=0, tb_prefix=""):
    model.train(train)
    total_loss = 0.0
    total_count = 0
    errors_ms = []
    desc = "train" if train else "eval"

    for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
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
        if writer is not None and train:
            writer.add_scalar(f"{tb_prefix}/batch_loss", loss.item(), global_step + batch_idx)

    errors = np.asarray(errors_ms, dtype=np.float32)
    metrics = {
        "loss": total_loss / max(total_count, 1),
        "mae_ms": float(errors.mean()) if errors.size else 0.0,
        "median_ms": float(np.median(errors)) if errors.size else 0.0,
        "p90_ms": float(np.percentile(errors, 90)) if errors.size else 0.0,
        "event_count": int(errors.size),
        "window_count": int(total_count),
    }
    return metrics, global_step + len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg):
    metrics, _ = run_epoch(model, loader, loss_fn, None, device, cfg, train=False)
    return metrics


@torch.no_grad()
def collect_eval_output(model, loader, device):
    model.eval()
    logits_list = []
    probs_list = []
    targets_list = []
    pred_frames = []
    target_frames = []
    anchor_classes = []
    anchor_frames = []
    for batch in tqdm(loader, desc="save_eval_output", leave=False):
        batch = batch_to_device(batch, device)
        logits = model(batch["joint"])
        probs = torch.sigmoid(logits)
        logits_list.append(logits.detach().cpu())
        probs_list.append(probs.detach().cpu())
        targets_list.append(batch["event_heatmap"].detach().cpu())
        pred_frames.append(probs.argmax(dim=1).detach().cpu())
        target_frames.append(batch["event_heatmap"].argmax(dim=1).detach().cpu())
        anchor_classes.append(batch["anchor_class"].detach().cpu())
        anchor_frames.append(batch["anchor_frame"].detach().cpu())
    return {
        "predictions": {
            "event_heatmap_logits": torch.cat(logits_list).numpy(),
            "event_heatmap": torch.cat(probs_list).numpy(),
            "event_frame": torch.cat(pred_frames).numpy(),
        },
        "targets": {
            "event_heatmap": torch.cat(targets_list).numpy(),
            "event_frame": torch.cat(target_frames).numpy(),
        },
        "anchors": {
            "event_class": torch.cat(anchor_classes).numpy(),
            "event_frame": torch.cat(anchor_frames).numpy(),
        },
    }


def make_output_dirs(out_root, subject):
    dirs = {
        "checkpoint": out_root / "checkpoint" / subject,
        "eval": out_root / "eval",
        "eval_output": out_root / "eval" / "output",
        "log": out_root / "log",
        "tensorboard": out_root / "Tensorboard" / subject,
        "curves": out_root / "visualizations" / subject / "learning_curves",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_config_copy(config_path, out_root):
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_root / "config.yaml")


def write_fold_log(log_path, lines):
    with open(log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def plot_learning_curves(history, curves_dir):
    train_rows = [row for row in history if row["split"] == "train"]
    val_rows = [row for row in history if row["split"] == "val"]

    def _plot(rows, key, title, filename, ylabel):
        if not rows:
            return
        plt.figure(figsize=(8, 5))
        plt.plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(curves_dir / filename, dpi=300)
        plt.close()

    _plot(train_rows, "loss", "Train Loss", "train_losses.png", "Loss")
    _plot(val_rows, "loss", "Val Loss", "val_losses.png", "Loss")
    _plot(val_rows, "mae_ms", "Val Event Timing MAE", "val_mae_ms.png", "Milliseconds")


def train_fold(cfg, args, test_subject):
    out_root = Path(cfg.default.output_dir) / Path(args.config).stem
    dirs = make_output_dirs(out_root, test_subject)
    save_config_copy(args.config, out_root)
    log_path = dirs["log"] / f"{test_subject}.log"

    train_loader, val_loader, test_loader = make_loaders(cfg, [test_subject], args.limit_files)
    train_dataset = train_loader.dataset
    header = f"\n=== UnderPressure LOSO | test={test_subject} ==="
    size_line = (
        f"windows: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )
    class_line = f"event output classes: {train_dataset.event_names}"
    print(header, flush=True)
    print(size_line, flush=True)
    print(class_line, flush=True)
    write_fold_log(log_path, [header, size_line, class_line])

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
    best_path = dirs["checkpoint"] / "best.pth"
    final_path = dirs["checkpoint"] / "final.pth"
    history = []
    writer = SummaryWriter(log_dir=dirs["tensorboard"]) if SummaryWriter is not None else None
    global_step = 0

    for epoch in range(cfg.training.epochs):
        train_metrics, global_step = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            cfg,
            train=True,
            writer=writer,
            global_step=global_step,
            tb_prefix=f"Subject_{test_subject}/train",
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, cfg)
        history.append({"epoch": epoch, "split": "train", **train_metrics})
        history.append({"epoch": epoch, "split": "val", **val_metrics})
        if writer is not None:
            writer.add_scalar(f"Loss/Subject_{test_subject}/train", train_metrics["loss"], epoch)
            writer.add_scalar(f"Loss/Subject_{test_subject}/val", val_metrics["loss"], epoch)
            writer.add_scalar(f"Metrics/Subject_{test_subject}/val_mae_ms", val_metrics["mae_ms"], epoch)
        epoch_line = (
            f"epoch {epoch:03d} | train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_mae={val_metrics['mae_ms']:.2f} ms"
        )
        print(epoch_line, flush=True)
        write_fold_log(log_path, [epoch_line])
        if val_metrics["mae_ms"] < best_val:
            best_val = val_metrics["mae_ms"]
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "event_names": train_dataset.event_names,
                    "test_subject": test_subject,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.training.patience:
            stop_line = f"early stopping at epoch {epoch}"
            print(stop_line, flush=True)
            write_fold_log(log_path, [stop_line])
            break

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "event_names": train_dataset.event_names,
            "test_subject": test_subject,
            "epoch": history[-1]["epoch"] if history else -1,
        },
        final_path,
    )
    if writer is not None:
        writer.flush()
        writer.close()

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, loss_fn, device, cfg)
    test_line = (
        f"test {test_subject}: MAE={test_metrics['mae_ms']:.2f} ms "
        f"median={test_metrics['median_ms']:.2f} ms p90={test_metrics['p90_ms']:.2f} ms"
    )
    print(test_line, flush=True)
    write_fold_log(log_path, [test_line])

    with open(dirs["log"] / f"{test_subject}_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "split", "loss", "mae_ms", "median_ms", "p90_ms", "event_count", "window_count"],
        )
        writer.writeheader()
        writer.writerows(history)
    with open(dirs["eval"] / f"subject{test_subject}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"test_subject": test_subject, "test": test_metrics, "event_names": train_dataset.event_names},
            f,
            indent=2,
        )
    eval_output = collect_eval_output(model, test_loader, device)
    eval_output["event_names"] = train_dataset.event_names
    eval_output["test_subject"] = test_subject
    eval_output["fps"] = cfg.data.fps
    with open(dirs["eval_output"] / f"subject{test_subject}_output.pkl", "wb") as f:
        pickle.dump(eval_output, f)
    plot_learning_curves(history, dirs["curves"])
    return {"subject": test_subject, **test_metrics}


def write_loso_summary(cfg, args, rows):
    out_root = Path(cfg.default.output_dir) / Path(args.config).stem
    out_root.mkdir(parents=True, exist_ok=True)
    eval_dir = out_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "underpressure_event_heatmap_loso_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "loss", "mae_ms", "median_ms", "p90_ms", "event_count", "window_count"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "subjects": len(rows),
        "mean_mae_ms": float(np.mean([r["mae_ms"] for r in rows])) if rows else 0.0,
        "mean_median_ms": float(np.mean([r["median_ms"] for r in rows])) if rows else 0.0,
        "mean_p90_ms": float(np.mean([r["p90_ms"] for r in rows])) if rows else 0.0,
    }
    with open(eval_dir / "underpressure_event_heatmap_loso_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {csv_path}")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train UnderPressure no-pooling event heatmap detector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", default="all", help="'all', one subject, or comma list like S1,S2.")
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(getattr(cfg.default, "seed", 0))
    rows = []
    for subject in parse_subjects(args.subject):
        rows.append(train_fold(cfg, args, subject))
        write_loso_summary(cfg, args, rows)


if __name__ == "__main__":
    main()
