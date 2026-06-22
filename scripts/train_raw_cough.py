from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.models import RawWaveformCoughCNN
from cough_analysis.paths import project_path
from cough_analysis.raw_baselines import (
    RawWaveformDataset,
    augment_raw_batch,
    build_raw_dataset,
)
from cough_analysis.v3 import resolve_device, split_records_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_or_absolute(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def window_settings(cfg: dict) -> dict:
    win_cfg = cfg["windowing"]
    return {
        "window_sec": float(win_cfg["window_sec"]),
        "hop_sec": float(win_cfg["hop_sec"]),
        "label_rule": str(win_cfg["label_rule"]),
        "center_fraction": float(win_cfg.get("center_fraction", 0.2)),
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    labels_all = []
    preds_all = []
    loss_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            motion = batch["motion"].to(device)
            labels = batch["label"].to(device)
            logits = model(audio, motion)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * audio.size(0)
            preds = (torch.sigmoid(logits) >= threshold).int().cpu().numpy()
            preds_all.extend(preds.tolist())
            labels_all.extend(labels.int().cpu().numpy().tolist())

    labels_np = np.asarray(labels_all)
    preds_np = np.asarray(preds_all)
    return {
        "loss": loss_sum / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.dry_run:
        model = RawWaveformCoughCNN().to(device)
        window_sec = float(cfg["windowing"]["window_sec"])
        dummy_audio = torch.randn(2, 2, int(round(window_sec * 4800))).to(device)
        dummy_motion = torch.randn(2, 2, int(round(window_sec * 100))).to(device)
        with torch.no_grad():
            out = model(dummy_audio, dummy_motion)
        print(f"Dry run output shape: {tuple(out.shape)}")
        return 0

    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    if args.max_records is not None:
        metadata = metadata.head(args.max_records).copy()
    data_root = project_or_absolute(cfg["data"].get("data_root"))
    train_ids, val_ids, test_ids = split_records_from_config(
        metadata,
        cfg.get("split"),
        random_state=args.seed,
    )
    print(
        f"Record split -> Train: {len(train_ids)} | "
        f"Val: {len(val_ids)} | Test: {len(test_ids)}"
    )

    settings = window_settings(cfg)
    X_a_train, X_m_train, y_train = build_raw_dataset(
        train_ids,
        metadata,
        data_root=data_root,
        **settings,
    )
    X_a_val, X_m_val, y_val = build_raw_dataset(
        val_ids,
        metadata,
        data_root=data_root,
        **settings,
    )
    print(
        f"Train: Audio {X_a_train.shape} | Motion {X_m_train.shape} | "
        f"Positive: {int(np.sum(y_train))} ({np.mean(y_train) * 100:.1f}%)"
    )
    print(
        f"Val:   Audio {X_a_val.shape} | Motion {X_m_val.shape} | "
        f"Positive: {int(np.sum(y_val))} ({np.mean(y_val) * 100:.1f}%)"
    )

    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    epochs = args.epochs if args.epochs is not None else int(cfg["training"]["epochs"])
    train_loader = DataLoader(
        RawWaveformDataset(X_a_train, X_m_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        RawWaveformDataset(X_a_val, X_m_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    model = RawWaveformCoughCNN().to(device)
    num_pos = int(np.sum(y_train))
    num_neg = int(len(y_train) - num_pos)
    if num_pos == 0:
        raise ValueError("No positive cough windows found in the training split.")
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    use_scheduler = str(cfg["training"].get("scheduler", "")).lower() == "reduce_lr_on_plateau"
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    aug_cfg = cfg.get("augmentation", {})
    use_awgn = bool(aug_cfg.get("awgn", False))
    audio_noise_std = float(aug_cfg.get("audio_noise_std", 0.02 if use_awgn else 0.0))
    motion_noise_std = float(aug_cfg.get("motion_noise_std", 0.01 if use_awgn else 0.0))

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            audio = batch["audio"].to(device)
            motion = batch["motion"].to(device)
            labels = batch["label"].to(device)
            if use_awgn:
                audio, motion = augment_raw_batch(
                    audio,
                    motion,
                    audio_noise_std=audio_noise_std,
                    motion_noise_std=motion_noise_std,
                )

            optimizer.zero_grad()
            logits = model(audio, motion)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * audio.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        val_metrics = evaluate_loader(model, val_loader, criterion, device=device)
        if use_scheduler:
            scheduler.step(val_metrics["loss"])
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch + 1,
            "learning_rate": lr,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | LR: {lr:.5f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.1f}%"
        )

    output_path = project_or_absolute(args.output)
    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": "RawWaveformCoughCNN",
        "model_state_dict": best_state or model.state_dict(),
        "config": cfg,
        "record_split": {
            "train": [int(x) for x in train_ids],
            "val": [int(x) for x in val_ids],
            "test": [int(x) for x in test_ids],
        },
        "history": history,
        "pos_weight": float(pos_weight.item()),
        "seed": args.seed,
    }
    torch.save(checkpoint, output_path)
    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {output_path}")
    print(f"Saved history: {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
