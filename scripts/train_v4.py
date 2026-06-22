from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.models import V4ActivityCNN, V4CoughFrameCNN
from cough_analysis.paths import project_path
from cough_analysis.v4 import (
    V4ActivityWindowDataset,
    V4CoughChunkDataset,
    activity_class_map,
    cough_frame_count,
    cough_gt_events,
    event_summary,
    frame_predictions_to_events,
    predict_cough_probabilities_for_record,
    resize_frame_logits,
    resolve_device,
    split_records_v4,
)
from cough_analysis.preprocessing import load_record_preprocessed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V4 cough-event and activity models.")
    parser.add_argument("--config", default="configs/v4.yaml")
    parser.add_argument("--output-dir", default="artifacts/models/v4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_path(path)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def loader_frame_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    frame_count: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"].to(device)
            motion = batch["motion"].to(device)
            labels = batch["label"].to(device)
            logits = resize_frame_logits(model(spec, motion), frame_count)
            loss = criterion(logits, labels)
            total_loss += loss.item() * spec.size(0)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_preds.extend((probs >= 0.5).astype(int).tolist())
            all_labels.extend(labels.cpu().numpy().ravel().astype(int).tolist())

    labels_np = np.asarray(all_labels)
    preds_np = np.asarray(all_preds)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
    }


def train_cough_model(
    cfg: dict,
    spec_name: str,
    spec_cfg: dict,
    metadata,
    split,
    output_dir: Path,
    device: torch.device,
) -> dict:
    train_cfg = cfg["training"]
    cough_cfg = cfg["cough"]
    data_root = project_or_absolute(cfg["data"]["data_root"])
    frame_count = cough_frame_count(cough_cfg)

    train_ds = V4CoughChunkDataset(split.train, metadata, cough_cfg, spec_cfg, data_root=data_root)
    val_ds = V4CoughChunkDataset(split.val, metadata, cough_cfg, spec_cfg, data_root=data_root)
    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = V4CoughFrameCNN().to(device)
    pos = float(train_ds.labels.sum().item())
    neg = float(train_ds.labels.numel() - pos)
    pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg["scheduler_factor"]),
        patience=int(train_cfg["scheduler_patience"]),
    )

    best_state = None
    best_loss = float("inf")
    patience_left = int(train_cfg["early_stopping_patience"])
    history = []

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            spec = batch["spec"].to(device)
            motion = batch["motion"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = resize_frame_logits(model(spec, motion), frame_count)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * spec.size(0)

        train_loss /= max(len(train_loader.dataset), 1)
        val_metrics = loader_frame_metrics(model, val_loader, criterion, frame_count, device)
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[{spec_name}] epoch {epoch + 1:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1']:.3f}"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(train_cfg["early_stopping_patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    sweep = select_cough_postprocessing(
        cfg=cfg,
        model=model,
        spec_cfg=spec_cfg,
        metadata=metadata,
        record_ids=split.val,
        device=device,
        batch_size=batch_size,
    )

    checkpoint = {
        "model_id": f"v4_cough_{spec_name}",
        "model_name": "V4CoughFrameCNN",
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "spec_name": spec_name,
        "spec_config": spec_cfg,
        "record_split": split.as_dict(),
        "history": history,
        "pos_weight": float(pos_weight.item()),
        "selected_postprocessing": sweep["selected"],
        "validation_sweep": sweep["candidates"],
    }
    checkpoint_path = output_dir / f"v4_cough_{spec_name}.pt"
    torch.save(checkpoint, checkpoint_path)
    (output_dir / f"v4_cough_{spec_name}.history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    print(f"Saved cough checkpoint: {checkpoint_path}")
    return {
        "checkpoint": str(checkpoint_path),
        "spec_name": spec_name,
        "selected_postprocessing": sweep["selected"],
        "best_val_loss": float(best_loss),
    }


def select_cough_postprocessing(
    cfg: dict,
    model: nn.Module,
    spec_cfg: dict,
    metadata,
    record_ids: list[int],
    device: torch.device,
    batch_size: int,
) -> dict:
    cough_cfg = cfg["cough"]
    post_cfg = cough_cfg["postprocessing"]
    frame_rate = int(round(int(cfg["sampling"]["audio_hz"]) / int(cough_cfg["frame_hop_samples"])))
    data_root = project_or_absolute(cfg["data"]["data_root"])

    record_cache = []
    for record_id in record_ids:
        record = load_record_preprocessed(int(record_id), metadata=metadata, data_root=data_root)
        probs = predict_cough_probabilities_for_record(
            model,
            record,
            cough_cfg,
            spec_cfg,
            device=device,
            batch_size=batch_size,
        )
        record_cache.append(
            {
                "record_id": int(record_id),
                "duration_sec": float(record["duration_sec"]),
                "probabilities": probs,
                "gt_events": cough_gt_events(record, cough_cfg),
            }
        )

    candidates = []
    for threshold in post_cfg["thresholds"]:
        for merge_gap in post_cfg["pred_merge_gap_sec"]:
            for min_duration in post_cfg["pred_min_duration_sec"]:
                gt_by_record = {}
                pred_by_record = {}
                for item in record_cache:
                    record_id = int(item["record_id"])
                    gt_by_record[record_id] = item["gt_events"]
                    pred_by_record[record_id] = frame_predictions_to_events(
                        item["probabilities"],
                        frame_rate=frame_rate,
                        threshold=float(threshold),
                        min_duration_sec=float(min_duration),
                        merge_gap_sec=float(merge_gap),
                        duration_sec=float(item["duration_sec"]),
                    )
                metrics = event_summary(
                    gt_by_record,
                    pred_by_record,
                    iou_threshold=float(post_cfg["event_iou_threshold"]),
                )
                candidates.append(
                    {
                        "threshold": float(threshold),
                        "pred_merge_gap_sec": float(merge_gap),
                        "pred_min_duration_sec": float(min_duration),
                        **metrics,
                    }
                )

    selected = sorted(
        candidates,
        key=lambda row: (
            row["f1"],
            row["mean_matched_iou"],
            row["precision"],
            -row["predicted_events"],
        ),
        reverse=True,
    )[0]
    return {"selected": selected, "candidates": candidates}


def loader_activity_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    labels = []
    preds = []

    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            y = batch["label"].to(device)
            logits = model(motion)
            loss = criterion(logits, y)
            total_loss += loss.item() * motion.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "precision_macro": precision_score(labels_np, preds_np, average="macro", zero_division=0),
        "recall_macro": recall_score(labels_np, preds_np, average="macro", zero_division=0),
        "f1_macro": f1_score(labels_np, preds_np, average="macro", zero_division=0),
    }


def train_activity_model(cfg: dict, metadata, split, output_dir: Path, device: torch.device) -> dict:
    activity_cfg = cfg["activity"]
    train_cfg = cfg["training"]
    data_root = project_or_absolute(cfg["data"]["data_root"])

    train_ds = V4ActivityWindowDataset(split.train, metadata, activity_cfg, data_root=data_root)
    val_ds = V4ActivityWindowDataset(split.val, metadata, activity_cfg, data_root=data_root)
    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(activity_cfg["classes"])
    model = V4ActivityCNN(num_classes=num_classes).to(device)
    counts = torch.bincount(train_ds.labels, minlength=num_classes).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0) / num_classes
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg["scheduler_factor"]),
        patience=int(train_cfg["scheduler_patience"]),
    )

    best_state = None
    best_loss = float("inf")
    patience_left = int(train_cfg["early_stopping_patience"])
    history = []

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            motion = batch["motion"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(motion), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * motion.size(0)

        train_loss /= max(len(train_loader.dataset), 1)
        val_metrics = loader_activity_metrics(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[activity] epoch {epoch + 1:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['f1_macro']:.3f}"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(train_cfg["early_stopping_patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = output_dir / "v4_activity.pt"
    checkpoint = {
        "model_id": "v4_activity",
        "model_name": "V4ActivityCNN",
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "classes": activity_cfg["classes"],
        "class_to_idx": activity_class_map(activity_cfg),
        "record_split": split.as_dict(),
        "history": history,
        "class_weights": weights.tolist(),
    }
    torch.save(checkpoint, checkpoint_path)
    (output_dir / "v4_activity.history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    print(f"Saved activity checkpoint: {checkpoint_path}")
    return {
        "checkpoint": str(checkpoint_path),
        "best_val_loss": float(best_loss),
    }


def dry_run(cfg: dict, device: torch.device) -> None:
    cough_cfg = cfg["cough"]
    spec_cfg = next(iter(cough_cfg["specs"].values()))
    frame_count = cough_frame_count(cough_cfg)
    cough_model = V4CoughFrameCNN().to(device)
    spec = torch.randn(2, 2, int(spec_cfg["n_mels"]), 501, device=device)
    motion = torch.randn(2, 2, int(float(cough_cfg["chunk_sec"]) * cfg["sampling"]["motion_hz"]), device=device)
    cough_out = resize_frame_logits(cough_model(spec, motion), frame_count)
    print(f"Cough dry-run output shape: {tuple(cough_out.shape)}")

    activity_model = V4ActivityCNN(num_classes=len(cfg["activity"]["classes"])).to(device)
    activity_motion = torch.randn(
        2,
        2,
        int(float(cfg["activity"]["window_sec"]) * cfg["sampling"]["motion_hz"]),
        device=device,
    )
    activity_out = activity_model(activity_motion)
    print(f"Activity dry-run output shape: {tuple(activity_out.shape)}")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["split"]["seed"]))
    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.dry_run:
        dry_run(cfg, device)
        return 0

    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    split = split_records_v4(metadata, cfg["split"])
    output_dir = project_or_absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cough_runs = {}
    for spec_name, spec_cfg in cfg["cough"]["specs"].items():
        cough_runs[spec_name] = train_cough_model(
            cfg=cfg,
            spec_name=spec_name,
            spec_cfg=spec_cfg,
            metadata=metadata,
            split=split,
            output_dir=output_dir,
            device=device,
        )

    primary_spec = sorted(
        cough_runs,
        key=lambda name: (
            cough_runs[name]["selected_postprocessing"]["f1"],
            cough_runs[name]["selected_postprocessing"]["mean_matched_iou"],
        ),
        reverse=True,
    )[0]
    activity_run = train_activity_model(cfg, metadata, split, output_dir, device)

    summary = {
        "model_id": "v4",
        "config_path": args.config,
        "record_split": split.as_dict(),
        "primary_cough_spec": primary_spec,
        "cough_runs": cough_runs,
        "activity_run": activity_run,
    }
    summary_path = output_dir / "v4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Primary cough spec: {primary_spec}")
    print(f"Saved V4 summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
