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
from cough_analysis.input_ablation import INPUT_ABLATION_MODES, apply_input_ablation
from cough_analysis.models import Spec2DCoughCNN
from cough_analysis.paths import project_path
from cough_analysis.v3 import (
    SpectrogramDataset,
    build_dataset,
    resolve_device,
    split_records_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--output", default="artifacts/models/v3_cough.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="v3_training")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument(
        "--input-ablation",
        choices=INPUT_ABLATION_MODES,
        default="full",
        help="Mask selected input groups while keeping the same model architecture.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def config_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(path)


def manifest_mlflow_params(path: str | None) -> dict:
    if not path:
        return {}
    manifest = load_config(path)
    hashes = manifest.get("hashes", {})
    return {
        "dataset_manifest": path,
        "dataset_id": manifest.get("dataset_id", ""),
        "dataset_record_count": manifest.get("record_count", ""),
        "dataset_hash": hashes.get("combined_sha256", ""),
        "dataset_metadata_hash": hashes.get("metadata_rows_sha256", ""),
        "dataset_files_hash": hashes.get("data_files_sha256", ""),
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    input_ablation: str = "full",
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"].to(device)
            motion = batch["motion"].to(device)
            spec, motion = apply_input_ablation(spec, motion, mode=input_ablation)
            labels = batch["label"].to(device)
            logits = model(spec, motion)
            loss = criterion(logits, labels)
            total_loss += loss.item() * spec.size(0)
            preds = (torch.sigmoid(logits) >= threshold).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.int().cpu().numpy())

    labels_np = np.asarray(all_labels)
    preds_np = np.asarray(all_preds)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
    }


def setup_mlflow(args: argparse.Namespace):
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed. Install tracking dependencies first: "
            ".venv/bin/python -m pip install -r requirements-tracking.txt"
        ) from exc

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.start_run(run_name=args.mlflow_run_name or "train_v3")
    return mlflow


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.dry_run:
        model = Spec2DCoughCNN(num_classes=1).to(device)
        n_mels = int(cfg["spectrogram"]["n_mels"])
        dummy_spec = torch.randn(2, 2, n_mels, 38).to(device)
        dummy_motion = torch.randn(2, 2, int(cfg["windowing"]["window_sec"] * 100)).to(device)
        with torch.no_grad():
            out = model(dummy_spec, dummy_motion)
        print(f"Dry run output shape: {tuple(out.shape)}")
        return 0

    metadata_path = config_path(cfg["data"]["metadata"])
    metadata = load_metadata(metadata_path)
    if args.max_records is not None:
        metadata = metadata.head(args.max_records).copy()
    data_root = config_path(cfg["data"]["data_root"]) if cfg["data"].get("data_root") else None

    train_ids, val_ids, test_ids = split_records_from_config(
        metadata,
        cfg.get("split"),
        random_state=args.seed,
    )
    print(
        f"Record split -> Train: {len(train_ids)} | "
        f"Val: {len(val_ids)} | Test: {len(test_ids)}"
    )

    window_cfg = cfg["windowing"]
    spec_cfg = cfg["spectrogram"]
    X_s_train, X_m_train, y_train = build_dataset(
        train_ids,
        metadata,
        data_root=data_root,
        window_sec=float(window_cfg["window_sec"]),
        hop_sec=float(window_cfg["hop_sec"]),
        center_fraction=float(window_cfg["center_fraction"]),
        spectrogram_config=spec_cfg,
    )
    X_s_val, X_m_val, y_val = build_dataset(
        val_ids,
        metadata,
        data_root=data_root,
        window_sec=float(window_cfg["window_sec"]),
        hop_sec=float(window_cfg["hop_sec"]),
        center_fraction=float(window_cfg["center_fraction"]),
        spectrogram_config=spec_cfg,
    )

    print(
        f"Train: Spec {X_s_train.shape} | Motion {X_m_train.shape} | "
        f"Positive: {int(np.sum(y_train))} ({np.mean(y_train) * 100:.1f}%)"
    )
    print(
        f"Val:   Spec {X_s_val.shape} | Motion {X_m_val.shape} | "
        f"Positive: {int(np.sum(y_val))} ({np.mean(y_val) * 100:.1f}%)"
    )

    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    epochs = args.epochs if args.epochs is not None else int(cfg["training"]["epochs"])
    train_loader = DataLoader(
        SpectrogramDataset(X_s_train, X_m_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SpectrogramDataset(X_s_val, X_m_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    model = Spec2DCoughCNN(num_classes=1).to(device)
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
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    mlflow = None
    if args.mlflow:
        mlflow = setup_mlflow(args)
        params = {
                "script": "train_v3",
                "config": args.config,
                "output": args.output,
                "model_id": args.model_id or "",
                "input_ablation": args.input_ablation,
                "seed": args.seed,
                "device": str(device),
                "max_records": args.max_records or "",
                "train_records": len(train_ids),
                "val_records": len(val_ids),
                "test_records": len(test_ids),
                "train_windows": len(y_train),
                "val_windows": len(y_val),
                "train_cough_windows": int(np.sum(y_train)),
                "val_cough_windows": int(np.sum(y_val)),
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": cfg["training"]["optimizer"],
                "learning_rate": float(cfg["training"]["learning_rate"]),
                "weight_decay": float(cfg["training"]["weight_decay"]),
                "scheduler": cfg["training"]["scheduler"],
                "pos_weight": float(pos_weight.item()),
                "window_sec": float(window_cfg["window_sec"]),
                "hop_sec": float(window_cfg["hop_sec"]),
                "center_fraction": float(window_cfg["center_fraction"]),
                "n_mels": int(spec_cfg["n_mels"]),
                "n_fft": int(spec_cfg["n_fft"]),
                "mel_hop_length": int(spec_cfg["hop_length"]),
                "f_min": float(spec_cfg["f_min"]),
                "f_max": float(spec_cfg["f_max"]),
        }
        params.update(manifest_mlflow_params(args.dataset_manifest))
        mlflow.log_params(params)

    best_val_loss = float("inf")
    best_state = None
    best_metrics = None
    history = []

    try:
        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0.0
            for batch in train_loader:
                spec = batch["spec"].to(device)
                motion = batch["motion"].to(device)
                spec, motion = apply_input_ablation(spec, motion, mode=args.input_ablation)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                logits = model(spec, motion)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * spec.size(0)

            train_loss = train_loss_sum / len(train_loader.dataset)
            val_metrics = evaluate_loader(
                model,
                val_loader,
                device=device,
                input_ablation=args.input_ablation,
            )
            scheduler.step(val_metrics["loss"])
            lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch + 1,
                "learning_rate": lr,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_metrics = val_metrics.copy()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if mlflow is not None:
                mlflow.log_metrics(
                    {
                        "train_loss": float(train_loss),
                        "learning_rate": float(lr),
                        "val_loss": float(val_metrics["loss"]),
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "val_precision": float(val_metrics["precision"]),
                        "val_recall": float(val_metrics["recall"]),
                        "val_f1": float(val_metrics["f1"]),
                    },
                    step=epoch + 1,
                )

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | LR: {lr:.5f} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy'] * 100:.1f}%"
            )

        output_path = config_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_name": "Spec2DCoughCNN",
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
            "input_ablation": args.input_ablation,
        }
        torch.save(checkpoint, output_path)
        history_path = output_path.with_suffix(".history.json")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if mlflow is not None:
            if best_metrics is not None:
                mlflow.log_metrics(
                    {
                        "best_val_loss": float(best_metrics["loss"]),
                        "best_val_accuracy": float(best_metrics["accuracy"]),
                        "best_val_precision": float(best_metrics["precision"]),
                        "best_val_recall": float(best_metrics["recall"]),
                        "best_val_f1": float(best_metrics["f1"]),
                    }
                )
            mlflow.log_artifact(str(output_path), artifact_path="model")
            mlflow.log_artifact(str(history_path), artifact_path="training")
            mlflow.log_artifact(str(config_path(args.config)), artifact_path="config")

        print(f"Saved checkpoint: {output_path}")
        print(f"Saved history: {history_path}")
    finally:
        if mlflow is not None:
            mlflow.end_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
