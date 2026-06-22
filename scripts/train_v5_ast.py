from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.models import ASTMotionFusionHead
from cough_analysis.paths import project_path
from cough_analysis.v3 import resolve_device, split_records_from_config
from cough_analysis.v5_ast import (
    ASTFusionDataset,
    ast_event_metrics_for_table,
    ast_split_summary,
    build_ast_window_table,
    build_gt_event_cache,
    evaluate_ast_fusion_model,
    extract_ast_embeddings,
    run_ast_postprocessing_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final/v5_ast_clean.yaml")
    parser.add_argument("--output-dir", default="artifacts/final/v5_ast_clean")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--overwrite-embeddings", action="store_true")
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


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )


def compact_window_metrics(split: str, metrics: dict) -> dict:
    return {
        "split": split,
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }


def load_ast_components(cfg: dict, device: torch.device):
    try:
        from transformers import ASTModel, AutoFeatureExtractor
    except ImportError as exc:
        raise RuntimeError("transformers is required for V5 AST runs.") from exc

    model_name = str(cfg["ast"]["model_name"])
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        ast_model = ASTModel.from_pretrained(model_name).to(device)
    except Exception as exc:
        raise RuntimeError(
            "Could not load AST. The first run needs internet access or an existing "
            "Hugging Face cache."
        ) from exc

    ast_model.eval()
    for parameter in ast_model.parameters():
        parameter.requires_grad = False
    return feature_extractor, ast_model


def load_or_extract_embeddings(
    split: str,
    table: dict,
    cache_dir: Path,
    cfg: dict,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
    feature_extractor=None,
    ast_model=None,
) -> torch.Tensor:
    cache_path = cache_dir / f"{split}_ast_embeddings.pt"
    if cache_path.exists() and not overwrite:
        return torch.load(cache_path, map_location="cpu")

    ast_cfg = cfg["ast"]
    if feature_extractor is None or ast_model is None:
        feature_extractor, ast_model = load_ast_components(cfg, device)

    embeddings = extract_ast_embeddings(
        table["audio"],
        feature_extractor=feature_extractor,
        ast_model=ast_model,
        device=device,
        batch_size=batch_size,
        ast_sample_rate=int(ast_cfg["sample_rate"]),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    return embeddings


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["split"]["seed"])
    set_seed(seed)
    device = resolve_device(args.device)
    output_dir = project_or_absolute(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    if args.dry_run:
        model = ASTMotionFusionHead(audio_dim=int(cfg["ast"].get("embedding_dim", 768))).to(device)
        window_sec = float(cfg["windowing"]["window_sec"])
        dummy_embedding = torch.randn(2, int(cfg["ast"].get("embedding_dim", 768))).to(device)
        dummy_motion = torch.randn(2, 2, int(round(window_sec * 100))).to(device)
        with torch.no_grad():
            out = model(dummy_embedding, dummy_motion)
        print(f"Device: {device}")
        print(f"Dry run output shape: {tuple(out.shape)}")
        return 0

    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    data_root = project_or_absolute(cfg["data"].get("data_root"))
    train_ids, val_ids, test_ids = split_records_from_config(
        metadata,
        cfg.get("split"),
        random_state=seed,
    )
    split_ids = {
        "train": [int(x) for x in train_ids],
        "val": [int(x) for x in val_ids],
        "test": [int(x) for x in test_ids],
    }
    print(
        f"Record split -> Train: {len(train_ids)} | "
        f"Val: {len(val_ids)} | Test: {len(test_ids)}"
    )

    win_cfg = cfg["windowing"]
    tables = {
        split: build_ast_window_table(
            ids,
            metadata=metadata,
            data_root=data_root,
            window_sec=float(win_cfg["window_sec"]),
            hop_sec=float(win_cfg["hop_sec"]),
            center_fraction=float(win_cfg["center_fraction"]),
        )
        for split, ids in split_ids.items()
    }
    pd.DataFrame(ast_split_summary(tables)).to_csv(
        tables_dir / "split_window_summary.csv",
        index=False,
    )

    embedding_batch_size = (
        args.embedding_batch_size
        if args.embedding_batch_size is not None
        else int(cfg["ast"].get("embedding_batch_size", 32))
    )
    needs_embedding_extraction = any(
        args.overwrite_embeddings or not (cache_dir / f"{split}_ast_embeddings.pt").exists()
        for split in tables
    )
    ast_components = (
        load_ast_components(cfg, device)
        if needs_embedding_extraction
        else (None, None)
    )
    embeddings = {
        split: load_or_extract_embeddings(
            split,
            table,
            cache_dir=cache_dir,
            cfg=cfg,
            device=device,
            batch_size=embedding_batch_size,
            overwrite=args.overwrite_embeddings,
            feature_extractor=ast_components[0],
            ast_model=ast_components[1],
        )
        for split, table in tables.items()
    }

    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    epochs = args.epochs if args.epochs is not None else int(cfg["training"]["epochs"])
    train_ds = ASTFusionDataset(embeddings["train"], tables["train"]["motion"], tables["train"]["labels"])
    val_ds = ASTFusionDataset(embeddings["val"], tables["val"]["motion"], tables["val"]["labels"])
    test_ds = ASTFusionDataset(embeddings["test"], tables["test"]["motion"], tables["test"]["labels"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = ASTMotionFusionHead(audio_dim=embeddings["train"].shape[1]).to(device)
    num_pos = float(np.sum(tables["train"]["labels"]))
    num_neg = float(len(tables["train"]["labels"]) - num_pos)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    history = []
    best_val_f1 = -1.0
    best_epoch = None
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            audio_embedding = batch["audio_embedding"].to(device)
            motion = batch["motion"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(audio_embedding, motion)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(labels)

        train_loss = train_loss_sum / len(train_ds)
        val_metrics_epoch = evaluate_ast_fusion_model(
            model,
            val_loader,
            criterion=criterion,
            device=device,
        )
        if val_metrics_epoch["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics_epoch["f1"])
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{
                f"val_{key}": float(val_metrics_epoch[key])
                for key in ["loss", "accuracy", "precision", "recall", "f1"]
            },
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics_epoch['loss']:.4f} | "
            f"val_f1={val_metrics_epoch['f1']:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate_ast_fusion_model(model, val_loader, criterion, device=device)
    test_metrics = evaluate_ast_fusion_model(model, test_loader, criterion, device=device)
    pd.DataFrame(history).to_csv(tables_dir / "training_history.csv", index=False)
    pd.DataFrame(
        [
            compact_window_metrics("val", val_metrics),
            compact_window_metrics("test", test_metrics),
        ]
    ).to_csv(tables_dir / "window_metrics.csv", index=False)

    event_cfg = cfg["event"]
    gt_caches = {
        "val": build_gt_event_cache(
            split_ids["val"],
            metadata,
            data_root,
            gt_min_duration_sec=float(event_cfg["gt_min_duration_sec"]),
            gt_merge_gap_sec=float(event_cfg["gt_merge_gap_sec"]),
        ),
        "test": build_gt_event_cache(
            split_ids["test"],
            metadata,
            data_root,
            gt_min_duration_sec=float(event_cfg["gt_min_duration_sec"]),
            gt_merge_gap_sec=float(event_cfg["gt_merge_gap_sec"]),
        ),
    }
    fixed_event_rows = []
    for split, metrics in [("val", val_metrics), ("test", test_metrics)]:
        fixed = ast_event_metrics_for_table(
            tables[split],
            metrics["probs"],
            split_ids[split],
            gt_caches[split],
            threshold=0.5,
            smoothing_sec=0.0,
            span_mode="full",
            pred_min_duration_sec=float(event_cfg["pred_min_duration_sec"]),
            pred_merge_gap_sec=float(event_cfg["pred_merge_gap_sec"]),
            center_fraction=float(win_cfg["center_fraction"]),
            event_iou_threshold=float(event_cfg["iou_threshold"]),
        )
        fixed_event_rows.append({"split": split, "threshold": 0.5, **fixed})
    pd.DataFrame(fixed_event_rows).to_csv(tables_dir / "event_metrics_threshold_0p5.csv", index=False)

    sweep_rows = run_ast_postprocessing_sweep(
        table=tables["val"],
        probs=val_metrics["probs"],
        record_ids=split_ids["val"],
        gt_cache=gt_caches["val"],
        sweep_cfg=cfg["postprocessing_sweep"],
        center_fraction=float(win_cfg["center_fraction"]),
        event_iou_threshold=float(event_cfg["iou_threshold"]),
    )
    val_sweep = pd.DataFrame(sweep_rows)
    val_sweep.to_csv(tables_dir / "postprocessing_sweep_val.csv", index=False)
    selection_cols = ["f1", "mean_matched_iou", "precision", "predicted_events"]
    selected = (
        val_sweep.sort_values(selection_cols, ascending=[False, False, False, True])
        .iloc[0]
        .to_dict()
    )
    selected_test = ast_event_metrics_for_table(
        table=tables["test"],
        probs=test_metrics["probs"],
        record_ids=split_ids["test"],
        gt_cache=gt_caches["test"],
        threshold=float(selected["threshold"]),
        smoothing_sec=float(selected["smoothing_sec"]),
        span_mode=str(selected["span_mode"]),
        pred_min_duration_sec=float(selected["pred_min_duration_sec"]),
        pred_merge_gap_sec=float(selected["pred_merge_gap_sec"]),
        center_fraction=float(win_cfg["center_fraction"]),
        event_iou_threshold=float(event_cfg["iou_threshold"]),
    )
    selected_summary = pd.DataFrame(
        [
            {"split": "val", **selected},
            {
                "split": "test",
                "threshold": selected["threshold"],
                "smoothing_sec": selected["smoothing_sec"],
                "span_mode": selected["span_mode"],
                "pred_min_duration_sec": selected["pred_min_duration_sec"],
                "pred_merge_gap_sec": selected["pred_merge_gap_sec"],
                **selected_test,
            },
        ]
    )
    selected_summary.to_csv(tables_dir / "postprocessing_selected_val_test.csv", index=False)

    checkpoint = {
        "model_name": "ASTMotionFusionHead",
        "model_state_dict": best_state or model.state_dict(),
        "config": cfg,
        "record_split": split_ids,
        "history": history,
        "best_epoch": best_epoch,
        "pos_weight": float(pos_weight.item()),
        "seed": seed,
        "ast_embedding_dim": int(embeddings["train"].shape[1]),
        "selected_postprocessing": selected,
    }
    checkpoint_path = output_dir / "fusion_head.pt"
    torch.save(checkpoint, checkpoint_path)
    save_json(
        output_dir / "summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "window_metrics": [
                compact_window_metrics("val", val_metrics),
                compact_window_metrics("test", test_metrics),
            ],
            "selected_postprocessing": selected_summary.to_dict(orient="records"),
        },
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved tables: {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
