from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.event_metrics import (
    binary_labels_to_events,
    event_level_metrics,
    probabilities_to_predictions,
    smooth_probabilities,
    window_predictions_to_events,
)
from cough_analysis.models import RawWaveformCoughCNN
from cough_analysis.paths import project_path
from cough_analysis.raw_baselines import (
    RawWaveformDataset,
    build_raw_dataset,
    build_raw_record_dataset,
)
from cough_analysis.v3 import resolve_device, split_records_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--record-ids", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event-iou-threshold", type=float, default=0.2)
    parser.add_argument("--gt-min-duration-sec", type=float, default=0.0)
    parser.add_argument("--gt-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--pred-min-duration-sec", type=float, default=0.0)
    parser.add_argument("--pred-merge-gap-sec", type=float, default=0.0)
    parser.add_argument(
        "--pred-span-mode",
        choices=["full", "center", "hop"],
        default=None,
        help="Defaults to center for center-positive configs, otherwise full.",
    )
    parser.add_argument("--prob-smoothing-sec", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def project_or_absolute(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def parse_record_ids(value: str) -> list[int]:
    record_ids = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", maxsplit=1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            record_ids.extend(range(start, end + 1))
        else:
            record_ids.append(int(token))
    return record_ids


def window_settings(cfg: dict) -> dict:
    win_cfg = cfg["windowing"]
    return {
        "window_sec": float(win_cfg["window_sec"]),
        "hop_sec": float(win_cfg["hop_sec"]),
        "label_rule": str(win_cfg["label_rule"]),
        "center_fraction": float(win_cfg.get("center_fraction", 0.2)),
    }


def predict_arrays(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_all = []
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            motion = batch["motion"].to(device)
            labels = batch["label"].int().cpu().numpy()
            logits = model(audio, motion)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            labels_all.extend(labels.tolist())
            preds_all.extend(preds.tolist())
            probs_all.extend(probs.tolist())
    return (
        np.asarray(labels_all),
        np.asarray(preds_all),
        np.asarray(probs_all),
    )


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(project_or_absolute(args.checkpoint), map_location=device)
    cfg = checkpoint.get("config") or load_config(args.config)
    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    data_root = project_or_absolute(cfg["data"].get("data_root"))

    split_map = checkpoint.get("record_split")
    if args.record_ids:
        record_ids = parse_record_ids(args.record_ids)
    elif split_map and args.split in split_map:
        record_ids = [int(x) for x in split_map[args.split]]
    else:
        _, val_ids, test_ids = split_records_from_config(metadata, cfg.get("split"))
        record_ids = [int(x) for x in (val_ids if args.split == "val" else test_ids)]

    settings = window_settings(cfg)
    pred_span_mode = args.pred_span_mode
    if pred_span_mode is None:
        pred_span_mode = "center" if settings["label_rule"] == "center_positive" else "full"

    audio, motion, labels = build_raw_dataset(
        record_ids,
        metadata,
        data_root=data_root,
        **settings,
    )
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    loader = DataLoader(
        RawWaveformDataset(audio, motion, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    model = RawWaveformCoughCNN().to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    labels_np, preds_np, probs_np = predict_arrays(
        model,
        loader,
        device=device,
        threshold=args.threshold,
    )
    report = classification_report(
        labels_np,
        preds_np,
        target_names=["Non-Cough", "Cough"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    avg_precision = (
        average_precision_score(labels_np, probs_np)
        if len(np.unique(labels_np)) > 1
        else 0.0
    )

    output_dir = project_or_absolute(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.split}_classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{args.split}_confusion_matrix.json").write_text(
        json.dumps(cm.tolist(), indent=2),
        encoding="utf-8",
    )
    with (output_dir / f"{args.split}_predictions.csv").open("w", encoding="utf-8") as f:
        f.write("label,prediction,probability\n")
        for label, pred, prob in zip(labels_np, preds_np, probs_np):
            f.write(f"{int(label)},{int(pred)},{float(prob):.8f}\n")

    total_event_counts = {
        "true_events": 0,
        "predicted_events": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    per_record = []
    for record_id in record_ids:
        record_data = build_raw_record_dataset(
            record_id,
            metadata,
            data_root=data_root,
            **settings,
        )
        record_loader = DataLoader(
            RawWaveformDataset(
                record_data["audio"],
                record_data["motion"],
                record_data["labels"],
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        _, _, record_probs = predict_arrays(
            model,
            record_loader,
            device=device,
            threshold=args.threshold,
        )
        gt_events = binary_labels_to_events(
            record_data["record"]["cough_label"],
            sample_rate=int(record_data["record"]["fs_audio"]),
            min_duration_sec=args.gt_min_duration_sec,
            merge_gap_sec=args.gt_merge_gap_sec,
        )
        event_probs = smooth_probabilities(
            record_probs,
            record_data["spans"],
            smoothing_sec=args.prob_smoothing_sec,
        )
        event_preds = probabilities_to_predictions(event_probs, threshold=args.threshold)
        pred_events = window_predictions_to_events(
            record_data["spans"],
            event_preds,
            min_duration_sec=args.pred_min_duration_sec,
            merge_gap_sec=args.pred_merge_gap_sec,
            span_mode=pred_span_mode,
            center_fraction=settings["center_fraction"],
        )
        metrics = event_level_metrics(
            gt_events,
            pred_events,
            iou_threshold=args.event_iou_threshold,
        )
        for key in total_event_counts:
            total_event_counts[key] += int(metrics[key])
        per_record.append({"record_id": int(record_id), **metrics})

    tp = total_event_counts["true_positive"]
    fp = total_event_counts["false_positive"]
    fn = total_event_counts["false_negative"]
    event_precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    event_recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall > 0
        else 0.0
    )
    event_summary = {
        **total_event_counts,
        "precision": event_precision,
        "recall": event_recall,
        "f1": event_f1,
        "average_precision": float(avg_precision),
        "threshold": args.threshold,
        "event_iou_threshold": args.event_iou_threshold,
        "gt_min_duration_sec": args.gt_min_duration_sec,
        "gt_merge_gap_sec": args.gt_merge_gap_sec,
        "pred_min_duration_sec": args.pred_min_duration_sec,
        "pred_merge_gap_sec": args.pred_merge_gap_sec,
        "pred_span_mode": pred_span_mode,
        "pred_center_fraction": settings["center_fraction"],
        "prob_smoothing_sec": args.prob_smoothing_sec,
        "per_record": per_record,
    }
    events_path = output_dir / f"{args.split}_event_metrics.json"
    events_path.write_text(json.dumps(event_summary, indent=2), encoding="utf-8")

    print(classification_report(labels_np, preds_np, target_names=["Non-Cough", "Cough"]))
    print(
        "Event-level: "
        f"P={event_precision:.3f} R={event_recall:.3f} F1={event_f1:.3f} "
        f"TP={tp} FP={fp} FN={fn}"
    )
    print(f"Saved event metrics: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
