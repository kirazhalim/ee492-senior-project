from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.event_metrics import (
    binary_labels_to_events,
    match_events,
    probabilities_to_predictions,
    smooth_probabilities,
    window_predictions_to_events,
)
from cough_analysis.models import ASTMotionFusionHead, V4ActivityCNN
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import load_record_preprocessed
from cough_analysis.v3 import resolve_device
from cough_analysis.v4 import (
    activity_target_label,
    assign_activity_to_event,
    event_summary,
    predict_activity_probabilities_for_record,
)
from cough_analysis.v5_ast import ASTFusionDataset, build_ast_window_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V5 AST cough events with the V4 activity classifier."
    )
    parser.add_argument("--v5-model-dir", default="artifacts/final/v5_ast_clean")
    parser.add_argument("--v4-model-dir", default="artifacts/models/final_v4_clean")
    parser.add_argument("--v4-config", default="configs/final/v4_clean.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--event-iou-threshold", type=float, default=None)
    parser.add_argument("--gt-min-duration-sec", type=float, default=None)
    parser.add_argument("--gt-merge-gap-sec", type=float, default=None)
    parser.add_argument("--pred-min-duration-sec", type=float, default=None)
    parser.add_argument("--pred-merge-gap-sec", type=float, default=None)
    parser.add_argument("--pred-span-mode", choices=["full", "center", "hop"], default=None)
    parser.add_argument("--prob-smoothing-sec", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="artifacts/evaluations/final_v5_ast_v4_activity")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_path(path)


def load_v5_model(model_dir: Path, device: torch.device) -> tuple[dict, ASTMotionFusionHead]:
    checkpoint = torch.load(model_dir / "fusion_head.pt", map_location=device, weights_only=False)
    audio_dim = int(checkpoint.get("ast_embedding_dim", checkpoint["config"]["ast"]["embedding_dim"]))
    model = ASTMotionFusionHead(audio_dim=audio_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def load_v4_activity_model(model_dir: Path, v4_cfg: dict, device: torch.device) -> V4ActivityCNN:
    checkpoint = torch.load(model_dir / "v4_activity.pt", map_location=device)
    model = V4ActivityCNN(num_classes=len(v4_cfg["activity"]["classes"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_v5_probs(
    model: ASTMotionFusionHead,
    embeddings: torch.Tensor,
    table: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        ASTFusionDataset(embeddings, table["motion"], table["labels"]),
        batch_size=batch_size,
        shuffle=False,
    )
    probs = []
    with torch.no_grad():
        for batch in loader:
            audio_embedding = batch["audio_embedding"].to(device)
            motion = batch["motion"].to(device)
            logits = model(audio_embedding, motion)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(probs, dtype=np.float32)


def merge_stationary(label: str) -> str:
    return "stationary" if label in {"sitting", "standing"} else label


def activity_reports(rows: list[dict], classes: list[str]) -> dict:
    matched = [row for row in rows if row["matched_gt"]]
    if not matched:
        return {
            "matched_cough_events": 0,
            "matched_with_correct_activity": 0,
            "matched_activity_accuracy": 0.0,
            "classification_report": {},
            "confusion_matrix": [],
            "merged3": {},
        }

    y_true = [row["true_activity"] for row in matched]
    y_pred = [row["activity"] for row in matched]
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    merged_classes = ["stationary", "walking", "running"]
    y_true_merged = [merge_stationary(x) for x in y_true]
    y_pred_merged = [merge_stationary(x) for x in y_pred]
    merged_correct = sum(int(t == p) for t, p in zip(y_true_merged, y_pred_merged))
    merged_report = classification_report(
        y_true_merged,
        y_pred_merged,
        labels=merged_classes,
        target_names=merged_classes,
        output_dict=True,
        zero_division=0,
    )
    merged_cm = confusion_matrix(y_true_merged, y_pred_merged, labels=merged_classes)

    return {
        "matched_cough_events": len(matched),
        "matched_with_correct_activity": int(correct),
        "matched_activity_accuracy": float(correct / len(matched)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "merged3": {
            "classes": merged_classes,
            "matched_with_correct_activity": int(merged_correct),
            "matched_activity_accuracy": float(merged_correct / len(matched)),
            "classification_report": merged_report,
            "confusion_matrix": merged_cm.tolist(),
        },
    }


def write_event_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def selected_or_arg(args_value, selected: dict, key: str, fallback):
    return args_value if args_value is not None else selected.get(key, fallback)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    v5_model_dir = project_or_absolute(args.v5_model_dir)
    v4_model_dir = project_or_absolute(args.v4_model_dir)
    v4_cfg = load_config(args.v4_config)

    v5_checkpoint, v5_model = load_v5_model(v5_model_dir, device)
    v5_cfg = v5_checkpoint["config"]
    v4_activity_model = load_v4_activity_model(v4_model_dir, v4_cfg, device)
    selected = v5_checkpoint.get("selected_postprocessing", {})
    event_cfg = v5_cfg["event"]
    window_cfg = v5_cfg["windowing"]

    threshold = float(selected_or_arg(args.threshold, selected, "threshold", 0.5))
    event_iou_threshold = float(
        args.event_iou_threshold
        if args.event_iou_threshold is not None
        else event_cfg.get("iou_threshold", 0.2)
    )
    gt_min_duration_sec = float(
        args.gt_min_duration_sec
        if args.gt_min_duration_sec is not None
        else event_cfg.get("gt_min_duration_sec", 0.1)
    )
    gt_merge_gap_sec = float(
        args.gt_merge_gap_sec
        if args.gt_merge_gap_sec is not None
        else event_cfg.get("gt_merge_gap_sec", 0.1)
    )
    pred_min_duration_sec = float(
        selected_or_arg(
            args.pred_min_duration_sec,
            selected,
            "pred_min_duration_sec",
            event_cfg.get("pred_min_duration_sec", 0.1),
        )
    )
    pred_merge_gap_sec = float(
        selected_or_arg(
            args.pred_merge_gap_sec,
            selected,
            "pred_merge_gap_sec",
            event_cfg.get("pred_merge_gap_sec", 0.1),
        )
    )
    pred_span_mode = str(selected_or_arg(args.pred_span_mode, selected, "span_mode", "full"))
    prob_smoothing_sec = float(selected_or_arg(args.prob_smoothing_sec, selected, "smoothing_sec", 0.0))

    metadata = load_metadata(project_or_absolute(v5_cfg["data"]["metadata"]))
    data_root = project_or_absolute(v5_cfg["data"].get("data_root", "data"))
    record_ids = [int(x) for x in v5_checkpoint["record_split"][args.split]]
    table = build_ast_window_table(
        record_ids,
        metadata=metadata,
        data_root=data_root,
        window_sec=float(window_cfg["window_sec"]),
        hop_sec=float(window_cfg["hop_sec"]),
        center_fraction=float(window_cfg["center_fraction"]),
    )
    cache_path = v5_model_dir / "cache" / f"{args.split}_ast_embeddings.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing V5 AST embedding cache: {cache_path}. "
            "Run scripts/train_v5_ast.py first."
        )
    embeddings = torch.load(cache_path, map_location="cpu")
    probs = predict_v5_probs(
        v5_model,
        embeddings,
        table,
        batch_size=args.batch_size,
        device=device,
    )

    gt_by_record = {}
    pred_by_record = {}
    event_rows = []
    for record_id in record_ids:
        mask = table["record_ids"] == int(record_id)
        spans = list(zip(table["span_start"][mask], table["span_end"][mask]))
        record_probs = smooth_probabilities(
            probs[mask],
            spans,
            smoothing_sec=prob_smoothing_sec,
        )
        preds = probabilities_to_predictions(record_probs, threshold=threshold)
        pred_events = window_predictions_to_events(
            spans,
            preds,
            min_duration_sec=pred_min_duration_sec,
            merge_gap_sec=pred_merge_gap_sec,
            span_mode=pred_span_mode,
            center_fraction=float(window_cfg["center_fraction"]),
        )

        record = load_record_preprocessed(int(record_id), metadata=metadata, data_root=data_root)
        gt_events = binary_labels_to_events(
            record["cough_label"],
            sample_rate=int(record["fs_audio"]),
            min_duration_sec=gt_min_duration_sec,
            merge_gap_sec=gt_merge_gap_sec,
        )
        gt_by_record[int(record_id)] = gt_events
        pred_by_record[int(record_id)] = pred_events

        activity_centers, activity_probs = predict_activity_probabilities_for_record(
            v4_activity_model,
            record,
            v4_cfg["activity"],
            device=device,
            batch_size=args.batch_size,
        )
        matches = match_events(gt_events, pred_events, iou_threshold=event_iou_threshold)
        matched_pred = {pred_idx: gt_idx for gt_idx, pred_idx, _ in matches}
        true_activity = activity_target_label(str(record["activity"]), v4_cfg["activity"])

        for pred_idx, event in enumerate(pred_events):
            assigned = assign_activity_to_event(
                event,
                activity_centers,
                activity_probs,
                v4_cfg["activity"]["classes"],
                context_sec=float(v4_cfg["activity"]["attribution_context_sec"]),
            )
            is_matched = pred_idx in matched_pred
            activity = str(assigned["activity"])
            event_rows.append(
                {
                    "record_id": int(record_id),
                    "cough_start": float(event.start),
                    "cough_end": float(event.end),
                    "matched_gt": bool(is_matched),
                    "true_activity": true_activity,
                    "activity": activity,
                    "activity_confidence": float(assigned["activity_confidence"]),
                    "activity_correct_if_matched": bool(is_matched and activity == true_activity),
                    "merged_activity_correct_if_matched": bool(
                        is_matched and merge_stationary(activity) == merge_stationary(true_activity)
                    ),
                }
            )

    cough_metrics = event_summary(gt_by_record, pred_by_record, iou_threshold=event_iou_threshold)
    activity_metrics = activity_reports(event_rows, v4_cfg["activity"]["classes"])
    report = {
        "pipeline": "v5_ast_cough_v4_activity",
        "v5_model_dir": str(v5_model_dir),
        "v4_model_dir": str(v4_model_dir),
        "split": args.split,
        "record_ids": record_ids,
        "postprocessing": {
            "threshold": threshold,
            "event_iou_threshold": event_iou_threshold,
            "gt_min_duration_sec": gt_min_duration_sec,
            "gt_merge_gap_sec": gt_merge_gap_sec,
            "pred_min_duration_sec": pred_min_duration_sec,
            "pred_merge_gap_sec": pred_merge_gap_sec,
            "pred_span_mode": pred_span_mode,
            "prob_smoothing_sec": prob_smoothing_sec,
        },
        "cough": cough_metrics,
        "activity_on_matched_v5_cough_events": activity_metrics,
    }

    output_dir = project_or_absolute(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "v5_ast_cough_v4_activity_evaluation.json"
    events_path = output_dir / "v5_ast_cough_v4_activity_events.csv"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_event_rows(events_path, event_rows)

    print(
        f"V5 cough event F1={cough_metrics['f1']:.3f} "
        f"P={cough_metrics['precision']:.3f} R={cough_metrics['recall']:.3f}"
    )
    print(
        "Activity accuracy on matched V5 cough events="
        f"{activity_metrics['matched_activity_accuracy']:.3f}"
    )
    print(
        "Merged stationary activity accuracy="
        f"{activity_metrics.get('merged3', {}).get('matched_activity_accuracy', 0.0):.3f}"
    )
    print(f"Saved report: {report_path}")
    print(f"Saved events: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
