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
from cough_analysis.input_ablation import apply_input_ablation
from cough_analysis.models import Spec2DCoughCNN, V4ActivityCNN
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import load_record_preprocessed
from cough_analysis.v3 import SpectrogramDataset, build_record_dataset, resolve_device
from cough_analysis.v4 import (
    activity_target_label,
    assign_activity_to_event,
    predict_activity_probabilities_for_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V3 cough events with the V4 activity classifier."
    )
    parser.add_argument("--v3-checkpoint", default="artifacts/models/v3_all_records.pt")
    parser.add_argument("--v4-model-dir", default="artifacts/models/v4")
    parser.add_argument("--v4-config", default="configs/v4.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--record-ids", default=None)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--event-iou-threshold", type=float, default=0.2)
    parser.add_argument("--gt-min-duration-sec", type=float, default=0.1)
    parser.add_argument("--gt-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--pred-min-duration-sec", type=float, default=0.2)
    parser.add_argument("--pred-merge-gap-sec", type=float, default=0.1)
    parser.add_argument("--pred-span-mode", choices=["full", "center", "hop"], default="center")
    parser.add_argument("--prob-smoothing-sec", type=float, default=0.0)
    parser.add_argument("--hysteresis-low-threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="artifacts/evaluations/v3_activity")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_path(path)


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


def load_v3_model(checkpoint_path: Path, device: torch.device) -> tuple[dict, Spec2DCoughCNN]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Spec2DCoughCNN(num_classes=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def load_v4_activity_model(model_dir: Path, v4_cfg: dict, device: torch.device) -> V4ActivityCNN:
    checkpoint = torch.load(model_dir / "v4_activity.pt", map_location=device)
    model = V4ActivityCNN(num_classes=len(v4_cfg["activity"]["classes"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_v3_record_probs(
    model: Spec2DCoughCNN,
    record_data: dict,
    batch_size: int,
    device: torch.device,
    input_ablation: str,
) -> np.ndarray:
    loader = DataLoader(
        SpectrogramDataset(
            record_data["spec"],
            record_data["motion"],
            record_data["labels"],
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    probs = []
    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"].to(device)
            motion = batch["motion"].to(device)
            spec, motion = apply_input_ablation(spec, motion, mode=input_ablation)
            logits = model(spec, motion)
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


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    v3_checkpoint_path = project_or_absolute(args.v3_checkpoint)
    v4_model_dir = project_or_absolute(args.v4_model_dir)
    v4_cfg = load_config(args.v4_config)

    v3_checkpoint, v3_model = load_v3_model(v3_checkpoint_path, device)
    v3_cfg = v3_checkpoint["config"]
    v4_activity_model = load_v4_activity_model(v4_model_dir, v4_cfg, device)
    input_ablation = v3_checkpoint.get("input_ablation", "full")

    metadata = load_metadata(project_or_absolute(v3_cfg["data"]["metadata"]))
    v3_data_root = project_or_absolute(v3_cfg["data"].get("data_root", "data"))
    record_split = v3_checkpoint.get("record_split", {})
    if args.record_ids:
        record_ids = parse_record_ids(args.record_ids)
    elif args.split in record_split:
        record_ids = [int(x) for x in record_split[args.split]]
    else:
        raise ValueError(f"Checkpoint has no split named {args.split!r}; pass --record-ids.")

    window_cfg = v3_cfg["windowing"]
    spec_cfg = v3_cfg["spectrogram"]
    gt_by_record = {}
    pred_by_record = {}
    event_rows = []

    for record_id in record_ids:
        record_data = build_record_dataset(
            int(record_id),
            metadata,
            data_root=v3_data_root,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
            spectrogram_config=spec_cfg,
        )
        probs = predict_v3_record_probs(
            v3_model,
            record_data,
            batch_size=args.batch_size,
            device=device,
            input_ablation=input_ablation,
        )
        event_probs = smooth_probabilities(
            probs,
            record_data["spans"],
            smoothing_sec=args.prob_smoothing_sec,
        )
        preds = probabilities_to_predictions(
            event_probs,
            threshold=args.threshold,
            hysteresis_low_threshold=args.hysteresis_low_threshold,
        )
        gt_events = binary_labels_to_events(
            record_data["record"]["cough_label"],
            sample_rate=int(record_data["record"]["fs_audio"]),
            min_duration_sec=args.gt_min_duration_sec,
            merge_gap_sec=args.gt_merge_gap_sec,
        )
        pred_events = window_predictions_to_events(
            record_data["spans"],
            preds,
            min_duration_sec=args.pred_min_duration_sec,
            merge_gap_sec=args.pred_merge_gap_sec,
            span_mode=args.pred_span_mode,
            center_fraction=float(window_cfg["center_fraction"]),
        )

        record = load_record_preprocessed(
            int(record_id),
            metadata=metadata,
            data_root=v3_data_root,
        )
        activity_centers, activity_probs = predict_activity_probabilities_for_record(
            v4_activity_model,
            record,
            v4_cfg["activity"],
            device=device,
            batch_size=args.batch_size,
        )

        matches = match_events(
            gt_events,
            pred_events,
            iou_threshold=args.event_iou_threshold,
        )
        matched_pred = {pred_idx: gt_idx for gt_idx, pred_idx, _ in matches}
        true_activity = activity_target_label(
            str(record_data["record"]["activity"]),
            v4_cfg["activity"],
        )

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

        gt_by_record[int(record_id)] = gt_events
        pred_by_record[int(record_id)] = pred_events

    totals = {
        "true_events": 0,
        "predicted_events": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    matched_ious = []
    for record_id in record_ids:
        gt_events = gt_by_record[int(record_id)]
        pred_events = pred_by_record[int(record_id)]
        matches = match_events(
            gt_events,
            pred_events,
            iou_threshold=args.event_iou_threshold,
        )
        totals["true_events"] += len(gt_events)
        totals["predicted_events"] += len(pred_events)
        totals["true_positive"] += len(matches)
        totals["false_positive"] += len(pred_events) - len(matches)
        totals["false_negative"] += len(gt_events) - len(matches)
        matched_ious.extend(match[2] for match in matches)

    tp = totals["true_positive"]
    fp = totals["false_positive"]
    fn = totals["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    cough_metrics = {
        **totals,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "iou_threshold": float(args.event_iou_threshold),
        "threshold": float(args.threshold),
        "pred_span_mode": args.pred_span_mode,
        "pred_min_duration_sec": float(args.pred_min_duration_sec),
        "pred_merge_gap_sec": float(args.pred_merge_gap_sec),
        "gt_min_duration_sec": float(args.gt_min_duration_sec),
        "gt_merge_gap_sec": float(args.gt_merge_gap_sec),
    }

    activity_metrics = activity_reports(event_rows, v4_cfg["activity"]["classes"])
    report = {
        "pipeline": "v3_cough_v4_activity",
        "v3_checkpoint": str(v3_checkpoint_path),
        "v4_model_dir": str(v4_model_dir),
        "split": args.split,
        "record_ids": record_ids,
        "cough": cough_metrics,
        "activity_on_matched_v3_cough_events": activity_metrics,
    }

    output_dir = project_or_absolute(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "v3_cough_v4_activity_evaluation.json"
    events_path = output_dir / "v3_cough_v4_activity_events.csv"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_event_rows(events_path, event_rows)

    print(
        f"V3 cough event F1={cough_metrics['f1']:.3f} "
        f"P={cough_metrics['precision']:.3f} R={cough_metrics['recall']:.3f}"
    )
    print(
        "Activity accuracy on matched V3 cough events="
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
