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
from cough_analysis.event_metrics import match_events
from cough_analysis.models import V4ActivityCNN, V4CoughFrameCNN
from cough_analysis.paths import project_path
from cough_analysis.v4 import (
    V4ActivityWindowDataset,
    activity_target_label,
    assign_activity_to_event,
    cough_gt_events,
    event_summary,
    frame_predictions_to_events,
    predict_activity_probabilities_for_record,
    predict_cough_probabilities_for_record,
    resolve_device,
)
from cough_analysis.preprocessing import load_record_preprocessed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V4 cough-event and activity models.")
    parser.add_argument("--model-dir", default="artifacts/models/v4")
    parser.add_argument("--config", default="configs/v4.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", default="artifacts/evaluations/v4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_path(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def load_v4_models(model_dir: Path, summary: dict, cfg: dict, device: torch.device):
    spec_name = summary["primary_cough_spec"]
    cough_ckpt = load_checkpoint(model_dir / f"v4_cough_{spec_name}.pt", device)
    cough_model = V4CoughFrameCNN().to(device)
    cough_model.load_state_dict(cough_ckpt["model_state_dict"])
    cough_model.eval()

    activity_ckpt = load_checkpoint(model_dir / "v4_activity.pt", device)
    activity_model = V4ActivityCNN(num_classes=len(cfg["activity"]["classes"])).to(device)
    activity_model.load_state_dict(activity_ckpt["model_state_dict"])
    activity_model.eval()
    return spec_name, cough_ckpt, cough_model, activity_model


def evaluate_cough_split(
    cfg: dict,
    metadata,
    record_ids: list[int],
    spec_cfg: dict,
    post_cfg: dict,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, dict[int, list], dict[int, list], dict[int, np.ndarray]]:
    data_root = project_or_absolute(cfg["data"]["data_root"])
    frame_rate = int(round(int(cfg["sampling"]["audio_hz"]) / int(cfg["cough"]["frame_hop_samples"])))
    gt_by_record = {}
    pred_by_record = {}
    probabilities = {}

    for record_id in record_ids:
        record = load_record_preprocessed(int(record_id), metadata=metadata, data_root=data_root)
        probs = predict_cough_probabilities_for_record(
            model,
            record,
            cfg["cough"],
            spec_cfg,
            device=device,
            batch_size=batch_size,
        )
        gt_by_record[int(record_id)] = cough_gt_events(record, cfg["cough"])
        pred_by_record[int(record_id)] = frame_predictions_to_events(
            probs,
            frame_rate=frame_rate,
            threshold=float(post_cfg["threshold"]),
            min_duration_sec=float(post_cfg["pred_min_duration_sec"]),
            merge_gap_sec=float(post_cfg["pred_merge_gap_sec"]),
            duration_sec=float(record["duration_sec"]),
        )
        probabilities[int(record_id)] = probs

    metrics = event_summary(
        gt_by_record,
        pred_by_record,
        iou_threshold=float(cfg["cough"]["postprocessing"]["event_iou_threshold"]),
    )
    metrics["threshold"] = float(post_cfg["threshold"])
    metrics["pred_min_duration_sec"] = float(post_cfg["pred_min_duration_sec"])
    metrics["pred_merge_gap_sec"] = float(post_cfg["pred_merge_gap_sec"])
    return metrics, gt_by_record, pred_by_record, probabilities


def evaluate_activity_split(
    cfg: dict,
    metadata,
    record_ids: list[int],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    data_root = project_or_absolute(cfg["data"]["data_root"])
    dataset = V4ActivityWindowDataset(
        record_ids,
        metadata,
        cfg["activity"],
        data_root=data_root,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels = []
    preds = []

    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            logits = model(motion)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())

    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    report = classification_report(
        labels_np,
        preds_np,
        labels=list(range(len(cfg["activity"]["classes"]))),
        target_names=cfg["activity"]["classes"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        labels_np,
        preds_np,
        labels=list(range(len(cfg["activity"]["classes"]))),
    )
    return {"classification_report": report, "confusion_matrix": cm.tolist()}, labels_np, preds_np


def evaluate_end_to_end(
    cfg: dict,
    metadata,
    record_ids: list[int],
    gt_by_record: dict[int, list],
    pred_by_record: dict[int, list],
    activity_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    data_root = project_or_absolute(cfg["data"]["data_root"])
    classes = cfg["activity"]["classes"]
    context_sec = float(cfg["activity"]["attribution_context_sec"])
    iou_threshold = float(cfg["cough"]["postprocessing"]["event_iou_threshold"])

    rows = []
    correct = 0
    matched = 0

    for record_id in record_ids:
        row = metadata.loc[metadata["record_id"] == int(record_id)].iloc[0]
        true_activity = activity_target_label(str(row["activity"]), cfg["activity"])
        record = load_record_preprocessed(int(record_id), metadata=metadata, data_root=data_root)
        centers, activity_probs = predict_activity_probabilities_for_record(
            activity_model,
            record,
            cfg["activity"],
            device=device,
            batch_size=batch_size,
        )
        matches = match_events(
            gt_by_record[int(record_id)],
            pred_by_record[int(record_id)],
            iou_threshold=iou_threshold,
        )
        pred_match_map = {pred_idx: gt_idx for gt_idx, pred_idx, _ in matches}
        for pred_idx, event in enumerate(pred_by_record[int(record_id)]):
            assigned = assign_activity_to_event(
                event,
                centers,
                activity_probs,
                classes,
                context_sec=context_sec,
            )
            is_matched = pred_idx in pred_match_map
            activity_correct = assigned["activity"] == true_activity
            if is_matched:
                matched += 1
                correct += int(activity_correct)
            rows.append(
                {
                    "record_id": int(record_id),
                    "cough_start": float(event.start),
                    "cough_end": float(event.end),
                    "matched_gt": bool(is_matched),
                    "true_activity": true_activity,
                    **assigned,
                    "activity_correct_if_matched": bool(is_matched and activity_correct),
                }
            )

    return {
        "matched_cough_events": int(matched),
        "matched_with_correct_activity": int(correct),
        "matched_activity_accuracy": float(correct / matched) if matched else 0.0,
    }, rows


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
    cfg = load_config(args.config)
    model_dir = project_or_absolute(args.model_dir)
    summary = json.loads((model_dir / "v4_summary.json").read_text(encoding="utf-8"))
    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    record_ids = [int(x) for x in summary["record_split"][args.split]]
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])

    spec_name, cough_ckpt, cough_model, activity_model = load_v4_models(
        model_dir,
        summary,
        cfg,
        device,
    )
    cough_metrics, gt_by_record, pred_by_record, _ = evaluate_cough_split(
        cfg,
        metadata,
        record_ids,
        spec_cfg=cough_ckpt["spec_config"],
        post_cfg=cough_ckpt["selected_postprocessing"],
        model=cough_model,
        device=device,
        batch_size=batch_size,
    )
    activity_metrics, _, _ = evaluate_activity_split(
        cfg,
        metadata,
        record_ids,
        activity_model,
        device,
        batch_size=batch_size,
    )
    end_to_end_metrics, event_rows = evaluate_end_to_end(
        cfg,
        metadata,
        record_ids,
        gt_by_record,
        pred_by_record,
        activity_model,
        device,
        batch_size=batch_size,
    )

    output_dir = project_or_absolute(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id": "v4",
        "split": args.split,
        "record_ids": record_ids,
        "primary_cough_spec": spec_name,
        "cough": cough_metrics,
        "activity": activity_metrics,
        "end_to_end": end_to_end_metrics,
    }
    report_path = output_dir / "v4_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_event_rows(output_dir / "v4_event_activity_predictions.csv", event_rows)

    print(
        f"Cough event F1={cough_metrics['f1']:.3f} "
        f"IoU={cough_metrics['mean_matched_iou']:.3f}"
    )
    print(
        "End-to-end matched activity accuracy="
        f"{end_to_end_metrics['matched_activity_accuracy']:.3f}"
    )
    print(f"Saved evaluation: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
