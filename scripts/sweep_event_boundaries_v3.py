from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import numpy as np
import torch
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
from cough_analysis.input_ablation import INPUT_ABLATION_MODES, apply_input_ablation
from cough_analysis.models import Spec2DCoughCNN
from cough_analysis.paths import project_path
from cough_analysis.v3 import (
    SpectrogramDataset,
    build_record_dataset,
    resolve_device,
    split_records_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--thresholds", default="0.4,0.5,0.6,0.7")
    parser.add_argument("--span-modes", default="full,hop,center")
    parser.add_argument("--event-iou-threshold", type=float, default=0.2)
    parser.add_argument("--gt-min-duration-sec", type=float, default=0.1)
    parser.add_argument("--gt-merge-gap-sec", type=float, default=0.1)
    parser.add_argument("--pred-min-duration-sec", type=float, default=0.1)
    parser.add_argument("--pred-min-duration-secs", default=None)
    parser.add_argument("--pred-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--pred-merge-gap-secs", default=None)
    parser.add_argument("--pred-center-fraction", type=float, default=None)
    parser.add_argument("--smoothing-secs", default="0.0")
    parser.add_argument(
        "--hysteresis-low-thresholds",
        default=None,
        help=(
            "Comma-separated low thresholds for hysteresis event construction. "
            "When omitted, plain single-threshold predictions are used."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--input-ablation",
        choices=INPUT_ABLATION_MODES,
        default=None,
        help="Input group to use. Defaults to the checkpoint ablation mode, then full.",
    )
    parser.add_argument(
        "--record-ids",
        nargs="+",
        type=int,
        default=None,
        help="Evaluate specific record ids instead of the configured split.",
    )
    parser.add_argument("--output-csv", default="artifacts/error_analysis/v3/boundary_sweep.csv")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_float_grid(value: str | None, fallback: float) -> list[float]:
    return parse_float_list(value) if value is not None else [float(fallback)]


def parse_optional_float_grid(value: str | None) -> list[float | None]:
    return [None] if value is None else parse_float_list(value)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(project_or_absolute(path), map_location=device)


def predict_record_probs(
    model: torch.nn.Module,
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


def boundary_error_summary(
    gt_events,
    pred_events,
    matches,
) -> dict:
    if not matches:
        return {
            "mean_matched_iou": 0.0,
            "mean_start_error_sec": 0.0,
            "mean_end_error_sec": 0.0,
            "mean_duration_ratio": 0.0,
        }

    ious = []
    start_errors = []
    end_errors = []
    duration_ratios = []
    for gt_idx, pred_idx, iou in matches:
        gt_event = gt_events[gt_idx]
        pred_event = pred_events[pred_idx]
        ious.append(float(iou))
        start_errors.append(abs(pred_event.start - gt_event.start))
        end_errors.append(abs(pred_event.end - gt_event.end))
        duration_ratios.append(
            pred_event.duration / gt_event.duration
            if gt_event.duration > 0
            else 0.0
        )

    return {
        "mean_matched_iou": float(np.mean(ious)),
        "mean_start_error_sec": float(np.mean(start_errors)),
        "mean_end_error_sec": float(np.mean(end_errors)),
        "mean_duration_ratio": float(np.mean(duration_ratios)),
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    cfg = checkpoint.get("config") or load_config(args.config)
    input_ablation = args.input_ablation or checkpoint.get("input_ablation", "full")
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    thresholds = parse_float_list(args.thresholds)
    span_modes = parse_str_list(args.span_modes)
    pred_min_duration_secs = resolve_float_grid(
        args.pred_min_duration_secs,
        args.pred_min_duration_sec,
    )
    pred_merge_gap_secs = resolve_float_grid(
        args.pred_merge_gap_secs,
        args.pred_merge_gap_sec,
    )
    smoothing_secs = parse_float_list(args.smoothing_secs)
    hysteresis_low_thresholds = parse_optional_float_grid(args.hysteresis_low_thresholds)

    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    data_root = project_or_absolute(cfg["data"].get("data_root"))
    split_map = checkpoint.get("record_split")
    if args.record_ids is not None:
        record_ids = [int(x) for x in args.record_ids]
        split_label = "record_ids"
    elif split_map and args.split in split_map:
        record_ids = [int(x) for x in split_map[args.split]]
        split_label = args.split
    else:
        _, val_ids, test_ids = split_records_from_config(metadata, cfg.get("split"))
        selected = val_ids if args.split == "val" else test_ids
        record_ids = [int(x) for x in selected]
        split_label = args.split

    model = Spec2DCoughCNN(num_classes=1).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    window_cfg = cfg["windowing"]
    spec_cfg = cfg["spectrogram"]
    pred_center_fraction = (
        float(window_cfg["center_fraction"])
        if args.pred_center_fraction is None
        else args.pred_center_fraction
    )

    records = []
    for record_id in record_ids:
        record_data = build_record_dataset(
            record_id,
            metadata,
            data_root=data_root,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
            spectrogram_config=spec_cfg,
        )
        probs = predict_record_probs(
            model,
            record_data,
            batch_size=batch_size,
            device=device,
            input_ablation=input_ablation,
        )
        gt_events = binary_labels_to_events(
            record_data["record"]["cough_label"],
            sample_rate=int(record_data["record"]["fs_audio"]),
            min_duration_sec=args.gt_min_duration_sec,
            merge_gap_sec=args.gt_merge_gap_sec,
        )
        records.append((record_data, probs, gt_events))

    rows = []
    for threshold in thresholds:
        for span_mode in span_modes:
            for pred_min_duration_sec in pred_min_duration_secs:
                for pred_merge_gap_sec in pred_merge_gap_secs:
                    for smoothing_sec in smoothing_secs:
                        for hysteresis_low_threshold in hysteresis_low_thresholds:
                            if (
                                hysteresis_low_threshold is not None
                                and hysteresis_low_threshold > threshold
                            ):
                                continue

                            tp = fp = fn = 0
                            matched_iou_sum = 0.0
                            start_error_sum = 0.0
                            end_error_sum = 0.0
                            duration_ratio_sum = 0.0
                            matched_count = 0

                            for record_data, probs, gt_events in records:
                                event_probs = smooth_probabilities(
                                    probs,
                                    record_data["spans"],
                                    smoothing_sec=smoothing_sec,
                                )
                                preds = probabilities_to_predictions(
                                    event_probs,
                                    threshold=threshold,
                                    hysteresis_low_threshold=hysteresis_low_threshold,
                                )
                                pred_events = window_predictions_to_events(
                                    record_data["spans"],
                                    preds,
                                    min_duration_sec=pred_min_duration_sec,
                                    merge_gap_sec=pred_merge_gap_sec,
                                    span_mode=span_mode,
                                    center_fraction=pred_center_fraction,
                                )
                                matches = match_events(
                                    gt_events,
                                    pred_events,
                                    iou_threshold=args.event_iou_threshold,
                                )
                                stats = boundary_error_summary(gt_events, pred_events, matches)
                                tp += len(matches)
                                fp += len(pred_events) - len(matches)
                                fn += len(gt_events) - len(matches)
                                matched_iou_sum += stats["mean_matched_iou"] * len(matches)
                                start_error_sum += stats["mean_start_error_sec"] * len(matches)
                                end_error_sum += stats["mean_end_error_sec"] * len(matches)
                                duration_ratio_sum += stats["mean_duration_ratio"] * len(matches)
                                matched_count += len(matches)

                            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                            f1 = (
                                2 * precision * recall / (precision + recall)
                                if precision + recall > 0
                                else 0.0
                            )
                            rows.append(
                                {
                                    "split": split_label,
                                    "input_ablation": input_ablation,
                                    "threshold": threshold,
                                    "hysteresis_low_threshold": (
                                        ""
                                        if hysteresis_low_threshold is None
                                        else hysteresis_low_threshold
                                    ),
                                    "pred_span_mode": span_mode,
                                    "pred_min_duration_sec": pred_min_duration_sec,
                                    "pred_merge_gap_sec": pred_merge_gap_sec,
                                    "smoothing_sec": smoothing_sec,
                                    "tp": tp,
                                    "fp": fp,
                                    "fn": fn,
                                    "precision": precision,
                                    "recall": recall,
                                    "f1": f1,
                                    "mean_matched_iou": (
                                        matched_iou_sum / matched_count if matched_count else 0.0
                                    ),
                                    "mean_start_error_sec": (
                                        start_error_sum / matched_count if matched_count else 0.0
                                    ),
                                    "mean_end_error_sec": (
                                        end_error_sum / matched_count if matched_count else 0.0
                                    ),
                                    "mean_duration_ratio": (
                                        duration_ratio_sum / matched_count if matched_count else 0.0
                                    ),
                                }
                            )

    rows.sort(
        key=lambda row: (
            -row["f1"],
            -row["mean_matched_iou"],
            abs(row["mean_duration_ratio"] - 1.0),
            row["threshold"],
        )
    )
    output_csv = project_or_absolute(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved sweep: {output_csv}")
    for row in rows[:10]:
        print(
            f"{row['split']} thr={row['threshold']:.2f} mode={row['pred_span_mode']} "
            f"input={row['input_ablation']} "
            f"F1={row['f1']:.3f} P={row['precision']:.3f} R={row['recall']:.3f} "
            f"IoU={row['mean_matched_iou']:.3f} dur_ratio={row['mean_duration_ratio']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
