from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
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
from cough_analysis.input_ablation import INPUT_ABLATION_MODES, apply_input_ablation
from cough_analysis.models import Spec2DCoughCNN
from cough_analysis.paths import project_path
from cough_analysis.v3 import (
    SpectrogramDataset,
    build_dataset,
    build_record_dataset,
    resolve_device,
    split_records_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--record-ids",
        default=None,
        help="Explicit record ids or ranges, e.g. '85-95' or '87,94'. Overrides --split.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event-iou-threshold", type=float, default=0.2)
    parser.add_argument("--event-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--gt-min-duration-sec", type=float, default=0.0)
    parser.add_argument("--gt-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--pred-min-duration-sec", type=float, default=0.0)
    parser.add_argument("--pred-merge-gap-sec", type=float, default=None)
    parser.add_argument(
        "--pred-span-mode",
        choices=["full", "center", "hop"],
        default="full",
    )
    parser.add_argument("--pred-center-fraction", type=float, default=None)
    parser.add_argument("--prob-smoothing-sec", type=float, default=0.0)
    parser.add_argument("--hysteresis-low-threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Evaluation output directory. If omitted, a descriptive path is "
            "created under artifacts/evaluations based on the checkpoint, "
            "record count, split, threshold, and event settings."
        ),
    )
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="v3_evaluation")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument(
        "--input-ablation",
        choices=INPUT_ABLATION_MODES,
        default=None,
        help="Input group to use. Defaults to the checkpoint ablation mode, then full.",
    )
    return parser.parse_args()


def project_or_absolute(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(project_or_absolute(path), map_location=device)


def parse_record_ids(value: str) -> list[int]:
    record_ids: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", maxsplit=1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                raise ValueError(f"Invalid record id range: {token}")
            record_ids.extend(range(start, end + 1))
        else:
            record_ids.append(int(token))
    return record_ids


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


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-Cough", "Cough"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-Cough", "Cough"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("V3 Cough Detection Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> dict:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    average_precision = average_precision_score(labels, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP={average_precision:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "average_precision": float(average_precision),
        "points": [
            {
                "precision": float(p),
                "recall": float(r),
                "threshold": (
                    ""
                    if idx >= len(thresholds)
                    else float(thresholds[idx])
                ),
            }
            for idx, (p, r) in enumerate(zip(precision, recall))
        ],
    }


def predict_arrays(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    input_ablation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"].to(device)
            motion = batch["motion"].to(device)
            spec, motion = apply_input_ablation(spec, motion, mode=input_ablation)
            batch_labels = batch["label"].int().cpu().numpy()
            logits = model(spec, motion)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.tolist())
    return np.asarray(all_labels), np.asarray(all_preds), np.asarray(all_probs)


def effective_pred_merge_gap(args: argparse.Namespace) -> float:
    return (
        args.event_merge_gap_sec
        if args.pred_merge_gap_sec is None
        else args.pred_merge_gap_sec
    )


def token_float(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p").replace("-", "m")


def checkpoint_label(checkpoint_path: str) -> str:
    return Path(checkpoint_path).name.replace(".", "_")


def event_settings_label(
    args: argparse.Namespace,
    pred_merge_gap_sec: float,
) -> str:
    uses_default_event_processing = (
        args.gt_min_duration_sec == 0.0
        and args.gt_merge_gap_sec == 0.0
        and args.pred_min_duration_sec == 0.0
        and pred_merge_gap_sec == 0.0
        and args.pred_center_fraction is None
        and args.prob_smoothing_sec == 0.0
        and args.hysteresis_low_threshold is None
    )
    if args.pred_span_mode == "full" and uses_default_event_processing:
        label = "baseline_full_eval"
    elif args.pred_span_mode == "full":
        label = "postprocessed_eval"
    else:
        label = f"{args.pred_span_mode}_eval"

    # Keep common V3 threshold folders readable. Non-standard thresholds are
    # still made visible to avoid accidental overwrites during experiments.
    if abs(float(args.threshold) - 0.6) > 1.0e-12:
        label = f"{label}_t{token_float(args.threshold)}"
    if args.prob_smoothing_sec > 0.0:
        label = f"{label}_smooth{token_float(args.prob_smoothing_sec)}"
    if args.hysteresis_low_threshold is not None:
        label = f"{label}_hyst{token_float(args.hysteresis_low_threshold)}"
    return label


def default_output_dir(
    args: argparse.Namespace,
    record_ids,
    pred_merge_gap_sec: float,
) -> Path:
    dataset_label = f"{args.split}_split_{len(record_ids):03d}_records"
    return project_path(
        Path("artifacts")
        / "evaluations"
        / checkpoint_label(args.checkpoint)
        / dataset_label
        / event_settings_label(args, pred_merge_gap_sec)
    )


def log_to_mlflow(
    args: argparse.Namespace,
    cfg: dict,
    report: dict,
    pr_data: dict,
    event_summary: dict,
    record_ids,
    output_paths: list[Path],
    batch_size: int,
    device: torch.device,
    input_ablation: str,
) -> None:
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

    run_name = args.mlflow_run_name or f"evaluate_v3_{args.split}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "script": "evaluate_v3",
                "checkpoint": args.checkpoint,
                "config": args.config,
                "model_id": args.model_id or "",
                "input_ablation": input_ablation,
                "split": args.split,
                "record_ids": args.record_ids or "",
                "threshold": args.threshold,
                "event_iou_threshold": args.event_iou_threshold,
                "event_merge_gap_sec": args.event_merge_gap_sec,
                "gt_min_duration_sec": args.gt_min_duration_sec,
                "gt_merge_gap_sec": args.gt_merge_gap_sec,
                "pred_min_duration_sec": args.pred_min_duration_sec,
                "pred_merge_gap_sec": effective_pred_merge_gap(args),
                "pred_span_mode": args.pred_span_mode,
                "pred_center_fraction": (
                    float(cfg["windowing"]["center_fraction"])
                    if args.pred_center_fraction is None
                    else args.pred_center_fraction
                ),
                "prob_smoothing_sec": args.prob_smoothing_sec,
                "hysteresis_low_threshold": (
                    "" if args.hysteresis_low_threshold is None else args.hysteresis_low_threshold
                ),
                "batch_size": batch_size,
                "device": str(device),
                "record_count": len(record_ids),
                "window_sec": float(cfg["windowing"]["window_sec"]),
                "hop_sec": float(cfg["windowing"]["hop_sec"]),
                "center_fraction": float(cfg["windowing"]["center_fraction"]),
                "n_mels": int(cfg["spectrogram"]["n_mels"]),
                "n_fft": int(cfg["spectrogram"]["n_fft"]),
                "mel_hop_length": int(cfg["spectrogram"]["hop_length"]),
                **manifest_mlflow_params(args.dataset_manifest),
            }
        )
        mlflow.log_metrics(
            {
                "window_accuracy": float(report["accuracy"]),
                "window_cough_precision": float(report["Cough"]["precision"]),
                "window_cough_recall": float(report["Cough"]["recall"]),
                "window_cough_f1": float(report["Cough"]["f1-score"]),
                "window_macro_f1": float(report["macro avg"]["f1-score"]),
                "window_weighted_f1": float(report["weighted avg"]["f1-score"]),
                "window_average_precision": float(pr_data["average_precision"]),
                "event_precision": float(event_summary["precision"]),
                "event_recall": float(event_summary["recall"]),
                "event_f1": float(event_summary["f1"]),
                "event_true_events": float(event_summary["true_events"]),
                "event_predicted_events": float(event_summary["predicted_events"]),
                "event_tp": float(event_summary["true_positive"]),
                "event_fp": float(event_summary["false_positive"]),
                "event_fn": float(event_summary["false_negative"]),
            }
        )
        for path in output_paths:
            mlflow.log_artifact(str(path), artifact_path="evaluation")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    pred_merge_gap_sec = effective_pred_merge_gap(args)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    cfg = checkpoint.get("config") or load_config(args.config)
    input_ablation = args.input_ablation or checkpoint.get("input_ablation", "full")

    split_map = checkpoint.get("record_split")
    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    data_root = project_or_absolute(cfg["data"].get("data_root"))
    if args.record_ids:
        record_ids = parse_record_ids(args.record_ids)
    elif split_map and args.split in split_map:
        record_ids = split_map[args.split]
    else:
        _, val_ids, test_ids = split_records_from_config(metadata, cfg.get("split"))
        record_ids = val_ids if args.split == "val" else test_ids

    window_cfg = cfg["windowing"]
    pred_center_fraction = (
        float(window_cfg["center_fraction"])
        if args.pred_center_fraction is None
        else args.pred_center_fraction
    )
    spec_cfg = cfg["spectrogram"]
    X_spec, X_motion, labels = build_dataset(
        record_ids,
        metadata,
        data_root=data_root,
        window_sec=float(window_cfg["window_sec"]),
        hop_sec=float(window_cfg["hop_sec"]),
        center_fraction=float(window_cfg["center_fraction"]),
        spectrogram_config=spec_cfg,
    )
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    loader = DataLoader(
        SpectrogramDataset(X_spec, X_motion, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    model = Spec2DCoughCNN(num_classes=1).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    labels_np, preds_np, probs_np = predict_arrays(
        model,
        loader,
        device=device,
        threshold=args.threshold,
        input_ablation=input_ablation,
    )
    report = classification_report(
        labels_np,
        preds_np,
        target_names=["Non-Cough", "Cough"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])

    output_dir = (
        project_or_absolute(args.output_dir)
        if args.output_dir
        else default_output_dir(args, record_ids, pred_merge_gap_sec)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.split}_classification_report.json"
    cm_path = output_dir / f"{args.split}_confusion_matrix.png"
    pr_curve_path = output_dir / f"{args.split}_precision_recall_curve.png"
    pr_data_path = output_dir / f"{args.split}_precision_recall_curve.json"
    preds_path = output_dir / f"{args.split}_predictions.csv"
    events_path = output_dir / f"{args.split}_event_metrics.json"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix(cm, cm_path)
    pr_data = save_precision_recall_curve(labels_np, probs_np, pr_curve_path)
    pr_data_path.write_text(json.dumps(pr_data, indent=2), encoding="utf-8")
    with preds_path.open("w", encoding="utf-8") as f:
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
    per_record_events = []
    for record_id in record_ids:
        record_data = build_record_dataset(
            int(record_id),
            metadata,
            data_root=data_root,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
            spectrogram_config=spec_cfg,
        )
        record_loader = DataLoader(
            SpectrogramDataset(
                record_data["spec"],
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
            input_ablation=input_ablation,
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
        record_event_preds = probabilities_to_predictions(
            event_probs,
            threshold=args.threshold,
            hysteresis_low_threshold=args.hysteresis_low_threshold,
        )
        pred_events = window_predictions_to_events(
            record_data["spans"],
            record_event_preds,
            min_duration_sec=args.pred_min_duration_sec,
            merge_gap_sec=pred_merge_gap_sec,
            span_mode=args.pred_span_mode,
            center_fraction=pred_center_fraction,
        )
        record_metrics = event_level_metrics(
            gt_events,
            pred_events,
            iou_threshold=args.event_iou_threshold,
        )
        for key in total_event_counts:
            total_event_counts[key] += int(record_metrics[key])
        per_record_events.append(
            {
                "record_id": int(record_id),
                **record_metrics,
            }
        )

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
        "threshold": args.threshold,
        "iou_threshold": args.event_iou_threshold,
        "gt_min_duration_sec": args.gt_min_duration_sec,
        "gt_merge_gap_sec": args.gt_merge_gap_sec,
        "pred_min_duration_sec": args.pred_min_duration_sec,
        "pred_merge_gap_sec": pred_merge_gap_sec,
        "pred_span_mode": args.pred_span_mode,
        "pred_center_fraction": pred_center_fraction,
        "prob_smoothing_sec": args.prob_smoothing_sec,
        "hysteresis_low_threshold": args.hysteresis_low_threshold,
        "input_ablation": input_ablation,
        "per_record": per_record_events,
    }
    events_path.write_text(json.dumps(event_summary, indent=2), encoding="utf-8")

    if args.mlflow:
        log_to_mlflow(
            args=args,
            cfg=cfg,
            report=report,
            pr_data=pr_data,
            event_summary=event_summary,
            record_ids=record_ids,
            output_paths=[
                report_path,
                cm_path,
                pr_curve_path,
                pr_data_path,
                preds_path,
                events_path,
            ],
            batch_size=batch_size,
            device=device,
            input_ablation=input_ablation,
        )

    print(classification_report(labels_np, preds_np, target_names=["Non-Cough", "Cough"]))
    print(
        "Event-level: "
        f"P={event_precision:.3f} R={event_recall:.3f} F1={event_f1:.3f} "
        f"TP={tp} FP={fp} FN={fn}"
    )
    print(f"Input ablation: {input_ablation}")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved precision-recall curve: {pr_curve_path}")
    print(f"Saved predictions: {preds_path}")
    print(f"Saved event metrics: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
