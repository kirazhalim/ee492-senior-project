from __future__ import annotations

import argparse
import csv
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
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.event_metrics import (
    Event,
    binary_labels_to_events,
    event_iou,
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
    parser.add_argument("--split", choices=["val", "test"], default="test")
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
        help="Analyze specific record ids instead of the configured split.",
    )
    parser.add_argument(
        "--possible-missing-label-record-ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Record ids where an unmatched prediction should be marked as "
            "possible_missing_label instead of hard_negative_fp."
        ),
    )
    parser.add_argument("--plot-records", choices=["problem", "all", "none"], default="problem")
    parser.add_argument("--output-dir", default="artifacts/error_analysis/v3")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(project_or_absolute(path), map_location=device)


def effective_pred_merge_gap(args: argparse.Namespace) -> float:
    return (
        args.event_merge_gap_sec
        if args.pred_merge_gap_sec is None
        else args.pred_merge_gap_sec
    )


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


def overlapping_prob_stats(
    event: Event,
    spans: list[tuple[float, float]],
    probs: np.ndarray,
) -> tuple[float, float]:
    selected = [
        float(prob)
        for (start, end), prob in zip(spans, probs)
        if max(float(start), event.start) < min(float(end), event.end)
    ]
    if not selected:
        return 0.0, 0.0
    return float(np.max(selected)), float(np.mean(selected))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def optional_fmt_float(value: float | None, digits: int = 3) -> str:
    return "" if value is None else fmt_float(value, digits=digits)


def event_gap(a: Event, b: Event) -> float:
    if max(a.start, b.start) < min(a.end, b.end):
        return 0.0
    return min(abs(a.start - b.end), abs(b.start - a.end))


def nearest_event(
    target: Event,
    candidates: list[Event],
) -> tuple[Event | None, float | None, float | None]:
    if not candidates:
        return None, None, None
    scored = [
        (candidate, event_iou(target, candidate), event_gap(target, candidate))
        for candidate in candidates
    ]
    scored.sort(key=lambda item: (-item[1], item[2]))
    event, iou, gap = scored[0]
    return event, float(iou), float(gap)


def classify_fn_error(
    gt_event: Event,
    pred_events: list[Event],
    max_probability: float,
    threshold: float,
    event_iou_threshold: float,
) -> tuple[str, Event | None, float | None, float | None, str]:
    nearest_pred, nearest_iou, nearest_gap = nearest_event(gt_event, pred_events)
    has_near_prediction = (
        nearest_pred is not None
        and (
            (nearest_iou or 0.0) > 0.0
            or (nearest_gap is not None and nearest_gap <= 0.3)
        )
    )
    if max_probability >= threshold or has_near_prediction:
        note = (
            "Model scored this region, but event conversion/matching did not "
            f"produce a one-to-one event match at IoU >= {event_iou_threshold:.2f}."
        )
        return "boundary_mismatch_fn", nearest_pred, nearest_iou, nearest_gap, note

    return (
        "score_miss_fn",
        nearest_pred,
        nearest_iou,
        nearest_gap,
        "Model probability stayed below threshold around the GT event.",
    )


def classify_fp_error(
    pred_event: Event,
    gt_events: list[Event],
    max_probability: float,
    record_id: int,
    possible_missing_label_record_ids: set[int],
    threshold: float,
    event_iou_threshold: float,
) -> tuple[str, Event | None, float | None, float | None, str]:
    nearest_gt, nearest_iou, nearest_gap = nearest_event(pred_event, gt_events)
    if record_id in possible_missing_label_record_ids:
        return (
            "possible_missing_label_fp",
            nearest_gt,
            nearest_iou,
            nearest_gap,
            "Manual review note: this unmatched prediction may be an unlabeled cough.",
        )

    if (nearest_iou or 0.0) > 0.0 or (
        nearest_gap is not None and nearest_gap <= 0.3
    ):
        note = (
            "Prediction is close to or partially overlaps GT, but did not "
            f"receive an IoU >= {event_iou_threshold:.2f} match."
        )
        return "boundary_mismatch_fp", nearest_gt, nearest_iou, nearest_gap, note

    if max_probability >= threshold:
        return (
            "hard_negative_fp",
            nearest_gt,
            nearest_iou,
            nearest_gap,
            "High-confidence cough-like prediction away from annotated GT.",
        )

    return (
        "low_confidence_fp",
        nearest_gt,
        nearest_iou,
        nearest_gap,
        "Unmatched prediction with weak mean/peak confidence.",
    )


def classify_events(
    record_data: dict,
    probs: np.ndarray,
    threshold: float,
    event_iou_threshold: float,
    gt_min_duration_sec: float,
    gt_merge_gap_sec: float,
    pred_min_duration_sec: float,
    pred_merge_gap_sec: float,
    pred_span_mode: str,
    pred_center_fraction: float,
    prob_smoothing_sec: float,
    hysteresis_low_threshold: float | None,
) -> tuple[list[Event], list[Event], list[tuple[int, int, float]], list[int], list[int]]:
    gt_events = binary_labels_to_events(
        record_data["record"]["cough_label"],
        sample_rate=int(record_data["record"]["fs_audio"]),
        min_duration_sec=gt_min_duration_sec,
        merge_gap_sec=gt_merge_gap_sec,
    )
    event_probs = smooth_probabilities(
        probs,
        record_data["spans"],
        smoothing_sec=prob_smoothing_sec,
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
        span_mode=pred_span_mode,
        center_fraction=pred_center_fraction,
    )
    matches = match_events(
        gt_events,
        pred_events,
        iou_threshold=event_iou_threshold,
    )
    matched_gt = {gt_idx for gt_idx, _, _ in matches}
    matched_pred = {pred_idx for _, pred_idx, _ in matches}
    fn_indices = [idx for idx in range(len(gt_events)) if idx not in matched_gt]
    fp_indices = [idx for idx in range(len(pred_events)) if idx not in matched_pred]
    return gt_events, pred_events, matches, fn_indices, fp_indices


def add_event_error_rows(
    rows: list[dict],
    record_data: dict,
    gt_events: list[Event],
    pred_events: list[Event],
    matches: list[tuple[int, int, float]],
    fn_indices: list[int],
    fp_indices: list[int],
    probs: np.ndarray,
    threshold: float,
    event_iou_threshold: float,
    possible_missing_label_record_ids: set[int],
) -> None:
    record = record_data["record"]
    record_id = int(record["record_id"])
    base = {
        "record_id": record_id,
        "filename": record["filename"],
        "activity": record["activity"],
        "context": record["context"],
    }
    for gt_idx in fn_indices:
        event = gt_events[gt_idx]
        max_prob, mean_prob = overlapping_prob_stats(
            event,
            record_data["spans"],
            probs,
        )
        category, nearest_pred, nearest_iou, nearest_gap, note = classify_fn_error(
            event,
            pred_events,
            max_probability=max_prob,
            threshold=threshold,
            event_iou_threshold=event_iou_threshold,
        )
        rows.append(
            {
                **base,
                "error_type": "FN",
                "error_category": category,
                "gt_start_sec": fmt_float(event.start),
                "gt_end_sec": fmt_float(event.end),
                "pred_start_sec": "",
                "pred_end_sec": "",
                "duration_sec": fmt_float(event.duration),
                "matched_iou": "",
                "nearest_event_start_sec": optional_fmt_float(
                    nearest_pred.start if nearest_pred else None
                ),
                "nearest_event_end_sec": optional_fmt_float(
                    nearest_pred.end if nearest_pred else None
                ),
                "nearest_event_iou": optional_fmt_float(nearest_iou),
                "nearest_event_gap_sec": optional_fmt_float(nearest_gap),
                "max_probability": fmt_float(max_prob),
                "mean_probability": fmt_float(mean_prob),
                "review_note": note,
            }
        )

    for pred_idx in fp_indices:
        event = pred_events[pred_idx]
        max_prob, mean_prob = overlapping_prob_stats(
            event,
            record_data["spans"],
            probs,
        )
        category, nearest_gt, nearest_iou, nearest_gap, note = classify_fp_error(
            event,
            gt_events,
            max_probability=max_prob,
            record_id=record_id,
            possible_missing_label_record_ids=possible_missing_label_record_ids,
            threshold=threshold,
            event_iou_threshold=event_iou_threshold,
        )
        rows.append(
            {
                **base,
                "error_type": "FP",
                "error_category": category,
                "gt_start_sec": "",
                "gt_end_sec": "",
                "pred_start_sec": fmt_float(event.start),
                "pred_end_sec": fmt_float(event.end),
                "duration_sec": fmt_float(event.duration),
                "matched_iou": "",
                "nearest_event_start_sec": optional_fmt_float(
                    nearest_gt.start if nearest_gt else None
                ),
                "nearest_event_end_sec": optional_fmt_float(
                    nearest_gt.end if nearest_gt else None
                ),
                "nearest_event_iou": optional_fmt_float(nearest_iou),
                "nearest_event_gap_sec": optional_fmt_float(nearest_gap),
                "max_probability": fmt_float(max_prob),
                "mean_probability": fmt_float(mean_prob),
                "review_note": note,
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["record_id"]),
            float(row["gt_start_sec"] or row["pred_start_sec"]),
            row["error_type"],
        )
    )


def event_spans_from_indices(events: list[Event], indices: list[int]) -> list[Event]:
    return [events[idx] for idx in indices]


def plot_event_bars(
    ax,
    events: list[Event],
    color: str,
    label: str,
    alpha: float = 0.8,
) -> None:
    for event in events:
        ax.broken_barh(
            [(event.start, event.duration)],
            (0.2, 0.6),
            facecolors=color,
            alpha=alpha,
            edgecolors=color,
            linewidth=1.4,
        )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([label])
    ax.set_ylabel(label)


def add_event_backgrounds(
    ax,
    gt_events: list[Event],
    pred_events: list[Event],
) -> None:
    for event in gt_events:
        ax.axvspan(event.start, event.end, color="tab:red", alpha=0.10, linewidth=0)
    for event in pred_events:
        ax.axvspan(event.start, event.end, color="tab:orange", alpha=0.08, linewidth=0)


def robust_scaled(values: np.ndarray, center: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if center:
        values = values - float(np.median(values))
    scale = float(np.percentile(np.abs(values), 99))
    if scale <= 1.0e-12:
        scale = float(np.max(np.abs(values))) or 1.0
    return np.clip(values / scale, -1.0, 1.0)


def save_timeline(
    record_data: dict,
    probs: np.ndarray,
    threshold: float,
    gt_events: list[Event],
    pred_events: list[Event],
    fn_indices: list[int],
    fp_indices: list[int],
    output_path: Path,
) -> None:
    record = record_data["record"]
    fs_audio = int(record["fs_audio"])
    fs_motion = int(record["fs_motion"])
    audio_time = np.arange(len(record["pulm_bp"])) / fs_audio
    motion_time = np.arange(len(record["stretch_lp"])) / fs_motion
    spans = record_data["spans"]
    centers = np.asarray([(start + end) / 2 for start, end in spans], dtype=np.float32)
    preds = (probs >= threshold).astype(float)
    fn_events = event_spans_from_indices(gt_events, fn_indices)
    fp_events = event_spans_from_indices(pred_events, fp_indices)

    fig, axes = plt.subplots(
        8,
        1,
        figsize=(18, 13),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.2, 1.2, 1.0, 1.0, 0.65, 0.65, 0.75, 1.25],
        },
    )

    title = (
        f"Record {record['record_id']} | {record['filename']} | "
        f"{record['activity']} / {record['context']}"
    )
    sensor_axes = axes[:4]
    for ax in sensor_axes:
        add_event_backgrounds(ax, gt_events, pred_events)

    axes[0].plot(
        audio_time,
        robust_scaled(record["pulm_bp"], center=False),
        color="tab:blue",
        linewidth=0.55,
    )
    axes[0].set_ylabel("Pulm mic")
    axes[0].set_title(title)
    axes[0].set_ylim(-1.05, 1.05)

    axes[1].plot(
        audio_time,
        robust_scaled(record["amb_bp"], center=False),
        color="tab:cyan",
        linewidth=0.55,
    )
    axes[1].set_ylabel("Amb mic")
    axes[1].set_ylim(-1.05, 1.05)

    axes[2].plot(
        motion_time,
        robust_scaled(record["stretch_lp"]),
        color="tab:green",
        linewidth=0.9,
    )
    axes[2].set_ylabel("Stretch")
    axes[2].set_ylim(-1.05, 1.05)

    axes[3].plot(
        motion_time,
        robust_scaled(record["accz_lp"]),
        color="tab:brown",
        linewidth=0.9,
    )
    axes[3].set_ylabel("Acc Z")
    axes[3].set_ylim(-1.05, 1.05)
    axes[0].legend(
        handles=[
            Patch(facecolor="tab:red", alpha=0.18, label="GT region"),
            Patch(facecolor="tab:orange", alpha=0.16, label="Pred region"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )

    plot_event_bars(axes[4], gt_events, color="tab:red", label="GT", alpha=0.72)
    axes[4].set_title("Ground Truth Events", loc="left", fontsize=10, pad=2)
    plot_event_bars(axes[5], pred_events, color="tab:orange", label="Pred", alpha=0.72)
    axes[5].set_title("Predicted Events", loc="left", fontsize=10, pad=2)

    for event in fn_events:
        axes[6].broken_barh(
            [(event.start, event.duration)],
            (0.58, 0.28),
            facecolors="tab:red",
            edgecolors="tab:red",
            alpha=0.9,
            linewidth=1.4,
        )
    for event in fp_events:
        axes[6].broken_barh(
            [(event.start, event.duration)],
            (0.14, 0.28),
            facecolors="tab:purple",
            edgecolors="tab:purple",
            alpha=0.9,
            linewidth=1.4,
        )
    axes[6].set_ylim(0, 1)
    axes[6].set_yticks([0.28, 0.72])
    axes[6].set_yticklabels(["FP", "FN"])
    axes[6].set_ylabel("Errors")
    axes[6].set_title("Event-level Errors", loc="left", fontsize=10, pad=2)

    axes[7].plot(centers, probs, color="tab:blue", linewidth=1.2, marker="o", markersize=2)
    axes[7].fill_between(
        centers,
        threshold,
        probs,
        where=probs >= threshold,
        color="tab:orange",
        alpha=0.16,
        interpolate=True,
    )
    axes[7].step(centers, preds, where="mid", color="tab:orange", alpha=0.45, linewidth=1.1)
    axes[7].axhline(threshold, color="black", linestyle="--", linewidth=0.9)
    axes[7].set_ylabel("P(cough)")
    axes[7].set_xlabel("Time (s)")
    axes[7].set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_xlim(0, max(audio_time[-1], centers[-1] if len(centers) else 0))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    pred_merge_gap_sec = effective_pred_merge_gap(args)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    cfg = checkpoint.get("config") or load_config(args.config)
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    input_ablation = args.input_ablation or checkpoint.get("input_ablation", "full")

    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    split_map = checkpoint.get("record_split")
    if args.record_ids is not None:
        record_ids = [int(x) for x in args.record_ids]
    elif split_map and args.split in split_map:
        record_ids = [int(x) for x in split_map[args.split]]
    else:
        _, val_ids, test_ids = split_records_from_config(metadata, cfg.get("split"))
        selected = val_ids if args.split == "val" else test_ids
        record_ids = [int(x) for x in selected]

    model = Spec2DCoughCNN(num_classes=1).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    window_cfg = cfg["windowing"]
    pred_center_fraction = (
        float(window_cfg["center_fraction"])
        if args.pred_center_fraction is None
        else args.pred_center_fraction
    )
    spec_cfg = cfg["spectrogram"]
    output_dir = project_or_absolute(args.output_dir)
    timeline_dir = output_dir / "timelines"
    possible_missing_label_record_ids = set(args.possible_missing_label_record_ids or [])
    event_rows = []
    record_rows = []
    timeline_paths = []

    for record_id in record_ids:
        record_data = build_record_dataset(
            record_id,
            metadata,
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
        event_probs = smooth_probabilities(
            probs,
            record_data["spans"],
            smoothing_sec=args.prob_smoothing_sec,
        )
        gt_events, pred_events, matches, fn_indices, fp_indices = classify_events(
            record_data,
            probs=probs,
            threshold=args.threshold,
            event_iou_threshold=args.event_iou_threshold,
            gt_min_duration_sec=args.gt_min_duration_sec,
            gt_merge_gap_sec=args.gt_merge_gap_sec,
            pred_min_duration_sec=args.pred_min_duration_sec,
            pred_merge_gap_sec=pred_merge_gap_sec,
            pred_span_mode=args.pred_span_mode,
            pred_center_fraction=pred_center_fraction,
            prob_smoothing_sec=args.prob_smoothing_sec,
            hysteresis_low_threshold=args.hysteresis_low_threshold,
        )

        record = record_data["record"]
        record_rows.append(
            {
                "record_id": int(record["record_id"]),
                "filename": record["filename"],
                "activity": record["activity"],
                "context": record["context"],
                "true_events": len(gt_events),
                "predicted_events": len(pred_events),
                "tp": len(matches),
                "fp": len(fp_indices),
                "fn": len(fn_indices),
            }
        )
        add_event_error_rows(
            event_rows,
            record_data,
            gt_events,
            pred_events,
            matches,
            fn_indices,
            fp_indices,
            event_probs,
            threshold=args.threshold,
            event_iou_threshold=args.event_iou_threshold,
            possible_missing_label_record_ids=possible_missing_label_record_ids,
        )

        should_plot = args.plot_records == "all" or (
            args.plot_records == "problem" and (fn_indices or fp_indices)
        )
        if should_plot:
            output_path = timeline_dir / f"record_{int(record['record_id']):03d}_timeline.png"
            save_timeline(
                record_data,
                probs=event_probs,
                threshold=args.threshold,
                gt_events=gt_events,
                pred_events=pred_events,
                fn_indices=fn_indices,
                fp_indices=fp_indices,
                output_path=output_path,
            )
            timeline_paths.append(output_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    record_summary_path = output_dir / "record_error_summary.csv"
    event_errors_path = output_dir / "event_errors.csv"
    summary_path = output_dir / "summary.json"

    write_csv(
        record_summary_path,
        record_rows,
        [
            "record_id",
            "filename",
            "activity",
            "context",
            "true_events",
            "predicted_events",
            "tp",
            "fp",
            "fn",
        ],
    )
    write_csv(
        event_errors_path,
        event_rows,
        [
            "record_id",
            "filename",
            "activity",
            "context",
            "error_type",
            "error_category",
            "gt_start_sec",
            "gt_end_sec",
            "pred_start_sec",
            "pred_end_sec",
            "duration_sec",
            "matched_iou",
            "nearest_event_start_sec",
            "nearest_event_end_sec",
            "nearest_event_iou",
            "nearest_event_gap_sec",
            "max_probability",
            "mean_probability",
            "review_note",
        ],
    )

    totals = {
        "split": args.split,
        "threshold": args.threshold,
        "event_iou_threshold": args.event_iou_threshold,
        "event_merge_gap_sec": args.event_merge_gap_sec,
        "gt_min_duration_sec": args.gt_min_duration_sec,
        "gt_merge_gap_sec": args.gt_merge_gap_sec,
        "pred_min_duration_sec": args.pred_min_duration_sec,
        "pred_merge_gap_sec": pred_merge_gap_sec,
        "pred_span_mode": args.pred_span_mode,
        "pred_center_fraction": pred_center_fraction,
        "prob_smoothing_sec": args.prob_smoothing_sec,
        "hysteresis_low_threshold": args.hysteresis_low_threshold,
        "input_ablation": input_ablation,
        "possible_missing_label_record_ids": sorted(possible_missing_label_record_ids),
        "records": len(record_rows),
        "true_events": int(sum(row["true_events"] for row in record_rows)),
        "predicted_events": int(sum(row["predicted_events"] for row in record_rows)),
        "tp": int(sum(row["tp"] for row in record_rows)),
        "fp": int(sum(row["fp"] for row in record_rows)),
        "fn": int(sum(row["fn"] for row in record_rows)),
        "timeline_count": len(timeline_paths),
        "timelines": [str(path) for path in timeline_paths],
    }
    summary_path.write_text(json.dumps(totals, indent=2), encoding="utf-8")

    print(
        f"Events: TP={totals['tp']} FP={totals['fp']} FN={totals['fn']} | "
        f"Timelines: {totals['timeline_count']}"
    )
    print(f"Saved record summary: {record_summary_path}")
    print(f"Saved event errors: {event_errors_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved timelines: {timeline_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
