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
from torch.utils.data import DataLoader

from cough_analysis.data import decode_channel3, load_metadata, load_record_array
from cough_analysis.event_metrics import (
    binary_labels_to_events,
    event_level_metrics,
    probabilities_to_predictions,
    smooth_probabilities,
    window_predictions_to_events,
)
from cough_analysis.models import Spec2DCoughCNN
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import FS_AUDIO, FS_MOTION
from cough_analysis.v3 import SpectrogramDataset, build_record_dataset, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record-id", type=int)
    source.add_argument("--record-path")
    parser.add_argument("--metadata", default="data/metadata.csv")
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="artifacts/predictions")
    return parser.parse_args()


def project_or_absolute(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(project_or_absolute(path), map_location=device)


def effective_pred_merge_gap(args: argparse.Namespace) -> float:
    return (
        args.event_merge_gap_sec
        if args.pred_merge_gap_sec is None
        else args.pred_merge_gap_sec
    )


def preprocess_external_record(record_path: Path) -> dict:
    raw = load_record_array(record_path)
    stretch, cough_label = decode_channel3(raw[:, 2])
    record = {
        "record_id": 0,
        "filename": record_path.name,
        "date": "",
        "subject": "unknown",
        "activity": "unknown",
        "context": "unknown",
        "path": str(record_path),
        "pulmonary": raw[:, 0].astype(np.float32),
        "ambient": raw[:, 1].astype(np.float32),
        "stretch": stretch,
        "accel_z": raw[:, 3].astype(np.float32),
        "cough_label": cough_label,
        "num_samples": raw.shape[0],
    }

    from scipy import signal
    from cough_analysis.preprocessing import butter_bandpass, butter_lowpass

    pulmonary = record["pulmonary"].astype(np.float64)
    ambient = record["ambient"].astype(np.float64)
    accz = record["accel_z"].astype(np.float64)
    stretch_f = record["stretch"].astype(np.float64)

    b_bp, a_bp = butter_bandpass(60, 2200, FS_AUDIO, order=4)
    pulmonary_bp = signal.filtfilt(b_bp, a_bp, pulmonary - np.median(pulmonary))
    ambient_bp = signal.filtfilt(b_bp, a_bp, ambient - np.median(ambient))
    n_motion = int(len(stretch_f) * (FS_MOTION / FS_AUDIO))
    stretch_resampled = signal.resample(stretch_f - np.median(stretch_f), n_motion)
    accz_resampled = signal.resample(accz, n_motion)
    b_lp, a_lp = butter_lowpass(20, FS_MOTION, order=4)

    return {
        **record,
        "pulm_bp": pulmonary_bp.astype(np.float32),
        "amb_bp": ambient_bp.astype(np.float32),
        "stretch_lp": signal.filtfilt(b_lp, a_lp, stretch_resampled).astype(np.float32),
        "accz_lp": signal.filtfilt(b_lp, a_lp, accz_resampled).astype(np.float32),
        "duration_sec": len(pulmonary_bp) / FS_AUDIO,
        "fs_audio": FS_AUDIO,
        "fs_motion": FS_MOTION,
    }


def robust_scaled(values: np.ndarray, center: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if center:
        values = values - float(np.median(values))
    scale = float(np.percentile(np.abs(values), 99))
    if scale <= 1.0e-12:
        scale = float(np.max(np.abs(values))) or 1.0
    return np.clip(values / scale, -1.0, 1.0)


def add_gt_backgrounds(ax, gt_events) -> None:
    for event in gt_events:
        ax.axvspan(event.start, event.end, color="tab:red", alpha=0.10, linewidth=0)


def plot_event_bars(ax, events, color: str, label: str, alpha: float = 0.8) -> None:
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


def save_timeline(
    record: dict,
    spans: list[tuple[float, float]],
    probs: np.ndarray,
    gt_events,
    pred_events,
    threshold: float,
    output_path: Path,
) -> None:
    fs_audio = int(record["fs_audio"])
    fs_motion = int(record["fs_motion"])
    audio_time = np.arange(len(record["pulm_bp"])) / fs_audio
    motion_time = np.arange(len(record["stretch_lp"])) / fs_motion
    centers = np.asarray([(start + end) / 2 for start, end in spans], dtype=np.float32)
    preds = (probs >= threshold).astype(float)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(18, 11),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.2, 1.2, 1.0, 1.0, 0.65, 0.65, 1.25],
        },
    )

    title = f"Record prediction | {record['filename']}"
    if record.get("record_id", 0) != 0:
        title = f"Record {record['record_id']} | {record['filename']}"
    if record.get("activity") and record.get("context"):
        title = f"{title} | {record['activity']} / {record['context']}"

    for ax in axes[:4]:
        add_gt_backgrounds(ax, gt_events)

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

    plot_event_bars(axes[4], gt_events, color="tab:red", label="GT", alpha=0.72)
    axes[4].set_title("Ground Truth Events", loc="left", fontsize=10, pad=2)

    plot_event_bars(axes[5], pred_events, color="tab:orange", label="Pred", alpha=0.70)
    axes[5].set_title("Predicted Events", loc="left", fontsize=10, pad=2)

    axes[6].plot(centers, probs, color="tab:blue", linewidth=1.2, marker="o", markersize=2)
    axes[6].fill_between(
        centers,
        threshold,
        probs,
        where=probs >= threshold,
        color="tab:orange",
        alpha=0.16,
        interpolate=True,
    )
    axes[6].step(centers, preds, where="mid", color="tab:orange", alpha=0.45, linewidth=1.1)
    axes[6].axhline(threshold, color="black", linestyle="--", linewidth=0.9)
    axes[6].set_ylabel("P(cough)")
    axes[6].set_xlabel("Time (s)")
    axes[6].set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_xlim(0, max(audio_time[-1], centers[-1] if len(centers) else 0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    pred_merge_gap_sec = effective_pred_merge_gap(args)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    cfg = checkpoint["config"]

    metadata = load_metadata(project_or_absolute(args.metadata))
    window_cfg = cfg["windowing"]
    spec_cfg = cfg["spectrogram"]
    pred_center_fraction = (
        float(window_cfg["center_fraction"])
        if args.pred_center_fraction is None
        else args.pred_center_fraction
    )

    if args.record_id is not None:
        record_data = build_record_dataset(
            args.record_id,
            metadata,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
            spectrogram_config=spec_cfg,
        )
    else:
        record_path = project_or_absolute(args.record_path)
        record = preprocess_external_record(record_path)
        from cough_analysis.v3 import audio_to_log_mel, build_centered_windows, make_mel_transform

        windows = build_centered_windows(
            record,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
        )
        mel_transform = make_mel_transform(spectrogram_config=spec_cfg)
        specs = audio_to_log_mel(
            windows["audio"],
            mel_transform=mel_transform,
            log_eps=float(spec_cfg.get("log_eps", 1.0e-9)),
        )
        record_data = {
            "record": record,
            "spec": specs,
            "motion": windows["motion"],
            "labels": windows["labels"],
            "spans": windows["spans"],
        }

    model = Spec2DCoughCNN(num_classes=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        SpectrogramDataset(
            record_data["spec"],
            record_data["motion"],
            record_data["labels"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["spec"].to(device), batch["motion"].to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    probs_np = np.asarray(probs)
    event_probs_np = smooth_probabilities(
        probs_np,
        record_data["spans"],
        smoothing_sec=args.prob_smoothing_sec,
    )
    preds_np = probabilities_to_predictions(
        event_probs_np,
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
        preds_np,
        min_duration_sec=args.pred_min_duration_sec,
        merge_gap_sec=pred_merge_gap_sec,
        span_mode=args.pred_span_mode,
        center_fraction=pred_center_fraction,
    )
    metrics = event_level_metrics(
        gt_events,
        pred_events,
        iou_threshold=args.event_iou_threshold,
    )

    output_dir = project_or_absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record_data["record"]["filename"]).stem
    pred_csv = output_dir / f"{stem}_window_predictions.csv"
    events_json = output_dir / f"{stem}_events.json"
    timeline_png = output_dir / f"{stem}_timeline.png"

    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_sec",
                "end_sec",
                "probability",
                "event_probability",
                "prediction",
                "label",
            ]
        )
        for (start, end), prob, event_prob, pred, label in zip(
            record_data["spans"],
            probs_np,
            event_probs_np,
            preds_np,
            record_data["labels"],
        ):
            writer.writerow([start, end, float(prob), float(event_prob), int(pred), int(label)])

    events_json.write_text(
        json.dumps(
            {
                "record": record_data["record"]["filename"],
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
                "event_metrics": metrics,
                "predicted_events": [event.__dict__ for event in pred_events],
                "ground_truth_events": [event.__dict__ for event in gt_events],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_timeline(
        record_data["record"],
        record_data["spans"],
        event_probs_np,
        gt_events,
        pred_events,
        args.threshold,
        timeline_png,
    )

    print(
        f"Predicted events: {len(pred_events)} | "
        f"GT events: {len(gt_events)} | "
        f"Event F1: {metrics['f1']:.3f}"
    )
    print(f"Saved window predictions: {pred_csv}")
    print(f"Saved events: {events_json}")
    print(f"Saved timeline: {timeline_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
