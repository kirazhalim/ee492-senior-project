from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.models import V4ActivityCNN, V4CoughFrameCNN
from cough_analysis.paths import project_path
from cough_analysis.v4 import (
    assign_activity_to_event,
    cough_gt_events,
    frame_labels_from_samples,
    frame_predictions_to_events,
    predict_activity_probabilities_for_record,
    predict_cough_probabilities_for_record,
    remove_short_events,
    resolve_device,
)
from cough_analysis.preprocessing import load_record_preprocessed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one V4 record with GT labels, predictions, and activity probabilities."
    )
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--config", default="configs/v4.yaml")
    parser.add_argument("--record-id", type=int, default=None)
    parser.add_argument("--output-dir", default="artifacts/inspections/v4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_path(path)


def robust_scaled(values: np.ndarray, center: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if center:
        values = values - float(np.median(values))
    scale = float(np.percentile(np.abs(values), 99))
    if scale <= 1.0e-12:
        scale = float(np.max(np.abs(values))) or 1.0
    return np.clip(values / scale, -1.0, 1.0)


def find_v4_model_dirs() -> list[Path]:
    roots = [
        project_path("artifacts", "models", "v4"),
        project_path("artifacts", "models"),
    ]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        if (root / "v4_summary.json").exists():
            candidates.append(root)
        candidates.extend(path.parent for path in root.glob("**/v4_summary.json"))
    return sorted(set(candidates))


def choose_model_dir(model_dir_arg: str | None, no_prompt: bool) -> Path:
    if model_dir_arg:
        model_dir = project_or_absolute(model_dir_arg)
        if not (model_dir / "v4_summary.json").exists():
            raise FileNotFoundError(f"Missing V4 summary: {model_dir / 'v4_summary.json'}")
        return model_dir

    candidates = find_v4_model_dirs()
    if not candidates:
        raise FileNotFoundError(
            "No trained V4 model found. Run scripts/train_v4.py first, or pass --model-dir."
        )
    if len(candidates) == 1 or no_prompt:
        return candidates[0]

    print("Available V4 model directories:")
    for idx, candidate in enumerate(candidates, start=1):
        print(f"{idx}. {candidate}")
    selected = int(input("Select model number: ").strip())
    return candidates[selected - 1]


def choose_record_id(record_id_arg: int | None, metadata, no_prompt: bool) -> int:
    if record_id_arg is not None:
        return int(record_id_arg)
    if no_prompt:
        raise ValueError("--record-id is required when --no-prompt is set.")

    preview = metadata[["record_id", "subject", "activity", "context", "filename"]]
    print(preview.to_string(index=False, max_rows=30))
    if len(preview) > 30:
        print(f"... {len(preview) - 30} more records not shown")
    return int(input("Record id: ").strip())


def open_path(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError as exc:
        print(f"Could not open plot automatically: {exc}")


def add_event_backgrounds(ax, gt_events, pred_events) -> None:
    for event in gt_events:
        ax.axvspan(event.start, event.end, color="tab:red", alpha=0.11, linewidth=0)
    for event in pred_events:
        ax.axvspan(event.start, event.end, color="tab:orange", alpha=0.10, linewidth=0)


def plot_event_bars(ax, events, color: str, label: str, annotations: list[str] | None = None) -> None:
    for idx, event in enumerate(events):
        ax.broken_barh(
            [(event.start, event.duration)],
            (0.2, 0.6),
            facecolors=color,
            edgecolors=color,
            alpha=0.75,
            linewidth=1.2,
        )
        if annotations:
            ax.text(
                event.start + event.duration / 2,
                0.88,
                annotations[idx],
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([label])
    ax.set_ylabel(label)


def save_outputs(
    output_dir: Path,
    record: dict,
    summary: dict,
    post_cfg: dict,
    probs: np.ndarray,
    gt_frame_labels: np.ndarray,
    pred_events,
    gt_events,
    event_rows: list[dict],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"record_{int(record['record_id']):03d}_v4"
    json_path = output_dir / f"{stem}_inspection.json"
    csv_path = output_dir / f"{stem}_frame_predictions.csv"
    png_path = output_dir / f"{stem}_timeline.png"

    json_path.write_text(
        json.dumps(
            {
                "model_id": "v4",
                "model_dir": summary.get("model_dir", ""),
                "record_id": int(record["record_id"]),
                "filename": record["filename"],
                "true_activity": record["activity"],
                "primary_cough_spec": summary["primary_cough_spec"],
                "postprocessing": post_cfg,
                "ground_truth_events": [event.__dict__ for event in gt_events],
                "predicted_events": event_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    frame_rate = len(probs) / float(record["duration_sec"])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "cough_probability", "gt_label"])
        for idx, (prob, label) in enumerate(zip(probs, gt_frame_labels)):
            writer.writerow([idx / frame_rate, float(prob), int(label)])

    return {"json": json_path, "csv": csv_path, "png": png_path}


def save_timeline_plot(
    output_path: Path,
    record: dict,
    probs: np.ndarray,
    gt_frame_labels: np.ndarray,
    gt_events,
    pred_events,
    event_rows: list[dict],
    activity_centers: np.ndarray,
    activity_probs: np.ndarray,
    activity_classes: list[str],
    threshold: float,
) -> None:
    fs_audio = int(record["fs_audio"])
    fs_motion = int(record["fs_motion"])
    audio_time = np.arange(len(record["pulm_bp"]), dtype=np.float32) / fs_audio
    motion_time = np.arange(len(record["stretch_lp"]), dtype=np.float32) / fs_motion
    prob_time = np.arange(len(probs), dtype=np.float32) / (len(probs) / float(record["duration_sec"]))

    fig, axes = plt.subplots(
        8,
        1,
        figsize=(18, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.55, 0.65, 1.25, 1.1]},
    )
    title = (
        f"V4 inspection | Record {record['record_id']} | {record['filename']} | "
        f"{record['subject']} / {record['activity']} / {record['context']}"
    )

    sensor_rows = [
        ("Pulm mic", audio_time, robust_scaled(record["pulm_bp"], center=False), "tab:blue"),
        ("Amb mic", audio_time, robust_scaled(record["amb_bp"], center=False), "tab:cyan"),
        ("Stretch", motion_time, robust_scaled(record["stretch_lp"]), "tab:green"),
        ("Acc Z", motion_time, robust_scaled(record["accz_lp"]), "tab:brown"),
    ]
    for ax, (label, times, values, color) in zip(axes[:4], sensor_rows):
        add_event_backgrounds(ax, gt_events, pred_events)
        ax.plot(times, values, color=color, linewidth=0.65)
        ax.set_ylabel(label)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    axes[0].set_title(title)
    axes[4].step(prob_time, gt_frame_labels, where="post", color="tab:red", linewidth=1.1)
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].set_yticks([0, 1])
    axes[4].set_ylabel("GT label")
    axes[4].grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    annotations = [
        f"{row['activity']} {row['activity_confidence']:.2f}"
        for row in event_rows
    ]
    plot_event_bars(axes[5], gt_events, color="tab:red", label="GT")
    plot_event_bars(axes[6], pred_events, color="tab:orange", label="Pred", annotations=annotations)

    axes[7].plot(prob_time, probs, color="tab:blue", linewidth=1.2)
    axes[7].fill_between(
        prob_time,
        threshold,
        probs,
        where=probs >= threshold,
        color="tab:orange",
        alpha=0.18,
    )
    axes[7].axhline(threshold, color="black", linestyle="--", linewidth=0.9)
    axes[7].set_ylabel("P(cough)")
    axes[7].set_xlabel("Time (s)")
    axes[7].set_ylim(-0.05, 1.05)
    axes[7].grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    activity_ax = axes[7].twinx()
    for idx, activity in enumerate(activity_classes):
        activity_ax.plot(
            activity_centers,
            activity_probs[:, idx],
            linewidth=0.9,
            alpha=0.65,
            label=activity,
        )
    activity_ax.set_ylim(-0.05, 1.05)
    activity_ax.set_ylabel("P(activity)")
    activity_ax.legend(loc="upper right", ncol=2, fontsize=8)

    for ax in axes:
        ax.set_xlim(0, float(record["duration_sec"]))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    model_dir = choose_model_dir(args.model_dir, no_prompt=args.no_prompt)
    record_id = choose_record_id(args.record_id, metadata, no_prompt=args.no_prompt)
    device = resolve_device(args.device)
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])

    summary = json.loads((model_dir / "v4_summary.json").read_text(encoding="utf-8"))
    summary["model_dir"] = str(model_dir)
    spec_name = summary["primary_cough_spec"]
    cough_ckpt = torch.load(model_dir / f"v4_cough_{spec_name}.pt", map_location=device)
    activity_ckpt = torch.load(model_dir / "v4_activity.pt", map_location=device)

    cough_model = V4CoughFrameCNN().to(device)
    cough_model.load_state_dict(cough_ckpt["model_state_dict"])
    cough_model.eval()

    activity_model = V4ActivityCNN(num_classes=len(cfg["activity"]["classes"])).to(device)
    activity_model.load_state_dict(activity_ckpt["model_state_dict"])
    activity_model.eval()

    record = load_record_preprocessed(
        record_id,
        metadata=metadata,
        data_root=project_or_absolute(cfg["data"]["data_root"]),
    )
    probs = predict_cough_probabilities_for_record(
        cough_model,
        record,
        cfg["cough"],
        cough_ckpt["spec_config"],
        device=device,
        batch_size=batch_size,
    )
    frame_rate = int(round(int(cfg["sampling"]["audio_hz"]) / int(cfg["cough"]["frame_hop_samples"])))
    post_cfg = cough_ckpt["selected_postprocessing"]
    pred_events = frame_predictions_to_events(
        probs,
        frame_rate=frame_rate,
        threshold=float(post_cfg["threshold"]),
        min_duration_sec=float(post_cfg["pred_min_duration_sec"]),
        merge_gap_sec=float(post_cfg["pred_merge_gap_sec"]),
        duration_sec=float(record["duration_sec"]),
    )
    gt_events = cough_gt_events(record, cfg["cough"])
    gt_labels = remove_short_events(
        record["cough_label"],
        sample_rate=int(record["fs_audio"]),
        min_duration_sec=float(cfg["cough"]["min_gt_event_duration_sec"]),
    )
    gt_frame_labels = frame_labels_from_samples(
        gt_labels,
        frame_hop_samples=int(cfg["cough"]["frame_hop_samples"]),
        frame_count=len(probs),
    )
    activity_centers, activity_probs = predict_activity_probabilities_for_record(
        activity_model,
        record,
        cfg["activity"],
        device=device,
        batch_size=batch_size,
    )

    event_rows = []
    for event in pred_events:
        assigned = assign_activity_to_event(
            event,
            activity_centers,
            activity_probs,
            cfg["activity"]["classes"],
            context_sec=float(cfg["activity"]["attribution_context_sec"]),
        )
        start_frame = max(0, int(round(event.start * frame_rate)))
        end_frame = min(len(probs), int(round(event.end * frame_rate)))
        cough_confidence = float(probs[start_frame:end_frame].max()) if end_frame > start_frame else 0.0
        event_rows.append(
            {
                "cough_start": float(event.start),
                "cough_end": float(event.end),
                "cough_confidence": cough_confidence,
                **assigned,
            }
        )

    output_paths = save_outputs(
        project_or_absolute(args.output_dir),
        record,
        summary,
        post_cfg,
        probs,
        gt_frame_labels,
        pred_events,
        gt_events,
        event_rows,
    )
    save_timeline_plot(
        output_paths["png"],
        record,
        probs,
        gt_frame_labels,
        gt_events,
        pred_events,
        event_rows,
        activity_centers,
        activity_probs,
        cfg["activity"]["classes"],
        threshold=float(post_cfg["threshold"]),
    )

    print(f"Model dir: {model_dir}")
    print(f"Record: {record_id} | Predicted events: {len(pred_events)} | GT events: {len(gt_events)}")
    print(f"Saved plot: {output_paths['png']}")
    print(f"Saved event JSON: {output_paths['json']}")
    print(f"Saved frame CSV: {output_paths['csv']}")
    if not args.no_open:
        open_path(output_paths["png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
