from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.event_metrics import Event, event_iou
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import load_record_preprocessed
from cough_analysis.v4 import cough_gt_events


ACTIVITY_CLASSES = ["sitting", "standing", "walking", "running"]
MERGED_CLASSES = ["stationary", "walking", "running"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready V4 figures and examples.")
    parser.add_argument("--config", default="configs/v4.yaml")
    parser.add_argument("--evaluation-json", default="artifacts/evaluations/v4/test/v4_evaluation.json")
    parser.add_argument(
        "--event-csv",
        default="artifacts/evaluations/v4/test/v4_event_activity_predictions.csv",
    )
    parser.add_argument("--inspection-dir", default="artifacts/inspections/v4")
    parser.add_argument("--model-dir", default="artifacts/models/v4")
    parser.add_argument("--output-dir", default="artifacts/report_assets/v4")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_path(path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def merged_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [cm[0:2, 0:2].sum(), cm[0:2, 2].sum(), cm[0:2, 3].sum()],
            [cm[2, 0:2].sum(), cm[2, 2], cm[2, 3]],
            [cm[3, 0:2].sum(), cm[3, 2], cm[3, 3]],
        ],
        dtype=int,
    )


def read_event_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_matching(rows: list[dict], label: str, predicate) -> dict | None:
    for row in rows:
        if predicate(row):
            return {"example_type": label, **row}
    return None


def nearest_gt_stats(row: dict, gt_by_record: dict[int, list[Event]]) -> tuple[float, float]:
    record_id = int(row["record_id"])
    pred = Event(start=float(row["cough_start"]), end=float(row["cough_end"]))
    gt_events = gt_by_record.get(record_id, [])
    if not gt_events:
        return 0.0, float("inf")

    best_iou = 0.0
    best_gap = float("inf")
    for gt_event in gt_events:
        iou = event_iou(pred, gt_event)
        gap = (
            0.0
            if max(pred.start, gt_event.start) < min(pred.end, gt_event.end)
            else min(abs(pred.start - gt_event.end), abs(gt_event.start - pred.end))
        )
        if iou > best_iou or (iou == best_iou and gap < best_gap):
            best_iou = iou
            best_gap = gap
    return best_iou, best_gap


def is_hard_fp(row: dict, gt_by_record: dict[int, list[Event]]) -> bool:
    if row["matched_gt"] != "False":
        return False
    nearest_iou, nearest_gap = nearest_gt_stats(row, gt_by_record)
    return nearest_iou == 0.0 and nearest_gap >= 0.3


def write_event_examples(
    rows: list[dict],
    output_path: Path,
    gt_by_record: dict[int, list[Event]],
) -> list[dict]:
    examples = [
        first_matching(
            rows,
            "cough_tp_activity_correct",
            lambda row: row["matched_gt"] == "True"
            and row["activity_correct_if_matched"] == "True",
        ),
        first_matching(
            rows,
            "cough_tp_activity_wrong",
            lambda row: row["matched_gt"] == "True"
            and row["activity_correct_if_matched"] == "False",
        ),
        first_matching(
            rows,
            "cough_fp",
            lambda row: is_hard_fp(row, gt_by_record),
        ),
        first_matching(
            rows,
            "stationary_merge_fix",
            lambda row: row["matched_gt"] == "True"
            and {row["true_activity"], row["activity"]} == {"sitting", "standing"},
        ),
    ]
    clean = [row for row in examples if row is not None]

    fieldnames = [
        "example_type",
        "record_id",
        "cough_start",
        "cough_end",
        "matched_gt",
        "true_activity",
        "activity",
        "activity_confidence",
        "activity_correct_if_matched",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean)
    return clean


def find_tn_segments(inspection_dir: Path, threshold: float) -> list[dict]:
    examples = []
    for path in sorted(inspection_dir.glob("record_*_v4_frame_predictions.csv")):
        record_token = path.name.split("_")[1]
        rows = []
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "time_sec": float(row["time_sec"]),
                        "prob": float(row["cough_probability"]),
                        "gt": int(row["gt_label"]),
                    }
                )
        if len(rows) < 100:
            continue
        probs = np.asarray([row["prob"] for row in rows], dtype=np.float32)
        labels = np.asarray([row["gt"] for row in rows], dtype=np.int64)
        times = np.asarray([row["time_sec"] for row in rows], dtype=np.float32)
        frame_rate = 1.0 / float(np.median(np.diff(times)))
        width = max(1, int(round(frame_rate)))
        for start in range(0, len(rows) - width + 1, width):
            end = start + width
            if labels[start:end].sum() == 0 and float(probs[start:end].max()) < threshold:
                examples.append(
                    {
                        "example_type": "cough_tn_segment",
                        "record_id": int(record_token),
                        "start_sec": float(times[start]),
                        "end_sec": float(times[end - 1]),
                        "max_cough_probability": float(probs[start:end].max()),
                    }
                )
                break
    return examples


def write_tn_segments(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "example_type",
        "record_id",
        "start_sec",
        "end_sec",
        "max_cough_probability",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_gt_by_record(cfg: dict, record_ids: set[int]) -> dict[int, list[Event]]:
    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    data_root = project_or_absolute(cfg["data"]["data_root"])
    gt_by_record = {}
    for record_id in sorted(record_ids):
        record = load_record_preprocessed(record_id, metadata=metadata, data_root=data_root)
        gt_by_record[record_id] = cough_gt_events(record, cfg["cough"])
    return gt_by_record


def generate_example_timelines(
    examples: list[dict],
    model_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    timeline_dir = output_dir / "timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    for row in examples:
        record_id = int(row["record_id"])
        example_type = str(row["example_type"])
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "inspect_v4_record.py"),
                "--model-dir",
                str(model_dir),
                "--record-id",
                str(record_id),
                "--output-dir",
                str(timeline_dir),
                "--no-prompt",
                "--no-open",
            ],
            check=True,
        )
        source = timeline_dir / f"record_{record_id:03d}_v4_timeline.png"
        target = timeline_dir / f"{example_type}_record_{record_id:03d}_timeline.png"
        if source.exists():
            shutil.copyfile(source, target)
            generated[example_type] = str(target)

    return generated


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    evaluation_path = project_or_absolute(args.evaluation_json)
    event_csv_path = project_or_absolute(args.event_csv)
    inspection_dir = project_or_absolute(args.inspection_dir)
    model_dir = project_or_absolute(args.model_dir)
    output_dir = project_or_absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    cm4 = np.asarray(report["activity"]["confusion_matrix"], dtype=int)
    cm3 = merged_confusion_matrix(cm4)

    plot_confusion_matrix(
        cm4,
        ACTIVITY_CLASSES,
        "V4 Activity Confusion Matrix",
        output_dir / "v4_activity_confusion_matrix_4class.png",
    )
    plot_confusion_matrix(
        cm3,
        MERGED_CLASSES,
        "V4 Activity Confusion Matrix (Sitting+Standing Merged)",
        output_dir / "v4_activity_confusion_matrix_merged3.png",
    )

    event_rows = read_event_rows(event_csv_path)
    gt_by_record = load_gt_by_record(
        cfg,
        {int(row["record_id"]) for row in event_rows},
    )
    examples = write_event_examples(
        event_rows,
        output_dir / "v4_event_examples.csv",
        gt_by_record=gt_by_record,
    )
    example_timelines = generate_example_timelines(
        examples,
        model_dir=model_dir,
        output_dir=output_dir,
    )
    tn_segments = find_tn_segments(
        inspection_dir,
        threshold=float(report["cough"]["threshold"]),
    )
    write_tn_segments(tn_segments, output_dir / "v4_tn_segment_examples.csv")

    summary = {
        "source_evaluation": str(evaluation_path),
        "source_event_csv": str(event_csv_path),
        "activity_confusion_matrix_4class": str(output_dir / "v4_activity_confusion_matrix_4class.png"),
        "activity_confusion_matrix_merged3": str(output_dir / "v4_activity_confusion_matrix_merged3.png"),
        "event_examples": str(output_dir / "v4_event_examples.csv"),
        "example_timelines": example_timelines,
        "tn_segment_examples": str(output_dir / "v4_tn_segment_examples.csv"),
        "example_record_ids": sorted({int(row["record_id"]) for row in examples}),
        "tn_record_ids": sorted({int(row["record_id"]) for row in tn_segments}),
        "note": "TN is represented as a low-probability non-cough time segment, not an event-level count.",
    }
    summary_path = output_dir / "v4_report_assets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved report assets: {output_dir}")
    print(f"Example records: {summary['example_record_ids']}")
    print(f"TN segment records: {summary['tn_record_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
