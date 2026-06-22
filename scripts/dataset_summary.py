from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.event_metrics import binary_labels_to_events
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import load_record_preprocessed
from cough_analysis.v3 import build_centered_windows, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output-dir", default="artifacts/dataset_summary")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def split_lookup(metadata: pd.DataFrame, seed: int) -> tuple[dict[int, str], dict[str, list[int]]]:
    train_ids, val_ids, test_ids = split_records(metadata, random_state=seed)
    splits = {
        "train": [int(x) for x in train_ids],
        "val": [int(x) for x in val_ids],
        "test": [int(x) for x in test_ids],
    }
    lookup = {
        record_id: split_name
        for split_name, record_ids in splits.items()
        for record_id in record_ids
    }
    return lookup, splits


def summarize_record(
    record_id: int,
    split_name: str,
    metadata: pd.DataFrame,
    window_sec: float,
    hop_sec: float,
    center_fraction: float,
) -> dict:
    record = load_record_preprocessed(record_id, metadata=metadata)
    windows = build_centered_windows(
        record,
        window_sec=window_sec,
        hop_sec=hop_sec,
        center_fraction=center_fraction,
    )
    cough_label = record["cough_label"]
    gt_events = binary_labels_to_events(
        cough_label,
        sample_rate=int(record["fs_audio"]),
    )
    row = metadata.loc[metadata["record_id"] == record_id].iloc[0]

    cough_samples = int(np.sum(cough_label))
    total_windows = int(len(windows["labels"]))
    cough_windows = int(np.sum(windows["labels"]))

    return {
        "record_id": record_id,
        "split": split_name,
        "filename": str(row["filename"]),
        "subject": str(row["subject"]),
        "activity": str(row["activity"]),
        "context": str(row["context"]),
        "duration_sec": float(record["duration_sec"]),
        "num_samples": int(record["num_samples"]),
        "cough_samples": cough_samples,
        "cough_duration_sec": float(cough_samples / record["fs_audio"]),
        "cough_events": int(len(gt_events)),
        "windows": total_windows,
        "cough_windows": cough_windows,
        "non_cough_windows": int(total_windows - cough_windows),
        "cough_window_rate": float(cough_windows / total_windows) if total_windows else 0.0,
    }


def aggregate(rows: list[dict], split_name: str) -> dict:
    selected = rows if split_name == "all" else [r for r in rows if r["split"] == split_name]
    records = len(selected)
    duration_sec = sum(float(r["duration_sec"]) for r in selected)
    windows = sum(int(r["windows"]) for r in selected)
    cough_windows = sum(int(r["cough_windows"]) for r in selected)
    cough_events = sum(int(r["cough_events"]) for r in selected)
    cough_duration_sec = sum(float(r["cough_duration_sec"]) for r in selected)

    return {
        "split": split_name,
        "records": records,
        "duration_min": duration_sec / 60.0,
        "windows": windows,
        "cough_windows": cough_windows,
        "non_cough_windows": windows - cough_windows,
        "cough_window_rate": cough_windows / windows if windows else 0.0,
        "cough_events": cough_events,
        "cough_duration_sec": cough_duration_sec,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    metadata_path = args.metadata or cfg["data"]["metadata"]
    metadata = load_metadata(project_or_absolute(metadata_path))
    if args.max_records is not None:
        metadata = metadata.head(args.max_records).copy()

    window_cfg = cfg["windowing"]
    window_sec = float(window_cfg["window_sec"])
    hop_sec = float(window_cfg["hop_sec"])
    center_fraction = float(window_cfg["center_fraction"])

    lookup, splits = split_lookup(metadata, seed=args.seed)
    rows = []
    for record_id in metadata["record_id"].tolist():
        rows.append(
            summarize_record(
                int(record_id),
                lookup[int(record_id)],
                metadata,
                window_sec=window_sec,
                hop_sec=hop_sec,
                center_fraction=center_fraction,
            )
        )

    summary_rows = [aggregate(rows, split_name) for split_name in ["train", "val", "test", "all"]]
    summary_df = pd.DataFrame(summary_rows)
    records_df = pd.DataFrame(rows).sort_values(["split", "record_id"]).reset_index(drop=True)

    output_dir = project_or_absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary_by_split.csv"
    records_csv = output_dir / "summary_by_record.csv"
    summary_json = output_dir / "dataset_summary.json"

    summary_df.to_csv(summary_csv, index=False)
    records_df.to_csv(records_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "config": str(project_or_absolute(args.config)),
                "metadata": str(project_or_absolute(metadata_path)),
                "seed": args.seed,
                "windowing": {
                    "window_sec": window_sec,
                    "hop_sec": hop_sec,
                    "center_fraction": center_fraction,
                },
                "splits": splits,
                "summary_by_split": summary_rows,
                "summary_by_record": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    display = summary_df.copy()
    display["duration_min"] = display["duration_min"].map(lambda x: f"{x:.2f}")
    display["cough_window_rate"] = display["cough_window_rate"].map(lambda x: f"{x * 100:.1f}%")
    display["cough_duration_sec"] = display["cough_duration_sec"].map(lambda x: f"{x:.2f}")
    print(display.to_string(index=False))
    print(f"Saved split summary: {summary_csv}")
    print(f"Saved record summary: {records_csv}")
    print(f"Saved JSON summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
