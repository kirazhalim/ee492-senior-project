from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.event_metrics import binary_labels_to_events
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import load_record_preprocessed


GROUP_BY_COLUMNS = {
    "subject": ["subject"],
    "date": ["date"],
    "subject_date": ["subject", "date"],
    "activity_context": ["activity", "context"],
    "subject_activity_context": ["subject", "activity", "context"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped split protocol reports without training a model. "
            "This is an experiment-planning tool, not the stable V3 pipeline."
        )
    )
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--max-records", type=int, default=85)
    parser.add_argument(
        "--group-by",
        choices=sorted(GROUP_BY_COLUMNS),
        default="subject",
    )
    parser.add_argument(
        "--fold-mode",
        choices=["leave-one-group-out", "group-kfold"],
        default="leave-one-group-out",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-test-records", type=int, default=3)
    parser.add_argument("--include-event-counts", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/grouped_protocol",
    )
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def make_group_labels(metadata: pd.DataFrame, group_by: str) -> pd.Series:
    columns = GROUP_BY_COLUMNS[group_by]
    return metadata[columns].astype(str).agg("__".join, axis=1)


def parse_count_map(values: pd.Series) -> dict[str, int]:
    counts = Counter(str(value) for value in values.tolist())
    return dict(sorted(counts.items()))


def event_stats_for_record(record_id: int, metadata: pd.DataFrame) -> dict[str, float]:
    record = load_record_preprocessed(int(record_id), metadata=metadata)
    events = binary_labels_to_events(
        record["cough_label"],
        sample_rate=int(record["fs_audio"]),
        min_duration_sec=0.0,
        merge_gap_sec=0.0,
    )
    return {
        "cough_events": float(len(events)),
        "cough_duration_sec": float(sum(event.duration for event in events)),
        "duration_sec": float(record["duration_sec"]),
    }


def attach_event_stats(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record_id in metadata["record_id"].tolist():
        rows.append(event_stats_for_record(int(record_id), metadata))
    stats = pd.DataFrame(rows, index=metadata.index)
    return pd.concat([metadata, stats], axis=1)


def split_iterator(
    metadata: pd.DataFrame,
    groups: np.ndarray,
    fold_mode: str,
    n_splits: int,
):
    record_ids = metadata["record_id"].to_numpy()
    dummy_y = np.zeros(len(record_ids), dtype=int)
    if fold_mode == "leave-one-group-out":
        splitter = LeaveOneGroupOut()
    else:
        unique_groups = np.unique(groups)
        effective_splits = min(int(n_splits), len(unique_groups))
        if effective_splits < 2:
            raise ValueError("At least two groups are required for GroupKFold.")
        splitter = GroupKFold(n_splits=effective_splits)
    return splitter.split(record_ids, dummy_y, groups)


def split_summary(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    prefix: str,
    include_event_counts: bool,
) -> dict:
    subset = metadata.iloc[indices]
    row = {
        f"{prefix}_records": int(len(subset)),
        f"{prefix}_groups": int(subset["group_label"].nunique()),
        f"{prefix}_subjects": json.dumps(parse_count_map(subset["subject"])),
        f"{prefix}_activities": json.dumps(parse_count_map(subset["activity"])),
        f"{prefix}_contexts": json.dumps(parse_count_map(subset["context"])),
    }
    if include_event_counts:
        row[f"{prefix}_events"] = int(subset["cough_events"].sum())
        row[f"{prefix}_cough_duration_sec"] = round(float(subset["cough_duration_sec"].sum()), 3)
        row[f"{prefix}_duration_sec"] = round(float(subset["duration_sec"].sum()), 3)
    return row


def coverage_warnings(
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    min_test_records: int,
) -> list[str]:
    train = metadata.iloc[train_idx]
    test = metadata.iloc[test_idx]
    warnings = []

    if len(test) < min_test_records:
        warnings.append(f"small_test_records:{len(test)}")

    for column in ["activity", "context"]:
        train_values = set(train[column].astype(str))
        missing = sorted(set(test[column].astype(str)) - train_values)
        if missing:
            warnings.append(f"test_{column}_not_in_train:{'|'.join(missing)}")

    if len(set(train["subject"].astype(str)) & set(test["subject"].astype(str))) > 0:
        warnings.append("subject_overlap_train_test")

    return warnings


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    metadata_path = project_or_absolute(args.metadata or cfg["data"]["metadata"])
    metadata = load_metadata(metadata_path)
    if args.max_records is not None:
        metadata = metadata.head(args.max_records).copy()

    metadata = metadata.reset_index(drop=True)
    metadata["group_label"] = make_group_labels(metadata, args.group_by)
    if args.include_event_counts:
        metadata = attach_event_stats(metadata)

    groups = metadata["group_label"].to_numpy()
    fold_rows = []
    record_rows = []
    group_rows = []

    for group_label, subset in metadata.groupby("group_label", sort=True):
        row = {
            "group_label": group_label,
            "records": int(len(subset)),
            "subjects": json.dumps(parse_count_map(subset["subject"])),
            "activities": json.dumps(parse_count_map(subset["activity"])),
            "contexts": json.dumps(parse_count_map(subset["context"])),
            "record_ids": " ".join(str(int(x)) for x in subset["record_id"].tolist()),
        }
        if args.include_event_counts:
            row["events"] = int(subset["cough_events"].sum())
            row["cough_duration_sec"] = round(float(subset["cough_duration_sec"].sum()), 3)
        group_rows.append(row)

    for fold_idx, (train_idx, test_idx) in enumerate(
        split_iterator(metadata, groups, args.fold_mode, args.n_splits),
        start=1,
    ):
        train_groups = sorted(metadata.iloc[train_idx]["group_label"].unique())
        test_groups = sorted(metadata.iloc[test_idx]["group_label"].unique())
        warnings = coverage_warnings(
            metadata,
            train_idx=train_idx,
            test_idx=test_idx,
            min_test_records=args.min_test_records,
        )

        fold_row = {
            "fold": fold_idx,
            "test_group_labels": " ".join(test_groups),
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
            "warnings": ";".join(warnings),
        }
        fold_row.update(split_summary(metadata, train_idx, "train", args.include_event_counts))
        fold_row.update(split_summary(metadata, test_idx, "test", args.include_event_counts))
        fold_rows.append(fold_row)

        for split_name, split_idx in [("train", train_idx), ("test", test_idx)]:
            for _, row in metadata.iloc[split_idx].iterrows():
                record_rows.append(
                    {
                        "fold": fold_idx,
                        "split": split_name,
                        "record_id": int(row["record_id"]),
                        "group_label": row["group_label"],
                        "date": row["date"],
                        "subject": row["subject"],
                        "activity": row["activity"],
                        "context": row["context"],
                        "filename": row["filename"],
                    }
                )

    output_dir = (
        project_or_absolute(args.output_dir)
        / f"{args.group_by}_{args.fold_mode}_records_{len(metadata):03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    group_fieldnames = list(group_rows[0].keys()) if group_rows else []
    fold_fieldnames = list(fold_rows[0].keys()) if fold_rows else []
    record_fieldnames = list(record_rows[0].keys()) if record_rows else []

    write_csv(output_dir / "groups.csv", group_rows, group_fieldnames)
    write_csv(output_dir / "fold_summary.csv", fold_rows, fold_fieldnames)
    write_csv(output_dir / "fold_records.csv", record_rows, record_fieldnames)

    summary = {
        "config": args.config,
        "metadata": str(metadata_path),
        "record_count": int(len(metadata)),
        "group_by": args.group_by,
        "fold_mode": args.fold_mode,
        "n_folds": int(len(fold_rows)),
        "unique_groups": int(metadata["group_label"].nunique()),
        "include_event_counts": bool(args.include_event_counts),
        "output_dir": str(output_dir),
        "warnings": sorted(
            set(
                warning
                for row in fold_rows
                for warning in str(row.get("warnings", "")).split(";")
                if warning
            )
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved grouped protocol report: {output_dir}")
    print(
        f"Groups: {summary['unique_groups']} | folds: {summary['n_folds']} | "
        f"warnings: {', '.join(summary['warnings']) if summary['warnings'] else 'none'}"
    )
    for row in fold_rows[:10]:
        print(
            f"fold={row['fold']} test_groups={row['test_group_labels']} "
            f"train_records={row['train_records']} test_records={row['test_records']} "
            f"warnings={row['warnings'] or 'none'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
