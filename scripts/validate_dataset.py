from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cough_analysis.data import decode_channel3
from cough_analysis.paths import project_path


REQUIRED_COLUMNS = [
    "record_id",
    "filename",
    "date",
    "subject",
    "activity",
    "context",
    "clothing",
    "relative_path",
]
ACTIVITY_OPTIONS = {"sitting", "standing", "walking", "running"}
CONTEXT_OPTIONS = {
    "clean",
    "coughnoise",
    "musicnoise",
    "sneezenoise",
    "snoozenoise",
    "doornoise",
    "falsepositive",
    "noise",
}
CLOTHING_OPTIONS = {"underclothes", "overclothes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate metadata and curated CSV files.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--expected-columns", type=int, default=4)
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def validate_metadata(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return [f"metadata missing columns: {missing}"]

    if df["record_id"].duplicated().any():
        repeated = df.loc[df["record_id"].duplicated(), "record_id"].tolist()
        errors.append(f"record_id values are duplicated: {repeated}")

    ids = [int(x) for x in df["record_id"].tolist()]
    if ids != sorted(ids):
        errors.append("record_id values are not sorted in ascending order")
    if ids and ids != list(range(min(ids), max(ids) + 1)):
        errors.append("record_id values are not contiguous")

    invalid_activities = sorted(set(df["activity"]) - ACTIVITY_OPTIONS)
    invalid_contexts = sorted(set(df["context"]) - CONTEXT_OPTIONS)
    invalid_clothing = sorted(set(df["clothing"]) - CLOTHING_OPTIONS)
    if invalid_activities:
        errors.append(f"invalid activity values: {invalid_activities}")
    if invalid_contexts:
        errors.append(f"invalid context values: {invalid_contexts}")
    if invalid_clothing:
        errors.append(f"invalid clothing values: {invalid_clothing}")

    return errors


def validate_record_file(
    record_row: pd.Series,
    data_root: Path,
    expected_columns: int,
    min_samples: int,
) -> list[str]:
    errors: list[str] = []
    record_id = int(record_row["record_id"])
    path = data_root / str(record_row["relative_path"])

    if not path.exists():
        return [f"record {record_id}: missing file {path}"]
    if path.name != str(record_row["filename"]):
        errors.append(
            f"record {record_id}: filename does not match relative_path "
            f"({record_row['filename']} vs {path.name})"
        )

    try:
        df = pd.read_csv(path, header=None)
    except Exception as exc:
        return [f"record {record_id}: could not read CSV: {exc}"]

    if df.shape[1] != expected_columns:
        errors.append(
            f"record {record_id}: expected {expected_columns} columns, got {df.shape[1]}"
        )
        return errors
    if len(df) < min_samples:
        errors.append(f"record {record_id}: expected at least {min_samples} samples, got {len(df)}")

    try:
        values = df.apply(pd.to_numeric, errors="raise").to_numpy()
    except Exception as exc:
        errors.append(f"record {record_id}: non-numeric values found: {exc}")
        return errors

    _, cough_label = decode_channel3(values[:, 2].astype(np.int64))
    unique_labels = set(np.unique(cough_label).astype(int).tolist())
    if not unique_labels <= {0, 1}:
        errors.append(f"record {record_id}: decoded labels are not binary: {unique_labels}")

    return errors


def main() -> int:
    args = parse_args()
    metadata_path = project_or_absolute(args.metadata)
    data_root = project_or_absolute(args.data_root)
    df = pd.read_csv(metadata_path)

    errors = validate_metadata(df)
    for _, row in df.iterrows():
        errors.extend(
            validate_record_file(
                row,
                data_root=data_root,
                expected_columns=args.expected_columns,
                min_samples=args.min_samples,
            )
        )

    if errors:
        print(f"Dataset validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Dataset validation OK: {len(df)} records, "
        f"metadata={metadata_path}, data_root={data_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
