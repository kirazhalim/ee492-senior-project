from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cough_analysis.paths import project_path


REQUIRED_METADATA_COLUMNS = [
    "record_id",
    "filename",
    "date",
    "subject",
    "activity",
    "context",
    "relative_path",
]


def load_metadata(metadata_path: str | Path | None = None) -> pd.DataFrame:
    metadata_path = (
        project_path("data", "metadata.csv")
        if metadata_path is None
        else Path(metadata_path)
    )
    df = pd.read_csv(metadata_path)

    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    return df.sort_values("record_id").reset_index(drop=True)


def get_record_row(metadata: pd.DataFrame, record_id: int) -> pd.Series:
    row = metadata.loc[metadata["record_id"] == record_id]
    if len(row) == 0:
        raise ValueError(f"record_id {record_id} not found in metadata.")
    return row.iloc[0]


def resolve_record_path(
    record_row: pd.Series,
    data_root: str | Path | None = None,
) -> Path:
    rel_path = Path(record_row["relative_path"])
    root = project_path("data") if data_root is None else Path(data_root)
    return root / rel_path


def decode_channel3(raw_col3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_col3 = raw_col3.astype(np.int64)
    cough_label = raw_col3 & 1
    stretch_signal = raw_col3 >> 1
    return stretch_signal.astype(np.float32), cough_label.astype(np.int64)


def load_record_array(record_path: str | Path, dtype=np.int64) -> np.ndarray:
    record_path = Path(record_path)
    df = pd.read_csv(record_path, header=None, quotechar='"')
    arr = df.to_numpy(dtype=dtype)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"Expected shape (N, 4), got {arr.shape} for file: {record_path}"
        )

    return arr


def load_record(
    record_id: int,
    metadata: pd.DataFrame | None = None,
    data_root: str | Path | None = None,
) -> dict:
    metadata = load_metadata() if metadata is None else metadata
    row = get_record_row(metadata, record_id)
    record_path = resolve_record_path(row, data_root=data_root)
    raw = load_record_array(record_path)

    pulmonary = raw[:, 0].astype(np.float32)
    ambient = raw[:, 1].astype(np.float32)
    stretch, cough_label = decode_channel3(raw[:, 2])
    accel_z = raw[:, 3].astype(np.float32)

    return {
        "record_id": int(row["record_id"]),
        "filename": row["filename"],
        "date": row["date"],
        "subject": row["subject"],
        "activity": row["activity"],
        "context": row["context"],
        "path": str(record_path),
        "pulmonary": pulmonary,
        "ambient": ambient,
        "stretch": stretch,
        "accel_z": accel_z,
        "cough_label": cough_label,
        "num_samples": raw.shape[0],
    }


def stack_channels(record: dict) -> np.ndarray:
    x = np.stack(
        [
            record["pulmonary"],
            record["ambient"],
            record["stretch"],
            record["accel_z"],
        ],
        axis=0,
    )
    return x.astype(np.float32)

