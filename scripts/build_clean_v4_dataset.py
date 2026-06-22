from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FS_AUDIO = 4800
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


OUTPUT_ROOT = project_path("data", "clean_v4")
REVIEW_CSV = OUTPUT_ROOT / "review_candidates.csv"
OUTPUT_CURATED = OUTPUT_ROOT / "curated_csv"
REPORT_CSV = OUTPUT_ROOT / "label_cleaning_report.csv"
VALID_DECISIONS = {"set_cough": 1, "set_non_cough": 0}


def apply_label_interval(
    encoded: np.ndarray,
    start_idx: int,
    end_idx: int,
    label_value: int,
) -> np.ndarray:
    cleaned = np.asarray(encoded, dtype=np.int64).copy()
    labels = cleaned & 1
    stretch = cleaned >> 1
    labels[start_idx:end_idx] = int(label_value)
    return (stretch << 1) | labels


def apply_decisions_to_encoded(
    encoded: np.ndarray,
    decisions: list[dict],
    fs_audio: int = FS_AUDIO,
) -> np.ndarray:
    cleaned = np.asarray(encoded, dtype=np.int64).copy()
    for row in decisions:
        decision = str(row.get("decision", "")).strip()
        if decision not in VALID_DECISIONS:
            continue

        start_idx = max(0, int(round(float(row["start_sec"]) * fs_audio)))
        end_idx = min(len(cleaned), int(round(float(row["end_sec"]) * fs_audio)))
        if end_idx > start_idx:
            cleaned = apply_label_interval(
                cleaned,
                start_idx,
                end_idx,
                VALID_DECISIONS[decision],
            )
    return cleaned


def decisions_for_record(review_df: pd.DataFrame, record_id: int) -> list[dict]:
    if review_df.empty:
        return []
    rows = review_df.loc[review_df["record_id"].astype(int) == int(record_id)]
    return rows.to_dict("records")


def main() -> int:
    metadata = pd.read_csv(project_path("data", "metadata.csv"))
    review_df = pd.read_csv(REVIEW_CSV) if REVIEW_CSV.exists() else pd.DataFrame()

    OUTPUT_CURATED.mkdir(parents=True, exist_ok=True)
    report_rows = []

    clean_metadata = metadata.copy()
    clean_metadata["relative_path"] = clean_metadata["filename"].map(
        lambda name: f"clean_v4/curated_csv/{name}"
    )

    for _, row in metadata.iterrows():
        record_id = int(row["record_id"])
        source_path = project_path("data") / str(row["relative_path"])
        output_path = OUTPUT_CURATED / str(row["filename"])

        df = pd.read_csv(source_path, header=None)
        original_encoded = df.iloc[:, 2].to_numpy(dtype=np.int64).copy()
        clean_encoded = apply_decisions_to_encoded(
            original_encoded,
            decisions_for_record(review_df, record_id),
        )

        df.iloc[:, 2] = clean_encoded
        df.to_csv(output_path, header=False, index=False)

        changed = int(np.sum((original_encoded & 1) != (clean_encoded & 1)))
        if changed:
            report_rows.append(
                {
                    "record_id": record_id,
                    "filename": row["filename"],
                    "changed_samples": changed,
                    "changed_duration_sec": changed / FS_AUDIO,
                }
            )

    clean_metadata.to_csv(OUTPUT_ROOT / "metadata.csv", index=False)
    pd.DataFrame(
        report_rows,
        columns=["record_id", "filename", "changed_samples", "changed_duration_sec"],
    ).to_csv(REPORT_CSV, index=False)

    print(f"Saved clean dataset: {OUTPUT_ROOT}")
    print(f"Changed records: {len(report_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
