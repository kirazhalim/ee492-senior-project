"""Curated raw-CSV presets for the live demo.

The preset list is generated from the **test split** of the shared
record-holdout split used by every final model (V3 / V4 / V5 / classical XGB).
That split is defined by the seed=42 stratified record split in
``configs/final/*.yaml`` and is identical across all four model checkpoints — we
verified it once by reading the ``record_split["test"]`` field of
``artifacts/final_report_results/clean_v4_shared_split/models/v3_main.pt``.

Each preset points at the curated CSV under ``data/clean_v4/curated_csv``. That
file is byte-identical (modulo the optional CSV quotes) to the corresponding
``data/raw_csv`` file: 4 integer columns, no header, with the cough label still
bit-packed into column 3. ``preprocess_raw_csv`` works on it unchanged.

If we ever need to expand the demo beyond the test split, add manual entries to
``EXTRA_PRESETS`` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from cough_analysis.paths import project_path


REPO_ROOT = Path(project_path()).resolve()
METADATA_PATH = REPO_ROOT / "data" / "clean_v4" / "metadata.csv"

# Shared test split (record_ids), seed=42 stratified by activity.
# Source: artifacts/final_report_results/clean_v4_shared_split/models/v3_main.pt
TEST_SPLIT_RECORD_IDS: tuple[int, ...] = (
    5, 16, 27, 30, 33, 34, 42, 46, 59, 63, 67, 80, 82, 91, 95, 99, 106,
)


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    relative_path: str

    @property
    def absolute_path(self) -> Path:
        return REPO_ROOT / self.relative_path


def _build_test_split_presets(record_ids: Iterable[int]) -> list[Preset]:
    """Read metadata.csv once, emit one Preset per requested record_id."""
    if not METADATA_PATH.exists():
        # Demo can still launch (with an empty preset list) — useful when the
        # clean dataset hasn't been built yet.
        return []
    df = pd.read_csv(METADATA_PATH)
    wanted = set(int(x) for x in record_ids)
    rows = df.loc[df["record_id"].isin(wanted)].copy()
    rows = rows.sort_values("record_id")

    presets: list[Preset] = []
    for _, row in rows.iterrows():
        rid = int(row["record_id"])
        activity = str(row["activity"]).strip().lower() or "unknown"
        rel_path = str(row["relative_path"]).strip()
        # metadata.csv stores paths as ``clean_v4/curated_csv/...``; the
        # ``data`` prefix is implicit. Normalise to ``data/...`` for consistency
        # with the rest of the demo code.
        if not rel_path.startswith("data/"):
            rel_path = "data/" + rel_path.lstrip("./")
        presets.append(Preset(
            key=f"record_{rid:03d}",
            label=f"Record {rid:03d} ({activity})",
            description="",
            relative_path=rel_path,
        ))
    return presets


# Optional manual additions — append to PRESETS below if you want non-test
# records (e.g. an interesting walking+cough case from data/raw_csv). Leave
# empty by default so the demo strictly reflects the held-out test set.
EXTRA_PRESETS: list[Preset] = []


PRESETS: list[Preset] = _build_test_split_presets(TEST_SPLIT_RECORD_IDS) + EXTRA_PRESETS
PRESETS_BY_KEY: dict[str, Preset] = {p.key: p for p in PRESETS}


__all__ = [
    "EXTRA_PRESETS",
    "Preset",
    "PRESETS",
    "PRESETS_BY_KEY",
    "TEST_SPLIT_RECORD_IDS",
]
