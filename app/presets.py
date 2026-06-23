"""Curated raw-CSV presets for the live demo.

The public repository includes only the held-out test preset CSVs required for
the demo, with anonymized filenames under ``app/demo_records``. If the private
``data/clean_v4/metadata.csv`` file is available locally, the app can also build
the same preset list directly from that metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from cough_analysis.paths import project_path


REPO_ROOT = Path(project_path()).resolve()
METADATA_PATH = REPO_ROOT / "data" / "clean_v4" / "metadata.csv"
PUBLIC_PRESET_ROOT = REPO_ROOT / "app" / "demo_records"

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


def _build_public_demo_presets() -> list[Preset]:
    """Use the anonymized CSV files bundled with the public demo."""
    presets: list[Preset] = []
    for path in sorted(PUBLIC_PRESET_ROOT.glob("record_*.csv")):
        stem = path.stem
        parts = stem.split("_", maxsplit=3)
        if len(parts) < 4:
            continue
        rid = parts[1]
        activity = parts[2]
        context = parts[3]
        presets.append(Preset(
            key=f"record_{rid}",
            label=f"Record {rid} ({activity})",
            description=context,
            relative_path=str(path.relative_to(REPO_ROOT)),
        ))
    return presets


PRESETS: list[Preset] = _build_test_split_presets(TEST_SPLIT_RECORD_IDS)
if not PRESETS:
    PRESETS = _build_public_demo_presets()
PRESETS_BY_KEY: dict[str, Preset] = {p.key: p for p in PRESETS}


__all__ = [
    "Preset",
    "PRESETS",
    "PRESETS_BY_KEY",
    "TEST_SPLIT_RECORD_IDS",
]
