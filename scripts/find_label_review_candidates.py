from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FS_AUDIO = 4800
SHORT_SEC = 0.15
ZOOM_CONTEXT_SEC = 1.5


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


OUTPUT_ROOT = project_path("data", "clean_v4")
PLOT_ROOT = project_path("artifacts", "label_review_v4")


def label_events(labels: np.ndarray) -> list[tuple[int, int]]:
    events = []
    start = None
    for idx, active in enumerate(np.asarray(labels, dtype=bool)):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            events.append((start, idx))
            start = None
    if start is not None:
        events.append((start, len(labels)))
    return events


def candidate_rows(record_id: int, labels: np.ndarray) -> list[dict]:
    events = label_events(labels)
    rows = []
    short_samples = int(round(SHORT_SEC * FS_AUDIO))

    for start, end in events:
        if end - start >= short_samples:
            continue
        candidate_type = (
            "edge_short_cough"
            if start == 0 or end == len(labels)
            else "short_cough"
        )
        rows.append(
            {
                "record_id": record_id,
                "start_sec": start / FS_AUDIO,
                "end_sec": end / FS_AUDIO,
                "type": candidate_type,
                "suggestion": "set_non_cough",
            }
        )

    for (_, prev_end), (next_start, _) in zip(events, events[1:]):
        if next_start - prev_end < short_samples:
            rows.append(
                {
                    "record_id": record_id,
                    "start_sec": prev_end / FS_AUDIO,
                    "end_sec": next_start / FS_AUDIO,
                    "type": "short_gap",
                    "suggestion": "set_cough",
                }
            )
    return rows


def scaled(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = np.percentile(values, 95) - np.percentile(values, 5)
    if scale <= 0:
        return values - np.median(values)
    return (values - np.median(values)) / scale


def save_candidate_plot(
    df: pd.DataFrame,
    row: dict,
    output_path: Path,
    full_record: bool,
) -> None:
    labels = (df.iloc[:, 2].to_numpy(dtype=np.int64) & 1).astype(float)
    if full_record:
        start_idx = 0
        end_idx = len(df)
    else:
        center = (float(row["start_sec"]) + float(row["end_sec"])) / 2
        start_idx = max(0, int(round((center - ZOOM_CONTEXT_SEC) * FS_AUDIO)))
        end_idx = min(len(df), int(round((center + ZOOM_CONTEXT_SEC) * FS_AUDIO)))
    t = np.arange(start_idx, end_idx) / FS_AUDIO

    fig, axes = plt.subplots(5, 1, figsize=(12, 6), sharex=True)
    names = ["pulmonary", "ambient", "stretch", "accel_z"]
    series = [
        df.iloc[start_idx:end_idx, 0].to_numpy(),
        df.iloc[start_idx:end_idx, 1].to_numpy(),
        (df.iloc[start_idx:end_idx, 2].to_numpy(dtype=np.int64) >> 1),
        df.iloc[start_idx:end_idx, 3].to_numpy(),
    ]

    for ax, name, values in zip(axes[:4], names, series):
        ax.plot(t, scaled(values), linewidth=0.7)
        ax.axvspan(row["start_sec"], row["end_sec"], color="tab:red", alpha=0.18)
        ax.set_ylabel(name)
        ax.grid(True, linewidth=0.3, alpha=0.3)

    axes[4].step(t, labels[start_idx:end_idx], where="post", color="tab:red")
    axes[4].axvspan(row["start_sec"], row["end_sec"], color="tab:red", alpha=0.18)
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("label")
    axes[4].set_xlabel("time (s)")
    axes[4].grid(True, linewidth=0.3, alpha=0.3)

    fig.suptitle(
        f"record={row['record_id']} {row['type']} "
        f"{row['start_sec']:.3f}-{row['end_sec']:.3f}s "
        f"suggestion={row['suggestion']}"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def existing_decisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    existing = pd.read_csv(path, dtype={"candidate_id": str})
    existing["candidate_id"] = existing["candidate_id"].astype(str).str.zfill(5)
    return {
        row["candidate_id"]: {
            "decision": "" if pd.isna(row.get("decision", "")) else str(row.get("decision", "")),
            "notes": "" if pd.isna(row.get("notes", "")) else str(row.get("notes", "")),
        }
        for _, row in existing.iterrows()
    }


def main() -> int:
    metadata = pd.read_csv(project_path("data", "metadata.csv"))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    output_csv = OUTPUT_ROOT / "review_candidates.csv"
    previous = existing_decisions(output_csv)
    rows = []
    for _, meta in metadata.iterrows():
        record_id = int(meta["record_id"])
        path = project_path("data") / str(meta["relative_path"])
        df = pd.read_csv(path, header=None)
        labels = df.iloc[:, 2].to_numpy(dtype=np.int64) & 1

        for row in candidate_rows(record_id, labels):
            candidate_id = f"{len(rows):05d}"
            plot_path = PLOT_ROOT / f"{candidate_id}_record_{record_id:03d}_{row['type']}.png"
            zoom_plot_path = (
                PLOT_ROOT / f"{candidate_id}_record_{record_id:03d}_{row['type']}_zoom.png"
            )
            saved = previous.get(candidate_id, {"decision": "", "notes": ""})
            row = {
                "candidate_id": candidate_id,
                **row,
                "plot_path": str(plot_path.relative_to(project_path())),
                "zoom_plot_path": str(zoom_plot_path.relative_to(project_path())),
                "decision": saved["decision"],
                "notes": saved["notes"],
            }
            save_candidate_plot(df, row, plot_path, full_record=True)
            save_candidate_plot(df, row, zoom_plot_path, full_record=False)
            rows.append(row)

    pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "record_id",
            "start_sec",
            "end_sec",
            "type",
            "suggestion",
            "plot_path",
            "zoom_plot_path",
            "decision",
            "notes",
        ],
    ).to_csv(output_csv, index=False)

    print(f"Saved review candidates: {output_csv}")
    print(f"Saved plots: {PLOT_ROOT}")
    print(f"Candidates: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
