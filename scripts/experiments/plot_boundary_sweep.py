from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cough_analysis.paths import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-csv", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in [
            "threshold",
            "pred_min_duration_sec",
            "pred_merge_gap_sec",
            "smoothing_sec",
            "precision",
            "recall",
            "f1",
            "mean_matched_iou",
            "mean_duration_ratio",
        ]:
            row[key] = float(row[key])
        row["hysteresis_low_threshold"] = (
            None
            if not row.get("hysteresis_low_threshold")
            else float(row["hysteresis_low_threshold"])
        )
    return rows


def format_label(row: dict) -> str:
    hysteresis = ""
    if row.get("hysteresis_low_threshold") is not None:
        hysteresis = f", low={row['hysteresis_low_threshold']:.1f}"
    return (
        f"thr={row['threshold']:.1f}, {row['pred_span_mode']}, "
        f"min={row['pred_min_duration_sec']:.1f}, "
        f"gap={row['pred_merge_gap_sec']:.1f}, "
        f"smooth={row['smoothing_sec']:.1f}"
        f"{hysteresis}"
    )


def save_precision_recall(rows: list[dict], output_path: Path, title: str) -> None:
    best_rows = sorted(
        rows,
        key=lambda row: (
            -row["f1"],
            -row["mean_matched_iou"],
            abs(row["mean_duration_ratio"] - 1.0),
        ),
    )[:3]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        [row["recall"] for row in rows],
        [row["precision"] for row in rows],
        c=[row["f1"] for row in rows],
        s=42,
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    for row in best_rows:
        ax.scatter(
            row["recall"],
            row["precision"],
            s=110,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.8,
        )
        ax.annotate(
            format_label(row),
            (row["recall"], row["precision"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title(f"{title}: Event Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Event F1")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_boundary_tradeoff(rows: list[dict], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        [row["mean_duration_ratio"] for row in rows],
        [row["mean_matched_iou"] for row in rows],
        c=[row["f1"] for row in rows],
        s=42,
        cmap="plasma",
        alpha=0.75,
        edgecolors="none",
    )
    top_iou = sorted(
        rows,
        key=lambda row: (-row["mean_matched_iou"], -row["f1"]),
    )[:3]
    for row in top_iou:
        ax.scatter(
            row["mean_duration_ratio"],
            row["mean_matched_iou"],
            s=110,
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=1.8,
        )
        ax.annotate(
            format_label(row),
            (row["mean_duration_ratio"], row["mean_matched_iou"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_title(f"{title}: Boundary Trade-off")
    ax.set_xlabel("Mean Duration Ratio")
    ax.set_ylabel("Mean Matched IoU")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Event F1")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(project_or_absolute(args.sweep_csv))
    output_dir = project_or_absolute(args.output_dir)
    save_precision_recall(
        rows,
        output_dir / f"{args.prefix}_precision_recall.png",
        args.title,
    )
    save_boundary_tradeoff(
        rows,
        output_dir / f"{args.prefix}_boundary_tradeoff.png",
        args.title,
    )
    print(f"Saved plots under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
