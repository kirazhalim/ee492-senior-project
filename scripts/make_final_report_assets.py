from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import load_record_preprocessed
from cough_analysis.v4 import cough_gt_events


MODEL_LABELS = {
    "EE491 Classical XGBoost": "EE491 Classical",
    "V1 Raw Waveform CNN": "V1 Raw CNN",
    "V2 Raw Waveform CNN": "V2 Aug. Raw CNN",
    "V3 Log-Mel CNN": "V3 Standard CNN",
    "V3 0.4s/0.1s Boundary": "V3-Fine CNN",
    "V4 Event + Activity": "V4 Pipeline",
    "V5 AST + Motion": "V5 AST-Motion",
}

PIPELINE_LABELS = {
    "V3 Log-Mel CNN + V4 Activity": "V3 + V4-Activity",
    "V3 Boundary + V4 Activity": "V3-Fine + V4-Activity",
    "V5 AST + V4 Activity": "V5 + V4-Activity",
}

ACTIVITY_CLASSES = ["stationary", "walking", "running"]
COLORS = {
    "precision": "#3b82f6",
    "recall": "#10b981",
    "f1": "#f59e0b",
    "activity": "#8b5cf6",
    "strict": "#ef4444",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready final EE492 assets.")
    parser.add_argument(
        "--results-root",
        default="artifacts/final_report_results/clean_v4_shared_split",
    )
    parser.add_argument("--metadata", default="data/clean_v4/metadata.csv")
    parser.add_argument(
        "--window-hop-summary",
        default="artifacts/window_hop_analysis/boundary_focused_summary.csv",
    )
    parser.add_argument("--v4-config", default="configs/final/v4_clean.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_path(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def setup_axes(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(
    df: pd.DataFrame,
    label_col: str,
    metric_cols: list[str],
    metric_labels: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
    ylim: tuple[float, float] = (0.0, 1.05),
) -> None:
    labels = df[label_col].tolist()
    x = np.arange(len(labels))
    width = min(0.22, 0.72 / max(len(metric_cols), 1))
    fig, ax = plt.subplots(figsize=(max(9.5, len(labels) * 1.25), 5.2))
    offsets = (np.arange(len(metric_cols)) - (len(metric_cols) - 1) / 2) * width
    color_values = [COLORS.get(name, f"C{idx}") for idx, name in enumerate(metric_cols)]

    for offset, col, label, color in zip(offsets, metric_cols, metric_labels, color_values):
        values = df[col].astype(float).to_numpy()
        bars = ax.bar(x + offset, values, width=width, label=label, color=color, alpha=0.88)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, ylim[1] - 0.02),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(*ylim)
    setup_axes(ax, title=title, ylabel=ylabel)
    ax.legend(ncol=len(metric_cols), loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    normalize: bool = False,
) -> None:
    matrix = np.asarray(matrix)
    display = matrix.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(display, cmap="Blues", vmin=0.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)

    threshold = float(display.max()) / 2.0 if display.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            raw = int(matrix[row, col])
            text = f"{raw}\n{display[row, col]:.0%}" if normalize else str(raw)
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if display[row, col] > threshold else "black",
                fontweight="bold" if raw else "normal",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, output_path)


def activity_target(activity: str) -> str:
    return "stationary" if activity in {"sitting", "standing"} else activity


def load_record_split(results_root: Path) -> dict[str, list[int]]:
    candidates = [
        results_root / "ee491_classical" / "summary.json",
        results_root / "evaluations" / "v5_activity" / "test" / "v5_ast_cough_v4_activity_evaluation.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = load_json(path)
        if "record_split" in data:
            return {
                split: [int(record_id) for record_id in ids]
                for split, ids in data["record_split"].items()
            }
    raise FileNotFoundError(f"Could not find record split metadata under {results_root}")


def make_dataset_assets(metadata_path: Path, results_root: Path, output_dir: Path) -> None:
    metadata = pd.read_csv(metadata_path)
    raw_order = ["sitting", "standing", "walking", "running"]
    merged_order = ["stationary", "walking", "running"]
    metadata["activity_merged"] = metadata["activity"].map(activity_target)

    raw_counts = metadata["activity"].value_counts().reindex(raw_order, fill_value=0)
    merged_counts = metadata["activity_merged"].value_counts().reindex(merged_order, fill_value=0)
    split_ids = load_record_split(results_root)
    split_lookup = {
        record_id: split for split, ids in split_ids.items() for record_id in ids
    }
    metadata["split"] = metadata["record_id"].map(split_lookup)

    save_table(
        pd.DataFrame(
            {
                "activity": raw_counts.index,
                "record_count": raw_counts.values,
            }
        ),
        output_dir / "dataset_activity_distribution.csv",
    )
    split_counts = (
        metadata.groupby(["split", "activity_merged"])["record_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(["train", "val", "test"])
        .reindex(columns=merged_order, fill_value=0)
    )
    split_counts.to_csv(output_dir / "dataset_split_activity_distribution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].bar(raw_counts.index, raw_counts.values, color="#64748b")
    setup_axes(axes[0], "Original Activity Labels", "Records")
    for idx, value in enumerate(raw_counts.values):
        axes[0].text(idx, value + 0.8, str(int(value)), ha="center", fontsize=9)
    axes[1].bar(merged_counts.index, merged_counts.values, color=["#64748b", "#3b82f6", "#10b981"])
    setup_axes(axes[1], "Merged Activity Labels", "Records")
    for idx, value in enumerate(merged_counts.values):
        axes[1].text(idx, value + 0.8, str(int(value)), ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, output_dir / "dataset_activity_distribution.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bottom = np.zeros(len(split_counts))
    colors = ["#64748b", "#3b82f6", "#10b981"]
    for activity, color in zip(merged_order, colors):
        values = split_counts[activity].to_numpy()
        ax.bar(split_counts.index, values, bottom=bottom, label=activity, color=color)
        for idx, value in enumerate(values):
            if value:
                ax.text(idx, bottom[idx] + value / 2, str(int(value)), ha="center", va="center", color="white")
        bottom += values
    setup_axes(ax, "Shared Record Split by Activity", "Records")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    save_fig(fig, output_dir / "dataset_split_activity_distribution.png")


def make_model_tables(results_root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = pd.read_csv(results_root / "summaries" / "final_model_comparison.csv")
    activity_df = pd.read_csv(results_root / "summaries" / "final_activity_pipelines.csv")
    model_df.insert(0, "report_name", model_df["model"].map(MODEL_LABELS).fillna(model_df["model"]))
    activity_df.insert(
        0,
        "report_name",
        activity_df["pipeline"].map(PIPELINE_LABELS).fillna(activity_df["pipeline"]),
    )

    save_table(model_df, output_dir / "report_model_metrics.csv")
    save_table(activity_df, output_dir / "report_activity_pipeline_metrics.csv")
    return model_df, activity_df


def make_metric_assets(model_df: pd.DataFrame, activity_df: pd.DataFrame, results_root: Path, output_dir: Path) -> None:
    event_df = model_df.loc[
        model_df["report_name"].isin(
            [
                "EE491 Classical",
                "V3 Standard CNN",
                "V3-Fine CNN",
                "V4 Pipeline",
                "V5 AST-Motion",
            ]
        )
    ].copy()
    event_df = event_df[
        ["report_name", "event_precision", "event_recall", "event_f1", "event_tp", "event_fp", "event_fn"]
    ].rename(
        columns={
            "event_precision": "precision",
            "event_recall": "recall",
            "event_f1": "f1",
        }
    )
    save_table(event_df, output_dir / "cough_event_metrics_main.csv")
    plot_grouped_bars(
        event_df,
        label_col="report_name",
        metric_cols=["precision", "recall", "f1"],
        metric_labels=["Precision", "Recall", "F1"],
        title="Event-Level Cough Detection on Shared Test Split",
        ylabel="Score",
        output_path=output_dir / "model_event_metrics_bar.png",
    )

    strict_df = build_strict_end_to_end_table(results_root)
    save_table(strict_df, output_dir / "strict_end_to_end_metrics.csv")
    plot_grouped_bars(
        strict_df,
        label_col="report_name",
        metric_cols=["cough_event_f1", "matched_activity_accuracy", "strict_end_to_end_f1"],
        metric_labels=["Cough Event F1", "Matched Activity Acc.", "Strict E2E F1"],
        title="End-to-End Cough Event and Activity Assignment",
        ylabel="Score",
        output_path=output_dir / "activity_pipeline_bar.png",
    )
    plot_grouped_bars(
        strict_df,
        label_col="report_name",
        metric_cols=["strict_end_to_end_precision", "strict_end_to_end_recall", "strict_end_to_end_f1"],
        metric_labels=["Strict Precision", "Strict Recall", "Strict F1"],
        title="Strict End-to-End Metric: Correct Event and Correct Activity",
        ylabel="Score",
        output_path=output_dir / "strict_end_to_end_metrics_bar.png",
    )

    raw_baselines = model_df.loc[
        model_df["report_name"].isin(["V1 Raw CNN", "V2 Aug. Raw CNN", "V3 Standard CNN", "V3-Fine CNN"])
    ][["report_name", "window_f1", "event_f1"]].copy()
    raw_baselines = raw_baselines.rename(columns={"window_f1": "window_f1", "event_f1": "event_f1"})
    save_table(raw_baselines, output_dir / "window_vs_event_metrics.csv")
    plot_grouped_bars(
        raw_baselines,
        label_col="report_name",
        metric_cols=["window_f1", "event_f1"],
        metric_labels=["Window F1", "Event F1"],
        title="Window-Level vs Event-Level Cough Metrics",
        ylabel="F1",
        output_path=output_dir / "window_vs_event_f1_bar.png",
    )


def build_strict_end_to_end_table(results_root: Path) -> pd.DataFrame:
    sources = [
        (
            "V4 Pipeline",
            results_root / "evaluations" / "v4" / "test" / "v4_evaluation.json",
            "v4",
        ),
        (
            "V3 + V4-Activity",
            results_root / "evaluations" / "v3_main_activity" / "test" / "v3_cough_v4_activity_evaluation.json",
            "v3",
        ),
        (
            "V3-Fine + V4-Activity",
            results_root / "evaluations" / "v3_window04_activity" / "test" / "v3_cough_v4_activity_evaluation.json",
            "v3",
        ),
        (
            "V5 + V4-Activity",
            results_root / "evaluations" / "v5_activity" / "test" / "v5_ast_cough_v4_activity_evaluation.json",
            "v5",
        ),
    ]
    rows = []
    for report_name, path, kind in sources:
        data = load_json(path)
        cough = data["cough"]
        if kind == "v4":
            correct = int(data["end_to_end"]["matched_with_correct_activity"])
            matched_acc = float(data["end_to_end"]["matched_activity_accuracy"])
            matched = int(data["end_to_end"]["matched_cough_events"])
        elif kind == "v5":
            activity = data["activity_on_matched_v5_cough_events"]
            correct = int(activity["matched_with_correct_activity"])
            matched_acc = float(activity["matched_activity_accuracy"])
            matched = int(activity["matched_cough_events"])
        else:
            activity = data["activity_on_matched_v3_cough_events"]
            correct = int(activity["matched_with_correct_activity"])
            matched_acc = float(activity["matched_activity_accuracy"])
            matched = int(activity["matched_cough_events"])

        true_events = int(cough["true_events"])
        predicted_events = int(cough["predicted_events"])
        strict_precision = correct / predicted_events if predicted_events else 0.0
        strict_recall = correct / true_events if true_events else 0.0
        strict_f1 = (
            2 * strict_precision * strict_recall / (strict_precision + strict_recall)
            if strict_precision + strict_recall
            else 0.0
        )
        rows.append(
            {
                "report_name": report_name,
                "true_events": true_events,
                "predicted_events": predicted_events,
                "matched_cough_events": matched,
                "correct_event_and_activity": correct,
                "cough_event_f1": float(cough["f1"]),
                "matched_activity_accuracy": matched_acc,
                "strict_end_to_end_precision": strict_precision,
                "strict_end_to_end_recall": strict_recall,
                "strict_end_to_end_f1": strict_f1,
            }
        )
    return pd.DataFrame(rows)


def make_activity_confusion_assets(results_root: Path, output_dir: Path) -> None:
    v4 = load_json(results_root / "evaluations" / "v4" / "test" / "v4_evaluation.json")
    plot_confusion_matrix(
        np.asarray(v4["activity"]["confusion_matrix"]),
        ACTIVITY_CLASSES,
        "V4-Activity Standalone Window Classification",
        output_dir / "activity_confusion_matrix_v4_windows.png",
        normalize=True,
    )

    event_sources = [
        (
            "V3 + V4-Activity Matched Events",
            results_root / "evaluations" / "v3_main_activity" / "test" / "v3_cough_v4_activity_evaluation.json",
            "activity_on_matched_v3_cough_events",
            output_dir / "activity_confusion_matrix_v3_standard_events.png",
        ),
        (
            "V3-Fine + V4-Activity Matched Events",
            results_root / "evaluations" / "v3_window04_activity" / "test" / "v3_cough_v4_activity_evaluation.json",
            "activity_on_matched_v3_cough_events",
            output_dir / "activity_confusion_matrix_v3_fine_events.png",
        ),
        (
            "V5 + V4-Activity Matched Events",
            results_root / "evaluations" / "v5_activity" / "test" / "v5_ast_cough_v4_activity_evaluation.json",
            "activity_on_matched_v5_cough_events",
            output_dir / "activity_confusion_matrix_v5_events.png",
        ),
    ]
    for title, path, key, output_path in event_sources:
        data = load_json(path)
        plot_confusion_matrix(
            np.asarray(data[key]["confusion_matrix"]),
            ACTIVITY_CLASSES,
            title,
            output_path,
            normalize=True,
        )


def make_pr_curve_asset(results_root: Path, output_dir: Path) -> None:
    sources = [
        ("V3 Standard CNN", results_root / "evaluations" / "v3_main" / "test_precision_recall_curve.json"),
        ("V3-Fine CNN", results_root / "evaluations" / "v3_window04" / "test_precision_recall_curve.json"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, path in sources:
        data = load_json(path)
        points = pd.DataFrame(data["points"])
        ax.plot(points["recall"], points["precision"], linewidth=2.0, label=f"{label} (AP={data['average_precision']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    setup_axes(ax, "Window-Level Precision-Recall Curves")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save_fig(fig, output_dir / "pr_curve_v3_standard_vs_fine.png")


def make_window_hop_asset(summary_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(summary_path)
    top = df.head(14).copy()
    selected = df[(np.isclose(df["window_sec"], 0.4)) & (np.isclose(df["hop_sec"], 0.1))].head(1)
    standard = df[(np.isclose(df["window_sec"], 1.0)) & (np.isclose(df["hop_sec"], 0.25))].head(1)
    compact = pd.concat([top, selected, standard]).drop_duplicates(["window_sec", "hop_sec"])
    compact = compact.sort_values(["rank"])
    save_table(
        compact[
            [
                "rank",
                "window_sec",
                "hop_sec",
                "total_windows",
                "oracle_event_f1",
                "oracle_mean_iou",
                "boundary_score",
                "mean_start_error_sec",
                "mean_end_error_sec",
            ]
        ],
        output_dir / "window_hop_candidate_summary.csv",
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    scatter = ax.scatter(
        df["total_windows"] / 1000,
        df["oracle_mean_iou"],
        c=df["window_sec"],
        s=70,
        cmap="viridis_r",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
    )
    for label, row_df, color in [
        ("selected 0.4s/0.1s", selected, "#ef4444"),
        ("standard 1.0s/0.25s", standard, "#111827"),
    ]:
        if not row_df.empty:
            row = row_df.iloc[0]
            ax.scatter(
                row["total_windows"] / 1000,
                row["oracle_mean_iou"],
                s=170,
                color=color,
                edgecolor="white",
                linewidth=1.4,
                zorder=4,
            )
            ax.annotate(
                label,
                xy=(row["total_windows"] / 1000, row["oracle_mean_iou"]),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xlabel("Total windows (thousands)")
    ax.set_ylabel("Oracle matched-event IoU")
    setup_axes(ax, "Annotation-Only Window/Hop Trade-Off")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Window size (s)")
    fig.tight_layout()
    save_fig(fig, output_dir / "window_hop_summary.png")


def draw_box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color,
        edgecolor="#1f2937",
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, fontweight="bold")


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#374151")
    ax.add_patch(arrow)


def make_pipeline_diagram(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.04, 0.62, 0.17, 0.16, "Audio + Motion\nSignals", "#e5e7eb")
    draw_box(ax, 0.28, 0.62, 0.20, 0.16, "Cough Detector\nV3 / V3-Fine / V5", "#dbeafe")
    draw_box(ax, 0.56, 0.62, 0.17, 0.16, "Event\nPost-processing", "#fef3c7")
    draw_box(ax, 0.80, 0.62, 0.16, 0.16, "Cough\nEvents", "#dcfce7")

    draw_box(ax, 0.04, 0.22, 0.17, 0.16, "Motion\nSignals", "#e5e7eb")
    draw_box(ax, 0.28, 0.22, 0.20, 0.16, "V4-Activity\nClassifier", "#ede9fe")
    draw_box(ax, 0.56, 0.22, 0.17, 0.16, "Event Activity\nAssignment", "#fee2e2")
    draw_box(ax, 0.80, 0.22, 0.16, 0.16, "Cough Event\n+ Activity", "#ccfbf1")

    draw_arrow(ax, (0.21, 0.70), (0.28, 0.70))
    draw_arrow(ax, (0.48, 0.70), (0.56, 0.70))
    draw_arrow(ax, (0.73, 0.70), (0.80, 0.70))
    draw_arrow(ax, (0.21, 0.30), (0.28, 0.30))
    draw_arrow(ax, (0.48, 0.30), (0.56, 0.30))
    draw_arrow(ax, (0.73, 0.30), (0.80, 0.30))
    draw_arrow(ax, (0.88, 0.62), (0.88, 0.38))

    ax.text(0.5, 0.93, "End-to-End Evaluation Pipeline", ha="center", va="center", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, output_dir / "pipeline_overview.png")


def make_v5_architecture_diagram(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.04, 0.66, 0.16, 0.14, "Pulmonary\nAudio", "#e5e7eb")
    draw_box(ax, 0.28, 0.66, 0.18, 0.14, "Frozen AST\nEncoder", "#dbeafe")
    draw_box(ax, 0.54, 0.66, 0.17, 0.14, "768-dim Audio\nEmbedding", "#bfdbfe")

    draw_box(ax, 0.04, 0.26, 0.16, 0.14, "Stretch + Acc Z\nMotion", "#e5e7eb")
    draw_box(ax, 0.28, 0.26, 0.18, 0.14, "Motion Conv1D\nBranch", "#ede9fe")
    draw_box(ax, 0.54, 0.26, 0.17, 0.14, "Motion\nEmbedding", "#ddd6fe")

    draw_box(ax, 0.78, 0.46, 0.17, 0.16, "Fusion Head\n+ Cough Logit", "#dcfce7")
    draw_box(ax, 0.78, 0.20, 0.17, 0.12, "Event\nPost-processing", "#fef3c7")

    draw_arrow(ax, (0.20, 0.73), (0.28, 0.73))
    draw_arrow(ax, (0.46, 0.73), (0.54, 0.73))
    draw_arrow(ax, (0.20, 0.33), (0.28, 0.33))
    draw_arrow(ax, (0.46, 0.33), (0.54, 0.33))
    draw_arrow(ax, (0.71, 0.73), (0.78, 0.56))
    draw_arrow(ax, (0.71, 0.33), (0.78, 0.52))
    draw_arrow(ax, (0.865, 0.46), (0.865, 0.32))

    ax.text(0.5, 0.92, "V5 AST-Motion Fusion", ha="center", va="center", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, output_dir / "v5_ast_motion_architecture.png")


def make_window_setting_table(output_dir: Path) -> None:
    rows = [
        ["EE491", "Classical Baseline", "0.2 s", "0.05 s", "handcrafted features"],
        ["V1", "Raw CNN", "1.0 s", "0.5 s", "raw waveform baseline"],
        ["V2", "Aug. Raw CNN", "1.0 s", "0.25 s", "augmentation baseline"],
        ["V3", "Standard CNN", "1.0 s", "0.25 s", "standard spectrogram CNN"],
        ["V3-Fine", "Fine-Resolution CNN", "0.4 s", "0.1 s", "event localization"],
        ["V4-Cough", "Frame Cough Detector", "5.0 s chunk", "frame hop 48 samples", "frame-level events"],
        ["V4-Activity", "Motion Activity Classifier", "3.0 s", "0.5 s", "activity assignment"],
        ["V5", "AST-Motion Fusion", "0.4 s", "0.1 s", "pretrained AST + motion"],
    ]
    df = pd.DataFrame(rows, columns=["version", "report_name", "window", "hop", "role"])
    save_table(df, output_dir / "model_window_hop_settings.csv")

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e5e7eb")
        elif row % 2:
            cell.set_facecolor("#f9fafb")
    ax.set_title("Model Variants and Window/Hop Settings", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, output_dir / "model_window_hop_settings.png")


def event_barh(
    ax,
    start: float,
    end: float,
    y: float,
    height: float,
    color: str,
    label: str | None = None,
) -> None:
    ax.broken_barh(
        [(start, max(0.02, end - start))],
        (y, height),
        facecolors=color,
        edgecolors="#111827",
        linewidth=0.75,
        alpha=0.86,
        label=label,
    )


def timeline_title(record_row: pd.Series, outcome: str) -> str:
    merged_activity = activity_target(str(record_row["activity"]))
    return (
        f"Record {int(record_row['record_id']):03d} | {merged_activity} | "
        f"{record_row['context']} | {outcome}"
    )


def choose_timeline_records(events: pd.DataFrame) -> list[tuple[str, int]]:
    grouped = events.groupby("record_id", sort=True)
    errors = []
    for record_id, group in grouped:
        matched = group["matched_gt"].astype(bool)
        correct = group["merged_activity_correct_if_matched"].astype(bool)
        if (matched & ~correct).any():
            errors.append((int(record_id), len(group)))

    selected: list[tuple[str, int]] = []
    if errors:
        selected.append(("activity_error", sorted(errors, key=lambda item: item[0])[0][0]))
    return selected


def make_timeline_plot(
    events: pd.DataFrame,
    gt_events,
    record_row: pd.Series,
    output_path: Path,
    outcome: str,
) -> None:
    duration = float(max(record_row.get("duration_sec", 0.0), events["cough_end"].max(), 1.0))
    fig, ax = plt.subplots(figsize=(10.5, 2.7))
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 3)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([2.25, 1.35, 0.45])
    ax.set_yticklabels(["GT cough", "Predicted cough", "Assigned activity"])
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(timeline_title(record_row, outcome), fontsize=12.5, fontweight="bold", pad=10)

    for idx, event in enumerate(gt_events):
        event_barh(
            ax,
            float(event.start),
            float(event.end),
            y=2.05,
            height=0.35,
            color="#ef4444",
            label="Ground truth" if idx == 0 else None,
        )

    activity_colors = {
        "stationary": "#64748b",
        "walking": "#3b82f6",
        "running": "#10b981",
    }
    status_colors = {
        "correct": "#22c55e",
        "wrong_activity": "#f59e0b",
        "false_positive": "#7f1d1d",
    }
    for _, row in events.iterrows():
        matched = bool(row["matched_gt"])
        correct = bool(row["merged_activity_correct_if_matched"])
        status = "correct" if matched and correct else "wrong_activity" if matched else "false_positive"
        start = float(row["cough_start"])
        end = float(row["cough_end"])
        activity = str(row["activity"])
        event_barh(ax, start, end, y=1.15, height=0.35, color=status_colors[status])
        event_barh(ax, start, end, y=0.25, height=0.35, color=activity_colors.get(activity, "#94a3b8"))
        ax.text(
            start + (end - start) / 2,
            0.68,
            f"{activity}\n{float(row['activity_confidence']):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#ef4444", label="Ground truth event"),
        plt.Rectangle((0, 0), 1, 1, color="#22c55e", label="Matched + activity correct"),
        plt.Rectangle((0, 0), 1, 1, color="#f59e0b", label="Matched + activity wrong"),
        plt.Rectangle((0, 0), 1, 1, color="#7f1d1d", label="False positive"),
    ]
    ax.legend(handles=legend_handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    save_fig(fig, output_path)


def make_qualitative_timeline_assets(results_root: Path, metadata_path: Path, v4_config_path: Path, output_dir: Path) -> None:
    events_path = results_root / "evaluations" / "v5_activity" / "test" / "v5_ast_cough_v4_activity_events.csv"
    if not events_path.exists():
        return
    events = pd.read_csv(events_path)
    if events.empty:
        return

    cfg = load_config(v4_config_path)
    metadata = load_metadata(metadata_path)
    selected = choose_timeline_records(events)
    if not selected:
        selected = []

    for stale_path in output_dir.glob("timeline_v5_record_*_*.png"):
        stale_path.unlink()

    record_cache = {}
    gt_cache = {}
    success_candidates = []
    for record_id, group in events.groupby("record_id", sort=True):
        record_id = int(record_id)
        record = load_record_preprocessed(
            record_id,
            metadata=metadata,
            data_root=project_or_absolute(cfg["data"]["data_root"]),
        )
        gt_events = cough_gt_events(record, cfg["cough"])
        record_cache[record_id] = record
        gt_cache[record_id] = gt_events
        matched = group["matched_gt"].astype(bool)
        correct = group["merged_activity_correct_if_matched"].astype(bool)
        no_false_positive = matched.all()
        no_false_negative = int(matched.sum()) == len(gt_events)
        if len(group) >= 2 and no_false_positive and no_false_negative and correct.all():
            success_candidates.append((len(group), record_id))

    if success_candidates:
        selected.insert(0, ("success", max(success_candidates)[1]))

    if not selected:
        return

    exported_rows = []
    for outcome, record_id in selected:
        record_events = events[events["record_id"] == record_id].copy()
        record_row = metadata.loc[metadata["record_id"] == record_id].iloc[0].copy()
        record = record_cache.get(record_id)
        if record is None:
            record = load_record_preprocessed(
                record_id,
                metadata=metadata,
                data_root=project_or_absolute(cfg["data"]["data_root"]),
            )
        record_row["duration_sec"] = float(record["duration_sec"])
        gt_events = gt_cache.get(record_id) or cough_gt_events(record, cfg["cough"])
        output_path = output_dir / f"timeline_v5_record_{record_id:03d}_{outcome}.png"
        make_timeline_plot(record_events, gt_events, record_row, output_path, outcome.replace("_", " "))
        record_events.insert(0, "example", outcome)
        exported_rows.append(record_events)

    if exported_rows:
        save_table(pd.concat(exported_rows, ignore_index=True), output_dir / "qualitative_timeline_events.csv")


def make_all_assets(args: argparse.Namespace) -> Path:
    results_root = project_or_absolute(args.results_root)
    output_dir = (
        project_or_absolute(args.output_dir)
        if args.output_dir
        else results_root / "report_assets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = project_or_absolute(args.metadata)
    make_dataset_assets(metadata_path, results_root, output_dir)
    model_df, activity_df = make_model_tables(results_root, output_dir)
    make_metric_assets(model_df, activity_df, results_root, output_dir)
    make_activity_confusion_assets(results_root, output_dir)
    make_pr_curve_asset(results_root, output_dir)
    make_window_hop_asset(project_or_absolute(args.window_hop_summary), output_dir)
    make_pipeline_diagram(output_dir)
    make_v5_architecture_diagram(output_dir)
    make_window_setting_table(output_dir)
    make_qualitative_timeline_assets(
        results_root,
        metadata_path,
        project_or_absolute(args.v4_config),
        output_dir,
    )

    manifest = {
        "output_dir": str(output_dir),
        "figures": sorted(path.name for path in output_dir.glob("*.png")),
        "tables": sorted(path.name for path in output_dir.glob("*.csv")),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def main() -> int:
    output_dir = make_all_assets(parse_args())
    print(f"Saved report assets: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
