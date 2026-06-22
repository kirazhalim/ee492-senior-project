"""Generate additional event-level result assets for the final report.

Produces:
  * event_tpfpfn_breakdown.png  -- stacked TP/FP/FN bars per model
  * event_count_summary.csv     -- predicted vs ground-truth event counts per model
  * test_window_pr_curves.png   -- window-level PR curves on the test split
"""

from __future__ import annotations

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
from sklearn.metrics import precision_recall_curve, average_precision_score


RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "final_report_results" / "clean_v4_shared_split"
OUT_DIR = RESULTS_ROOT / "report_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Event metric data sources, in the order they should appear in the report.
EVENT_SOURCES = [
    {
        "label": "EE491 ML Model",
        "json": RESULTS_ROOT / "ee491_classical" / "tables" / "postprocessing_selected_val_test.csv",
        "kind": "csv_test_row",
    },
    {
        "label": "Raw Waveform CNN",
        "json": RESULTS_ROOT / "evaluations" / "v1" / "test_event_metrics.json",
        "kind": "event_json",
    },
    {
        "label": "Spectrogram 2D CNN (V3)",
        "json": RESULTS_ROOT / "evaluations" / "v3_main" / "test_event_metrics.json",
        "kind": "event_json",
    },
    {
        "label": "Boundary-refined V3",
        "json": RESULTS_ROOT / "evaluations" / "v3_window04" / "test_event_metrics.json",
        "kind": "event_json",
    },
    {
        "label": "Frame-Level CNN (V4)",
        "json": RESULTS_ROOT / "evaluations" / "v4" / "test" / "v4_evaluation.json",
        "kind": "v4_json",
    },
    {
        "label": "Pretrained AST Model (V5)",
        "json": RESULTS_ROOT / "v5_ast" / "tables" / "postprocessing_selected_val_test.csv",
        "kind": "csv_test_row",
    },
]

# Window-level test predictions for the PR curve figure (label + probability columns).
PR_SOURCES = [
    ("EE491 ML Model",
     RESULTS_ROOT / "ee491_classical" / "tables" / "test_predictions.csv",
     "y_cough", "probability", "#6b7280"),
    ("Raw Waveform CNN",
     RESULTS_ROOT / "evaluations" / "v1" / "test_predictions.csv",
     "label", "probability", "#f59e0b"),
    ("Spectrogram 2D CNN (V3)",
     RESULTS_ROOT / "evaluations" / "v3_main" / "test_predictions.csv",
     "label", "probability", "#3b82f6"),
    ("Boundary-refined V3",
     RESULTS_ROOT / "evaluations" / "v3_window04" / "test_predictions.csv",
     "label", "probability", "#10b981"),
    ("Pretrained AST Model (V5)",
     RESULTS_ROOT / "evaluations" / "v5_activity" / "test" / "v5_ast_cough_v4_activity_events.csv",
     None, None, None),  # special-case handled below
]


def load_event_metrics() -> pd.DataFrame:
    rows = []
    for src in EVENT_SOURCES:
        path = src["json"]
        if src["kind"] == "event_json":
            data = json.loads(path.read_text())
            tp = int(data["true_positive"])
            fp = int(data["false_positive"])
            fn = int(data["false_negative"])
            true_events = int(data["true_events"])
            predicted_events = int(data["predicted_events"])
            precision = float(data["precision"])
            recall = float(data["recall"])
            f1 = float(data["f1"])
        elif src["kind"] == "v4_json":
            data = json.loads(path.read_text())["cough"]
            tp = int(data["true_positive"])
            fp = int(data["false_positive"])
            fn = int(data["false_negative"])
            true_events = int(data["true_events"])
            predicted_events = int(data["predicted_events"])
            precision = float(data["precision"])
            recall = float(data["recall"])
            f1 = float(data["f1"])
        elif src["kind"] == "csv_test_row":
            df = pd.read_csv(path)
            row = df[df["split"] == "test"].iloc[0]
            tp = int(row["true_positive"])
            fp = int(row["false_positive"])
            fn = int(row["false_negative"])
            true_events = int(row["true_events"])
            predicted_events = int(row["predicted_events"])
            precision = float(row["precision"])
            recall = float(row["recall"])
            f1 = float(row["f1"])
        else:
            raise ValueError(src["kind"])

        rows.append({
            "Model": src["label"],
            "Ground-truth events": true_events,
            "Predicted events": predicted_events,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })
    return pd.DataFrame(rows)


def plot_tpfpfn(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = df["Model"].tolist()
    tp = df["TP"].to_numpy()
    fp = df["FP"].to_numpy()
    fn = df["FN"].to_numpy()

    x = np.arange(len(labels))
    width = 0.65

    p_tp = ax.bar(x, tp, width, label="True Positive", color="#10b981")
    p_fp = ax.bar(x, fp, width, bottom=tp, label="False Positive", color="#ef4444")
    p_fn = ax.bar(x, fn, width, bottom=tp + fp, label="False Negative", color="#9ca3af")

    gt_total = int(df["Ground-truth events"].iloc[0])
    ax.axhline(gt_total, color="black", linestyle="--", linewidth=1.0,
               label=f"Ground-truth events ({gt_total})")

    for i, (t, f_p, f_n) in enumerate(zip(tp, fp, fn)):
        predicted = int(t + f_p)
        ax.text(i, t + f_p + f_n + 1.0, f"Pred={predicted}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Number of cough events")
    ax.set_title("Event-level outcome breakdown on the final test split")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, max(tp + fp + fn) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_pr_pair(path: Path, label_col: str, prob_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    y_true = df[label_col].to_numpy().astype(int)
    y_score = df[prob_col].to_numpy().astype(float)
    return y_true, y_score


def plot_pr_curves(event_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    drawn_any = False
    for (label, path, label_col, prob_col, color) in PR_SOURCES:
        if label_col is None:
            continue
        if not path.exists():
            continue
        y_true, y_score = _load_pr_pair(path, label_col, prob_col)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})",
                color=color, linewidth=1.8)
        drawn_any = True

    # Overlay event-level operating points from the final post-processed event detector.
    for _, row in event_df.iterrows():
        marker_label = row["Model"]
        ax.scatter(row["Recall"], row["Precision"],
                   marker="*", s=120, edgecolor="black", linewidth=0.6,
                   color="#fbbf24", zorder=5,
                   label="Event-level operating point" if marker_label == event_df["Model"].iloc[0] else None)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Window-level PR curves (test split) and final event-level operating points")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    if not drawn_any:
        raise RuntimeError("No PR sources were drawn; check input paths.")


def main() -> None:
    df = load_event_metrics()
    summary_path = OUT_DIR / "event_count_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    bar_path = OUT_DIR / "event_tpfpfn_breakdown.png"
    plot_tpfpfn(df, bar_path)
    print(f"wrote {bar_path}")

    pr_path = OUT_DIR / "test_window_pr_curves.png"
    plot_pr_curves(df, pr_path)
    print(f"wrote {pr_path}")


if __name__ == "__main__":
    main()
