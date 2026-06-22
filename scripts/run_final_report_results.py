from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from cough_analysis.paths import project_path


PYTHON = os.environ.get("PYTHON", sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and collect final EE492 report results.")
    parser.add_argument(
        "--output-root",
        default="artifacts/final_report_results/clean_v4_shared_split",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-v5", action="store_true", help="Skip the slow AST run.")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_path(path)


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_step(name: str, cmd: list[str], log_dir: Path, env: dict, skip_if: Path | None = None) -> None:
    if skip_if is not None and skip_if.exists():
        print(f"[skip] {name}: {skip_if}")
        return

    log_path = log_dir / f"{name}.log"
    print(f"\n[run] {name}")
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=project_path(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{name}] {line}", end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Step failed: {name}. See log: {log_path}")


def select_best_sweep(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if "predicted_events" not in df.columns and {"tp", "fp"}.issubset(df.columns):
        df["predicted_events"] = df["tp"] + df["fp"]
    selection_cols = ["f1", "mean_matched_iou", "precision", "predicted_events"]
    row = (
        df.sort_values(selection_cols, ascending=[False, False, False, True])
        .iloc[0]
        .to_dict()
    )
    return row


def value_is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return str(value) != ""


def v3_eval_args(row: dict) -> list[str]:
    args = [
        "--threshold",
        str(float(row["threshold"])),
        "--event-iou-threshold",
        "0.2",
        "--gt-min-duration-sec",
        "0.1",
        "--gt-merge-gap-sec",
        "0.1",
        "--pred-min-duration-sec",
        str(float(row["pred_min_duration_sec"])),
        "--pred-merge-gap-sec",
        str(float(row["pred_merge_gap_sec"])),
        "--pred-span-mode",
        str(row["pred_span_mode"]),
        "--prob-smoothing-sec",
        str(float(row["smoothing_sec"])),
    ]
    if value_is_present(row.get("hysteresis_low_threshold")):
        args.extend(["--hysteresis-low-threshold", str(float(row["hysteresis_low_threshold"]))])
    return args


def summarize_raw_model(root: Path, name: str, label: str) -> dict:
    event_path = root / "evaluations" / name / "test_event_metrics.json"
    report_path = root / "evaluations" / name / "test_classification_report.json"
    if not event_path.exists():
        return {"model": label, "available": False}
    event = json_load(event_path)
    report = json_load(report_path)
    cough_report = report.get("Cough", {})
    return {
        "model": label,
        "available": True,
        "window_f1": cough_report.get("f1-score"),
        "window_precision": cough_report.get("precision"),
        "window_recall": cough_report.get("recall"),
        "event_f1": event.get("f1"),
        "event_precision": event.get("precision"),
        "event_recall": event.get("recall"),
        "event_tp": event.get("true_positive"),
        "event_fp": event.get("false_positive"),
        "event_fn": event.get("false_negative"),
    }


def summarize_v3_model(root: Path, name: str, label: str) -> dict:
    event_path = root / "evaluations" / name / "test_event_metrics.json"
    report_path = root / "evaluations" / name / "test_classification_report.json"
    if not event_path.exists():
        return {"model": label, "available": False}
    event = json_load(event_path)
    report = json_load(report_path)
    cough_report = report.get("Cough", {})
    return {
        "model": label,
        "available": True,
        "window_f1": cough_report.get("f1-score"),
        "window_precision": cough_report.get("precision"),
        "window_recall": cough_report.get("recall"),
        "event_f1": event.get("f1"),
        "event_precision": event.get("precision"),
        "event_recall": event.get("recall"),
        "event_tp": event.get("true_positive"),
        "event_fp": event.get("false_positive"),
        "event_fn": event.get("false_negative"),
        "post_threshold": event.get("threshold"),
        "post_span_mode": event.get("pred_span_mode"),
        "post_smoothing_sec": event.get("prob_smoothing_sec"),
        "post_pred_min_duration_sec": event.get("pred_min_duration_sec"),
        "post_pred_merge_gap_sec": event.get("pred_merge_gap_sec"),
    }


def summarize_classical(root: Path) -> dict:
    base = root / "ee491_classical" / "tables"
    selected_path = base / "postprocessing_selected_val_test.csv"
    window_path = base / "window_metrics_summary.csv"
    if not selected_path.exists():
        return {"model": "EE491 Classical XGBoost", "available": False}
    event_row = pd.read_csv(selected_path).query("split == 'test'").iloc[0].to_dict()
    window_rows = pd.read_csv(window_path)
    window_row = window_rows.query("split == 'test'").iloc[0].to_dict()
    return {
        "model": "EE491 Classical XGBoost",
        "available": True,
        "window_f1": window_row.get("f1"),
        "window_precision": window_row.get("precision"),
        "window_recall": window_row.get("recall"),
        "event_f1": event_row.get("f1"),
        "event_precision": event_row.get("precision"),
        "event_recall": event_row.get("recall"),
        "event_tp": event_row.get("true_positive"),
        "event_fp": event_row.get("false_positive"),
        "event_fn": event_row.get("false_negative"),
        "post_threshold": event_row.get("threshold"),
        "post_pred_min_duration_sec": event_row.get("pred_min_duration_sec"),
        "post_pred_merge_gap_sec": event_row.get("pred_merge_gap_sec"),
    }


def summarize_v4(root: Path) -> dict:
    path = root / "evaluations" / "v4" / "test" / "v4_evaluation.json"
    if not path.exists():
        return {"model": "V4 Event + Activity", "available": False}
    data = json_load(path)
    cough = data.get("cough", {})
    end_to_end = data.get("end_to_end", {})
    return {
        "model": "V4 Event + Activity",
        "available": True,
        "event_f1": cough.get("f1"),
        "event_precision": cough.get("precision"),
        "event_recall": cough.get("recall"),
        "event_tp": cough.get("true_positive"),
        "event_fp": cough.get("false_positive"),
        "event_fn": cough.get("false_negative"),
        "activity_matched_accuracy": end_to_end.get("matched_activity_accuracy"),
    }


def summarize_v5(root: Path) -> dict:
    summary_path = root / "v5_ast" / "summary.json"
    selected_path = root / "v5_ast" / "tables" / "postprocessing_selected_val_test.csv"
    if not summary_path.exists() or not selected_path.exists():
        return {"model": "V5 AST + Motion", "available": False}
    summary = json_load(summary_path)
    window = {row["split"]: row for row in summary.get("window_metrics", [])}.get("test", {})
    event_row = pd.read_csv(selected_path).query("split == 'test'").iloc[0].to_dict()
    return {
        "model": "V5 AST + Motion",
        "available": True,
        "window_f1": window.get("f1"),
        "window_precision": window.get("precision"),
        "window_recall": window.get("recall"),
        "event_f1": event_row.get("f1"),
        "event_precision": event_row.get("precision"),
        "event_recall": event_row.get("recall"),
        "event_tp": event_row.get("true_positive"),
        "event_fp": event_row.get("false_positive"),
        "event_fn": event_row.get("false_negative"),
        "post_threshold": event_row.get("threshold"),
        "post_span_mode": event_row.get("span_mode"),
        "post_smoothing_sec": event_row.get("smoothing_sec"),
        "post_pred_min_duration_sec": event_row.get("pred_min_duration_sec"),
        "post_pred_merge_gap_sec": event_row.get("pred_merge_gap_sec"),
    }


def summarize_activity(root: Path, name: str, filename: str, label: str, metric_key: str) -> dict:
    path = root / "evaluations" / name / "test" / filename
    if not path.exists():
        return {"pipeline": label, "available": False}
    data = json_load(path)
    cough = data.get("cough", {})
    activity = data.get(metric_key, {})
    merged = activity.get("merged3", {})
    return {
        "pipeline": label,
        "available": True,
        "event_f1": cough.get("f1"),
        "event_precision": cough.get("precision"),
        "event_recall": cough.get("recall"),
        "matched_activity_accuracy": activity.get("matched_activity_accuracy"),
        "matched_activity_accuracy_merged3": merged.get("matched_activity_accuracy"),
        "matched_cough_events": activity.get("matched_cough_events"),
    }


def collect_results(root: Path) -> None:
    summaries_dir = root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    model_rows = [
        summarize_classical(root),
        summarize_raw_model(root, "v1", "V1 Raw Waveform CNN"),
        summarize_raw_model(root, "v2", "V2 Raw Waveform CNN"),
        summarize_v3_model(root, "v3_main", "V3 Log-Mel CNN"),
        summarize_v3_model(root, "v3_window04", "V3 0.4s/0.1s Boundary"),
        summarize_v4(root),
        summarize_v5(root),
    ]
    activity_rows = [
        summarize_activity(
            root,
            "v3_main_activity",
            "v3_cough_v4_activity_evaluation.json",
            "V3 Log-Mel CNN + V4 Activity",
            "activity_on_matched_v3_cough_events",
        ),
        summarize_activity(
            root,
            "v3_window04_activity",
            "v3_cough_v4_activity_evaluation.json",
            "V3 Boundary + V4 Activity",
            "activity_on_matched_v3_cough_events",
        ),
        summarize_activity(
            root,
            "v5_activity",
            "v5_ast_cough_v4_activity_evaluation.json",
            "V5 AST + V4 Activity",
            "activity_on_matched_v5_cough_events",
        ),
    ]
    pd.DataFrame(model_rows).to_csv(summaries_dir / "final_model_comparison.csv", index=False)
    pd.DataFrame(activity_rows).to_csv(summaries_dir / "final_activity_pipelines.csv", index=False)
    (summaries_dir / "final_model_comparison.json").write_text(
        json.dumps(model_rows, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )
    (summaries_dir / "final_activity_pipelines.json").write_text(
        json.dumps(activity_rows, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    root = project_or_absolute(args.output_root)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" if not env.get("PYTHONPATH") else f"src{os.pathsep}{env['PYTHONPATH']}"
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env.setdefault(key, "1")

    models_dir = root / "models"
    eval_dir = root / "evaluations"
    sweep_dir = root / "sweeps"
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    py = PYTHON if Path(PYTHON).exists() else sys.executable

    run_step(
        "ee491_classical",
        [
            py,
            "scripts/train_classical_ml.py",
            "--config",
            "configs/final/ee491_classical_clean.yaml",
            "--output-dir",
            str(root / "ee491_classical"),
        ],
        log_dir,
        env,
        skip_if=(root / "ee491_classical" / "summary.json") if args.skip_existing else None,
    )

    for name, config in [
        ("v1", "configs/final/v1_clean_raw_waveform.yaml"),
        ("v2", "configs/final/v2_clean_raw_waveform.yaml"),
    ]:
        ckpt = models_dir / f"{name}.pt"
        run_step(
            f"train_{name}",
            [py, "scripts/train_raw_cough.py", "--config", config, "--output", str(ckpt)],
            log_dir,
            env,
            skip_if=ckpt if args.skip_existing else None,
        )
        run_step(
            f"eval_{name}",
            [
                py,
                "scripts/evaluate_raw_cough.py",
                "--checkpoint",
                str(ckpt),
                "--config",
                config,
                "--split",
                "test",
                "--output-dir",
                str(eval_dir / name),
            ],
            log_dir,
            env,
            skip_if=(eval_dir / name / "test_event_metrics.json") if args.skip_existing else None,
        )

    v3_runs = [
        ("v3_main", "configs/final/v3_clean_all_records.yaml"),
        ("v3_window04", "configs/final/v3_clean_window04_hop01.yaml"),
    ]
    selected_v3 = {}
    for name, config in v3_runs:
        ckpt = models_dir / f"{name}.pt"
        run_step(
            f"train_{name}",
            [
                py,
                "scripts/train_v3.py",
                "--config",
                config,
                "--output",
                str(ckpt),
                "--model-id",
                name,
            ],
            log_dir,
            env,
            skip_if=ckpt if args.skip_existing else None,
        )
        sweep_csv = sweep_dir / f"{name}_val_sweep.csv"
        run_step(
            f"sweep_{name}",
            [
                py,
                "scripts/sweep_event_boundaries_v3.py",
                "--checkpoint",
                str(ckpt),
                "--config",
                config,
                "--split",
                "val",
                "--thresholds",
                "0.4,0.5,0.6,0.7,0.8,0.9",
                "--span-modes",
                "full,hop,center",
                "--gt-min-duration-sec",
                "0.1",
                "--gt-merge-gap-sec",
                "0.1",
                "--pred-min-duration-secs",
                "0.0,0.1,0.2",
                "--pred-merge-gap-secs",
                "0.0,0.1,0.2,0.3",
                "--smoothing-secs",
                "0.0,0.1,0.2,0.3",
                "--output-csv",
                str(sweep_csv),
            ],
            log_dir,
            env,
            skip_if=sweep_csv if args.skip_existing else None,
        )
        selected = select_best_sweep(sweep_csv)
        selected_v3[name] = selected
        (sweep_dir / f"{name}_selected_postprocessing.json").write_text(
            json.dumps(selected, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
            encoding="utf-8",
        )
        run_step(
            f"eval_{name}",
            [
                py,
                "scripts/evaluate_v3.py",
                "--checkpoint",
                str(ckpt),
                "--config",
                config,
                "--split",
                "test",
                "--output-dir",
                str(eval_dir / name),
                *v3_eval_args(selected),
            ],
            log_dir,
            env,
            skip_if=(eval_dir / name / "test_event_metrics.json") if args.skip_existing else None,
        )

    v4_dir = models_dir / "v4"
    run_step(
        "train_v4",
        [py, "scripts/train_v4.py", "--config", "configs/final/v4_clean.yaml", "--output-dir", str(v4_dir)],
        log_dir,
        env,
        skip_if=(v4_dir / "v4_summary.json") if args.skip_existing else None,
    )
    run_step(
        "eval_v4",
        [
            py,
            "scripts/evaluate_v4.py",
            "--config",
            "configs/final/v4_clean.yaml",
            "--model-dir",
            str(v4_dir),
            "--split",
            "test",
            "--output-dir",
            str(eval_dir / "v4"),
        ],
        log_dir,
        env,
        skip_if=(eval_dir / "v4" / "test" / "v4_evaluation.json") if args.skip_existing else None,
    )

    for name, _config in v3_runs:
        run_step(
            f"activity_{name}",
            [
                py,
                "scripts/evaluate_v3_activity.py",
                "--v3-checkpoint",
                str(models_dir / f"{name}.pt"),
                "--v4-model-dir",
                str(v4_dir),
                "--v4-config",
                "configs/final/v4_clean.yaml",
                "--split",
                "test",
                "--output-dir",
                str(eval_dir / f"{name}_activity"),
                *v3_eval_args(selected_v3[name]),
            ],
            log_dir,
            env,
            skip_if=(eval_dir / f"{name}_activity" / "test" / "v3_cough_v4_activity_evaluation.json")
            if args.skip_existing
            else None,
        )

    if not args.skip_v5:
        v5_dir = root / "v5_ast"
        run_step(
            "train_v5_ast",
            [py, "scripts/train_v5_ast.py", "--config", "configs/final/v5_ast_clean.yaml", "--output-dir", str(v5_dir)],
            log_dir,
            env,
            skip_if=(v5_dir / "summary.json") if args.skip_existing else None,
        )
        run_step(
            "activity_v5_ast",
            [
                py,
                "scripts/evaluate_v5_activity.py",
                "--v5-model-dir",
                str(v5_dir),
                "--v4-model-dir",
                str(v4_dir),
                "--v4-config",
                "configs/final/v4_clean.yaml",
                "--split",
                "test",
                "--output-dir",
                str(eval_dir / "v5_activity"),
            ],
            log_dir,
            env,
            skip_if=(eval_dir / "v5_activity" / "test" / "v5_ast_cough_v4_activity_evaluation.json")
            if args.skip_existing
            else None,
        )

    collect_results(root)
    print(f"\nDone. Results root: {root}")
    print(f"Summaries: {root / 'summaries'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
