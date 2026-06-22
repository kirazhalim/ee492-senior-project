from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cough_analysis.classical_ml import (
    FEATURE_COLUMNS,
    build_classical_feature_table,
    feature_matrix,
    labels,
)
from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.event_metrics import (
    binary_labels_to_events,
    probabilities_to_predictions,
    window_predictions_to_events,
)
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import load_record_preprocessed
from cough_analysis.v3 import split_records_from_config
from cough_analysis.v4 import event_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final/ee491_classical_clean.yaml")
    parser.add_argument("--output-dir", default="artifacts/final/ee491_classical_clean")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_or_absolute(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )


def build_model(model_cfg: dict, y_train: np.ndarray, seed: int):
    model_type = str(model_cfg.get("type", "xgboost")).lower()
    pos = max(float(np.sum(y_train)), 1.0)
    neg = float(len(y_train) - np.sum(y_train))
    scale_pos_weight = neg / pos

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=int(model_cfg.get("n_estimators", 400)),
                max_depth=int(model_cfg.get("max_depth", 4)),
                learning_rate=float(model_cfg.get("learning_rate", 0.05)),
                subsample=float(model_cfg.get("subsample", 0.8)),
                colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
                reg_lambda=float(model_cfg.get("reg_lambda", 1.0)),
                min_child_weight=float(model_cfg.get("min_child_weight", 1.0)),
                gamma=float(model_cfg.get("gamma", 0.0)),
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
            )
        except ImportError:
            fallback = str(model_cfg.get("fallback", "hist_gradient_boosting"))
            if fallback != "hist_gradient_boosting":
                raise

    if model_type == "hist_gradient_boosting" or model_cfg.get("fallback") == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=int(model_cfg.get("max_iter", 400)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            max_leaf_nodes=int(model_cfg.get("max_leaf_nodes", 31)),
            l2_regularization=float(model_cfg.get("l2_regularization", 0.0)),
            class_weight={0: 1.0, 1: scale_pos_weight},
            random_state=seed,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if model_type == "svm_rbf":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported classical model type: {model_type}")


def window_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    report = classification_report(
        y_true,
        preds,
        target_names=["Non-Cough", "Cough"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
        "average_precision": float(average_precision_score(y_true, probs))
        if len(np.unique(y_true)) > 1
        else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0,
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def event_metrics_for_table(
    table: pd.DataFrame,
    probs: np.ndarray,
    record_ids: list[int],
    metadata: pd.DataFrame,
    data_root: Path | None,
    event_cfg: dict,
    threshold: float,
    pred_min_duration_sec: float,
    pred_merge_gap_sec: float,
) -> dict:
    gt_by_record = {}
    pred_by_record = {}
    for record_id in [int(x) for x in record_ids]:
        record = load_record_preprocessed(record_id, metadata=metadata, data_root=data_root)
        gt_by_record[record_id] = binary_labels_to_events(
            record["cough_label"],
            sample_rate=int(record["fs_audio"]),
            min_duration_sec=float(event_cfg["gt_min_duration_sec"]),
            merge_gap_sec=float(event_cfg["gt_merge_gap_sec"]),
        )
        mask = table["record_id"].to_numpy(dtype=int) == record_id
        if not np.any(mask):
            pred_by_record[record_id] = []
            continue
        spans = list(zip(table.loc[mask, "t0"].to_numpy(), table.loc[mask, "t1"].to_numpy()))
        preds = probabilities_to_predictions(probs[mask], threshold=threshold)
        pred_by_record[record_id] = window_predictions_to_events(
            spans,
            preds,
            min_duration_sec=pred_min_duration_sec,
            merge_gap_sec=pred_merge_gap_sec,
            span_mode="full",
            center_fraction=1.0,
        )
    return event_summary(
        gt_by_record,
        pred_by_record,
        iou_threshold=float(event_cfg["iou_threshold"]),
    )


def run_event_sweep(
    table: pd.DataFrame,
    probs: np.ndarray,
    record_ids: list[int],
    metadata: pd.DataFrame,
    data_root: Path | None,
    event_cfg: dict,
    sweep_cfg: dict,
) -> pd.DataFrame:
    rows = []
    for threshold in sweep_cfg["thresholds"]:
        for pred_min_duration_sec in sweep_cfg["pred_min_duration_sec"]:
            for pred_merge_gap_sec in sweep_cfg["pred_merge_gap_sec"]:
                metrics = event_metrics_for_table(
                    table,
                    probs,
                    record_ids,
                    metadata,
                    data_root,
                    event_cfg,
                    threshold=float(threshold),
                    pred_min_duration_sec=float(pred_min_duration_sec),
                    pred_merge_gap_sec=float(pred_merge_gap_sec),
                )
                rows.append(
                    {
                        "threshold": float(threshold),
                        "pred_min_duration_sec": float(pred_min_duration_sec),
                        "pred_merge_gap_sec": float(pred_merge_gap_sec),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["split"]["seed"])
    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    data_root = project_or_absolute(cfg["data"].get("data_root"))
    output_dir = project_or_absolute(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = split_records_from_config(metadata, cfg.get("split"), random_state=seed)
    split_ids = {
        "train": [int(x) for x in train_ids],
        "val": [int(x) for x in val_ids],
        "test": [int(x) for x in test_ids],
    }

    win_cfg = cfg["windowing"]
    if args.dry_run:
        table = build_classical_feature_table(
            [int(metadata.iloc[0]["record_id"])],
            metadata,
            data_root=data_root,
            window_sec=float(win_cfg["window_sec"]),
            hop_sec=float(win_cfg["hop_sec"]),
            label_overlap_tau=float(win_cfg["label_overlap_tau"]),
        )
        model = build_model(cfg["model"], labels(table), seed=seed)
        print(f"Dry run feature table shape: {table.shape}")
        print(f"Feature columns: {FEATURE_COLUMNS}")
        print(f"Model class: {model.__class__.__name__}")
        return 0

    tables = {
        split: build_classical_feature_table(
            ids,
            metadata,
            data_root=data_root,
            window_sec=float(win_cfg["window_sec"]),
            hop_sec=float(win_cfg["hop_sec"]),
            label_overlap_tau=float(win_cfg["label_overlap_tau"]),
            gt_min_duration_sec=float(cfg["event"]["gt_min_duration_sec"]),
            gt_merge_gap_sec=float(cfg["event"]["gt_merge_gap_sec"]),
        )
        for split, ids in split_ids.items()
    }
    for split, table in tables.items():
        table.to_csv(tables_dir / f"{split}_features.csv", index=False)

    X_train = feature_matrix(tables["train"])
    y_train = labels(tables["train"])
    model = build_model(cfg["model"], y_train, seed=seed)
    model.fit(X_train, y_train)

    results = []
    probs_by_split = {}
    for split in ["val", "test"]:
        X = feature_matrix(tables[split])
        probs = model.predict_proba(X)[:, 1]
        probs_by_split[split] = probs
        metrics = window_metrics(labels(tables[split]), probs, threshold=0.5)
        save_json(tables_dir / f"{split}_window_metrics.json", metrics)
        results.append(
            {
                "split": split,
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
        pred_table = tables[split][["record_id", "t0", "t1", "y_cough"]].copy()
        pred_table["probability"] = probs
        pred_table.to_csv(tables_dir / f"{split}_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(tables_dir / "window_metrics_summary.csv", index=False)

    val_sweep = run_event_sweep(
        tables["val"],
        probs_by_split["val"],
        split_ids["val"],
        metadata,
        data_root,
        event_cfg=cfg["event"],
        sweep_cfg=cfg["postprocessing_sweep"],
    )
    val_sweep.to_csv(tables_dir / "postprocessing_sweep_val.csv", index=False)
    selection_cols = ["f1", "mean_matched_iou", "precision", "predicted_events"]
    selected = (
        val_sweep.sort_values(selection_cols, ascending=[False, False, False, True])
        .iloc[0]
        .to_dict()
    )
    selected_test = event_metrics_for_table(
        tables["test"],
        probs_by_split["test"],
        split_ids["test"],
        metadata,
        data_root,
        event_cfg=cfg["event"],
        threshold=float(selected["threshold"]),
        pred_min_duration_sec=float(selected["pred_min_duration_sec"]),
        pred_merge_gap_sec=float(selected["pred_merge_gap_sec"]),
    )
    selected_summary = pd.DataFrame(
        [
            {"split": "val", **selected},
            {
                "split": "test",
                "threshold": selected["threshold"],
                "pred_min_duration_sec": selected["pred_min_duration_sec"],
                "pred_merge_gap_sec": selected["pred_merge_gap_sec"],
                **selected_test,
            },
        ]
    )
    selected_summary.to_csv(tables_dir / "postprocessing_selected_val_test.csv", index=False)

    model_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "config": cfg,
            "feature_columns": FEATURE_COLUMNS,
            "record_split": split_ids,
            "selected_postprocessing": selected,
        },
        model_path,
    )
    save_json(
        output_dir / "summary.json",
        {
            "model_path": str(model_path),
            "feature_columns": FEATURE_COLUMNS,
            "record_split": split_ids,
            "selected_postprocessing": selected_summary.to_dict(orient="records"),
            "window_metrics": results,
        },
    )
    print(f"Saved model: {model_path}")
    print(f"Saved tables: {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
