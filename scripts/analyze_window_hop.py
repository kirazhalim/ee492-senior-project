from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cough_analysis.data import load_metadata, load_record
from cough_analysis.event_metrics import (
    Event,
    binary_labels_to_events,
    match_events,
    window_predictions_to_events,
)
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import FS_AUDIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze candidate window/hop sizes using only ground-truth cough "
            "annotations. No model training or prediction is used."
        )
    )
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--record-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--fs-audio", type=int, default=FS_AUDIO)
    parser.add_argument("--windows", default="0.5,1.0,2.0,3.0,5.0")
    parser.add_argument("--hop-fractions", default="0.2,0.25,0.5")
    parser.add_argument(
        "--pairs",
        default=None,
        help=(
            "Optional comma-separated window:hop pairs, e.g. "
            "'0.5:0.1,1.0:0.25'. Overrides --windows and --hop-fractions."
        ),
    )
    parser.add_argument(
        "--label-rule",
        choices=["center_positive", "any_positive", "overlap"],
        default="center_positive",
    )
    parser.add_argument("--center-fraction", type=float, default=0.2)
    parser.add_argument("--overlap-threshold", type=float, default=0.2)
    parser.add_argument(
        "--span-mode",
        choices=["hop", "center", "full"],
        default="hop",
        help="How oracle positive windows are converted back to event spans.",
    )
    parser.add_argument("--event-iou-threshold", type=float, default=0.2)
    parser.add_argument("--gt-min-duration-sec", type=float, default=0.1)
    parser.add_argument("--gt-merge-gap-sec", type=float, default=0.1)
    parser.add_argument("--pred-min-duration-sec", type=float, default=0.0)
    parser.add_argument(
        "--pred-merge-gap-sec",
        type=float,
        default=1e-6,
        help="Tiny default tolerance prevents floating-point span fragmentation.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/window_hop_analysis/summary.csv",
    )
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.pairs:
        pairs = []
        for item in args.pairs.split(","):
            if not item.strip():
                continue
            window, hop = item.split(":")
            pairs.append((float(window), float(hop)))
    else:
        windows = parse_float_list(args.windows)
        hop_fractions = parse_float_list(args.hop_fractions)
        pairs = [
            (window_sec, window_sec * hop_fraction)
            for window_sec in windows
            for hop_fraction in hop_fractions
        ]

    unique_pairs = []
    seen = set()
    for window_sec, hop_sec in pairs:
        if window_sec <= 0 or hop_sec <= 0:
            raise ValueError("Window and hop sizes must be positive.")
        if hop_sec > window_sec:
            continue
        key = (round(window_sec, 6), round(hop_sec, 6))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((float(key[0]), float(key[1])))
    return unique_pairs


def overlap_sec(span: tuple[float, float], events: list[Event]) -> float:
    start, end = span
    return float(
        sum(max(0.0, min(end, event.end) - max(start, event.start)) for event in events)
    )


def window_decision_span(
    start_sample: int,
    window_samples: int,
    fs_audio: int,
    center_fraction: float,
) -> tuple[float, float]:
    center_len = max(1, int(round(window_samples * center_fraction)))
    center_lo = (window_samples - center_len) // 2
    center_hi = center_lo + center_len
    return (
        (start_sample + center_lo) / fs_audio,
        (start_sample + center_hi) / fs_audio,
    )


def label_windows(
    num_samples: int,
    events: list[Event],
    window_sec: float,
    hop_sec: float,
    fs_audio: int,
    label_rule: str,
    center_fraction: float,
    overlap_threshold: float,
) -> tuple[list[tuple[float, float]], np.ndarray, list[float], int]:
    window_samples = int(round(window_sec * fs_audio))
    hop_samples = int(round(hop_sec * fs_audio))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("Window and hop sizes must map to at least one sample.")

    spans: list[tuple[float, float]] = []
    labels: list[int] = []
    positive_purities: list[float] = []
    ambiguous_windows = 0

    for start_sample in range(0, num_samples - window_samples + 1, hop_samples):
        full_span = (
            start_sample / fs_audio,
            (start_sample + window_samples) / fs_audio,
        )
        full_overlap = overlap_sec(full_span, events)

        if label_rule == "any_positive":
            label = int(full_overlap > 0.0)
        elif label_rule == "overlap":
            label = int((full_overlap / window_sec) >= overlap_threshold)
        elif label_rule == "center_positive":
            decision_span = window_decision_span(
                start_sample,
                window_samples,
                fs_audio,
                center_fraction,
            )
            label = int(overlap_sec(decision_span, events) > 0.0)
        else:
            raise ValueError(f"Unknown label rule: {label_rule}")

        if label:
            positive_purities.append(min(1.0, full_overlap / window_sec))
        elif full_overlap > 0.0:
            ambiguous_windows += 1

        spans.append(full_span)
        labels.append(label)

    return spans, np.asarray(labels, dtype=np.int64), positive_purities, ambiguous_windows


def count_covered_events(
    events: list[Event],
    spans: list[tuple[float, float]],
    labels: np.ndarray,
) -> int:
    positive_spans = [span for span, label in zip(spans, labels) if int(label) == 1]
    return sum(
        int(any(overlap_sec(span, [event]) > 0.0 for span in positive_spans))
        for event in events
    )


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_candidate(
    records: list[dict],
    window_sec: float,
    hop_sec: float,
    args: argparse.Namespace,
) -> dict:
    total_gt_events = 0
    total_windows = 0
    total_positive_windows = 0
    total_ambiguous_windows = 0
    total_covered_events = 0
    total_pred_events = 0
    total_duration_sec = 0.0
    tp = 0
    fp = 0
    fn = 0
    matched_ious: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    duration_ratios: list[float] = []
    positive_purities: list[float] = []

    for record in records:
        events = record["events"]
        spans, labels, purities, ambiguous_windows = label_windows(
            num_samples=record["num_samples"],
            events=events,
            window_sec=window_sec,
            hop_sec=hop_sec,
            fs_audio=args.fs_audio,
            label_rule=args.label_rule,
            center_fraction=args.center_fraction,
            overlap_threshold=args.overlap_threshold,
        )
        pred_events = window_predictions_to_events(
            spans,
            labels,
            min_duration_sec=args.pred_min_duration_sec,
            merge_gap_sec=args.pred_merge_gap_sec,
            span_mode=args.span_mode,
            center_fraction=args.center_fraction,
        )
        matches = match_events(
            events,
            pred_events,
            iou_threshold=args.event_iou_threshold,
        )

        record_tp = len(matches)
        record_fp = len(pred_events) - record_tp
        record_fn = len(events) - record_tp

        total_gt_events += len(events)
        total_windows += len(labels)
        total_positive_windows += int(np.sum(labels))
        total_ambiguous_windows += ambiguous_windows
        total_covered_events += count_covered_events(events, spans, labels)
        total_pred_events += len(pred_events)
        total_duration_sec += record["duration_sec"]
        tp += record_tp
        fp += record_fp
        fn += record_fn
        for gt_idx, pred_idx, iou in matches:
            gt_event = events[gt_idx]
            pred_event = pred_events[pred_idx]
            matched_ious.append(float(iou))
            start_errors.append(abs(pred_event.start - gt_event.start))
            end_errors.append(abs(pred_event.end - gt_event.end))
            if gt_event.duration > 0:
                duration_ratios.append(pred_event.duration / gt_event.duration)
        positive_purities.extend(purities)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    purities = np.asarray(positive_purities, dtype=np.float32)

    return {
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "label_rule": args.label_rule,
        "span_mode": args.span_mode,
        "records": len(records),
        "duration_min": total_duration_sec / 60.0,
        "gt_events": total_gt_events,
        "total_windows": total_windows,
        "windows_per_record": safe_divide(total_windows, len(records)),
        "positive_windows": total_positive_windows,
        "positive_window_rate": safe_divide(total_positive_windows, total_windows),
        "positive_windows_per_event": safe_divide(
            total_positive_windows,
            total_gt_events,
        ),
        "event_coverage": safe_divide(total_covered_events, total_gt_events),
        "ambiguous_windows": total_ambiguous_windows,
        "ambiguous_window_rate": safe_divide(total_ambiguous_windows, total_windows),
        "positive_purity_mean": float(np.mean(purities)) if len(purities) else 0.0,
        "positive_purity_median": float(np.median(purities)) if len(purities) else 0.0,
        "positive_purity_p25": float(np.percentile(purities, 25)) if len(purities) else 0.0,
        "oracle_pred_events": total_pred_events,
        "oracle_event_precision": precision,
        "oracle_event_recall": recall,
        "oracle_event_f1": f1,
        "oracle_mean_iou": mean_iou,
        "boundary_score": f1 * mean_iou,
        "mean_start_error_sec": float(np.mean(start_errors)) if start_errors else 0.0,
        "mean_end_error_sec": float(np.mean(end_errors)) if end_errors else 0.0,
        "mean_duration_ratio": float(np.mean(duration_ratios)) if duration_ratios else 0.0,
        "oracle_tp": tp,
        "oracle_fp": fp,
        "oracle_fn": fn,
    }


def load_records(args: argparse.Namespace) -> list[dict]:
    metadata = load_metadata(project_or_absolute(args.metadata))
    if args.record_ids is not None:
        selected = set(args.record_ids)
        metadata = metadata.loc[metadata["record_id"].isin(selected)].copy()
    if args.max_records is not None:
        metadata = metadata.head(args.max_records).copy()

    records = []
    for record_id in metadata["record_id"].tolist():
        record = load_record(int(record_id), metadata=metadata)
        labels = record["cough_label"]
        events = binary_labels_to_events(
            labels,
            sample_rate=args.fs_audio,
            min_duration_sec=args.gt_min_duration_sec,
            merge_gap_sec=args.gt_merge_gap_sec,
        )
        records.append(
            {
                "record_id": int(record_id),
                "num_samples": int(record["num_samples"]),
                "duration_sec": float(record["num_samples"] / args.fs_audio),
                "events": events,
            }
        )
    return records


def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(
        [
            "boundary_score",
            "oracle_event_f1",
            "oracle_mean_iou",
            "event_coverage",
            "positive_purity_median",
            "ambiguous_window_rate",
            "total_windows",
        ],
        ascending=[False, False, False, False, False, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def print_top_configs(df: pd.DataFrame, top_n: int = 10) -> None:
    display_cols = [
        "rank",
        "window_sec",
        "hop_sec",
        "total_windows",
        "positive_windows_per_event",
        "event_coverage",
        "positive_purity_median",
        "ambiguous_window_rate",
        "oracle_event_f1",
        "oracle_mean_iou",
        "boundary_score",
        "mean_start_error_sec",
        "mean_end_error_sec",
        "mean_duration_ratio",
    ]
    display = df.loc[:, display_cols].head(top_n).copy()
    for col in [
        "positive_windows_per_event",
        "event_coverage",
        "positive_purity_median",
        "ambiguous_window_rate",
        "oracle_event_f1",
        "oracle_mean_iou",
        "boundary_score",
        "mean_start_error_sec",
        "mean_end_error_sec",
        "mean_duration_ratio",
    ]:
        display[col] = display[col].map(lambda value: f"{value:.3f}")
    print(display.to_string(index=False))


def print_event_duration_summary(records: list[dict]) -> None:
    durations = np.asarray(
        [event.duration for record in records for event in record["events"]],
        dtype=np.float32,
    )
    if len(durations) == 0:
        print("No ground-truth events found after filtering.")
        return

    print(
        "GT event durations (sec): "
        f"p25={np.percentile(durations, 25):.3f}, "
        f"median={np.median(durations):.3f}, "
        f"p75={np.percentile(durations, 75):.3f}, "
        f"max={np.max(durations):.3f}"
    )


def main() -> int:
    args = parse_args()
    records = load_records(args)
    if not records:
        raise ValueError("No records selected.")

    rows = [
        summarize_candidate(records, window_sec, hop_sec, args)
        for window_sec, hop_sec in parse_pairs(args)
    ]
    ranked = add_rank(pd.DataFrame(rows))

    output_csv = project_or_absolute(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_csv, index=False)

    print(f"Analyzed {len(records)} records.")
    print_event_duration_summary(records)
    print(f"Saved summary: {output_csv}")
    print()
    print_top_configs(ranked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
