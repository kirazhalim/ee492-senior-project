from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Event:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def binary_labels_to_events(
    labels: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 0.0,
    merge_gap_sec: float = 0.0,
) -> list[Event]:
    labels = np.asarray(labels).astype(bool)
    events = []
    start_idx = None

    for idx, active in enumerate(labels):
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            events.append(Event(start_idx / sample_rate, idx / sample_rate))
            start_idx = None

    if start_idx is not None:
        event = Event(start_idx / sample_rate, len(labels) / sample_rate)
        events.append(event)

    return post_process_events(
        events,
        min_duration_sec=min_duration_sec,
        merge_gap_sec=merge_gap_sec,
    )


def post_process_events(
    events: list[Event],
    min_duration_sec: float = 0.0,
    merge_gap_sec: float = 0.0,
) -> list[Event]:
    if not events:
        return []

    sorted_events = sorted(events, key=lambda event: (event.start, event.end))
    merged = []
    current = sorted_events[0]

    for event in sorted_events[1:]:
        if event.start <= current.end + merge_gap_sec:
            current = Event(current.start, max(current.end, event.end))
        else:
            merged.append(current)
            current = event
    merged.append(current)

    return [
        event
        for event in merged
        if event.duration >= min_duration_sec
    ]


def window_predictions_to_events(
    spans: list[tuple[float, float]],
    predictions: np.ndarray,
    min_duration_sec: float = 0.0,
    merge_gap_sec: float = 0.0,
    span_mode: str = "full",
    center_fraction: float = 0.2,
) -> list[Event]:
    predictions = np.asarray(predictions).astype(bool)
    decision_spans = prediction_decision_spans(
        spans,
        span_mode=span_mode,
        center_fraction=center_fraction,
    )
    active_spans = [
        (float(start), float(end))
        for (start, end), active in zip(decision_spans, predictions)
        if active
    ]
    if not active_spans:
        return []

    events = []
    current_start, current_end = active_spans[0]
    for start, end in active_spans[1:]:
        if start <= current_end + merge_gap_sec:
            current_end = max(current_end, end)
        else:
            events.append(Event(current_start, current_end))
            current_start, current_end = start, end

    events.append(Event(current_start, current_end))
    return post_process_events(events, min_duration_sec=min_duration_sec)


def prediction_decision_spans(
    spans: list[tuple[float, float]],
    span_mode: str = "full",
    center_fraction: float = 0.2,
) -> list[tuple[float, float]]:
    if span_mode not in {"full", "center", "hop"}:
        raise ValueError(f"Unknown span_mode: {span_mode}")
    if not spans:
        return []

    clean_spans = [(float(start), float(end)) for start, end in spans]
    if span_mode == "full":
        return clean_spans

    centers = np.asarray([(start + end) / 2 for start, end in clean_spans], dtype=np.float32)
    if span_mode == "center":
        widths = [
            max(0.0, (end - start) * center_fraction)
            for start, end in clean_spans
        ]
    else:
        if len(centers) > 1:
            width = float(np.median(np.diff(centers)))
        else:
            start, end = clean_spans[0]
            width = end - start
        widths = [max(0.0, width) for _ in clean_spans]

    return [
        (max(0.0, float(center) - width / 2), float(center) + width / 2)
        for center, width in zip(centers, widths)
    ]


def smooth_probabilities(
    probabilities: np.ndarray,
    spans: list[tuple[float, float]],
    smoothing_sec: float = 0.0,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if smoothing_sec <= 0.0 or len(probs) < 2:
        return probs

    centers = np.asarray([(start + end) / 2 for start, end in spans], dtype=np.float32)
    if len(centers) < 2:
        return probs

    hop_sec = float(np.median(np.diff(centers)))
    if hop_sec <= 0.0:
        return probs

    window_size = max(1, int(round(float(smoothing_sec) / hop_sec)))
    if window_size <= 1:
        return probs

    left_pad = window_size // 2
    right_pad = window_size - 1 - left_pad
    padded = np.pad(probs, (left_pad, right_pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def probabilities_to_predictions(
    probabilities: np.ndarray,
    threshold: float,
    hysteresis_low_threshold: float | None = None,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    high = float(threshold)
    if hysteresis_low_threshold is None:
        return (probs >= high).astype(int)

    low = float(hysteresis_low_threshold)
    if low > high:
        raise ValueError("hysteresis_low_threshold must be <= threshold")

    predictions = np.zeros(len(probs), dtype=int)
    active = False
    for idx, prob in enumerate(probs):
        if active:
            if prob >= low:
                predictions[idx] = 1
            else:
                active = False
        elif prob >= high:
            active = True
            predictions[idx] = 1
    return predictions


def event_iou(a: Event, b: Event) -> float:
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union > 0 else 0.0


def match_events(
    gt_events: list[Event],
    pred_events: list[Event],
    iou_threshold: float = 0.2,
) -> list[tuple[int, int, float]]:
    candidates = []
    for gt_idx, gt_event in enumerate(gt_events):
        for pred_idx, pred_event in enumerate(pred_events):
            iou = event_iou(gt_event, pred_event)
            if iou >= iou_threshold:
                candidates.append((gt_idx, pred_idx, iou))

    candidates.sort(key=lambda x: x[2], reverse=True)
    matched_gt = set()
    matched_pred = set()
    matches = []

    for gt_idx, pred_idx, iou in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    return matches


def event_level_metrics(
    gt_events: list[Event],
    pred_events: list[Event],
    iou_threshold: float = 0.2,
) -> dict:
    matches = match_events(gt_events, pred_events, iou_threshold=iou_threshold)
    tp = len(matches)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

    return {
        "true_events": len(gt_events),
        "predicted_events": len(pred_events),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
        "iou_threshold": iou_threshold,
    }
