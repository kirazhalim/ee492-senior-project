import numpy as np

from cough_analysis.event_metrics import (
    Event,
    binary_labels_to_events,
    event_iou,
    event_level_metrics,
    post_process_events,
    probabilities_to_predictions,
    prediction_decision_spans,
    smooth_probabilities,
    window_predictions_to_events,
)


def test_binary_labels_to_events():
    labels = np.array([0, 1, 1, 0, 1, 0])
    events = binary_labels_to_events(labels, sample_rate=2)
    assert events == [Event(0.5, 1.5), Event(2.0, 2.5)]


def test_window_predictions_to_events_merges_overlaps():
    spans = [(0.0, 1.0), (0.5, 1.5), (2.0, 3.0)]
    preds = np.array([1, 1, 1])
    events = window_predictions_to_events(spans, preds)
    assert events == [Event(0.0, 1.5), Event(2.0, 3.0)]


def test_window_predictions_to_events_can_use_hop_spans():
    spans = [(0.0, 1.0), (0.25, 1.25), (0.5, 1.5)]
    preds = np.array([1, 1, 1])
    events = window_predictions_to_events(spans, preds, span_mode="hop")
    assert events == [Event(0.375, 1.125)]


def test_prediction_decision_spans_center_mode():
    spans = [(0.0, 1.0)]
    decision_spans = prediction_decision_spans(
        spans,
        span_mode="center",
        center_fraction=0.2,
    )
    assert decision_spans == [(0.4, 0.6)]


def test_binary_labels_to_events_merges_small_gaps_before_filtering():
    labels = np.array([0, 1, 0, 1, 1, 0])
    events = binary_labels_to_events(
        labels,
        sample_rate=10,
        min_duration_sec=0.3,
        merge_gap_sec=0.11,
    )
    assert events == [Event(0.1, 0.5)]


def test_post_process_events_filters_short_events():
    events = [Event(0.0, 0.05), Event(1.0, 1.5)]
    assert post_process_events(events, min_duration_sec=0.1) == [Event(1.0, 1.5)]


def test_event_metrics_counts_matches():
    gt = [Event(0.0, 1.0), Event(3.0, 4.0)]
    pred = [Event(0.1, 1.1), Event(5.0, 6.0)]
    metrics = event_level_metrics(gt, pred, iou_threshold=0.2)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["f1"] == 0.5
    assert event_iou(gt[0], pred[0]) > 0.8


def test_smooth_probabilities_uses_span_hop():
    probs = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    spans = [(idx * 0.1, idx * 0.1 + 0.5) for idx in range(len(probs))]

    smoothed = smooth_probabilities(probs, spans, smoothing_sec=0.3)

    assert smoothed.shape == probs.shape
    assert np.allclose(smoothed, [0.0, 1 / 3, 1 / 3, 1 / 3, 0.0])


def test_probabilities_to_predictions_supports_hysteresis():
    probs = np.asarray([0.2, 0.8, 0.55, 0.45, 0.7, 0.4], dtype=np.float32)

    preds = probabilities_to_predictions(
        probs,
        threshold=0.7,
        hysteresis_low_threshold=0.5,
    )

    assert preds.tolist() == [0, 1, 1, 0, 1, 0]
