# V3 All-Records Results

This note records the V3 cough detector trained on the full 107-record metadata
set using the same record-level split as V4.

## Setup

- Config: `configs/v3_all_records.yaml`
- Checkpoint: `artifacts/models/v3_all_records.pt`
- Split: activity-stratified record holdout, seed 42
- Records: 74 train, 16 validation, 17 test
- Test records: same as V4
- Model: V3 log-mel audio branch + motion branch, late fusion

## Validation Selection

Post-processing was selected on the validation split using event F1, with event
IoU threshold 0.2.

Selected settings:

| Setting | Value |
| --- | ---: |
| Threshold | 0.8 |
| Prediction span mode | center |
| Prediction min duration | 0.2 s |
| Prediction merge gap | 0.1 s |
| GT min duration | 0.1 s |
| GT merge gap | 0.0 s |

Validation event result for the selected setting:

| Precision | Recall | F1 | TP / FP / FN |
| ---: | ---: | ---: | --- |
| 0.955 | 0.955 | 0.955 | 42 / 2 / 2 |

## Test Results

These results use the V4-compatible ground-truth event definition
(`gt_min_duration_sec=0.1`, `gt_merge_gap_sec=0.0`).

Window-level test results:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.915 |
| Cough precision | 0.868 |
| Cough recall | 0.787 |
| Cough F1 | 0.826 |
| Cough window support | 334 |

Event-level test results:

| Metric | Value |
| --- | ---: |
| True cough events | 61 |
| Predicted cough events | 54 |
| True positive events | 49 |
| False positive events | 5 |
| False negative events | 12 |
| Event precision | 0.907 |
| Event recall | 0.803 |
| Event F1 | 0.852 |

## Comparison With V4

Both rows below use the same 17-record test split and 61 true cough events.

| Model | Precision | Recall | F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | --- |
| V3 all-records | 0.907 | 0.803 | 0.852 | 49 / 5 / 12 |
| V4 cough detector | 0.821 | 0.902 | 0.859 | 55 / 12 / 6 |

Interpretation: V3 is more conservative and produces fewer false positives,
while V4 detects more true cough events. On this shared split, V4 has a slightly
higher event F1, but V3 has much higher precision.

## V3 Cough + V4 Activity

V3 was also evaluated as a hybrid pipeline: V3 produces cough events, then the
V4 motion-only activity classifier assigns activity to each matched cough event.
This does not retrain V3 for activity.

| Metric | Value |
| --- | ---: |
| V3 cough event F1 | 0.852 |
| Matched V3 cough events | 49 |
| 4-class activity accuracy on matched events | 0.816 |
| Sitting+standing merged activity accuracy | 0.939 |

Interpretation: activity assignment is almost the same as V4 because the same
V4 activity classifier is used. The main difference between the V3 and V4
end-to-end pipelines comes from the cough event detector, not from activity
classification.

## Source Files

- Validation sweep: `artifacts/error_analysis/v3_all_records/val_sweep_gtmerge00.csv`
- Test event metrics: `artifacts/evaluations/v3_all_records/test_selected_t08_center_min02_gap01_gtmerge00/test_event_metrics.json`
- Test classification report: `artifacts/evaluations/v3_all_records/test_selected_t08_center_min02_gap01_gtmerge00/test_classification_report.json`
- Hybrid V3+V4 activity report: `artifacts/evaluations/v3_activity/test/v3_cough_v4_activity_evaluation.json`
- Hybrid V3+V4 event activity CSV: `artifacts/evaluations/v3_activity/test/v3_cough_v4_activity_events.csv`
- V4 comparison source: `artifacts/evaluations/v4/test/v4_evaluation.json`
- Training history: `artifacts/models/v3_all_records.history.json`

## Reproduction Commands

```bash
PYTHONPATH=src .venv/bin/python scripts/train_v3.py \
  --config configs/v3_all_records.yaml \
  --output artifacts/models/v3_all_records.pt \
  --model-id v3_all_records

PYTHONPATH=src .venv/bin/python scripts/sweep_event_boundaries_v3.py \
  --checkpoint artifacts/models/v3_all_records.pt \
  --config configs/v3_all_records.yaml \
  --split val \
  --thresholds 0.3,0.4,0.5,0.6,0.7,0.8 \
  --span-modes full,hop,center \
  --gt-min-duration-sec 0.1 \
  --gt-merge-gap-sec 0.0 \
  --pred-min-duration-secs 0.1,0.2 \
  --pred-merge-gap-secs 0.1,0.2,0.3 \
  --smoothing-secs 0.0 \
  --output-csv artifacts/error_analysis/v3_all_records/val_sweep_gtmerge00.csv

PYTHONPATH=src .venv/bin/python scripts/evaluate_v3.py \
  --checkpoint artifacts/models/v3_all_records.pt \
  --config configs/v3_all_records.yaml \
  --split test \
  --threshold 0.8 \
  --gt-min-duration-sec 0.1 \
  --gt-merge-gap-sec 0.0 \
  --pred-min-duration-sec 0.2 \
  --pred-merge-gap-sec 0.1 \
  --pred-span-mode center \
  --output-dir artifacts/evaluations/v3_all_records/test_selected_t08_center_min02_gap01_gtmerge00

PYTHONPATH=src .venv/bin/python scripts/evaluate_v3_activity.py \
  --v3-checkpoint artifacts/models/v3_all_records.pt \
  --v4-model-dir artifacts/models/v4 \
  --split test
```
