# V3 Baseline Results

This note records the current V3 cough detection baseline. The V3 model uses
log-mel spectrogram features from the pulmonary and ambient microphone channels,
low-pass filtered stretch and accelerometer features, and a late-fusion neural
network classifier.

## Dataset Split

The split is record-level, so windows from the same recording do not appear in
multiple splits.

| Split | Records | Duration (min) | Windows | Cough windows | Cough events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 59 | 19.67 | 4543 | 788 | 157 |
| Validation | 13 | 4.33 | 1001 | 226 | 43 |
| Test | 13 | 4.33 | 1001 | 192 | 37 |
| All | 85 | 28.33 | 6545 | 1206 | 237 |

Source files:

- `artifacts/dataset_summary/summary_by_split.csv`
- `artifacts/dataset_summary/dataset_summary.json`

## Threshold Selection

The decision threshold was selected using the validation split. The tested
thresholds were 0.4, 0.5, 0.6, and 0.7. Among these runs, threshold 0.6 gave the
best validation event F1 while also keeping event precision high.

| Split | Threshold | Event precision | Event recall | Event F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.4 | 0.795 | 0.721 | 0.756 |
| Validation | 0.5 | 0.821 | 0.744 | 0.780 |
| Validation | 0.6 | 0.842 | 0.744 | 0.790 |
| Validation | 0.7 | 0.821 | 0.744 | 0.780 |

The selected threshold for the final V3 test evaluation is:

```text
threshold = 0.6
```

## Final Test Evaluation

The final test evaluation uses the selected validation threshold of 0.6.

Window-level results:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Non-Cough | 0.978 | 0.948 | 0.963 | 809 |
| Cough | 0.806 | 0.911 | 0.856 | 192 |
| Accuracy |  |  | 0.941 | 1001 |
| Macro avg | 0.892 | 0.930 | 0.909 | 1001 |
| Weighted avg | 0.945 | 0.941 | 0.942 | 1001 |

Event-level results:

| Metric | Value |
| --- | ---: |
| True cough events | 37 |
| Predicted cough events | 36 |
| True positive events | 33 |
| False positive events | 3 |
| False negative events | 4 |
| Event precision | 0.917 |
| Event recall | 0.892 |
| Event F1 | 0.904 |
| Event IoU threshold | 0.2 |
| Event merge gap (s) | 0.0 |

Source files:

- `artifacts/evaluations/v3_cough_pt/original_split_085_records/baseline_full_eval/test_classification_report.json`
- `artifacts/evaluations/v3_cough_pt/original_split_085_records/baseline_full_eval/test_event_metrics.json`
- `artifacts/evaluations/v3_cough_pt/original_split_085_records/baseline_full_eval/test_confusion_matrix.png`
- `artifacts/evaluations/v3_cough_pt/original_split_085_records/baseline_full_eval/test_predictions.csv`

Reproduction command:

```bash
make eval-v3-mlflow PYTHON=.venv/bin/python EVAL_ARGS="--split test --threshold 0.6 --mlflow-run-name final_v3_test"
```

## Interpretation

The model reaches high window-level accuracy on the test split, but the more
important result for the cough detection task is the event-level performance.
With threshold 0.6, the model detects 33 out of 37 test cough events and produces
3 false positive events. This corresponds to an event F1 score of 0.904.

The remaining analysis should focus on the records that contain false positive
and false negative events. Those cases are useful for understanding whether
errors are caused by short cough segments, noisy non-cough audio, motion
artifacts, or boundary mismatches between predicted and ground-truth events.
