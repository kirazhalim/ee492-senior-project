# V4 Event + Activity Summary

## Goal

V4 predicts cough events and assigns an activity label to each detected cough.
It uses two simple models instead of one joint model:

1. Cough event detector: audio + motion -> frame-level cough probability.
2. Activity classifier: motion-only -> activity probability over time.

The final output is a list of cough events with start/end time, cough
confidence, activity label, and activity confidence.

## Data And Split

- Dataset: all 107 records from `data/metadata.csv`.
- Split: record-level holdout, activity-stratified.
- Ratio: 70% train, 15% validation, 15% test.
- Seed: 42.
- Important limitation: this is not subject-level holdout, so results should
  not be described as unseen-subject performance.

## Input Configuration

Cough detector:

- Inputs: pulmonary mic, ambient mic, stretch, acc_z.
- Chunk length: 5.0 s.
- Chunk hop: 1.0 s.
- Frame step: 48 audio samples = 10 ms at 4800 Hz.
- Short GT cough events under 0.1 s are removed.
- Two spectrogram configs were trained:
  - `spec128`: `n_fft=128`, `hop_length=48`, `n_mels=48`.
  - `spec256`: `n_fft=256`, `hop_length=48`, `n_mels=64`.
- Frequency range: 60-2200 Hz.

Activity classifier:

- Inputs: stretch + acc_z only.
- Window length: 3.0 s.
- Window hop: 0.5 s.
- Original classes: sitting, standing, walking, running.

## Pipeline

Training:

1. Split records first.
2. Generate cough chunks and activity windows only inside each split.
3. Train both cough spectrogram variants.
4. Select the primary cough variant using validation event F1, with mean IoU
   as tie-breaker.
5. Train the motion-only activity classifier.

Inference:

1. Run cough detector over the full record and average overlapping chunk
   probabilities into one 10 ms probability timeline.
2. Convert cough probabilities to events using validation-selected
   post-processing.
3. Run activity classifier over the full record.
4. For each cough event, average activity probabilities from event start - 2 s
   to event end + 2 s.
5. Assign the highest-probability activity to the cough event.

## Post-Processing

Validation sweep:

- Thresholds: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8.
- Prediction merge gap: 0.1, 0.2, 0.3 s.
- Prediction minimum duration: 0.1, 0.2 s.
- Event match IoU threshold: 0.2.

Selected cough model:

- Primary spec: `spec128`.
- Threshold: 0.8.
- Merge gap: 0.3 s.
- Minimum predicted event duration: 0.2 s.

## Test Results

Cough event detection:

| Metric | Value |
| --- | ---: |
| True events | 61 |
| Predicted events | 67 |
| TP / FP / FN | 55 / 12 / 6 |
| Precision | 0.821 |
| Recall | 0.902 |
| F1 | 0.859 |
| Mean matched IoU | 0.723 |
| Mean onset error | 0.166 s |
| Mean offset error | 0.232 s |

Original 4-class activity:

| Metric | Value |
| --- | ---: |
| Window accuracy | 0.783 |
| Macro F1 | 0.697 |
| Weighted F1 | 0.750 |
| End-to-end matched activity accuracy | 0.818 |

Main weakness: standing is often confused with sitting.

Merged 3-class activity, with sitting + standing as `stationary`:

| Metric | Value |
| --- | ---: |
| Window accuracy | 0.933 |
| Macro F1 | 0.882 |
| Weighted F1 | 0.933 |
| End-to-end matched activity accuracy | 0.945 |

Merged confusion matrix:

| True \ Pred | stationary | walking | running |
| --- | ---: | ---: | ---: |
| stationary | 350 | 0 | 0 |
| walking | 20 | 170 | 20 |
| running | 0 | 0 | 35 |

Interpretation: using `stationary / walking / running` is reasonable if the
goal is coarse activity during cough. If the goal explicitly requires sitting
vs standing, the 4-class result should be reported with the standing weakness.

## Useful Files

- Config: `configs/v4.yaml`
- Training: `scripts/train_v4.py`
- Evaluation: `scripts/evaluate_v4.py`
- Single-record inspection: `scripts/inspect_v4_record.py`
- Report assets: `scripts/make_v4_report_assets.py`
- Test report: `artifacts/evaluations/v4/test/v4_evaluation.json`
- Example inspection output: `artifacts/inspections/v4/record_000_v4_timeline.png`
- 4-class confusion matrix: `artifacts/report_assets/v4/v4_activity_confusion_matrix_4class.png`
- Merged 3-class confusion matrix: `artifacts/report_assets/v4/v4_activity_confusion_matrix_merged3.png`
- TP/FP/activity-error examples: `artifacts/report_assets/v4/v4_event_examples.csv`
- TN segment example: `artifacts/report_assets/v4/v4_tn_segment_examples.csv`
- TP example timeline: `artifacts/report_assets/v4/timelines/cough_tp_activity_correct_record_005_timeline.png`
- FP example timeline: `artifacts/report_assets/v4/timelines/cough_fp_record_046_timeline.png`
- Activity-error timeline: `artifacts/report_assets/v4/timelines/cough_tp_activity_wrong_record_030_timeline.png`
