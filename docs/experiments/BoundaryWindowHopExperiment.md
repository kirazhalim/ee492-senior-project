# Boundary Window/Hop Experiment

This note records the first controlled boundary-precision experiment on
`dataset_v1_085_records`.

## Goal

Improve temporal alignment between predicted cough events and ground-truth
events without sacrificing event detection performance.

## Setup

Baseline V3 used:

```text
window_sec = 1.0
hop_sec = 0.25
```

This experiment uses:

```text
config = configs/experiments/v3_window05_hop01.yaml
window_sec = 0.5
hop_sec = 0.1
center_fraction = 0.2
```

Training command:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_v3.py \
  --config configs/experiments/v3_window05_hop01.yaml \
  --output artifacts/models/v3_window05_hop01.pt \
  --max-records 85
```

The `--max-records 85` flag keeps the experiment on the original V3 dataset
slice (`record_id` 0-84), so the comparison stays controlled.

## Validation Selection

Boundary sweep on validation suggested two useful candidates:

| Candidate | Threshold | Span mode | Pred merge gap | Event F1 | Mean IoU | Duration ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| High event F1 | 0.8 | full | 0.0 | 0.965 | 0.702 | 1.13 |
| Tighter duration | 0.5 | hop | 0.3 | 0.920 | 0.757 | 0.93 |

The `hop` mode needed a non-zero prediction merge gap. With no merge gap, it
fragmented events too aggressively.

## Test Results

| Candidate | Event precision | Event recall | Event F1 | TP | FP | FN | Mean IoU | Duration ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full, threshold 0.8 | 0.971 | 0.971 | 0.971 | 33 | 1 | 1 | 0.739 | 1.22 |
| hop, threshold 0.5, gap 0.3 | 0.917 | 0.971 | 0.943 | 33 | 3 | 1 | 0.762 | 1.00 |

## External New-Data Check

After the controlled 0-84 record experiment, the selected candidates were also
checked on the newly added records (`record_id` 85-95). These records were not
used for training the `v3_window05_hop01` model.

This check is useful for generalization analysis, but it must not be used as a
free threshold-tuning set. If we tune thresholds or post-processing directly on
these records, they stop being an independent external check and should be
renamed as a calibration/development set.

| Model / candidate | Threshold | Span mode | Pred merge gap | Event precision | Event recall | Event F1 | Mean IoU | Duration ratio | Window cough F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original V3 baseline | 0.6 | hop | 0.0 | 0.849 | 0.882 | 0.865 | 0.666 | 1.39 | 0.804 |
| v3_window05_hop01 full | 0.8 | full | 0.0 | 0.891 | 0.804 | 0.845 | 0.565 | 0.91 | 0.509 |
| v3_window05_hop01 hop | 0.5 | hop | 0.3 | 0.721 | 0.863 | 0.786 | 0.588 | 0.81 | 0.674 |

External-check artifacts:

```text
artifacts/evaluations/new_records_085_095_model_comparison/
```

External error-analysis timelines:

```text
artifacts/error_analysis/new_records_085_095_model_comparison/
```

## Interpretation

On the original 0-84 dataset split, the smaller window/hop setup improved
event-level detection and boundary behavior. The best test event F1 came from
`full` span mode with a high threshold. The `hop` candidate produced tighter
predicted durations and a slightly higher mean IoU, but it also introduced more
false positive events.

For reporting the main improved detector, `full, threshold=0.8` is the cleaner
choice. For a boundary-refinement discussion, the `hop, threshold=0.5,
pred_merge_gap=0.3` candidate is useful because its predicted duration ratio is
closer to 1.0.

However, the external new-data check changes the conclusion. On records 85-95,
the original V3 baseline still has the strongest overall event-level
generalization (`event_f1=0.865`, `mean_iou=0.666`, `window_cough_f1=0.804`).
The 0.5s/0.1s model is not clearly better on these records. Its `full`
candidate has stronger precision but weaker recall and window-level cough F1,
while its `hop` candidate has more controlled durations but worse event F1.

Therefore, the current honest conclusion is:

```text
The 0.5s/0.1s setup improves the original internal test split, but this
improvement does not yet generalize clearly to the newly added external records.
The original V3 baseline remains the stronger external-check model for now.
```

## External Error Patterns

Problem timelines were generated for the external records to understand why the
models behave differently.

| Model / candidate | TP | FP | FN | Main failure pattern |
| --- | ---: | ---: | ---: | --- |
| original V3 baseline | 45 | 8 | 6 | Better overall recall, but sometimes merges noisy regions into long predictions. |
| v3_window05_hop01 full | 41 | 5 | 10 | More conservative; fewer FPs, but misses more events on walking/noise records. |
| v3_window05_hop01 hop | 44 | 17 | 7 | Recovers more events than `full`, but creates many short FP fragments. |

Observed examples:

1. `record_id=87` is difficult for the original V3 baseline. It produces a very
   long early FP region and still misses two GT events. This suggests that noisy
   standing recordings can create broad high-probability regions that do not map
   cleanly to individual cough events.

2. `record_id=88` and `record_id=94` are difficult for the 0.5s/0.1s model,
   especially in `full` mode. Several walking/noise cough events have local
   probabilities below the selected threshold or only partial overlap with the
   GT span. This points to under-detection rather than only bad boundary
   formatting.

3. The 0.5s/0.1s `hop` candidate often detects more local evidence, but the
   resulting events can become small isolated FP fragments. This is a
   post-processing issue: threshold, merge gap, minimum duration, and smoothing
   interact strongly with the short hop size.

4. Some FNs have high local maximum probability but low mean probability over
   the GT event. That means the model may briefly notice the cough but the
   event-construction rule does not preserve enough continuous support to form a
   matched event.

These patterns suggest that the next improvement should not be just "train a
new model and hope." We need a controlled post-processing/calibration pass on
the validation data, followed by one untouched external check.

## Internal Post-Processing Calibration

We froze the newly added records (`record_id` 85-95) as an external holdout and
ran post-processing calibration only on the original 0-84 validation split.

The calibration grid included:

```text
threshold: 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
span_mode: full, hop
pred_min_duration_sec: 0.0, 0.1, 0.2
pred_merge_gap_sec: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
prob_smoothing_sec: 0.0, 0.2, 0.3, 0.5
```

Calibration artifacts:

```text
artifacts/error_analysis/postprocessing_calibration_0_84/
artifacts/evaluations/postprocessing_calibration_0_84/
```

Top validation candidates:

| Model / candidate | Threshold | Span mode | Min duration | Merge gap | Smoothing | Val F1 | Val precision | Val recall | Val IoU | Val duration ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original V3 baseline | 0.9 | hop | 0.0 | 0.0 | 0.0 | 0.941 | 0.909 | 0.976 | 0.719 | 1.13 |
| v3_window05_hop01 top-F1 | 0.5 | hop | 0.2 | 0.1 | 0.5 | 0.965 | 0.932 | 1.000 | 0.726 | 0.88 |
| v3_window05_hop01 boundary-balanced | 0.4 | hop | 0.2 | 0.3 | 0.2 | 0.953 | 0.911 | 1.000 | 0.744 | 0.97 |

These candidates were selected from validation only, then checked on the
original 0-84 internal test split:

| Model / candidate | Test F1 | Test precision | Test recall | Test TP | Test FP | Test FN | Test IoU | Test duration ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original V3 baseline | 0.986 | 0.971 | 1.000 | 34 | 1 | 0 | 0.724 | 1.01 |
| v3_window05_hop01 top-F1 | 0.986 | 0.971 | 1.000 | 34 | 1 | 0 | 0.761 | 0.95 |
| v3_window05_hop01 boundary-balanced | 0.958 | 0.919 | 1.000 | 34 | 3 | 0 | 0.775 | 1.03 |

The cleanest internal result is the `v3_window05_hop01 top-F1` candidate. It
matches the calibrated baseline on event F1 while improving internal test IoU
from `0.724` to `0.761` and keeping predicted durations close to GT
(`duration_ratio=0.95`).

## External Check After Internal Calibration

We then evaluated the validation-selected candidates on the newly added
external records (`record_id` 85-95). This is useful and honest, but it also
means these records have now been inspected. From this point on, we should not
use records 85-95 to choose more thresholds or post-processing settings unless
we explicitly relabel them as a development set and reserve a newer untouched
holdout for final reporting.

| Model / candidate | External F1 | External precision | External recall | External IoU | External duration ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| original V3 baseline, old setting | 0.865 | 0.849 | 0.882 | 0.666 | 1.39 |
| original V3 baseline, validation-calibrated | 0.833 | 0.789 | 0.882 | 0.650 | 1.09 |
| v3_window05_hop01 top-F1 | 0.792 | 0.800 | 0.784 | 0.589 | 0.84 |
| v3_window05_hop01 boundary-balanced | 0.841 | 0.804 | 0.882 | 0.638 | 0.90 |

External-calibration artifacts:

```text
artifacts/error_analysis/postprocessing_calibration_0_84/external_085_095_*.csv
```

This result supports the overfitting concern. The internally selected
`v3_window05_hop01 top-F1` candidate does not generalize well to records 85-95.
The boundary-balanced 0.5s/0.1s candidate generalizes better than the top-F1
candidate and has a more reasonable duration ratio, but it still does not beat
the old original V3 baseline on external event F1 or IoU.

Current honest conclusion:

```text
The 0.5s/0.1s model can improve internal boundary behavior after calibration,
but the improvement is not robust on the current external records. The old V3
baseline remains the best external F1/IoU result, while the boundary-balanced
0.5s/0.1s candidate is better controlled in predicted duration.
```

## Visual Diagnostics

The plots below are generated from validation-only calibration sweeps. They are
useful for understanding the trade-off between event detection and boundary
precision.

## Input-Source Ablation

We also ran a controlled input-source ablation after noticing that some external
timelines, especially `record_id=87`, show visible cough structure in the
stretch and accelerometer channels. The goal is diagnostic: determine whether
the model can use motion information, not declare a final model from two
interesting examples.

Methodology:

```text
training records: 0-84 only
validation selection: internal validation split only
external check: records 85-95 only after validation selection
window/hop: 0.5s / 0.1s
span mode: hop
```

Input modes:

```text
full         = pulmonary + ambient + stretch + accel
audio_only   = pulmonary + ambient
motion_only  = stretch + accel
stretch_only = stretch
accel_only   = accel
```

Implementation keeps the same model architecture and masks unused input
channels. This avoids comparing different model capacities during the first
diagnostic pass.

Validation-selected settings:

| Input mode | Threshold | Hysteresis low | Min duration | Merge gap | Smoothing | Val F1 | Val precision | Val recall | Val IoU | Val duration ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.5 | 0.4 | 0.3 | 0.3 | 0.2 | 0.976 | 0.953 | 1.000 | 0.738 | 0.96 |
| audio_only | 0.9 | 0.4 | 0.3 | 0.3 | 0.2 | 0.951 | 0.951 | 0.951 | 0.692 | 0.87 |
| motion_only | 0.7 | 0.3 | 0.1 | 0.1 | 0.0 | 0.964 | 0.952 | 0.976 | 0.718 | 1.08 |
| stretch_only | 0.7 | 0.4 | 0.1 | 0.1 | 0.5 | 0.881 | 0.860 | 0.902 | 0.635 | 1.14 |
| accel_only | 0.4 | 0.3 | 0.1 | 0.1 | 0.5 | 0.897 | 0.946 | 0.854 | 0.761 | 0.97 |

Internal test results using the validation-selected settings:

| Input mode | Test F1 | Test precision | Test recall | Test IoU | Test duration ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.971 | 0.944 | 1.000 | 0.774 | 1.01 |
| audio_only | 1.000 | 1.000 | 1.000 | 0.692 | 0.85 |
| motion_only | 0.842 | 0.762 | 0.941 | 0.708 | 1.08 |
| stretch_only | 0.790 | 0.681 | 0.941 | 0.655 | 1.31 |
| accel_only | 0.895 | 0.810 | 1.000 | 0.752 | 0.92 |

External 85-95 check using the same validation-selected settings:

| Input mode | External F1 | External precision | External recall | External IoU | External duration ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.860 | 0.878 | 0.843 | 0.636 | 0.89 |
| audio_only | 0.780 | 0.796 | 0.765 | 0.585 | 1.31 |
| motion_only | 0.909 | 0.847 | 0.980 | 0.681 | 0.97 |
| stretch_only | 0.902 | 0.902 | 0.902 | 0.637 | 1.24 |
| accel_only | 0.836 | 0.718 | 1.000 | 0.676 | 1.21 |

Case-study timelines were generated for records 87 and 94:

```text
artifacts/error_analysis/ablation_input_sources/full_records_087_094/
artifacts/error_analysis/ablation_input_sources/audio_only_records_087_094/
artifacts/error_analysis/ablation_input_sources/motion_only_records_087_094/
artifacts/error_analysis/ablation_input_sources/stretch_only_records_087_094/
```

On records 87 and 94 only:

| Input mode | TP | FP | FN | Interpretation |
| --- | ---: | ---: | ---: | --- |
| full | 4 | 2 | 4 | Detects record 87 well, misses most of record 94. |
| audio_only | 4 | 4 | 4 | Similar recall issue on record 94, more FP on record 87. |
| motion_only | 8 | 4 | 0 | Detects both records, but still creates extra FP events. |
| stretch_only | 3 | 1 | 5 | Good on record 87, fails record 94 almost completely. |

Interpretation:

```text
The external ablation supports the hypothesis that motion information is useful
and may generalize better than audio on the current new noisy/walking records.
However, this is not enough to declare motion-only as the final model because
records 85-95 have now been inspected and may have a different distribution
from the original dataset.
```

The most useful takeaway is methodological: motion is not just a decorative
input. The full fusion model may not be using motion optimally, especially on
walking/noise records. The next step should be grouped cross-validation and
activity/context-level reporting, followed by a cleaner fusion experiment.

V3 baseline validation sweep:

![V3 baseline validation precision-recall](../../artifacts/error_analysis/postprocessing_calibration_0_84/plots/v3_baseline_val_precision_recall.png)

![V3 baseline validation boundary trade-off](../../artifacts/error_analysis/postprocessing_calibration_0_84/plots/v3_baseline_val_boundary_tradeoff.png)

0.5s/0.1s validation sweep:

![0.5s validation precision-recall](../../artifacts/error_analysis/postprocessing_calibration_0_84/plots/v3_window05_hop01_val_precision_recall.png)

![0.5s validation boundary trade-off](../../artifacts/error_analysis/postprocessing_calibration_0_84/plots/v3_window05_hop01_val_boundary_tradeoff.png)

Internal-test confusion matrices:

![Calibrated V3 baseline confusion matrix](../../artifacts/evaluations/postprocessing_calibration_0_84/v3_baseline_hop_t09_internal_test/test_confusion_matrix.png)

![0.5s top-F1 confusion matrix](../../artifacts/evaluations/postprocessing_calibration_0_84/v3_window05_top_f1_internal_test/test_confusion_matrix.png)

![0.5s boundary-balanced confusion matrix](../../artifacts/evaluations/postprocessing_calibration_0_84/v3_window05_boundary_balanced_internal_test/test_confusion_matrix.png)

These visualizations reinforce an important point: the current scores are not
bad. The issue is not that the detector cannot find coughs at all. The issue is
that event-level F1 can remain acceptable while predicted boundaries are still
too wide, too shifted, or too fragmented for precise temporal localization.

Therefore, for this project, event F1 should be treated as a detection metric,
not as the only success criterion. Boundary quality should be judged with IoU,
start/end error, duration ratio, and timeline inspection.

## Toward More Precise Detection

The most realistic path to more precise boundaries is staged. We should avoid
making the model more complex until we know which part of the pipeline is
causing the wide intervals.

1. Define the boundary target precisely.

   A model can be "correct" for detection even if it predicts a span that is
   too wide. We need to decide what precision means for this project: acceptable
   start error, acceptable end error, acceptable duration ratio, and minimum IoU.
   Without this, threshold tuning can improve one metric while hurting another.

2. Separate detection from boundary refinement.

   The current CNN is a window-level cough detector. It was not explicitly
   trained to predict exact start/end timestamps. A good practical design is:
   first detect candidate cough regions, then refine boundaries inside those
   regions with signal/post-processing rules or a second lightweight model.

3. Use local evidence inside detected events.

   For wide predictions, trim event boundaries using local probability shape
   and sensor energy. Examples: keep only the contiguous high-confidence core,
   trim low-probability tails, or locate onset/offset using short-time audio
   energy plus stretch/accelerometer changes. This should be validated on
   records 0-84 only before touching any newer holdout.

4. Prefer robust post-processing before retraining.

   Good next sweeps are hysteresis thresholding, probability smoothing windows,
   minimum/maximum event duration, gap merging, and event-core extraction. These
   directly target over-wide or fragmented predictions and are easier to audit
   than a larger model.

5. Do not optimize on records 85-95 anymore.

   Since we inspected them, they are no longer a clean final test set. They are
   still useful for qualitative diagnosis, but any new threshold chosen because
   it improves 85-95 would be optimistic. For a final number, we need a fresh
   holdout or a clearly defined grouped cross-validation protocol.

6. Consider model changes only after post-processing is exhausted.

   If post-processing cannot solve the boundary issue, then we can try training
   with shorter labels, soft labels around cough boundaries, sequence models,
   or an onset/offset head. But that should come after grouped validation shows
   the current representation is the bottleneck.

## Boundary-Focused Hysteresis Trial

To target boundary precision more directly, we added hysteresis event
construction. Instead of a single threshold, this starts an event when the
probability reaches a high threshold and keeps the event active while the
probability remains above a lower threshold.

This was evaluated only on the original 0-84 validation/test split, not tuned
on records 85-95.

Validation sweep artifact:

```text
artifacts/error_analysis/boundary_refinement_0_84/v3_window05_hysteresis_val_sweep.csv
```

Selected validation candidate:

```text
model = v3_window05_hop01
threshold = 0.5
hysteresis_low_threshold = 0.3
span_mode = hop
pred_min_duration_sec = 0.2
pred_merge_gap_sec = 0.1
prob_smoothing_sec = 0.0
```

| Split | Event F1 | Precision | Recall | TP | FP | FN | Mean IoU | Duration ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.965 | 0.932 | 1.000 | 41 | 3 | 0 | 0.745 | 0.97 |
| internal test | 0.958 | 0.919 | 1.000 | 34 | 3 | 0 | 0.779 | 1.02 |

Hysteresis plots:

![0.5s hysteresis validation precision-recall](../../artifacts/error_analysis/boundary_refinement_0_84/plots/v3_window05_hysteresis_val_precision_recall.png)

![0.5s hysteresis validation boundary trade-off](../../artifacts/error_analysis/boundary_refinement_0_84/plots/v3_window05_hysteresis_val_boundary_tradeoff.png)

Internal-test confusion matrix:

![0.5s hysteresis confusion matrix](../../artifacts/evaluations/boundary_refinement_0_84/v3_window05_hysteresis_internal_test/test_confusion_matrix.png)

Problem timelines:

```text
artifacts/error_analysis/boundary_refinement_0_84/v3_window05_hysteresis_internal_test/timelines/
```

This candidate is objectively not the highest-F1 internal option, but it is one
of the best boundary-focused options so far. Compared with the calibrated
top-F1 candidate, it trades event F1 from `0.986` to `0.958`, but improves
internal test mean IoU from `0.761` to `0.779` and moves duration ratio from
`0.95` to `1.02`.

The remaining errors are false positives on `record_id=40` and `record_id=82`.
There are no false negatives on the internal test split. This means the
boundary-focused failure mode is currently more about extra detections/noisy
regions than missed coughs.

External check on records 85-95:

| Split | Event F1 | Precision | Recall | TP | FP | FN | Mean IoU | Duration ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| external 85-95 | 0.775 | 0.717 | 0.843 | 43 | 17 | 8 | 0.602 | 0.86 |

External artifacts:

```text
artifacts/error_analysis/boundary_refinement_0_84/external_085_095_v3_window05_hysteresis_selected.csv
artifacts/error_analysis/boundary_refinement_0_84/external_085_095_v3_window05_hysteresis/
```

This is not a strong external result. The hysteresis candidate improved
internal-test boundary behavior, but it does not generalize well to the current
external records. The external errors are both false positives and false
negatives, with `record_id=88` and `record_id=94` especially difficult. This
suggests that the issue is not only event-width post-processing; the model also
struggles to separate cough-like noise from true coughs in some walking/noise
recordings.

## What To Fix Next

The external-check result points to a generalization problem rather than only a
boundary-formatting problem. The next steps should separate model learning,
threshold selection, and final evaluation.

1. Treat records 85-95 as an inspected external check, not as a fresh untouched
   final test set anymore.

2. Do threshold and post-processing selection only on the training/validation
   portion of records 0-84. Do not pick new thresholds because they look better
   on records 85-95.

3. Run grouped cross-validation on records 0-84, preferably grouped by subject,
   to check whether the 0.5s/0.1s setup is consistently better or only lucky on
   one split.

4. Compare models using the same reporting table every time: window metrics,
   event precision/recall/F1, mean IoU, start/end boundary error, duration
   ratio, and per-record failures.

5. Inspect per-record failures before changing the model. In the external check,
   some records show under-detection and some show over-long predictions, so a
   single global threshold may not solve the whole problem.

6. If the smaller-window model remains unstable, try calibration and
   post-processing before increasing model complexity: threshold sweep,
   prediction merge gap sweep, minimum event duration sweep, and possibly
   probability smoothing.

7. Only after that, consider retraining with more data or a different loss
   strategy. If we need a publishable final number, collect or reserve a newer
   untouched holdout because records 85-95 have now influenced our analysis.

## Artifacts

```text
artifacts/models/v3_window05_hop01.pt
artifacts/error_analysis/v3_window05_hop01/
artifacts/evaluations/v3_window05_hop01/
```

Problem timelines:

```text
artifacts/error_analysis/v3_window05_hop01/full_thr08_test/timelines/
artifacts/error_analysis/v3_window05_hop01/hop_thr05_gap03_test/timelines/
```
