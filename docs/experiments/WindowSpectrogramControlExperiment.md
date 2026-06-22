# Window And Spectrogram Control Experiment

This experiment checks whether the 0.5s / 0.1s setup improves the underlying
representation for boundary-aware cough detection, or whether it only looks
better because it creates more overlapping windows.

This is a validation diagnostic, not final model selection. Records 85-95
remain external data and should not be used to choose thresholds or
post-processing settings.

## Question

```text
Is 0.5s better, or does it only look better because hop=0.1 creates more windows?
```

## Compared Runs

All runs use the same model architecture, seed, metadata, center-positive label
rule, and train/val/test split.

| Run | Window | Hop | n_fft | mel hop | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `v3_window05_hop01_fft512_hop128` | 0.5s | 0.1s | 512 | 128 | Current 0.5s experiment baseline |
| `v3_window05_hop01_fft256_hop64` | 0.5s | 0.1s | 256 | 64 | Same windows, finer spectrogram time resolution |
| `v3_window10_hop01_fft512_hop128` | 1.0s | 0.1s | 512 | 128 | Same hop density, longer context |

## Fixed Evaluation Protocol

Primary validation read:

```text
threshold = 0.5
pred_span_mode = hop
event_iou_threshold = 0.2
gt_min_duration_sec = 0.1
gt_merge_gap_sec = 0.1
pred_min_duration_sec = 0.1
pred_merge_gap_sec = 0.3
prob_smoothing_sec = 0.0
```

The same protocol is applied to all runs. This avoids choosing a setting after
looking at one run's result.

## Validation Results

These results use only records 0-84 for training and validation split creation.
An earlier diagnostic was accidentally run over all 96 records; that result is
superseded and should not be used for conclusions because it contaminated the
frozen external records.

| Run | Window F1 | AP | Event P | Event R | Event F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.5s / 512-128` | 0.888 | 0.901 | 0.870 | 0.976 | 0.920 | 40 | 6 | 1 |
| `0.5s / 256-64` | 0.895 | 0.904 | 0.769 | 0.976 | 0.860 | 40 | 12 | 1 |
| `1.0s / 512-128` | 0.902 | 0.930 | 0.870 | 0.976 | 0.920 | 40 | 6 | 1 |

Boundary summary for matched events:

| Run | Mean IoU | Mean Start Error | Mean End Error | Mean Duration Ratio |
| --- | ---: | ---: | ---: | ---: |
| `0.5s / 512-128` | 0.757 | 0.129s | 0.193s | 0.93 |
| `0.5s / 256-64` | 0.741 | 0.198s | 0.163s | 1.14 |
| `1.0s / 512-128` | 0.708 | 0.232s | 0.211s | 1.40 |

Without merging adjacent predicted fragments (`pred_merge_gap_sec=0.0`), all
runs produce hundreds of false positive event fragments. This indicates that
post-processing is part of the current window-classification method, not an
optional cosmetic step.

## Interpretation

The answer is not simply "0.5s is better." The 0.5s / 512-128 and 1.0s /
512-128 runs have the same validation event F1 under the fixed merge setting.
This means the previous improvement was not only caused by using a shorter
window. Hop density and event merging are major factors.

However, the 1.0s run has the weakest boundary behavior: lower matched IoU,
larger start/end error, and a mean duration ratio of 1.40. It detects events as
well as 0.5s / 512-128 in event count terms, but tends to produce broader
regions. That is not aligned with the current goal of precise boundaries.

The 0.5s / 256-64 run does not prove that finer spectrogram time resolution is
better. Its window-level F1 and AP are similar to 0.5s / 512-128, but its event
precision drops because it produces more false positive event fragments,
especially on hard running/noise-like records.

## Timeline Error Analysis

The main error pattern is not random failure. Several false positives occur on
physically cough-like transients where audio, stretch, and/or acceleration move
together outside the annotated GT interval.

- Record 64 has a consistent false positive around 13.0-14.3s across all three
  runs. The signal looks cough-like, so this should be reviewed as a possible
  missing label, non-cough body artifact, or hard negative.
- Record 80 shows why 1.0s is risky for boundary precision: it detects the true
  events but also creates extra broad predicted regions outside GT. The 0.5s /
  512-128 run is more conservative; the 0.5s / 256-64 run is more fragmented.
- Record 84 is a running/clean hard case. Periodic motion and frequent audio
  bursts make false positives likely. The 0.5s / 256-64 run is especially
  sensitive here and produces four FPs.
- Record 13 and record 28 show that some FNs are not complete score misses.
  The model often assigns high probability near the cough, but the predicted
  event is too short, too shifted, or merged in a way that fails the event IoU
  matching rule.

Focused review outputs for records 13, 28, 64, 72, 80, 84, 67, and 81:

```text
artifacts/error_analysis/experiments_085/window_spectrogram_control/review_records/
```

The focused review adds `error_category`, nearest-event timing, nearest-event
IoU/gap, and a short `review_note` to `event_errors.csv`.

| Record | Review Category | Interpretation |
| ---: | --- | --- |
| 13 | `boundary_mismatch_fn` + `boundary_mismatch_fp` | The model scores the region, but the predicted event is too short to become a one-to-one event match. |
| 28 | `boundary_mismatch_fn` | Not a true score miss. The cough is scored highly, but event conversion/greedy matching leaves one GT unmatched in the 0.5s/256 and 1.0s runs. |
| 64 | `possible_missing_label_fp` | Manual note: likely missing label because the button may not have been pressed during recording. This should be treated as label-quality review, not model failure. |
| 72 | `hard_negative_fp` | Early isolated transient away from GT; high confidence, so this is a useful hard negative candidate unless manual review shows an unlabeled cough. |
| 80 | `hard_negative_fp` | Sitting/clean but contains several cough-like audio/motion bursts outside GT; 1.0s broadens this problem. |
| 84 | `hard_negative_fp` | Running/clean produces periodic motion and audio bursts; this is an activity-confound hard case. |
| 67 | `hard_negative_fp` | Noise context; the 0.5s/256 run becomes sensitive to noise-like bursts. |
| 81 | `hard_negative_fp` | Noise context; extra FPs appear mainly in the 0.5s/256 run, suggesting finer spectrogram resolution can increase sensitivity to noise motifs. |

## Current Conclusion

For the current V3 architecture, the main bottleneck appears to be the
window-classification-to-event conversion plus label/boundary ambiguity, not
only the spectrogram resolution. The method can detect events well after fixed
merging, but precise boundaries remain limited by:

- center-positive window labels,
- overlapping window predictions,
- merge/min-duration post-processing,
- cough-like noise windows,
- activity-related motion confounds.

The current best baseline for a boundary-focused V3 direction is 0.5s / 0.1s /
512-128, not because it wins every score, but because it matches the best event
F1 while keeping predicted durations and IoU more reasonable than 1.0s. The
next methodological step should be label/error review for records 13, 28, 64,
72, 80, 84, 67, and 81, plus adding diagnostics that distinguish true score
misses from boundary/IoU matching failures.
