# Window-Hop and Boundary-Focused Model Summary

## Goal

The goal was to choose the window and hop size before training the model, using
only the ground-truth cough annotations. The final model was optimized for
event-level cough detection and temporal boundary alignment, not only for
window-level classification.

## Pre-Model Window-Hop Analysis

We added an annotation-only analysis script:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_window_hop.py
```

The script evaluates candidate window/hop configurations without training a
model. It computes event coverage, label purity, ambiguous windows, oracle
event F1, mean IoU, boundary score, start error, and end error.

Ground-truth cough event duration summary:

| Metric | Duration |
| --- | ---: |
| 25th percentile | 0.914 s |
| Median | 1.116 s |
| 75th percentile | 1.427 s |
| Max | 3.454 s |

The boundary-focused sweep showed that very small windows such as
`0.25s / 0.05s` gave the best theoretical boundary alignment, but produced too
many windows and risked losing useful cough context. A more balanced choice was:

```text
window_sec = 0.4
hop_sec = 0.1
```

This configuration preserved high boundary precision while keeping enough local
context for the model.

## Selected Model Configuration

Config file:

```text
configs/experiments/v3_window04_hop01_fft256_hop64.yaml
```

Key settings:

| Parameter | Value |
| --- | ---: |
| Window size | 0.4 s |
| Hop size | 0.1 s |
| Label rule | center_positive |
| Center fraction | 0.2 |
| Spectrogram FFT | 256 |
| Spectrogram hop length | 64 |

## Validation-Based Post-Processing

Post-processing was selected on the validation split only, then applied to the
test split.

Selected event construction settings:

| Parameter | Value |
| --- | ---: |
| Threshold | 0.90 |
| Prediction span mode | full |
| Prediction merge gap | 0.1 s |
| Probability smoothing | 0.3 s |
| GT min duration | 0.1 s |
| GT merge gap | 0.1 s |
| Event IoU threshold | 0.2 |

## Final Results

| Split | Event F1 | Precision | Recall | TP | FP | FN | Mean IoU | Start Error | End Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.990 | 0.980 | 1.000 | 49 | 1 | 0 | 0.782 | 130 ms | 202 ms |
| Test | 0.956 | 0.982 | 0.931 | 54 | 1 | 4 | 0.718 | 185 ms | 229 ms |

The selected test setting produced only one false positive and maintained close
temporal alignment with the annotated cough events. The predicted duration ratio
was approximately `0.95`, which means predicted event lengths were close to
ground-truth event lengths.

## Error Analysis

Error timeline plots were generated for the problematic test records:

```text
artifacts/error_analysis/window04_hop01/test_selected_t09_full_merge01_smooth03/timelines/
```

Problem records:

| Record | Error Summary |
| ---: | --- |
| 030 | 1 false negative |
| 094 | 1 boundary-mismatch false positive |
| 105 | 3 false negatives |

The main failure pattern was missed events where the model probability stayed
below the selected high threshold. The high threshold made the final model very
precise, but slightly reduced recall.

## Presentation Takeaway

This model is suitable to present as a boundary-focused event detection model.
The important point is that the model should not be framed only as a
window-level classifier. It is better described as an event-level cough detector
whose window/hop configuration was chosen with annotation-only analysis and
whose post-processing was calibrated on validation data.

Suggested summary sentence:

```text
Using annotation-only window-hop analysis, we selected a 0.4 s window and
0.1 s hop for boundary-focused cough detection. After validation-based
post-processing calibration, the model achieved 0.956 test Event F1 with high
precision and close temporal alignment to ground-truth cough events.
```
