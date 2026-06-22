# Spectrogram And Motion Visual Audit

This diagnostic checks whether our V3 representation is plausible before
changing the model. It is not a model-selection experiment and should not be
used to tune final thresholds.

## Questions

1. Do cough windows look visually separable from clean non-cough windows?
2. Do noisy non-cough windows look similar to cough windows?
3. Does the ambient channel help distinguish noise, or does it introduce
   confusing patterns?
4. Do stretch and accelerometer signals show cough-related structure?
5. Are the current spectrogram parameters too coarse for short cough events?

## Tool

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/visual_audit_windows.py \
  --config configs/v3.yaml \
  --max-records 85 \
  --samples-per-category 24 \
  --output-dir artifacts/experiments/visual_audit/v3_baseline
```

For the 0.5s / 0.1s experiment config:

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/visual_audit_windows.py \
  --config configs/experiments/v3_window05_hop01.yaml \
  --max-records 85 \
  --samples-per-category 24 \
  --output-dir artifacts/experiments/visual_audit/v3_window05_hop01
```

The tool creates:

```text
candidate_windows.csv
sampled_windows.csv
summary.json
cough_montage.png
clean_non_cough_montage.png
noise_non_cough_montage.png
walking_noise_non_cough_montage.png
```

Each montage shows pulmonary and ambient log-mel spectrograms with stretch and
accelerometer traces below the spectrogram panel.

## Refined Audit

The original category set is useful, but it is center-label based. A window can
therefore be labeled non-cough while still containing cough energy near its
edges. For boundary work, run the refined audit as well:

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/visual_audit_windows.py \
  --config configs/v3.yaml \
  --max-records 85 \
  --samples-per-category 24 \
  --category-mode refined \
  --output-dir artifacts/experiments/visual_audit_refined/v3_baseline
```

For the 0.5s / 0.1s experiment config:

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/visual_audit_windows.py \
  --config configs/experiments/v3_window05_hop01.yaml \
  --max-records 85 \
  --samples-per-category 24 \
  --category-mode refined \
  --output-dir artifacts/experiments/visual_audit_refined/v3_window05_hop01
```

## Categories

| Category | Meaning |
| --- | --- |
| `cough` | Window center label is cough-positive. |
| `clean_non_cough` | Non-cough window from clean context. |
| `noise_non_cough` | Non-cough window from non-clean context. |
| `walking_noise_non_cough` | Non-cough window from walking + non-clean context. |

Refined categories:

| Category | Meaning |
| --- | --- |
| `pure_cough_core` | Center label is cough-positive and at least 80% of the full window is cough. |
| `boundary_cough` | Center label is cough-positive but less than 80% of the full window is cough. |
| `pure_non_cough` | Clean-context non-cough window with no cough overlap in the full window. |
| `hard_noise_non_cough` | Non-clean non-cough window with no cough overlap in the full window. |

The default refined thresholds can be changed with
`--core-cough-min-fraction` and `--pure-noncough-max-fraction`.

## Interpretation Rules

The audit should be read qualitatively and conservatively.

- If cough and clean non-cough spectrograms are visually separable, the audio
  representation is plausible.
- If noisy non-cough windows resemble cough windows, the task may require
  better noise modeling, not just a larger CNN.
- If motion traces clearly align with cough windows, motion is a meaningful
  signal and fusion should be examined.
- If motion traces are dominated by walking/background activity, motion may
  require activity-aware normalization or features.
- If short cough patterns occupy only a few spectrogram time frames, smaller
  audio hop length or a temporal model may be justified.
- If `boundary_cough` is visually very different from `pure_cough_core`, event
  boundary prediction may need a separate boundary-aware method rather than
  only a stronger classifier.
- If `hard_noise_non_cough` resembles `pure_cough_core`, false positives should
  be treated as a data/representation problem, not simply as bad thresholding.

## Methodological Guardrail

Do not choose a final model by visually inspecting only a few attractive
examples. This audit is for hypothesis generation:

```text
visual evidence -> hypothesis -> controlled experiment -> held-out evaluation
```

Good next hypotheses after this audit may include:

- compare `n_fft=512/hop=128` with `n_fft=256/hop=64`,
- compare `1.0s/0.1s` with `0.5s/0.1s`,
- add stretch derivative and accelerometer magnitude,
- build a more conscious classical ML baseline,
- consider temporal segmentation only if window classification is visibly
  misaligned with cough boundaries.
