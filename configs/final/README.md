# Final Report Configs

These configs are reserved for the report-generation runs on the cleaned
`data/clean_v4` dataset. They intentionally do not overwrite the historical
experiment configs.

- `ee491_classical_clean.yaml`: EE491-style classical ML baseline with
  0.2 s / 0.05 s handcrafted feature windows and overlap-rule labels.
- `v3_clean_all_records.yaml`: V3 log-Mel CNN with the original 1.0 s / 0.25 s
  window setup.
- `v3_clean_window04_hop01.yaml`: boundary-focused V3-style setup with
  0.4 s / 0.1 s windows.
- `v1_clean_raw_waveform.yaml`: V1 raw-waveform 1D CNN baseline with
  1.0 s / 0.5 s windows and any-positive labels.
- `v2_clean_raw_waveform.yaml`: V2 raw-waveform 1D CNN baseline with
  1.0 s / 0.25 s windows, center-positive labels, AWGN augmentation, and
  validation-loss scheduling.
- `v4_clean.yaml`: V4 event/activity pipeline with a motion-only activity
  classifier and the activity classes `stationary`, `walking`, and `running`.
- `v5_ast_clean.yaml`: V5 frozen-AST pulmonary-audio embedding plus motion
  fusion head with the same clean split and validation-selected
  post-processing sweep.
