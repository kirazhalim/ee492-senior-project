# Experiment Workspace

This folder keeps research experiments separate from the stable project
backbone. Experiments can add configs, diagnostics, plots, and interpretation
notes without turning every exploratory result into the default workflow.

## Rules

1. Keep stable defaults in root configs such as `configs/v3.yaml`.
2. Put exploratory configs under `configs/experiments/`.
3. Put one-off experiment helpers under `scripts/experiments/`.
4. Put experiment writeups under `docs/experiments/`.
5. Treat `artifacts/` outputs as generated evidence, not source-of-truth code.
6. Do not choose thresholds or model settings from an inspected external set.

## Current Experiments

| Experiment | Config | Notes |
| --- | --- | --- |
| Boundary window/hop and input-source ablation | `configs/experiments/v3_window05_hop01.yaml` | `docs/experiments/BoundaryWindowHopExperiment.md` |
| Grouped validation protocol design | `configs/v3.yaml` | `docs/experiments/GroupedValidationProtocol.md` |
| Spectrogram and motion visual audit | `configs/v3.yaml` | `docs/experiments/VisualAuditProtocol.md` |

## Promotion Criteria

Only promote an experiment into the main workflow after it has:

- a clear methodological reason,
- validation-only selection of hyperparameters/post-processing,
- internal test results,
- grouped or external generalization evidence,
- reproducible commands or scripts,
- tests for shared utilities.

Until then, the experiment should stay on its branch or under the experiment
folders.
