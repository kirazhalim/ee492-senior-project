from __future__ import annotations

import torch


INPUT_ABLATION_MODES = (
    "full",
    "audio_only",
    "motion_only",
    "pulmonary_only",
    "ambient_only",
    "stretch_only",
    "accel_only",
)


def apply_input_ablation(
    spec: torch.Tensor,
    motion: torch.Tensor,
    mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask input channels while keeping the model architecture unchanged."""
    if mode not in INPUT_ABLATION_MODES:
        choices = ", ".join(INPUT_ABLATION_MODES)
        raise ValueError(f"Unknown input ablation mode '{mode}'. Choose one of: {choices}")

    if mode == "full":
        return spec, motion

    spec_out = torch.zeros_like(spec)
    motion_out = torch.zeros_like(motion)

    if mode == "audio_only":
        spec_out = spec
    elif mode == "motion_only":
        motion_out = motion
    elif mode == "pulmonary_only":
        spec_out[:, 0:1] = spec[:, 0:1]
    elif mode == "ambient_only":
        spec_out[:, 1:2] = spec[:, 1:2]
    elif mode == "stretch_only":
        motion_out[:, 0:1] = motion[:, 0:1]
    elif mode == "accel_only":
        motion_out[:, 1:2] = motion[:, 1:2]

    return spec_out, motion_out
