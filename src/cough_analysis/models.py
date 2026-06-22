from __future__ import annotations

import torch
from torch import nn


class Spec2DCoughCNN(nn.Module):
    """Spectrogram audio branch and motion branch with late fusion."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.audio_branch = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.motion_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, spec: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        xa = self.audio_branch(spec).flatten(1)
        xm = self.motion_branch(motion).flatten(1)
        out = self.classifier(torch.cat([xa, xm], dim=1))
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class RawWaveformCoughCNN(nn.Module):
    """Dual-branch 1D CNN used by the V1/V2 waveform baselines."""

    def __init__(self):
        super().__init__()
        self.audio_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.motion_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, audio: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        xa = self.audio_branch(audio).squeeze(-1)
        xm = self.motion_branch(motion).squeeze(-1)
        return self.classifier(torch.cat((xa, xm), dim=1)).squeeze(-1)


class V4CoughFrameCNN(nn.Module):
    """Small audio+motion model that keeps a time axis for cough detection."""

    def __init__(self):
        super().__init__()
        self.audio_branch = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.motion_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.temporal_head = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, spec: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        audio_features = self.audio_branch(spec).mean(dim=2)
        motion_features = self.motion_branch(motion)
        if motion_features.shape[-1] != audio_features.shape[-1]:
            motion_features = nn.functional.interpolate(
                motion_features,
                size=audio_features.shape[-1],
                mode="linear",
                align_corners=False,
            )
        fused = torch.cat([audio_features, motion_features], dim=1)
        return self.temporal_head(fused).squeeze(1)


class V4ActivityCNN(nn.Module):
    """Small motion-only classifier for sliding-window activity prediction."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        features = self.encoder(motion).flatten(1)
        return self.classifier(features)


class ASTMotionFusionHead(nn.Module):
    """Small V5 classifier on frozen AST embeddings plus motion windows."""

    def __init__(self, audio_dim: int):
        super().__init__()
        self.motion_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(audio_dim) + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, audio_embedding: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        motion_features = self.motion_branch(motion).flatten(1)
        fused = torch.cat([audio_embedding, motion_features], dim=1)
        return self.classifier(fused).squeeze(1)
