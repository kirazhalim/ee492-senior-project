from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cough_analysis.preprocessing import FS_AUDIO, FS_MOTION, load_record_preprocessed


def label_raw_window(
    label_window: np.ndarray,
    label_rule: str,
    center_fraction: float = 0.2,
) -> int:
    rule = str(label_rule)
    if rule in {"any", "any_positive"}:
        return int(np.any(label_window > 0))
    if rule == "center_positive":
        if not 0 < center_fraction <= 1:
            raise ValueError("center_fraction must be in (0, 1].")
        center_len = max(1, int(round(len(label_window) * center_fraction)))
        center_lo = (len(label_window) - center_len) // 2
        center_hi = center_lo + center_len
        return int(np.any(label_window[center_lo:center_hi] > 0))
    raise ValueError(f"Unsupported raw-window label rule: {label_rule}")


def build_raw_windows(
    record: dict,
    window_sec: float,
    hop_sec: float,
    label_rule: str,
    center_fraction: float = 0.2,
    fs_audio: int = FS_AUDIO,
    fs_motion: int = FS_MOTION,
) -> dict:
    audio_window = int(round(window_sec * fs_audio))
    audio_hop = int(round(hop_sec * fs_audio))
    motion_window = int(round(window_sec * fs_motion))

    audio_windows = []
    motion_windows = []
    labels = []
    spans = []

    for start in range(0, len(record["pulm_bp"]) - audio_window + 1, audio_hop):
        end = start + audio_window
        motion_start = int(round((start / fs_audio) * fs_motion))
        motion_end = motion_start + motion_window
        if motion_end > len(record["stretch_lp"]):
            break

        audio_windows.append(
            np.stack(
                [
                    record["pulm_bp"][start:end],
                    record["amb_bp"][start:end],
                ],
                axis=0,
            )
        )
        motion_windows.append(
            np.stack(
                [
                    record["stretch_lp"][motion_start:motion_end],
                    record["accz_lp"][motion_start:motion_end],
                ],
                axis=0,
            )
        )
        labels.append(
            label_raw_window(
                record["cough_label"][start:end],
                label_rule=label_rule,
                center_fraction=center_fraction,
            )
        )
        spans.append((start / fs_audio, end / fs_audio))

    return {
        "audio": np.asarray(audio_windows, dtype=np.float32),
        "motion": np.asarray(motion_windows, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "spans": spans,
    }


def build_raw_dataset(
    record_ids: list[int] | np.ndarray,
    metadata: pd.DataFrame,
    data_root: str | Path | None,
    window_sec: float,
    hop_sec: float,
    label_rule: str,
    center_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio_chunks = []
    motion_chunks = []
    label_chunks = []

    for record_id in record_ids:
        record = load_record_preprocessed(
            int(record_id),
            metadata=metadata,
            data_root=data_root,
        )
        windows = build_raw_windows(
            record,
            window_sec=window_sec,
            hop_sec=hop_sec,
            label_rule=label_rule,
            center_fraction=center_fraction,
        )
        if len(windows["labels"]) == 0:
            continue
        audio_chunks.append(windows["audio"])
        motion_chunks.append(windows["motion"])
        label_chunks.append(windows["labels"])

    audio_len = int(round(window_sec * FS_AUDIO))
    motion_len = int(round(window_sec * FS_MOTION))
    if not label_chunks:
        return (
            np.empty((0, 2, audio_len), dtype=np.float32),
            np.empty((0, 2, motion_len), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.vstack(audio_chunks).astype(np.float32),
        np.vstack(motion_chunks).astype(np.float32),
        np.concatenate(label_chunks).astype(np.int64),
    )


def build_raw_record_dataset(
    record_id: int,
    metadata: pd.DataFrame,
    data_root: str | Path | None,
    window_sec: float,
    hop_sec: float,
    label_rule: str,
    center_fraction: float = 0.2,
) -> dict:
    record = load_record_preprocessed(
        int(record_id),
        metadata=metadata,
        data_root=data_root,
    )
    windows = build_raw_windows(
        record,
        window_sec=window_sec,
        hop_sec=hop_sec,
        label_rule=label_rule,
        center_fraction=center_fraction,
    )
    return {
        "record": record,
        "audio": windows["audio"].astype(np.float32),
        "motion": windows["motion"].astype(np.float32),
        "labels": windows["labels"].astype(np.int64),
        "spans": windows["spans"],
    }


class RawWaveformDataset(Dataset):
    def __init__(
        self,
        audio: np.ndarray,
        motion: np.ndarray,
        labels: np.ndarray,
        label_dtype: torch.dtype = torch.float32,
    ):
        self.audio = torch.tensor(audio, dtype=torch.float32)
        self.motion = torch.tensor(motion, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=label_dtype)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "audio": self.audio[idx],
            "motion": self.motion[idx],
            "label": self.labels[idx],
        }


def augment_raw_batch(
    audio: torch.Tensor,
    motion: torch.Tensor,
    audio_noise_std: float = 0.0,
    motion_noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if audio_noise_std > 0.0:
        audio = audio + torch.randn_like(audio) * float(audio_noise_std)
    if motion_noise_std > 0.0:
        motion = motion + torch.randn_like(motion) * float(motion_noise_std)
    return audio, motion
