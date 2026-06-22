from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from cough_analysis.preprocessing import FS_AUDIO, FS_MOTION, load_record_preprocessed


DEFAULT_SPECTROGRAM = {
    "n_fft": 512,
    "hop_length": 128,
    "n_mels": 64,
    "f_min": 60,
    "f_max": 2200,
    "log_eps": 1.0e-9,
}


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_records(
    metadata: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = metadata["record_id"].unique()
    train_val, test = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=random_state,
    )
    return train, val, test


def split_records_from_config(
    metadata: pd.DataFrame,
    split_cfg: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not split_cfg:
        return split_records(metadata, random_state=random_state)

    strategy = str(split_cfg.get("strategy", "record_holdout"))
    if strategy != "record_holdout":
        raise ValueError(f"Unsupported V3 split strategy: {strategy}")

    train_size = float(split_cfg.get("train_size", 0.70))
    val_size = float(split_cfg.get("val_size", 0.15))
    test_size = float(split_cfg.get("test_size", 0.15))
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    seed = int(split_cfg.get("seed", random_state))
    record_ids = metadata["record_id"].to_numpy()
    stratify_by = split_cfg.get("stratify_by")
    stratify = None if stratify_by is None else metadata[str(stratify_by)].to_numpy()

    train_ids, temp_ids, _, temp_labels = train_test_split(
        record_ids,
        stratify if stratify is not None else record_ids,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )
    val_ratio = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_ratio,
        random_state=seed,
        stratify=temp_labels if stratify is not None else None,
    )
    return (
        np.asarray(sorted(int(x) for x in train_ids)),
        np.asarray(sorted(int(x) for x in val_ids)),
        np.asarray(sorted(int(x) for x in test_ids)),
    )


def build_centered_windows(
    record: dict,
    window_sec: float = 1.0,
    hop_sec: float = 0.25,
    fs_audio: int = FS_AUDIO,
    fs_motion: int = FS_MOTION,
    center_fraction: float = 0.2,
) -> dict:
    pulmonary = record["pulm_bp"]
    ambient = record["amb_bp"]
    stretch = record["stretch_lp"]
    accz = record["accz_lp"]
    labels_raw = record["cough_label"]

    a_win = int(window_sec * fs_audio)
    a_hop = int(hop_sec * fs_audio)
    m_win = int(window_sec * fs_motion)

    center_len = max(1, int(round(a_win * center_fraction)))
    center_lo = (a_win - center_len) // 2
    center_hi = center_lo + center_len

    audio_wins = []
    motion_wins = []
    labels = []
    spans = []

    for start in range(0, len(pulmonary) - a_win + 1, a_hop):
        end = start + a_win
        motion_start = int((start / fs_audio) * fs_motion)
        motion_end = motion_start + m_win
        if motion_end > len(stretch):
            break

        audio_wins.append(np.stack([pulmonary[start:end], ambient[start:end]], axis=0))
        motion_wins.append(
            np.stack([stretch[motion_start:motion_end], accz[motion_start:motion_end]], axis=0)
        )
        labels.append(
            int(np.any(labels_raw[start + center_lo : start + center_hi] > 0))
        )
        spans.append((start / fs_audio, end / fs_audio))

    return {
        "audio": np.asarray(audio_wins, dtype=np.float32),
        "motion": np.asarray(motion_wins, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "spans": spans,
    }


def make_mel_transform(
    sample_rate: int = FS_AUDIO,
    spectrogram_config: dict[str, Any] | None = None,
) -> torchaudio.transforms.MelSpectrogram:
    cfg = {**DEFAULT_SPECTROGRAM, **(spectrogram_config or {})}
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        n_mels=int(cfg["n_mels"]),
        f_min=float(cfg["f_min"]),
        f_max=float(cfg["f_max"]),
    )


def audio_to_log_mel(
    audio: np.ndarray,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    log_eps: float = 1.0e-9,
) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(audio, dtype=torch.float32)
        mel = mel_transform(tensor)
        log_mel = torch.log(mel + log_eps)
    return log_mel.numpy()


def build_dataset(
    record_ids: list[int] | np.ndarray,
    metadata: pd.DataFrame,
    data_root: str | Path | None = None,
    window_sec: float = 1.0,
    hop_sec: float = 0.25,
    center_fraction: float = 0.2,
    spectrogram_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mel_transform = make_mel_transform(
        sample_rate=FS_AUDIO,
        spectrogram_config=spectrogram_config,
    )
    log_eps = float((spectrogram_config or {}).get("log_eps", 1.0e-9))

    all_spec = []
    all_motion = []
    all_labels = []

    for rid in record_ids:
        record = load_record_preprocessed(int(rid), metadata=metadata, data_root=data_root)
        windows = build_centered_windows(
            record,
            window_sec=window_sec,
            hop_sec=hop_sec,
            center_fraction=center_fraction,
        )
        if len(windows["labels"]) == 0:
            continue

        specs = audio_to_log_mel(
            windows["audio"],
            mel_transform=mel_transform,
            log_eps=log_eps,
        )
        all_spec.append(specs)
        all_motion.append(windows["motion"])
        all_labels.append(windows["labels"])

    if not all_labels:
        n_mels = int((spectrogram_config or DEFAULT_SPECTROGRAM)["n_mels"])
        return (
            np.empty((0, 2, n_mels, 0), dtype=np.float32),
            np.empty((0, 2, int(window_sec * FS_MOTION)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.vstack(all_spec).astype(np.float32),
        np.vstack(all_motion).astype(np.float32),
        np.concatenate(all_labels).astype(np.int64),
    )


def build_record_dataset(
    record_id: int,
    metadata: pd.DataFrame,
    data_root: str | Path | None = None,
    window_sec: float = 1.0,
    hop_sec: float = 0.25,
    center_fraction: float = 0.2,
    spectrogram_config: dict[str, Any] | None = None,
) -> dict:
    mel_transform = make_mel_transform(
        sample_rate=FS_AUDIO,
        spectrogram_config=spectrogram_config,
    )
    log_eps = float((spectrogram_config or {}).get("log_eps", 1.0e-9))
    record = load_record_preprocessed(
        int(record_id),
        metadata=metadata,
        data_root=data_root,
    )
    windows = build_centered_windows(
        record,
        window_sec=window_sec,
        hop_sec=hop_sec,
        center_fraction=center_fraction,
    )
    specs = audio_to_log_mel(
        windows["audio"],
        mel_transform=mel_transform,
        log_eps=log_eps,
    )
    return {
        "record": record,
        "spec": specs.astype(np.float32),
        "motion": windows["motion"].astype(np.float32),
        "labels": windows["labels"].astype(np.int64),
        "spans": windows["spans"],
    }


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        X_spec: np.ndarray,
        X_motion: np.ndarray,
        labels: np.ndarray,
        label_dtype: torch.dtype = torch.float32,
    ):
        self.spec = torch.tensor(X_spec, dtype=torch.float32)
        self.motion = torch.tensor(X_motion, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=label_dtype)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "spec": self.spec[idx],
            "motion": self.motion[idx],
            "label": self.labels[idx],
        }
