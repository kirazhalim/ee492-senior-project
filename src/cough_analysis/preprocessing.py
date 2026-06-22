from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from cough_analysis import data as data_utils


FS_AUDIO = 4800
FS_MOTION = 100


def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    nyquist = 0.5 * fs
    return signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")


def butter_lowpass(cutoff: float, fs: int, order: int = 4):
    nyquist = 0.5 * fs
    return signal.butter(order, cutoff / nyquist, btype="low")


def load_record_preprocessed(
    record_id: int,
    metadata: pd.DataFrame | None = None,
    data_root: str | Path | None = None,
    fs_audio: int = FS_AUDIO,
    fs_motion: int = FS_MOTION,
) -> dict:
    record = data_utils.load_record(record_id, metadata=metadata, data_root=data_root)

    pulmonary = record["pulmonary"].astype(np.float64)
    ambient = record["ambient"].astype(np.float64)
    stretch = record["stretch"].astype(np.float64)
    accz = record["accel_z"].astype(np.float64)

    b_bp, a_bp = butter_bandpass(60, 2200, fs_audio, order=4)
    pulmonary_bp = signal.filtfilt(b_bp, a_bp, pulmonary - np.median(pulmonary))
    ambient_bp = signal.filtfilt(b_bp, a_bp, ambient - np.median(ambient))

    stretch_centered = stretch - np.median(stretch)
    n_motion = int(len(stretch_centered) * (fs_motion / fs_audio))
    stretch_resampled = signal.resample(stretch_centered, n_motion)
    accz_resampled = signal.resample(accz, n_motion)

    b_lp, a_lp = butter_lowpass(20, fs_motion, order=4)
    stretch_lp = signal.filtfilt(b_lp, a_lp, stretch_resampled)
    accz_lp = signal.filtfilt(b_lp, a_lp, accz_resampled)

    return {
        **record,
        "pulm_bp": pulmonary_bp.astype(np.float32),
        "amb_bp": ambient_bp.astype(np.float32),
        "stretch_lp": stretch_lp.astype(np.float32),
        "accz_lp": accz_lp.astype(np.float32),
        "duration_sec": len(pulmonary_bp) / fs_audio,
        "fs_audio": fs_audio,
        "fs_motion": fs_motion,
    }

