from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from cough_analysis.event_metrics import binary_labels_to_events
from cough_analysis.preprocessing import FS_AUDIO, FS_MOTION, load_record_preprocessed


EPS = 1.0e-12
FEATURE_COLUMNS = [
    "log_rms_p",
    "log_rms_ratio",
    "spec_centroid",
    "acc_var",
    "str_sharpness",
    "str_min",
    "kurt_audio",
    "kurt_stretch",
]


def window_starts(signal_length: int, window_samples: int, hop_samples: int) -> list[int]:
    if signal_length < window_samples:
        return []
    return list(range(0, signal_length - window_samples + 1, hop_samples))


def overlap_label_rule_c(
    start_sec: float,
    end_sec: float,
    gt_events,
    tau: float,
) -> tuple[int, float]:
    window_duration = end_sec - start_sec
    best_ratio = 0.0
    for event in gt_events:
        overlap = max(0.0, min(end_sec, event.end) - max(start_sec, event.start))
        denominator = min(window_duration, event.duration)
        ratio = overlap / denominator if denominator > 0 else 0.0
        best_ratio = max(best_ratio, ratio)
    return int(best_ratio >= tau), float(best_ratio)


def safe_kurtosis(values: np.ndarray) -> float:
    value = float(kurtosis(values, fisher=True, bias=False))
    return value if np.isfinite(value) else 0.0


def extract_ee491_features(
    pulmonary: np.ndarray,
    ambient: np.ndarray,
    accz: np.ndarray,
    stretch: np.ndarray,
    fs_audio: int = FS_AUDIO,
    fs_motion: int = FS_MOTION,
) -> dict[str, float]:
    rms_p = np.sqrt(np.mean(pulmonary**2) + EPS)
    rms_a = np.sqrt(np.mean(ambient**2) + EPS)
    log_rms_p = np.log(rms_p)
    log_rms_ratio = np.log(rms_p / rms_a)

    audio_window = np.hanning(len(pulmonary))
    audio_freqs = np.fft.rfftfreq(len(pulmonary), d=1 / fs_audio)
    spec_p = np.abs(np.fft.rfft(pulmonary * audio_window)) ** 2
    spec_centroid = np.sum(audio_freqs * spec_p) / (np.sum(spec_p) + EPS)

    acc_var = np.var(accz)
    str_sharpness = np.mean(np.diff(stretch) ** 2) if len(stretch) > 2 else 0.0
    str_min = np.min(stretch) if len(stretch) else 0.0

    return {
        "log_rms_p": float(log_rms_p),
        "log_rms_ratio": float(log_rms_ratio),
        "spec_centroid": float(spec_centroid),
        "acc_var": float(acc_var),
        "str_sharpness": float(str_sharpness),
        "str_min": float(str_min),
        "kurt_audio": safe_kurtosis(pulmonary),
        "kurt_stretch": safe_kurtosis(stretch),
    }


def build_classical_feature_table(
    record_ids: list[int] | np.ndarray,
    metadata: pd.DataFrame,
    data_root: str | Path | None,
    window_sec: float,
    hop_sec: float,
    label_overlap_tau: float,
    gt_min_duration_sec: float = 0.0,
    gt_merge_gap_sec: float = 0.0,
) -> pd.DataFrame:
    rows = []
    audio_window = int(round(window_sec * FS_AUDIO))
    audio_hop = int(round(hop_sec * FS_AUDIO))
    motion_window = int(round(window_sec * FS_MOTION))

    for record_id in [int(x) for x in record_ids]:
        record = load_record_preprocessed(record_id, metadata=metadata, data_root=data_root)
        gt_events = binary_labels_to_events(
            record["cough_label"],
            sample_rate=int(record["fs_audio"]),
            min_duration_sec=gt_min_duration_sec,
            merge_gap_sec=gt_merge_gap_sec,
        )
        for start in window_starts(len(record["pulm_bp"]), audio_window, audio_hop):
            end = start + audio_window
            motion_start = int(round((start / FS_AUDIO) * FS_MOTION))
            motion_end = motion_start + motion_window
            if motion_end > len(record["stretch_lp"]):
                break

            t0 = start / FS_AUDIO
            t1 = end / FS_AUDIO
            label, best_overlap_ratio = overlap_label_rule_c(
                t0,
                t1,
                gt_events,
                tau=label_overlap_tau,
            )
            features = extract_ee491_features(
                record["pulm_bp"][start:end],
                record["amb_bp"][start:end],
                record["accz_lp"][motion_start:motion_end],
                record["stretch_lp"][motion_start:motion_end],
                fs_audio=int(record["fs_audio"]),
                fs_motion=int(record["fs_motion"]),
            )
            rows.append(
                {
                    "record_id": record_id,
                    "activity": str(record["activity"]),
                    "t0": float(t0),
                    "t1": float(t1),
                    "win_start_idx": int(start),
                    "y_cough": int(label),
                    "best_overlap_ratio": best_overlap_ratio,
                    **features,
                }
            )

    return pd.DataFrame(rows)


def feature_matrix(table: pd.DataFrame) -> np.ndarray:
    return table[FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def labels(table: pd.DataFrame) -> np.ndarray:
    return table["y_cough"].to_numpy(dtype=int)


def split_tables_by_record(
    table: pd.DataFrame,
    record_ids: list[int] | np.ndarray,
) -> pd.DataFrame:
    return table.loc[table["record_id"].isin([int(x) for x in record_ids])].copy()
