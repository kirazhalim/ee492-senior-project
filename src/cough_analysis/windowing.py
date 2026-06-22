from __future__ import annotations

import numpy as np


def sliding_window_indices(
    signal_length: int,
    window_size: int,
    hop_size: int,
) -> list[tuple[int, int]]:
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive.")
    if signal_length < window_size:
        return []

    indices = []
    start = 0
    while start + window_size <= signal_length:
        end = start + window_size
        indices.append((start, end))
        start += hop_size
    return indices


def label_window_any_positive(label_window: np.ndarray) -> int:
    return int(np.any(label_window > 0))


def label_window_by_overlap(label_window: np.ndarray, threshold: float = 0.2) -> int:
    ratio = np.mean(label_window > 0)
    return int(ratio >= threshold)


def label_window_center_positive(
    label_window: np.ndarray,
    center_fraction: float = 0.2,
) -> int:
    if not 0 < center_fraction <= 1:
        raise ValueError("center_fraction must be in (0, 1].")

    n = len(label_window)
    center_len = max(1, int(round(n * center_fraction)))
    start = (n - center_len) // 2
    end = start + center_len
    return int(np.any(label_window[start:end] > 0))


def make_windows(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    hop_size: int,
    label_mode: str = "any_positive",
    overlap_threshold: float = 0.2,
    center_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    if x.ndim != 2:
        raise ValueError(f"x must have shape (C, T), got {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must have shape (T,), got {y.shape}")
    if x.shape[1] != len(y):
        raise ValueError("x and y length mismatch.")

    spans = sliding_window_indices(
        signal_length=x.shape[1],
        window_size=window_size,
        hop_size=hop_size,
    )

    Xw = []
    yw = []

    for start, end in spans:
        xw = x[:, start:end]
        yw_raw = y[start:end]

        if label_mode == "any_positive":
            label = label_window_any_positive(yw_raw)
        elif label_mode == "overlap":
            label = label_window_by_overlap(yw_raw, threshold=overlap_threshold)
        elif label_mode == "center_positive":
            label = label_window_center_positive(
                yw_raw,
                center_fraction=center_fraction,
            )
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        Xw.append(xw)
        yw.append(label)

    if len(Xw) == 0:
        return (
            np.empty((0, x.shape[0], window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            spans,
        )

    return (
        np.stack(Xw, axis=0).astype(np.float32),
        np.asarray(yw, dtype=np.int64),
        spans,
    )

