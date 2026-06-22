from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import Dataset

from cough_analysis.event_metrics import (
    binary_labels_to_events,
    probabilities_to_predictions,
    smooth_probabilities,
    window_predictions_to_events,
)
from cough_analysis.preprocessing import FS_AUDIO, load_record_preprocessed
from cough_analysis.v3 import build_centered_windows
from cough_analysis.v4 import event_summary


class ASTFusionDataset(Dataset):
    def __init__(self, audio_embeddings: torch.Tensor, motion: np.ndarray, labels: np.ndarray):
        self.audio_embeddings = torch.as_tensor(audio_embeddings, dtype=torch.float32)
        self.motion = torch.as_tensor(motion, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "audio_embedding": self.audio_embeddings[idx],
            "motion": self.motion[idx],
            "label": self.labels[idx],
        }


def build_ast_window_table(
    record_ids: list[int] | np.ndarray,
    metadata: pd.DataFrame,
    data_root: str | Path | None,
    window_sec: float,
    hop_sec: float,
    center_fraction: float,
) -> dict[str, np.ndarray]:
    audio_chunks = []
    motion_chunks = []
    label_chunks = []
    record_index = []
    span_start = []
    span_end = []

    for record_id in [int(x) for x in record_ids]:
        record = load_record_preprocessed(record_id, metadata=metadata, data_root=data_root)
        windows = build_centered_windows(
            record,
            window_sec=window_sec,
            hop_sec=hop_sec,
            center_fraction=center_fraction,
        )
        if len(windows["labels"]) == 0:
            continue
        audio_chunks.append(windows["audio"])
        motion_chunks.append(windows["motion"])
        label_chunks.append(windows["labels"])
        record_index.extend([record_id] * len(windows["labels"]))
        span_start.extend([float(start) for start, _ in windows["spans"]])
        span_end.extend([float(end) for _, end in windows["spans"]])

    if not label_chunks:
        raise ValueError("No AST windows were created. Check data paths and window settings.")

    return {
        "audio": np.vstack(audio_chunks).astype(np.float32),
        "motion": np.vstack(motion_chunks).astype(np.float32),
        "labels": np.concatenate(label_chunks).astype(np.int64),
        "record_ids": np.asarray(record_index, dtype=np.int64),
        "span_start": np.asarray(span_start, dtype=np.float32),
        "span_end": np.asarray(span_end, dtype=np.float32),
    }


def ast_split_summary(split_tables: dict[str, dict[str, np.ndarray]]) -> list[dict[str, float]]:
    rows = []
    for split, table in split_tables.items():
        labels = table["labels"]
        rows.append(
            {
                "split": split,
                "windows": int(len(labels)),
                "cough_windows": int(np.sum(labels)),
                "positive_rate": float(np.mean(labels)) if len(labels) else 0.0,
            }
        )
    return rows


def pulmonary_to_ast_waveform(
    audio_window: np.ndarray,
    resampler: torchaudio.transforms.Resample,
) -> np.ndarray:
    pulmonary = audio_window[0].astype(np.float32)
    pulmonary = pulmonary - float(np.mean(pulmonary))
    peak = float(np.max(np.abs(pulmonary)))
    if peak > 0:
        pulmonary = pulmonary / peak
    with torch.no_grad():
        waveform_16k = resampler(torch.tensor(pulmonary)).numpy()
    return waveform_16k.astype(np.float32)


@torch.no_grad()
def extract_ast_embeddings(
    audio_windows: np.ndarray,
    feature_extractor: Any,
    ast_model: nn.Module,
    device: torch.device,
    batch_size: int,
    ast_sample_rate: int,
) -> torch.Tensor:
    resampler = torchaudio.transforms.Resample(orig_freq=FS_AUDIO, new_freq=ast_sample_rate)
    embeddings = []
    total = len(audio_windows)

    for start in range(0, total, batch_size):
        batch_audio = audio_windows[start : start + batch_size]
        waveforms = [pulmonary_to_ast_waveform(window, resampler) for window in batch_audio]
        inputs = feature_extractor(
            waveforms,
            sampling_rate=ast_sample_rate,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        outputs = ast_model(**inputs)
        embeddings.append(outputs.pooler_output.detach().cpu())

    return torch.cat(embeddings, dim=0)


def evaluate_ast_fusion_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    losses = []
    labels = []
    probs = []

    with torch.no_grad():
        for batch in loader:
            audio_embedding = batch["audio_embedding"].to(device)
            motion = batch["motion"].to(device)
            y = batch["label"].to(device)
            logits = model(audio_embedding, motion)
            loss = criterion(logits, y)
            losses.append(float(loss.item()) * len(y))
            labels.extend(y.cpu().numpy().astype(int).tolist())
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    labels_np = np.asarray(labels)
    probs_np = np.asarray(probs, dtype=np.float32)
    preds_np = (probs_np >= threshold).astype(int)
    return {
        "loss": sum(losses) / max(len(labels_np), 1),
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
        "labels": labels_np,
        "probs": probs_np,
        "preds": preds_np,
    }


def build_gt_event_cache(
    record_ids: list[int] | np.ndarray,
    metadata: pd.DataFrame,
    data_root: str | Path | None,
    gt_min_duration_sec: float,
    gt_merge_gap_sec: float,
) -> dict[int, list]:
    cache = {}
    for record_id in [int(x) for x in record_ids]:
        record = load_record_preprocessed(record_id, metadata=metadata, data_root=data_root)
        cache[record_id] = binary_labels_to_events(
            record["cough_label"],
            sample_rate=int(record["fs_audio"]),
            min_duration_sec=gt_min_duration_sec,
            merge_gap_sec=gt_merge_gap_sec,
        )
    return cache


def ast_event_metrics_for_table(
    table: dict[str, np.ndarray],
    probs: np.ndarray,
    record_ids: list[int] | np.ndarray,
    gt_cache: dict[int, list],
    threshold: float,
    smoothing_sec: float,
    span_mode: str,
    pred_min_duration_sec: float,
    pred_merge_gap_sec: float,
    center_fraction: float,
    event_iou_threshold: float,
) -> dict[str, float | int]:
    gt_by_record = {}
    pred_by_record = {}

    for record_id in [int(x) for x in record_ids]:
        mask = table["record_ids"] == record_id
        if not np.any(mask):
            continue
        spans = list(zip(table["span_start"][mask], table["span_end"][mask]))
        record_probs = smooth_probabilities(probs[mask], spans, smoothing_sec=smoothing_sec)
        preds = probabilities_to_predictions(record_probs, threshold=threshold)
        gt_by_record[record_id] = gt_cache[record_id]
        pred_by_record[record_id] = window_predictions_to_events(
            spans,
            preds,
            min_duration_sec=pred_min_duration_sec,
            merge_gap_sec=pred_merge_gap_sec,
            span_mode=span_mode,
            center_fraction=center_fraction,
        )

    return event_summary(gt_by_record, pred_by_record, iou_threshold=event_iou_threshold)


def run_ast_postprocessing_sweep(
    table: dict[str, np.ndarray],
    probs: np.ndarray,
    record_ids: list[int] | np.ndarray,
    gt_cache: dict[int, list],
    sweep_cfg: dict[str, Any],
    center_fraction: float,
    event_iou_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for threshold in sweep_cfg["thresholds"]:
        for smoothing_sec in sweep_cfg["smoothing_sec"]:
            for span_mode in sweep_cfg["span_modes"]:
                for pred_min_duration_sec in sweep_cfg["pred_min_duration_sec"]:
                    for pred_merge_gap_sec in sweep_cfg["pred_merge_gap_sec"]:
                        metrics = ast_event_metrics_for_table(
                            table=table,
                            probs=probs,
                            record_ids=record_ids,
                            gt_cache=gt_cache,
                            threshold=float(threshold),
                            smoothing_sec=float(smoothing_sec),
                            span_mode=str(span_mode),
                            pred_min_duration_sec=float(pred_min_duration_sec),
                            pred_merge_gap_sec=float(pred_merge_gap_sec),
                            center_fraction=center_fraction,
                            event_iou_threshold=event_iou_threshold,
                        )
                        rows.append(
                            {
                                "threshold": float(threshold),
                                "smoothing_sec": float(smoothing_sec),
                                "span_mode": str(span_mode),
                                "pred_min_duration_sec": float(pred_min_duration_sec),
                                "pred_merge_gap_sec": float(pred_merge_gap_sec),
                                **metrics,
                            }
                        )
    return rows
