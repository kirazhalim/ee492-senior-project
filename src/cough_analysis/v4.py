from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from cough_analysis.event_metrics import (
    Event,
    binary_labels_to_events,
    match_events,
    post_process_events,
)
from cough_analysis.preprocessing import FS_AUDIO, FS_MOTION, load_record_preprocessed


@dataclass(frozen=True)
class V4Split:
    train: list[int]
    val: list[int]
    test: list[int]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_records_v4(metadata: pd.DataFrame, split_cfg: dict[str, Any]) -> V4Split:
    train_size = float(split_cfg["train_size"])
    val_size = float(split_cfg["val_size"])
    test_size = float(split_cfg["test_size"])
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    seed = int(split_cfg["seed"])
    record_ids = metadata["record_id"].to_numpy()
    stratify = metadata[str(split_cfg["stratify_by"])].to_numpy()

    train_ids, temp_ids, _, temp_labels = train_test_split(
        record_ids,
        stratify,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )
    val_ratio = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_ratio,
        random_state=seed,
        stratify=temp_labels,
    )
    return V4Split(
        train=sorted(int(x) for x in train_ids),
        val=sorted(int(x) for x in val_ids),
        test=sorted(int(x) for x in test_ids),
    )


def remove_short_events(
    labels: np.ndarray,
    sample_rate: int = FS_AUDIO,
    min_duration_sec: float = 0.1,
) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int64)
    if min_duration_sec <= 0.0:
        return labels.copy()

    filtered = np.zeros_like(labels)
    for event in binary_labels_to_events(labels, sample_rate=sample_rate):
        if event.duration < min_duration_sec:
            continue
        start = max(0, int(round(event.start * sample_rate)))
        end = min(len(labels), int(round(event.end * sample_rate)))
        filtered[start:end] = 1
    return filtered


def frame_labels_from_samples(
    labels: np.ndarray,
    frame_hop_samples: int,
    frame_count: int,
) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int64)
    frame_labels = np.zeros(frame_count, dtype=np.float32)
    for idx in range(frame_count):
        start = idx * frame_hop_samples
        end = min(len(labels), start + frame_hop_samples)
        if start < len(labels) and np.any(labels[start:end] > 0):
            frame_labels[idx] = 1.0
    return frame_labels


def make_mel_transform(
    sample_rate: int,
    spec_cfg: dict[str, Any],
) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=int(spec_cfg["n_fft"]),
        hop_length=int(spec_cfg["hop_length"]),
        n_mels=int(spec_cfg["n_mels"]),
        f_min=float(spec_cfg["f_min"]),
        f_max=float(spec_cfg["f_max"]),
    )


def audio_to_log_mel(
    audio: np.ndarray,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    log_eps: float,
) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(audio, dtype=torch.float32)
        mel = mel_transform(tensor)
        log_mel = torch.log(mel + float(log_eps))
    return log_mel.numpy().astype(np.float32)


def cough_frame_count(cough_cfg: dict[str, Any], fs_audio: int = FS_AUDIO) -> int:
    chunk_samples = int(round(float(cough_cfg["chunk_sec"]) * fs_audio))
    return int(round(chunk_samples / int(cough_cfg["frame_hop_samples"])))


def build_cough_record_arrays(
    record: dict,
    cough_cfg: dict[str, Any],
    spec_cfg: dict[str, Any],
) -> dict[str, np.ndarray | list[tuple[float, float]]]:
    fs_audio = int(record.get("fs_audio", FS_AUDIO))
    fs_motion = int(record.get("fs_motion", FS_MOTION))
    chunk_samples = int(round(float(cough_cfg["chunk_sec"]) * fs_audio))
    chunk_hop_samples = int(round(float(cough_cfg["chunk_hop_sec"]) * fs_audio))
    motion_samples = int(round(float(cough_cfg["chunk_sec"]) * fs_motion))
    frame_hop_samples = int(cough_cfg["frame_hop_samples"])
    frames_per_chunk = cough_frame_count(cough_cfg, fs_audio=fs_audio)

    clean_labels = remove_short_events(
        record["cough_label"],
        sample_rate=fs_audio,
        min_duration_sec=float(cough_cfg["min_gt_event_duration_sec"]),
    )
    mel_transform = make_mel_transform(sample_rate=fs_audio, spec_cfg=spec_cfg)
    log_eps = float(spec_cfg.get("log_eps", 1.0e-9))

    audio_chunks = []
    motion_chunks = []
    label_chunks = []
    spans = []

    for start in range(0, len(record["pulm_bp"]) - chunk_samples + 1, chunk_hop_samples):
        end = start + chunk_samples
        motion_start = int(round((start / fs_audio) * fs_motion))
        motion_end = motion_start + motion_samples
        if motion_end > len(record["stretch_lp"]):
            break

        audio_chunks.append(
            np.stack(
                [
                    record["pulm_bp"][start:end],
                    record["amb_bp"][start:end],
                ],
                axis=0,
            )
        )
        motion_chunks.append(
            np.stack(
                [
                    record["stretch_lp"][motion_start:motion_end],
                    record["accz_lp"][motion_start:motion_end],
                ],
                axis=0,
            )
        )
        label_chunks.append(
            frame_labels_from_samples(
                clean_labels[start:end],
                frame_hop_samples=frame_hop_samples,
                frame_count=frames_per_chunk,
            )
        )
        spans.append((start / fs_audio, end / fs_audio))

    if not audio_chunks:
        n_mels = int(spec_cfg["n_mels"])
        return {
            "spec": np.empty((0, 2, n_mels, 0), dtype=np.float32),
            "motion": np.empty((0, 2, motion_samples), dtype=np.float32),
            "labels": np.empty((0, frames_per_chunk), dtype=np.float32),
            "spans": [],
        }

    specs = audio_to_log_mel(
        np.asarray(audio_chunks, dtype=np.float32),
        mel_transform=mel_transform,
        log_eps=log_eps,
    )
    return {
        "spec": specs,
        "motion": np.asarray(motion_chunks, dtype=np.float32),
        "labels": np.asarray(label_chunks, dtype=np.float32),
        "spans": spans,
    }


class V4CoughChunkDataset(Dataset):
    def __init__(
        self,
        record_ids: list[int] | np.ndarray,
        metadata: pd.DataFrame,
        cough_cfg: dict[str, Any],
        spec_cfg: dict[str, Any],
        data_root: str | Path | None = None,
    ):
        specs = []
        motions = []
        labels = []
        record_index = []

        for record_id in record_ids:
            record = load_record_preprocessed(
                int(record_id),
                metadata=metadata,
                data_root=data_root,
            )
            arrays = build_cough_record_arrays(record, cough_cfg, spec_cfg)
            if len(arrays["labels"]) == 0:
                continue
            specs.append(arrays["spec"])
            motions.append(arrays["motion"])
            labels.append(arrays["labels"])
            record_index.extend([int(record_id)] * len(arrays["labels"]))

        if labels:
            self.spec = torch.tensor(np.vstack(specs), dtype=torch.float32)
            self.motion = torch.tensor(np.vstack(motions), dtype=torch.float32)
            self.labels = torch.tensor(np.vstack(labels), dtype=torch.float32)
        else:
            n_mels = int(spec_cfg["n_mels"])
            motion_samples = int(round(float(cough_cfg["chunk_sec"]) * FS_MOTION))
            frames = cough_frame_count(cough_cfg)
            self.spec = torch.empty((0, 2, n_mels, 0), dtype=torch.float32)
            self.motion = torch.empty((0, 2, motion_samples), dtype=torch.float32)
            self.labels = torch.empty((0, frames), dtype=torch.float32)
        self.record_ids = np.asarray(record_index, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "spec": self.spec[idx],
            "motion": self.motion[idx],
            "label": self.labels[idx],
        }


def activity_class_map(activity_cfg: dict[str, Any]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(activity_cfg["classes"])}


def activity_target_label(activity: str, activity_cfg: dict[str, Any]) -> str:
    label_map = activity_cfg.get("label_map", {})
    activity = str(activity)
    return str(label_map.get(activity, activity))


def build_activity_record_arrays(
    record: dict,
    activity_label: str,
    activity_cfg: dict[str, Any],
    class_to_idx: dict[str, int],
) -> dict[str, np.ndarray | list[tuple[float, float]]]:
    fs_motion = int(record.get("fs_motion", FS_MOTION))
    window_samples = int(round(float(activity_cfg["window_sec"]) * fs_motion))
    hop_samples = int(round(float(activity_cfg["hop_sec"]) * fs_motion))
    target_label = activity_target_label(str(activity_label), activity_cfg)
    if target_label not in class_to_idx:
        raise ValueError(
            f"Activity label {activity_label!r} maps to {target_label!r}, "
            f"which is not in configured classes {list(class_to_idx)}."
        )
    label = class_to_idx[target_label]

    motion = np.stack([record["stretch_lp"], record["accz_lp"]], axis=0)
    windows = []
    labels = []
    spans = []

    for start in range(0, motion.shape[1] - window_samples + 1, hop_samples):
        end = start + window_samples
        windows.append(motion[:, start:end])
        labels.append(label)
        spans.append((start / fs_motion, end / fs_motion))

    if not windows:
        return {
            "motion": np.empty((0, 2, window_samples), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
            "spans": [],
        }

    return {
        "motion": np.asarray(windows, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "spans": spans,
    }


class V4ActivityWindowDataset(Dataset):
    def __init__(
        self,
        record_ids: list[int] | np.ndarray,
        metadata: pd.DataFrame,
        activity_cfg: dict[str, Any],
        data_root: str | Path | None = None,
    ):
        class_to_idx = activity_class_map(activity_cfg)
        motions = []
        labels = []
        record_index = []

        for record_id in record_ids:
            row = metadata.loc[metadata["record_id"] == int(record_id)].iloc[0]
            record = load_record_preprocessed(
                int(record_id),
                metadata=metadata,
                data_root=data_root,
            )
            arrays = build_activity_record_arrays(
                record,
                activity_label=str(row["activity"]),
                activity_cfg=activity_cfg,
                class_to_idx=class_to_idx,
            )
            if len(arrays["labels"]) == 0:
                continue
            motions.append(arrays["motion"])
            labels.append(arrays["labels"])
            record_index.extend([int(record_id)] * len(arrays["labels"]))

        if labels:
            self.motion = torch.tensor(np.vstack(motions), dtype=torch.float32)
            self.labels = torch.tensor(np.concatenate(labels), dtype=torch.long)
        else:
            window_samples = int(round(float(activity_cfg["window_sec"]) * FS_MOTION))
            self.motion = torch.empty((0, 2, window_samples), dtype=torch.float32)
            self.labels = torch.empty((0,), dtype=torch.long)
        self.record_ids = np.asarray(record_index, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "motion": self.motion[idx],
            "label": self.labels[idx],
        }


def resize_frame_logits(logits: torch.Tensor, frame_count: int) -> torch.Tensor:
    if logits.shape[-1] == frame_count:
        return logits
    return torch.nn.functional.interpolate(
        logits.unsqueeze(1),
        size=frame_count,
        mode="linear",
        align_corners=False,
    ).squeeze(1)


def cough_gt_events(record: dict, cough_cfg: dict[str, Any]) -> list[Event]:
    labels = remove_short_events(
        record["cough_label"],
        sample_rate=int(record.get("fs_audio", FS_AUDIO)),
        min_duration_sec=float(cough_cfg["min_gt_event_duration_sec"]),
    )
    return binary_labels_to_events(labels, sample_rate=int(record.get("fs_audio", FS_AUDIO)))


def frame_predictions_to_events(
    probabilities: np.ndarray,
    frame_rate: int,
    threshold: float,
    min_duration_sec: float,
    merge_gap_sec: float,
    duration_sec: float,
) -> list[Event]:
    predictions = np.asarray(probabilities) >= float(threshold)
    events = binary_labels_to_events(predictions, sample_rate=frame_rate)
    clipped = [
        Event(start=max(0.0, event.start), end=min(float(duration_sec), event.end))
        for event in events
        if event.start < float(duration_sec)
    ]
    return post_process_events(
        clipped,
        min_duration_sec=float(min_duration_sec),
        merge_gap_sec=float(merge_gap_sec),
    )


def event_summary(
    gt_by_record: dict[int, list[Event]],
    pred_by_record: dict[int, list[Event]],
    iou_threshold: float,
) -> dict[str, float | int]:
    tp = 0
    fp = 0
    fn = 0
    ious = []
    onset_errors = []
    offset_errors = []

    for record_id, gt_events in gt_by_record.items():
        pred_events = pred_by_record.get(record_id, [])
        matches = match_events(gt_events, pred_events, iou_threshold=iou_threshold)
        tp += len(matches)
        fp += len(pred_events) - len(matches)
        fn += len(gt_events) - len(matches)
        for gt_idx, pred_idx, iou in matches:
            gt_event = gt_events[gt_idx]
            pred_event = pred_events[pred_idx]
            ious.append(iou)
            onset_errors.append(abs(pred_event.start - gt_event.start))
            offset_errors.append(abs(pred_event.end - gt_event.end))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_events": int(sum(len(v) for v in gt_by_record.values())),
        "predicted_events": int(sum(len(v) for v in pred_by_record.values())),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_onset_error_sec": float(np.mean(onset_errors)) if onset_errors else 0.0,
        "mean_offset_error_sec": float(np.mean(offset_errors)) if offset_errors else 0.0,
    }


def predict_cough_probabilities_for_record(
    model: torch.nn.Module,
    record: dict,
    cough_cfg: dict[str, Any],
    spec_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    arrays = build_cough_record_arrays(record, cough_cfg, spec_cfg)
    frame_count = cough_frame_count(cough_cfg, fs_audio=int(record.get("fs_audio", FS_AUDIO)))
    duration_sec = float(record["duration_sec"])
    frame_rate = int(round(int(record.get("fs_audio", FS_AUDIO)) / int(cough_cfg["frame_hop_samples"])))
    total_frames = int(round(duration_sec * frame_rate))
    sums = np.zeros(total_frames, dtype=np.float32)
    counts = np.zeros(total_frames, dtype=np.float32)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(arrays["spec"], dtype=torch.float32),
        torch.tensor(arrays["motion"], dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    chunk_probs = []
    model.eval()
    with torch.no_grad():
        for spec, motion in loader:
            logits = model(spec.to(device), motion.to(device))
            logits = resize_frame_logits(logits, frame_count)
            probs = torch.sigmoid(logits).cpu().numpy()
            chunk_probs.extend(probs)

    for (start_sec, _), probs in zip(arrays["spans"], chunk_probs):
        start_frame = int(round(start_sec * frame_rate))
        end_frame = min(total_frames, start_frame + frame_count)
        usable = end_frame - start_frame
        if usable <= 0:
            continue
        sums[start_frame:end_frame] += probs[:usable]
        counts[start_frame:end_frame] += 1.0

    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def predict_activity_probabilities_for_record(
    model: torch.nn.Module,
    record: dict,
    activity_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    class_to_idx = activity_class_map(activity_cfg)
    arrays = build_activity_record_arrays(
        record,
        activity_label=activity_cfg["classes"][0],
        activity_cfg=activity_cfg,
        class_to_idx=class_to_idx,
    )
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(arrays["motion"], dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs = []
    model.eval()
    with torch.no_grad():
        for (motion,) in loader:
            logits = model(motion.to(device))
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    centers = np.asarray(
        [(start + end) / 2 for start, end in arrays["spans"]],
        dtype=np.float32,
    )
    return centers, np.asarray(probs, dtype=np.float32)


def assign_activity_to_event(
    event: Event,
    activity_centers: np.ndarray,
    activity_probs: np.ndarray,
    activity_classes: list[str],
    context_sec: float,
) -> dict[str, float | str]:
    if len(activity_probs) == 0:
        return {"activity": "", "activity_confidence": 0.0}

    lo = max(0.0, event.start - float(context_sec))
    hi = event.end + float(context_sec)
    mask = (activity_centers >= lo) & (activity_centers <= hi)
    if not np.any(mask):
        center = (event.start + event.end) / 2
        nearest = int(np.argmin(np.abs(activity_centers - center)))
        selected = activity_probs[[nearest]]
    else:
        selected = activity_probs[mask]

    averaged = selected.mean(axis=0)
    idx = int(np.argmax(averaged))
    return {
        "activity": activity_classes[idx],
        "activity_confidence": float(averaged[idx]),
    }
