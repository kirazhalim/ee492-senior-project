"""Unified inference wrapper for the four cough-detection models used in the report.

Public entry points
-------------------
- :func:`preprocess_raw_csv`  : raw 4-column CSV -> preprocessed record dict
- :func:`load_model_bundle`   : load a model + its post-processing config (cached)
- :func:`load_activity_bundle`: load the V4 activity head (cached)
- :func:`run_inference`       : full pipeline (preprocess + cough preds + activity preds)

Notes
-----
The function reuses the project's existing helpers (``cough_analysis.*``) so the demo
behaves identically to the report's evaluation scripts. Activity prediction always
comes from the V4 motion head, regardless of which cough model the user picks, so the
timeline plot stays consistent.
"""

from __future__ import annotations

# macOS arm64 + Python 3.12: torch (libomp.dylib) and xgboost (libomp.dylib)
# ship different libomp builds. Two things hit us if we don't pre-flight this:
#   1) joblib.load on the XGBoost pickle segfaults when torch initialises OpenMP
#      first (exit 139).
#   2) Even with the import order fixed, multi-threaded V4 activity inference
#      crashes the interpreter unless we cap OpenMP to a single thread.
# Both fixes are cheap (the demo's 20 s records are bottlenecked elsewhere), so
# we apply them unconditionally at module import time, before any heavy import.
import os as _os
# This MUST stay at "1". I tested OMP=2 and OMP=4 directly: torch and xgboost
# each ship their own libomp.dylib, and as soon as the second thread is spawned
# the duplicate OMP runtimes clash — V4 activity inference and classical
# joblib.load segfault the Python worker (exit 139, surfacing in macOS as a
# "Python quit unexpectedly" dialog and a Streamlit "CONNECTING…" hang). The
# right way to speed up V5_AST is the per-record embedding cache populated by
# ``app.warmup``, not more OMP threads.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import xgboost  # noqa: F401, E402  pylint: disable=unused-import,wrong-import-position

import functools
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy import signal

from cough_analysis.classical_ml import (
    FEATURE_COLUMNS,
    extract_ee491_features,
    window_starts,
)
from cough_analysis.config import load_config
from cough_analysis.data import decode_channel3, load_record_array
from cough_analysis.event_metrics import (
    Event,
    binary_labels_to_events,
    probabilities_to_predictions,
    smooth_probabilities,
    window_predictions_to_events,
)
from cough_analysis.models import (
    ASTMotionFusionHead,
    Spec2DCoughCNN,
    V4ActivityCNN,
    V4CoughFrameCNN,
)
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import (
    FS_AUDIO,
    FS_MOTION,
    butter_bandpass,
    butter_lowpass,
)
from cough_analysis.v3 import (
    audio_to_log_mel as v3_audio_to_log_mel,
    build_centered_windows,
    make_mel_transform as v3_make_mel_transform,
    resolve_device,
)
from cough_analysis.v4 import (
    assign_activity_to_event,
    frame_predictions_to_events,
    predict_activity_probabilities_for_record,
    predict_cough_probabilities_for_record,
)
from cough_analysis.v5_ast import extract_ast_embeddings


# ----- Model registry ------------------------------------------------------- #

REPO_ROOT = Path(project_path()).resolve()
CKPT_ROOT = REPO_ROOT / "artifacts" / "final_report_results" / "clean_v4_shared_split"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "v5_ast": {
        "display_name": "V5 Frozen AST",
        "config": REPO_ROOT / "configs" / "final" / "v5_ast_clean.yaml",
        "checkpoint": CKPT_ROOT / "v5_ast" / "fusion_head.pt",
        "kind": "v5_ast",
    },
    "v4": {
        "display_name": "V4 Spec CNN",
        "config": REPO_ROOT / "configs" / "final" / "v4_clean.yaml",
        "checkpoint": CKPT_ROOT / "models" / "v4" / "v4_cough_spec256.pt",
        "kind": "v4",
    },
    "v3": {
        "display_name": "V3 Log-Mel CNN",
        "config": REPO_ROOT / "configs" / "final" / "v3_clean_all_records.yaml",
        "checkpoint": CKPT_ROOT / "models" / "v3_main.pt",
        "kind": "v3",
        "postprocessing": {
            "threshold": 0.8,
            "smoothing_sec": 0.0,
            "span_mode": "center",
            "center_fraction": 0.2,
            "pred_min_duration_sec": 0.2,
            "pred_merge_gap_sec": 0.1,
        },
    },
    "classical": {
        "display_name": "Classical XGBoost (EE491)",
        "config": REPO_ROOT / "configs" / "final" / "ee491_classical_clean.yaml",
        "checkpoint": CKPT_ROOT / "ee491_classical" / "model.joblib",
        "kind": "classical",
        "postprocessing": {
            "threshold": 0.95,
            "pred_min_duration_sec": 0.2,
            "pred_merge_gap_sec": 0.2,
        },
    },
}

V4_ACTIVITY_CHECKPOINT = CKPT_ROOT / "models" / "v4" / "v4_activity.pt"


# ----- Preprocessing -------------------------------------------------------- #


@dataclass
class Record:
    filename: str
    path: str
    # Preprocessed signals (used by models and "Preprocessed" plot block)
    pulm_bp: np.ndarray
    amb_bp: np.ndarray
    stretch_lp: np.ndarray
    accz_lp: np.ndarray
    # Raw signals — unfiltered ADC integers, kept for the "Raw" plot block.
    # ``stretch_raw`` is already bit-decoded from column 3 (LSB cough label
    # stripped) because the LSB toggle would otherwise add visual noise to the
    # raw stretch trace without communicating anything useful.
    pulm_raw: np.ndarray
    amb_raw: np.ndarray
    stretch_raw: np.ndarray
    accz_raw: np.ndarray
    cough_label: np.ndarray  # binary, length == len(pulm_bp), audio rate
    fs_audio: int = FS_AUDIO
    fs_motion: int = FS_MOTION
    activity_gt: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return len(self.pulm_bp) / self.fs_audio

    def as_v4_dict(self) -> dict[str, Any]:
        """Adapter so existing project helpers that expect a record-dict keep working."""
        return {
            "filename": self.filename,
            "path": self.path,
            "pulm_bp": self.pulm_bp,
            "amb_bp": self.amb_bp,
            "stretch_lp": self.stretch_lp,
            "accz_lp": self.accz_lp,
            "pulmonary": self.pulm_raw,
            "cough_label": self.cough_label,
            "fs_audio": self.fs_audio,
            "fs_motion": self.fs_motion,
            "duration_sec": self.duration_sec,
            "activity": self.activity_gt,
        }


_ACTIVITY_PAT = re.compile(
    r"_(sitting|sittingup|standing|standingup|walking|running)",
    flags=re.IGNORECASE,
)
_ACTIVITY_MAP = {
    "sitting": "sitting",
    "sittingup": "sitting",
    "standing": "standing",
    "standingup": "standing",
    "walking": "walking",
    "running": "running",
}


def _activity_from_filename(filename: str) -> str:
    match = _ACTIVITY_PAT.search(filename)
    if not match:
        return "unknown"
    return _ACTIVITY_MAP.get(match.group(1).lower(), "unknown")


def preprocess_raw_csv(
    csv_path: str | Path,
    fs_audio: int = FS_AUDIO,
    fs_motion: int = FS_MOTION,
) -> Record:
    """Load a headerless 4-column raw CSV and apply the project's preprocessing chain.

    Mirrors ``cough_analysis.preprocessing.load_record_preprocessed`` but takes a
    direct path instead of a record_id, so user-uploaded CSVs work too.
    """
    csv_path = Path(csv_path)
    raw = load_record_array(csv_path)
    pulmonary_raw = raw[:, 0].astype(np.float64)
    ambient_raw = raw[:, 1].astype(np.float64)
    stretch_raw, cough_label = decode_channel3(raw[:, 2])
    accz_raw = raw[:, 3].astype(np.float64)

    b_bp, a_bp = butter_bandpass(60, 2200, fs_audio, order=4)
    pulm_bp = signal.filtfilt(b_bp, a_bp, pulmonary_raw - np.median(pulmonary_raw))
    amb_bp = signal.filtfilt(b_bp, a_bp, ambient_raw - np.median(ambient_raw))

    stretch_centered = stretch_raw.astype(np.float64) - np.median(stretch_raw)
    n_motion = int(len(stretch_centered) * (fs_motion / fs_audio))
    stretch_resampled = signal.resample(stretch_centered, n_motion)
    accz_resampled = signal.resample(accz_raw, n_motion)

    b_lp, a_lp = butter_lowpass(20, fs_motion, order=4)
    stretch_lp = signal.filtfilt(b_lp, a_lp, stretch_resampled)
    accz_lp = signal.filtfilt(b_lp, a_lp, accz_resampled)

    return Record(
        filename=csv_path.name,
        path=str(csv_path),
        pulm_bp=pulm_bp.astype(np.float32),
        amb_bp=amb_bp.astype(np.float32),
        stretch_lp=stretch_lp.astype(np.float32),
        accz_lp=accz_lp.astype(np.float32),
        pulm_raw=pulmonary_raw.astype(np.float32),
        amb_raw=ambient_raw.astype(np.float32),
        stretch_raw=stretch_raw.astype(np.float32),
        accz_raw=accz_raw.astype(np.float32),
        cough_label=cough_label.astype(np.int64),
        fs_audio=fs_audio,
        fs_motion=fs_motion,
        activity_gt=_activity_from_filename(csv_path.name),
    )


# ----- Model loaders (cached) ---------------------------------------------- #


# MPS hangs the torchaudio MelSpectrogram path in our V3/V5 pipelines on macOS
# (it stalls indefinitely with no progress). 20-second records are fast enough
# on CPU (~100 ms for V3/V4, ~10 s for V5_AST first time), so we default to CPU
# for the demo. Override with COUGH_DEMO_DEVICE=mps|cuda to force a device.
@functools.lru_cache(maxsize=1)
def get_device() -> torch.device:
    forced = _os.environ.get("COUGH_DEMO_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cpu")


@functools.lru_cache(maxsize=8)
def _load_torch_checkpoint(path_str: str) -> dict[str, Any]:
    return torch.load(path_str, map_location=get_device(), weights_only=False)


@functools.lru_cache(maxsize=1)
def load_activity_bundle() -> dict[str, Any]:
    ckpt = _load_torch_checkpoint(str(V4_ACTIVITY_CHECKPOINT))
    classes = list(ckpt["classes"])
    model = V4ActivityCNN(num_classes=len(classes)).to(get_device())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    activity_cfg = ckpt["config"]["activity"]
    return {"model": model, "classes": classes, "activity_cfg": activity_cfg}


@functools.lru_cache(maxsize=4)
def load_model_bundle(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id={model_id!r}")
    entry = MODEL_REGISTRY[model_id]
    cfg = load_config(str(entry["config"]))

    if entry["kind"] == "v4":
        ckpt = _load_torch_checkpoint(str(entry["checkpoint"]))
        model = V4CoughFrameCNN().to(get_device())
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return {
            "kind": "v4",
            "model": model,
            "config": cfg,
            "spec_config": ckpt["spec_config"],
            "post": ckpt["selected_postprocessing"],
        }

    if entry["kind"] == "v3":
        ckpt = _load_torch_checkpoint(str(entry["checkpoint"]))
        model = Spec2DCoughCNN(num_classes=1).to(get_device())
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return {
            "kind": "v3",
            "model": model,
            "config": ckpt.get("config", cfg),
            "post": entry["postprocessing"],
        }

    if entry["kind"] == "v5_ast":
        ckpt = _load_torch_checkpoint(str(entry["checkpoint"]))
        head = ASTMotionFusionHead(audio_dim=int(cfg["ast"]["embedding_dim"]))
        head = head.to(get_device())
        head.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        head.eval()
        ast_extractor, ast_model = _load_ast_backbone(cfg["ast"]["model_name"])
        return {
            "kind": "v5_ast",
            "fusion_head": head,
            "ast_extractor": ast_extractor,
            "ast_model": ast_model,
            "config": cfg,
            "post": {
                "threshold": 0.9,
                "smoothing_sec": 0.3,
                "span_mode": "full",
                "pred_min_duration_sec": 0.1,
                "pred_merge_gap_sec": 0.3,
            },
        }

    if entry["kind"] == "classical":
        # The saved joblib is a dict: {model, config, feature_columns,
        # record_split, selected_postprocessing}. Use the embedded
        # selected_postprocessing rather than the registry fallback so the demo
        # tracks the report exactly.
        loaded = joblib.load(entry["checkpoint"])
        if isinstance(loaded, dict):
            model = loaded["model"]
            saved_post = loaded.get("selected_postprocessing") or {}
            feature_columns = loaded.get("feature_columns")
        else:
            model = loaded
            saved_post = {}
            feature_columns = None
        post = {**entry["postprocessing"], **{k: v for k, v in saved_post.items() if v is not None}}
        return {
            "kind": "classical",
            "model": model,
            "feature_columns": feature_columns,
            "config": cfg,
            "post": post,
        }

    raise RuntimeError(f"Unhandled model kind: {entry['kind']!r}")


@functools.lru_cache(maxsize=1)
def _load_ast_backbone(model_name: str):
    from transformers import AutoFeatureExtractor, ASTModel

    device = get_device()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    ast_model = ASTModel.from_pretrained(model_name).to(device)
    ast_model.eval()
    return feature_extractor, ast_model


# ----- Per-model inference -------------------------------------------------- #


def _run_v4_cough(record: Record, bundle: dict[str, Any]) -> dict[str, Any]:
    cfg = bundle["config"]
    cough_cfg = cfg["cough"]
    spec_cfg = bundle["spec_config"]
    device = get_device()

    probs = predict_cough_probabilities_for_record(
        bundle["model"],
        record.as_v4_dict(),
        cough_cfg,
        spec_cfg,
        device=device,
        batch_size=32,
    )
    frame_rate = int(round(record.fs_audio / int(cough_cfg["frame_hop_samples"])))
    post = bundle["post"]
    events = frame_predictions_to_events(
        probs,
        frame_rate=frame_rate,
        threshold=float(post["threshold"]),
        min_duration_sec=float(post["pred_min_duration_sec"]),
        merge_gap_sec=float(post["pred_merge_gap_sec"]),
        duration_sec=record.duration_sec,
    )
    times = np.arange(len(probs)) / frame_rate
    return {
        "prob_time": times.astype(np.float32),
        "prob_value": probs.astype(np.float32),
        "events": events,
        "threshold": float(post["threshold"]),
        "post": post,
    }


def _v3_window_probs(record: Record, bundle: dict[str, Any]) -> tuple[np.ndarray, list[tuple[float, float]], dict]:
    cfg = bundle["config"]
    window_cfg = cfg["windowing"]
    spec_cfg = cfg["spectrogram"]
    device = get_device()

    record_dict = {
        "pulm_bp": record.pulm_bp,
        "amb_bp": record.amb_bp,
        "stretch_lp": record.stretch_lp,
        "accz_lp": record.accz_lp,
        "cough_label": record.cough_label,
    }
    windows = build_centered_windows(
        record_dict,
        window_sec=float(window_cfg["window_sec"]),
        hop_sec=float(window_cfg["hop_sec"]),
        center_fraction=float(window_cfg["center_fraction"]),
        fs_audio=record.fs_audio,
        fs_motion=record.fs_motion,
    )
    mel = v3_make_mel_transform(
        sample_rate=record.fs_audio,
        spectrogram_config=spec_cfg,
    )
    log_eps = float(spec_cfg.get("log_eps", 1e-9))
    specs = v3_audio_to_log_mel(windows["audio"], mel_transform=mel, log_eps=log_eps)

    spec_tensor = torch.tensor(specs, dtype=torch.float32, device=device)
    motion_tensor = torch.tensor(windows["motion"], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = bundle["model"](spec_tensor, motion_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return probs, windows["spans"], window_cfg


def _events_from_window_probs(
    probs: np.ndarray,
    spans: list[tuple[float, float]],
    post: dict[str, Any],
    center_fraction: float,
) -> list[Event]:
    smoothed = smooth_probabilities(
        probs,
        spans,
        smoothing_sec=float(post.get("smoothing_sec", 0.0)),
    )
    preds = probabilities_to_predictions(
        smoothed,
        threshold=float(post["threshold"]),
    )
    return window_predictions_to_events(
        spans,
        preds,
        min_duration_sec=float(post.get("pred_min_duration_sec", 0.0)),
        merge_gap_sec=float(post.get("pred_merge_gap_sec", 0.0)),
        span_mode=str(post.get("span_mode", "full")),
        center_fraction=float(center_fraction),
    )


def _window_probs_to_timeline(
    probs: np.ndarray,
    spans: list[tuple[float, float]],
    duration_sec: float,
    frame_rate: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Project window-level probabilities onto a dense time axis by averaging
    overlapping windows. Used for v3 / v5_ast / classical visualisation."""
    n = max(1, int(round(duration_sec * frame_rate)))
    times = np.arange(n) / frame_rate
    accum = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    for prob, (start, end) in zip(probs, spans):
        i0 = max(0, int(round(start * frame_rate)))
        i1 = min(n, int(round(end * frame_rate)))
        if i1 <= i0:
            continue
        accum[i0:i1] += float(prob)
        counts[i0:i1] += 1.0
    out = np.divide(accum, counts, out=np.zeros_like(accum), where=counts > 0)
    return times, out


def _run_v3_cough(record: Record, bundle: dict[str, Any]) -> dict[str, Any]:
    probs, spans, window_cfg = _v3_window_probs(record, bundle)
    post = bundle["post"]
    events = _events_from_window_probs(
        probs,
        spans,
        post,
        center_fraction=float(window_cfg["center_fraction"]),
    )
    times, dense = _window_probs_to_timeline(probs, spans, record.duration_sec)
    return {
        "prob_time": times,
        "prob_value": dense,
        "events": events,
        "threshold": float(post["threshold"]),
        "post": post,
    }


def _run_v5_ast_cough(record: Record, bundle: dict[str, Any]) -> dict[str, Any]:
    cfg = bundle["config"]
    window_cfg = cfg["windowing"]
    ast_cfg = cfg["ast"]
    device = get_device()

    record_dict = {
        "pulm_bp": record.pulm_bp,
        "amb_bp": record.amb_bp,
        "stretch_lp": record.stretch_lp,
        "accz_lp": record.accz_lp,
        "cough_label": record.cough_label,
    }
    windows = build_centered_windows(
        record_dict,
        window_sec=float(window_cfg["window_sec"]),
        hop_sec=float(window_cfg["hop_sec"]),
        center_fraction=float(window_cfg["center_fraction"]),
        fs_audio=record.fs_audio,
        fs_motion=record.fs_motion,
    )

    # No disk cache here — V5 inference must actually run each time so the
    # presentation conveys live computation rather than pre-baked results.
    # On a 20 s record this is ~75 s on CPU; the spinner in the Streamlit
    # frontend sets the right expectation.
    embeddings = extract_ast_embeddings(
        windows["audio"],
        feature_extractor=bundle["ast_extractor"],
        ast_model=bundle["ast_model"],
        device=device,
        batch_size=int(ast_cfg.get("embedding_batch_size", 32)),
        ast_sample_rate=int(ast_cfg["sample_rate"]),
    )

    motion_tensor = torch.tensor(windows["motion"], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = bundle["fusion_head"](embeddings.to(device), motion_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    post = bundle["post"]
    events = _events_from_window_probs(
        probs,
        windows["spans"],
        post,
        center_fraction=float(window_cfg["center_fraction"]),
    )
    times, dense = _window_probs_to_timeline(probs, windows["spans"], record.duration_sec)
    return {
        "prob_time": times,
        "prob_value": dense,
        "events": events,
        "threshold": float(post["threshold"]),
        "post": post,
    }


def _run_classical_cough(record: Record, bundle: dict[str, Any]) -> dict[str, Any]:
    cfg = bundle["config"]
    window_cfg = cfg["windowing"]
    window_sec = float(window_cfg["window_sec"])
    hop_sec = float(window_cfg["hop_sec"])

    audio_win = int(round(window_sec * record.fs_audio))
    audio_hop = int(round(hop_sec * record.fs_audio))
    motion_win = int(round(window_sec * record.fs_motion))

    starts = window_starts(len(record.pulm_bp), audio_win, audio_hop)
    rows: list[dict[str, float]] = []
    spans: list[tuple[float, float]] = []
    for start in starts:
        end = start + audio_win
        motion_start = int(round((start / record.fs_audio) * record.fs_motion))
        motion_end = motion_start + motion_win
        if motion_end > len(record.stretch_lp):
            break
        rows.append(
            extract_ee491_features(
                record.pulm_bp[start:end],
                record.amb_bp[start:end],
                record.accz_lp[motion_start:motion_end],
                record.stretch_lp[motion_start:motion_end],
                fs_audio=record.fs_audio,
                fs_motion=record.fs_motion,
            )
        )
        spans.append((start / record.fs_audio, end / record.fs_audio))

    if not rows:
        return {
            "prob_time": np.zeros(1, dtype=np.float32),
            "prob_value": np.zeros(1, dtype=np.float32),
            "events": [],
            "threshold": float(bundle["post"]["threshold"]),
            "post": bundle["post"],
        }

    table = pd.DataFrame(rows)
    feature_columns = bundle.get("feature_columns") or FEATURE_COLUMNS
    features = table[feature_columns].to_numpy(dtype=np.float32)
    probs = bundle["model"].predict_proba(features)[:, 1].astype(np.float32)

    post = bundle["post"]
    binary = (probs >= float(post["threshold"])).astype(np.int64)
    events = window_predictions_to_events(
        spans,
        binary,
        min_duration_sec=float(post.get("pred_min_duration_sec", 0.0)),
        merge_gap_sec=float(post.get("pred_merge_gap_sec", 0.0)),
        span_mode="full",
    )
    times, dense = _window_probs_to_timeline(probs, spans, record.duration_sec)
    return {
        "prob_time": times,
        "prob_value": dense,
        "events": events,
        "threshold": float(post["threshold"]),
        "post": post,
    }


def _run_activity(record: Record) -> dict[str, Any]:
    bundle = load_activity_bundle()
    centers, probs = predict_activity_probabilities_for_record(
        bundle["model"],
        record.as_v4_dict(),
        bundle["activity_cfg"],
        device=get_device(),
        batch_size=32,
    )
    return {
        "centers": centers,
        "probs": probs,
        "classes": bundle["classes"],
    }


# ----- Public entry point --------------------------------------------------- #


def run_inference(model_id: str, record: Record) -> dict[str, Any]:
    bundle = load_model_bundle(model_id)
    t0 = time.perf_counter()
    if bundle["kind"] == "v4":
        cough = _run_v4_cough(record, bundle)
    elif bundle["kind"] == "v3":
        cough = _run_v3_cough(record, bundle)
    elif bundle["kind"] == "v5_ast":
        cough = _run_v5_ast_cough(record, bundle)
    elif bundle["kind"] == "classical":
        cough = _run_classical_cough(record, bundle)
    else:
        raise RuntimeError(f"Unhandled bundle kind: {bundle['kind']!r}")
    cough_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    activity = _run_activity(record)
    # Activity is only meaningful in the context of a cough — attribute one
    # label per *predicted* cough event using ``assign_activity_to_event``
    # (same helper the V4 pipeline uses to produce its event tables). We keep
    # the full per-window activity probabilities around in ``activity`` for any
    # future use, but the demo plot only consumes ``event_activities``.
    event_activities: list[dict[str, Any]] = []
    for ev in cough["events"]:
        assigned = assign_activity_to_event(
            ev,
            activity["centers"],
            activity["probs"],
            activity["classes"],
            context_sec=2.0,  # matches configs/final/v4_clean.yaml attribution_context_sec
        )
        event_activities.append({
            "start": float(ev.start),
            "end": float(ev.end),
            "activity": str(assigned["activity"]),
            "confidence": float(assigned["activity_confidence"]),
        })
    activity_time = time.perf_counter() - t1

    gt_events: list[Event] = binary_labels_to_events(
        record.cough_label,
        sample_rate=record.fs_audio,
        min_duration_sec=0.1,
        merge_gap_sec=0.1,
    )

    return {
        "model_id": model_id,
        "display_name": MODEL_REGISTRY[model_id]["display_name"],
        "cough": cough,
        "activity": activity,
        "event_activities": event_activities,
        "gt_cough_events": gt_events,
        "timings": {
            "cough_sec": cough_time,
            "activity_sec": activity_time,
            "total_sec": cough_time + activity_time,
        },
    }


__all__ = [
    "MODEL_REGISTRY",
    "Record",
    "load_activity_bundle",
    "load_model_bundle",
    "preprocess_raw_csv",
    "run_inference",
]
