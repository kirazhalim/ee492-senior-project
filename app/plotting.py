"""Four-stage pipeline visualisation for the live demo.

The demo renders the input → preprocessing → model-perceptual-input → prediction
story as four separate matplotlib figures that the Streamlit app stacks
vertically:

  1. ``make_raw_figure``           — raw ADC integers for the 4 sensor channels
  2. ``make_preprocessed_figure``  — filtered / normalised signals + GT shading
  3. ``make_model_input_figure``   — what the *selected model* actually sees
                                     (log-mel spectrogram for V3/V4/V5, feature
                                     heat-map for classical XGBoost)
  4. ``make_predictions_figure``   — GT cough trace, predicted cough probability
                                     + events, and per-event activity attribution

Splitting the figures lets the app render stages 1–3 immediately after the CSV
is preprocessed (~few hundred ms total), then wait for inference to render
stage 4. The audience sees the signals and the model's input representation
right away instead of staring at a spinner during the ~75 s V5_AST forward pass.

Visual style for the preprocessed and prediction figures follows the report
(``report/figures/preprocessed_signals.png`` and the v5 event-attribution
timeline). The raw figure intentionally keeps the unscaled ADC y-axis so the
audience can see the DC offset and sensor scales that preprocessing removes.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from cough_analysis.event_metrics import Event, match_events

from app.inference import Record


EVENT_IOU_THRESHOLD = 0.2

TP_COLOR = "#22c55e"   # green — predicted cough event matched a GT event
FP_COLOR = "#ef4444"   # red   — predicted cough event has no matching GT
FN_COLOR = "#ef4444"   # red   — GT event was missed by the model


SIGNAL_COLORS = {
    "pulmonary": "#1f3fc4",   # deep blue
    "ambient":   "#c43030",   # red
    "stretch":   "#b836b8",   # magenta
    "accel":     "#3aa64a",   # green
}
GT_BAND_COLOR = "#9a9a9a"
GT_BAND_ALPHA = 0.22
PRED_EVENT_COLOR = "#ff8c1a"
PRED_EVENT_ALPHA = 0.35
PROB_LINE_COLOR = "#ff8c1a"
THRESHOLD_LINE_COLOR = "#666666"

ACTIVITY_COLORS = {
    "stationary": "#3d70b2",
    "walking":    "#f1a340",
    "running":    "#d6604d",
    "sitting":    "#3d70b2",
    "standing":   "#5b7ea8",
    "unknown":    "#bdbdbd",
}


# ---------------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------------- #


def _shade_events(ax: plt.Axes, events: list[Event], color: str, alpha: float) -> None:
    for event in events:
        ax.axvspan(event.start, event.end, color=color, alpha=alpha, linewidth=0)


def _plot_trace(ax: plt.Axes, t: np.ndarray, y: np.ndarray, color: str, max_points: int = 6000) -> None:
    """Plot a trace, decimating long signals so matplotlib stays responsive."""
    if len(y) > max_points:
        step = int(np.ceil(len(y) / max_points))
        ax.plot(t[::step], y[::step], color=color, linewidth=0.6)
    else:
        ax.plot(t, y, color=color, linewidth=0.9)


def robust_scale(values: np.ndarray) -> np.ndarray:
    """Centre + 99th-percentile-normalise a signal to roughly [-1, 1].

    Same approach used in ``report/generate_report_support_figures.py`` so the
    preprocessed-stage plots match the report's figures.
    """
    values = np.asarray(values, dtype=np.float32)
    values = values - float(np.median(values))
    scale = float(np.percentile(np.abs(values), 99))
    if scale <= 1.0e-9:
        scale = float(np.max(np.abs(values))) or 1.0
    return np.clip(values / scale, -1.2, 1.2)


def _shared_setup(ax: plt.Axes, title: str, ylabel: str, duration: float) -> None:
    ax.set_title(title, fontsize=9, fontweight="semibold", loc="left", pad=2)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, axis="x", alpha=0.15)
    ax.set_xlim(0, duration)


def _label_events_on_signal(ax: plt.Axes, gt_events: list[Event]) -> None:
    _shade_events(ax, gt_events, color=GT_BAND_COLOR, alpha=GT_BAND_ALPHA)


# ---------------------------------------------------------------------------- #
# Stage 1 — raw ADC integers
# ---------------------------------------------------------------------------- #


def make_raw_figure(
    record: Record,
    *,
    figsize: tuple[float, float] = (12.0, 4.4),
    dpi: int = 110,
) -> Figure:
    """Plot the raw 4-channel ADC integers exactly as they came off the sensor.

    No filtering, no centring, no amplitude rescaling. The accelerometer and
    stretch DC offsets are deliberately visible so the audience can compare
    against the preprocessed view in the next figure.
    """
    duration = len(record.pulm_raw) / record.fs_audio
    t_audio = np.linspace(0, duration, len(record.pulm_raw), endpoint=False)
    t_motion = np.linspace(0, duration, len(record.stretch_raw), endpoint=False) \
        if len(record.stretch_raw) != len(record.pulm_raw) else t_audio

    fig, axes = plt.subplots(
        4, 1, figsize=figsize, dpi=dpi, sharex=True,
        gridspec_kw={"hspace": 0.38},
    )

    rows = [
        (axes[0], "Pulmonary Microphone", t_audio,  record.pulm_raw,    SIGNAL_COLORS["pulmonary"]),
        (axes[1], "Ambient Microphone",   t_audio,  record.amb_raw,     SIGNAL_COLORS["ambient"]),
        (axes[2], "Stretch Sensor",       t_motion, record.stretch_raw, SIGNAL_COLORS["stretch"]),
        (axes[3], "Accelerometer Z",      t_audio,  record.accz_raw,    SIGNAL_COLORS["accel"]),
    ]
    for ax, title, t, y, color in rows:
        _plot_trace(ax, t, y, color)
        _shared_setup(ax, title, "ADC", duration)
    axes[-1].set_xlabel("Time (s)", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- #
# Stage 2 — preprocessed signals
# ---------------------------------------------------------------------------- #


def make_preprocessed_figure(
    record: Record,
    gt_events: list[Event],
    *,
    figsize: tuple[float, float] = (12.0, 5.2),
    dpi: int = 110,
) -> Figure:
    """Plot the bandpassed / low-passed signals after ``preprocess_raw_csv``,
    plus a ground-truth cough trace below them.

    Amplitude is normalised per channel via :func:`robust_scale` for fair side-
    by-side comparison, matching the report's preprocessed-signals figure.
    GT cough intervals are shaded in grey on every signal row so the audience
    can visually correlate signal morphology with the label; the dedicated GT
    row at the bottom shows the same intervals as a discrete binary trace.
    """
    duration = record.duration_sec
    t_audio = np.linspace(0, duration, len(record.pulm_bp), endpoint=False)
    t_motion = np.linspace(0, duration, len(record.stretch_lp), endpoint=False)

    fig, axes = plt.subplots(
        5, 1,
        figsize=figsize, dpi=dpi,
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.0, 1.0, 1.0, 1.0, 0.55],
            "hspace": 0.42,
        },
    )

    rows = [
        (axes[0], "Pulmonary Microphone", t_audio,  record.pulm_bp,     SIGNAL_COLORS["pulmonary"]),
        (axes[1], "Ambient Microphone",   t_audio,  record.amb_bp,      SIGNAL_COLORS["ambient"]),
        (axes[2], "Stretch Sensor",       t_motion, record.stretch_lp,  SIGNAL_COLORS["stretch"]),
        (axes[3], "Accelerometer Z",      t_motion, record.accz_lp,     SIGNAL_COLORS["accel"]),
    ]
    for ax, title, t, y, color in rows:
        _plot_trace(ax, t, robust_scale(y), color)
        _label_events_on_signal(ax, gt_events)
        _shared_setup(ax, title, "Amp", duration)
        ax.set_ylim(-1.25, 1.25)

    # --- Row 5: ground-truth cough binary trace --------------------------- #
    ax_gt = axes[4]
    gt_dense_t = np.linspace(0, duration, len(record.cough_label), endpoint=False)
    if len(record.cough_label) > 6000:
        step = int(np.ceil(len(record.cough_label) / 6000))
        ax_gt.fill_between(gt_dense_t[::step], 0, record.cough_label[::step],
                           step="post", color=GT_BAND_COLOR, alpha=0.6)
    else:
        ax_gt.fill_between(gt_dense_t, 0, record.cough_label,
                           step="post", color=GT_BAND_COLOR, alpha=0.6)
    ax_gt.set_ylim(-0.05, 1.05)
    ax_gt.set_yticks([0, 1])
    _shared_setup(
        ax_gt,
        f"Ground Truth — {len(gt_events)} cough event(s)",
        "GT", duration,
    )
    axes[-1].set_xlabel("Time (s)", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------- #
# Stage 3 — what the model sees (model-specific input representation)
# ---------------------------------------------------------------------------- #


def _log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    log_eps: float = 1.0e-9,
) -> np.ndarray:
    """Compute a log-mel spectrogram for visualisation, matching the project's
    train-time params per model. Returns shape (n_mels, n_frames)."""
    import torch
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    with torch.no_grad():
        spec = mel(torch.tensor(audio, dtype=torch.float32))
    return np.log(spec.numpy() + log_eps)


def _draw_motion_branch(
    ax: plt.Axes,
    record: Record,
    gt_events: list[Event],
    duration: float,
) -> None:
    """Plot the (stretch_lp, accz_lp) tensor that the deep models' 1D-CNN
    motion branch consumes. Both channels are robust-scaled for visual parity
    and overlaid on the same axes with a legend; GT cough intervals are
    outlined in yellow exactly like the spectrogram above.
    """
    t = np.linspace(0, duration, len(record.stretch_lp), endpoint=False)
    ax.plot(t, robust_scale(record.stretch_lp), color=SIGNAL_COLORS["stretch"],
            linewidth=0.9, label="stretch")
    ax.plot(t, robust_scale(record.accz_lp), color=SIGNAL_COLORS["accel"],
            linewidth=0.9, label="accel z")
    # Same cyan as the spectrogram above — stage 3 highlights the *same* cough
    # events in both panels, so they share the colour.
    gt_color = "#22d3ee"
    for ev in gt_events:
        ax.axvspan(
            ev.start, ev.end,
            facecolor="none",
            edgecolor=gt_color, linewidth=2.0, linestyle="-", alpha=0.95,
        )
        ax.axvline(ev.start, color=gt_color, linewidth=1.0, alpha=0.9)
        ax.axvline(ev.end,   color=gt_color, linewidth=1.0, alpha=0.9)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.25, 1.25)
    ax.set_ylabel("Motion", fontsize=8)
    ax.set_title(
        "Motion branch input · stretch + accel z @ 100 Hz "
        "(1D-CNN tensor shape: (2, T))",
        fontsize=9, fontweight="semibold", loc="left", pad=2,
    )
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, axis="x", alpha=0.15)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)


def _draw_spectrogram(
    ax: plt.Axes,
    spec: np.ndarray,
    duration: float,
    f_min: float,
    f_max: float,
    title: str,
    gt_events: list[Event],
) -> None:
    # ``magma`` matches the presentation slides (and the report's spectrogram
    # figures) so the audience can relate the demo view back to the deck.
    ax.imshow(
        spec, origin="lower", aspect="auto",
        extent=(0.0, duration, f_min, f_max),
        cmap="magma", interpolation="nearest",
    )
    # GT cough intervals are outlined in cyan — picked to maximise contrast
    # against the warm ``magma`` palette (which itself ranges purple → orange
    # → cream). Outline only, no fill, so the underlying spectral content
    # stays fully visible inside the cough interval.
    gt_color = "#22d3ee"
    for ev in gt_events:
        ax.axvspan(
            ev.start, ev.end,
            facecolor="none",
            edgecolor=gt_color, linewidth=2.4, linestyle="-", alpha=0.95,
        )
        ax.axvline(ev.start, color=gt_color, linewidth=1.2, alpha=0.9)
        ax.axvline(ev.end,   color=gt_color, linewidth=1.2, alpha=0.9)
    ax.set_ylabel("Frequency (Hz)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="semibold", loc="left", pad=2)
    ax.tick_params(axis="both", labelsize=7)


def _make_audio_plus_motion_figure(
    record: Record,
    gt_events: list[Event],
    spec: np.ndarray,
    spec_title: str,
    spec_f_min: float,
    spec_f_max: float,
    *,
    figsize: tuple[float, float],
    dpi: int,
) -> Figure:
    """Shared layout for the V3/V4/V5 stage-3 figure: spectrogram on top,
    motion-branch overlay below. The deep models have two branches (audio CNN
    + motion CNN) fused late; we show both so the audience can see the full
    fusion input rather than just the audio side.
    """
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.42},
    )
    _draw_spectrogram(
        axes[0], spec, duration=record.duration_sec,
        f_min=spec_f_min, f_max=spec_f_max,
        title=spec_title, gt_events=gt_events,
    )
    axes[0].set_xlim(0, record.duration_sec)
    _draw_motion_branch(axes[1], record, gt_events, duration=record.duration_sec)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout()
    return fig


def make_v3_spectrogram_figure(
    record: Record,
    gt_events: list[Event],
    *,
    figsize: tuple[float, float] = (12.0, 3.7),
    dpi: int = 110,
) -> Figure:
    """V3 stage-3 figure: log-mel spectrogram of the pulmonary mic (V3's
    audio-branch input) on top and the (stretch, accel z) motion-branch input
    below.
    """
    spec = _log_mel_spectrogram(
        record.pulm_bp, sample_rate=record.fs_audio,
        n_fft=512, hop_length=128, n_mels=64, f_min=60, f_max=2200,
    )
    return _make_audio_plus_motion_figure(
        record, gt_events, spec,
        spec_title="V3 audio branch · log-Mel · n_fft=512 · hop=128 · 64 mels · 60–2200 Hz",
        spec_f_min=60, spec_f_max=2200,
        figsize=figsize, dpi=dpi,
    )


def make_v4_spectrogram_figure(
    record: Record,
    gt_events: list[Event],
    *,
    figsize: tuple[float, float] = (12.0, 3.7),
    dpi: int = 110,
) -> Figure:
    """V4 stage-3 figure: log-mel spectrogram (V4 frame head's audio input) on
    top and the (stretch, accel z) motion-branch input below.
    """
    spec = _log_mel_spectrogram(
        record.pulm_bp, sample_rate=record.fs_audio,
        n_fft=256, hop_length=48, n_mels=64, f_min=60, f_max=2200,
    )
    return _make_audio_plus_motion_figure(
        record, gt_events, spec,
        spec_title="V4 audio branch · log-Mel · n_fft=256 · hop=48 · 64 mels · 60–2200 Hz",
        spec_f_min=60, spec_f_max=2200,
        figsize=figsize, dpi=dpi,
    )


def make_v5_spectrogram_figure(
    record: Record,
    gt_events: list[Event],
    *,
    figsize: tuple[float, float] = (12.0, 3.7),
    dpi: int = 110,
) -> Figure:
    """16 kHz log-mel spectrogram that the frozen AST backbone effectively sees.

    The actual AST feature extractor computes its own mel-spec internally on a
    16 kHz waveform; rather than duplicating its private params bit-for-bit we
    show the same waveform under conventional AST-style mel-spec params
    (25 ms / 10 ms / 128 mels), which is what the AudioSet checkpoint was
    pretrained on. The point is to convey that "AST is fed mel-spec too, just
    at a higher sampling rate and finer mel resolution".
    """
    import torch
    import torchaudio

    # Resample pulmonary to AST's 16 kHz.
    resampler = torchaudio.transforms.Resample(orig_freq=record.fs_audio, new_freq=16000)
    with torch.no_grad():
        waveform = resampler(torch.tensor(record.pulm_bp, dtype=torch.float32)).numpy()

    # AST-style mel-spec params. Pulmonary is bandpassed to 60–2200 Hz upstream
    # so anything above ~2.4 kHz is silent; we cap f_max at 4 kHz so the visible
    # band stays meaningful. n_fft=512 (257 freq bins) gives enough resolution
    # for 64 mel filters to span 0–4 kHz without torchaudio "filterbank has all
    # zero values" warnings.
    spec = _log_mel_spectrogram(
        waveform, sample_rate=16000,
        n_fft=512, hop_length=160, n_mels=64, f_min=0, f_max=4000,
    )
    return _make_audio_plus_motion_figure(
        record, gt_events, spec,
        spec_title="V5 audio branch · 16 kHz log-Mel · n_fft=512 · hop=160 · 64 mels · 0–4000 Hz "
                   "(frozen AST embeds this)",
        spec_f_min=0, spec_f_max=4000,
        figsize=figsize, dpi=dpi,
    )


def make_classical_features_figure(
    record: Record,
    gt_events: list[Event],
    *,
    figsize: tuple[float, float] = (12.0, 3.2),
    dpi: int = 110,
) -> Figure:
    """Per-window heat-map of the 8 hand-crafted EE491 features that drive the
    classical XGBoost classifier. Each row is z-scored across the record so the
    relative dynamics are visible despite very different absolute scales.
    """
    from cough_analysis.classical_ml import (
        FEATURE_COLUMNS,
        extract_ee491_features,
        window_starts,
    )

    # Use the same windowing as the classical config: 0.2 s window, 0.05 s hop.
    window_sec = 0.2
    hop_sec = 0.05
    audio_win = int(round(window_sec * record.fs_audio))
    audio_hop = int(round(hop_sec * record.fs_audio))
    motion_win = int(round(window_sec * record.fs_motion))

    starts = window_starts(len(record.pulm_bp), audio_win, audio_hop)
    feats: list[dict[str, float]] = []
    centers: list[float] = []
    for start in starts:
        end = start + audio_win
        m_start = int(round((start / record.fs_audio) * record.fs_motion))
        m_end = m_start + motion_win
        if m_end > len(record.stretch_lp):
            break
        feats.append(extract_ee491_features(
            record.pulm_bp[start:end],
            record.amb_bp[start:end],
            record.accz_lp[m_start:m_end],
            record.stretch_lp[m_start:m_end],
            fs_audio=record.fs_audio,
            fs_motion=record.fs_motion,
        ))
        centers.append((start + end) / 2 / record.fs_audio)

    if not feats:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "no feature windows", ha="center", va="center")
        return fig

    matrix = np.array(
        [[row[c] for row in feats] for c in FEATURE_COLUMNS],
        dtype=np.float32,
    )
    # Row-wise z-score so features with very different absolute scales (e.g.
    # spec_centroid in Hz vs. log_rms_ratio in nats) are visually comparable.
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True) + 1.0e-6
    matrix_z = (matrix - mean) / std

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        matrix_z, origin="lower", aspect="auto",
        extent=(0.0, record.duration_sec, -0.5, len(FEATURE_COLUMNS) - 0.5),
        cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest",
    )
    # Same highlight treatment as the spectrograms: thick solid border + sharp
    # vertical lines at the boundaries, no fill. Black reads well on the
    # diverging RdBu colormap regardless of cell colour.
    gt_color = "#111111"
    for ev in gt_events:
        ax.axvspan(
            ev.start, ev.end,
            facecolor="none",
            edgecolor=gt_color, linewidth=2.2, linestyle="-", alpha=0.85,
        )
        ax.axvline(ev.start, color=gt_color, linewidth=1.2, alpha=0.8)
        ax.axvline(ev.end,   color=gt_color, linewidth=1.2, alpha=0.8)
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=8)
    ax.set_xlim(0, record.duration_sec)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title(
        "Classical features · row-wise z-scored · window=0.2 s · hop=0.05 s · "
        f"{matrix.shape[1]} windows",
        fontsize=9, fontweight="semibold", loc="left", pad=2,
    )
    ax.tick_params(axis="x", labelsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("z-score", fontsize=8)
    fig.tight_layout()
    return fig


def make_model_input_figure(
    record: Record,
    gt_events: list[Event],
    model_id: str,
) -> Figure:
    """Dispatch to the appropriate per-model input visualisation."""
    if model_id == "v3":
        return make_v3_spectrogram_figure(record, gt_events)
    if model_id == "v4":
        return make_v4_spectrogram_figure(record, gt_events)
    if model_id == "v5_ast":
        return make_v5_spectrogram_figure(record, gt_events)
    if model_id == "classical":
        return make_classical_features_figure(record, gt_events)
    raise ValueError(f"Unknown model_id={model_id!r}")


# ---------------------------------------------------------------------------- #
# Stage 4 — model output
# ---------------------------------------------------------------------------- #


def make_predictions_figure(
    record: Record,
    result: dict[str, Any],
    *,
    figsize: tuple[float, float] = (12.0, 3.8),
    dpi: int = 110,
) -> Figure:
    """Plot ground truth, predicted cough probability + events, and per-event
    activity attribution. Visual style mirrors the report's prediction timeline
    and the v5 event-attribution figure.
    """
    cough = result["cough"]
    gt_events: list[Event] = result["gt_cough_events"]
    pred_events: list[Event] = cough["events"]
    duration = record.duration_sec

    # Event-level matching: which predictions are true positives vs false
    # positives, and which GT events the model missed. IoU 0.2 matches the
    # report's event_iou_threshold.
    matches = match_events(gt_events, pred_events, iou_threshold=EVENT_IOU_THRESHOLD)
    matched_pred_idx = {pred_idx for (_gt, pred_idx, _iou) in matches}
    matched_gt_idx = {gt_idx for (gt_idx, _pred, _iou) in matches}
    tp_count = len(matches)
    fp_count = len(pred_events) - tp_count
    fn_count = len(gt_events) - tp_count

    fig, axes = plt.subplots(
        3, 1,
        figsize=figsize, dpi=dpi,
        sharex=True,
        gridspec_kw={
            "height_ratios": [0.55, 1.1, 0.65],
            "hspace": 0.45,
        },
    )

    # Row 1: GT cough events, colour-coded by match status ----------------- #
    ax_gt = axes[0]
    for idx, ev in enumerate(gt_events):
        if idx in matched_gt_idx:
            # Matched GT event — neutral gray, same look as the preprocessed
            # figure's GT row so the audience can see "the model got this one".
            ax_gt.axvspan(ev.start, ev.end, color=GT_BAND_COLOR, alpha=0.6, linewidth=0)
        else:
            # Missed GT (FN) — red hatched border to call attention to it.
            ax_gt.axvspan(
                ev.start, ev.end,
                facecolor=FN_COLOR, alpha=0.18,
                edgecolor=FN_COLOR, linewidth=1.5, hatch="///",
            )
    ax_gt.set_ylim(-0.05, 1.05)
    ax_gt.set_yticks([0, 1])
    _shared_setup(
        ax_gt,
        f"Ground Truth — {len(gt_events)} cough event(s) · activity (filename): {record.activity_gt} "
        f"· missed (FN): {fn_count}",
        "GT", duration,
    )

    # Row 2: predicted cough probability + events colour-coded TP / FP ----- #
    ax_p = axes[1]
    prob_t = cough["prob_time"]
    prob_y = cough["prob_value"]
    ax_p.plot(prob_t, prob_y, color=PROB_LINE_COLOR, linewidth=1.2, label="P(cough)")
    ax_p.axhline(cough["threshold"], color=THRESHOLD_LINE_COLOR,
                 linestyle="--", linewidth=0.8, alpha=0.7,
                 label=f"threshold = {cough['threshold']:.2f}")
    for idx, ev in enumerate(pred_events):
        color = TP_COLOR if idx in matched_pred_idx else FP_COLOR
        ax_p.axvspan(ev.start, ev.end, color=color, alpha=0.32, linewidth=0)
    # Legend entries for TP/FP. Build them as proxy patches so they appear
    # alongside the P(cough) and threshold lines.
    tp_patch = Patch(facecolor=TP_COLOR, alpha=0.32, label=f"TP ({tp_count})")
    fp_patch = Patch(facecolor=FP_COLOR, alpha=0.32, label=f"FP ({fp_count})")
    fn_patch = Patch(facecolor=FN_COLOR, alpha=0.18, hatch="///",
                     edgecolor=FN_COLOR, label=f"FN ({fn_count})")
    ax_p.set_ylim(-0.02, 1.05)
    ax_p.set_yticks([0.0, 0.5, 1.0])
    _shared_setup(
        ax_p,
        f"Predicted cough events — {len(pred_events)} event(s) detected "
        f"(TP={tp_count}, FP={fp_count}, FN={fn_count}, IoU≥{EVENT_IOU_THRESHOLD})",
        "P(cough)", duration,
    )
    line_handles, line_labels = ax_p.get_legend_handles_labels()
    ax_p.legend(
        line_handles + [tp_patch, fp_patch, fn_patch],
        line_labels + [tp_patch.get_label(), fp_patch.get_label(), fn_patch.get_label()],
        loc="upper right", fontsize=7, frameon=False, ncol=5,
    )

    # Row 3: per-event activity attribution -------------------------------- #
    ax_a = axes[2]
    event_activities = result.get("event_activities", [])
    seen_classes: set[str] = set()
    for item in event_activities:
        start = float(item["start"])
        end = float(item["end"])
        cls = str(item["activity"])
        conf = float(item["confidence"])
        color = ACTIVITY_COLORS.get(cls, "#999999")
        ax_a.axvspan(start, end, color=color, alpha=0.75)
        seen_classes.add(cls)
        cx = (start + end) / 2
        if end - start >= 0.8:
            ax_a.text(
                cx, 0.5, f"{cls}\n{conf:.0%}",
                ha="center", va="center", fontsize=8, fontweight="medium",
                color="white",
            )
        else:
            ax_a.text(
                cx, 1.02, f"{cls} {conf:.0%}",
                ha="center", va="bottom", fontsize=7, fontweight="medium",
                color=color,
            )
    ax_a.set_ylim(0, 1)
    ax_a.set_yticks([])
    _shared_setup(
        ax_a,
        f"Activity assigned per predicted cough event "
        f"(V4 motion head, {len(event_activities)} event(s))",
        "Activity", duration,
    )
    ax_a.set_xlabel("Time (s)", fontsize=9)
    if seen_classes:
        handles = [
            Patch(facecolor=ACTIVITY_COLORS.get(c, "#999999"), alpha=0.75, label=c)
            for c in sorted(seen_classes)
        ]
        ax_a.legend(handles=handles, loc="upper right", fontsize=7,
                    frameon=False, ncol=len(handles))

    fig.tight_layout()
    return fig


__all__ = [
    "make_raw_figure",
    "make_preprocessed_figure",
    "make_model_input_figure",
    "make_v3_spectrogram_figure",
    "make_v4_spectrogram_figure",
    "make_v5_spectrogram_figure",
    "make_classical_features_figure",
    "make_predictions_figure",
    "robust_scale",
]
