"""Generate raw and preprocessed signal figures for the EE492 presentation.

Produces two stand-alone figures for record 91 (walking / noise, anonymized subject),
matching the slide flow: first the raw ADC signals, then the preprocessed
signals together with the ground-truth cough intervals. Run with the project
ML environment and PYTHONPATH=src.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cough_analysis import data as data_utils
from cough_analysis.preprocessing import load_record_preprocessed

RECORD_ID = 91
OUT_DIR = Path("presentation/figures")

C_PULM = "#1f5fd0"
C_AMB = "#d62728"
C_STRETCH = "#9b27b0"
C_ACC = "#2ca02c"
SHADE = "#f6b9b9"


def cough_intervals(label: np.ndarray, fs: int) -> list[tuple[float, float]]:
    """Contiguous runs of label==1 returned as (start_s, end_s)."""
    label = (np.asarray(label) > 0).astype(np.int8)
    if label.size == 0:
        return []
    diff = np.diff(label, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(s / fs, e / fs) for s, e in zip(starts, ends)]


def robust_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.median(x)
    scale = np.percentile(np.abs(x), 99)
    if scale <= 0:
        scale = np.max(np.abs(x)) or 1.0
    return np.clip(x / scale, -1.2, 1.2)


def shade(ax, intervals):
    for s, e in intervals:
        ax.axvspan(s, e, color=SHADE, alpha=0.6, lw=0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = data_utils.load_record(RECORD_ID)
    pre = load_record_preprocessed(RECORD_ID)

    fs_a = pre["fs_audio"]
    fs_m = pre["fs_motion"]
    t_a = np.arange(rec["pulmonary"].size) / fs_a
    t_m = np.arange(pre["stretch_lp"].size) / fs_m
    intervals = cough_intervals(rec["cough_label"], fs_a)
    n_events = len(intervals)

    # ---- Figure 1: raw ADC signals + ground truth ----
    fig, axes = plt.subplots(5, 1, figsize=(11, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.45]})
    panels = [
        ("Pulmonary Microphone", t_a, rec["pulmonary"], C_PULM),
        ("Ambient Microphone", t_a, rec["ambient"], C_AMB),
        ("Stretch Sensor", t_a, rec["stretch"], C_STRETCH),
        ("Accelerometer Z", t_a, rec["accel_z"], C_ACC),
    ]
    for ax, (title, t, y, c) in zip(axes[:4], panels):
        ax.plot(t, y, color=c, lw=0.6)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("ADC", fontsize=10)
        ax.grid(True, axis="x", alpha=0.25)
        ax.margins(x=0)

    gt = axes[4]
    gt.set_title(f"Ground Truth — {n_events} cough event(s)", loc="left",
                 fontsize=12, fontweight="bold")
    for s, e in intervals:
        gt.axvspan(s, e, color="#c0392b", alpha=0.85, lw=0)
    gt.set_ylim(0, 1)
    gt.set_yticks([])
    gt.set_ylabel("GT", fontsize=10)
    gt.margins(x=0)
    gt.set_xlabel("Time (s)", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "raw_signals_091.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: preprocessed signals + ground truth ----
    fig, axes = plt.subplots(5, 1, figsize=(11, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.45]})
    pp = [
        ("Pulmonary Microphone", t_a, robust_norm(pre["pulm_bp"]), C_PULM),
        ("Ambient Microphone", t_a, robust_norm(pre["amb_bp"]), C_AMB),
        ("Stretch Sensor", t_m, robust_norm(pre["stretch_lp"]), C_STRETCH),
        ("Accelerometer Z", t_m, robust_norm(pre["accz_lp"]), C_ACC),
    ]
    for ax, (title, t, y, c) in zip(axes[:4], pp):
        shade(ax, intervals)
        ax.plot(t, y, color=c, lw=0.7)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("Amp", fontsize=10)
        ax.set_ylim(-1.3, 1.3)
        ax.grid(True, axis="x", alpha=0.25)
        ax.margins(x=0)

    gt = axes[4]
    gt.set_title(f"Ground Truth — {n_events} cough event(s)", loc="left",
                 fontsize=12, fontweight="bold")
    for s, e in intervals:
        gt.axvspan(s, e, color="#c0392b", alpha=0.85, lw=0)
    gt.set_ylim(0, 1)
    gt.set_yticks([])
    gt.set_ylabel("GT", fontsize=10)
    gt.margins(x=0)
    gt.set_xlabel("Time (s)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "preprocessed_signals_091.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"record {RECORD_ID}: {rec['subject']} / {rec['activity']} / {rec['context']}")
    print(f"duration ~{t_a[-1]:.1f}s, {n_events} cough events")
    print(f"wrote {OUT_DIR/'raw_signals_091.png'} and {OUT_DIR/'preprocessed_signals_091.png'}")


if __name__ == "__main__":
    main()
