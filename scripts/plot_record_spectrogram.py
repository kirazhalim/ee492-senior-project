from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from cough_analysis.data import load_metadata
from cough_analysis.preprocessing import load_record_preprocessed


# Change these values while trying different spectrogram settings.
RECORD_ID = 8
N_FFT = 256 
HOP_LENGTH = 32
WINDOW = "hann"
F_MIN = 60
F_MAX = 2200
LOG_EPS = 1.0e-9

START_SEC = None
END_SEC = None
METADATA_PATH = "data/metadata.csv"
DATA_ROOT = None
OUTPUT_DIR = "artifacts/record_spectrograms"
DPI = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot preprocessed sensors, audio spectrograms, and cough labels for one record."
    )
    parser.add_argument("--record-id", type=int, default=RECORD_ID)
    parser.add_argument("--n-fft", type=int, default=N_FFT)
    parser.add_argument("--hop-length", type=int, default=HOP_LENGTH)
    parser.add_argument("--f-min", type=float, default=F_MIN)
    parser.add_argument("--f-max", type=float, default=F_MAX)
    parser.add_argument("--start-sec", type=float, default=START_SEC)
    parser.add_argument("--end-sec", type=float, default=END_SEC)
    parser.add_argument("--metadata", default=METADATA_PATH)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--save", action="store_true", help="Also save the plot as a PNG.")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])
    scale = float(q3 - q1)
    if scale <= 1.0e-9:
        scale = float(np.std(values))
    if scale <= 1.0e-9:
        return values - median
    return (values - median) / scale


def slice_signal(values: np.ndarray, fs: int, start_sec: float, end_sec: float) -> tuple[np.ndarray, np.ndarray]:
    start_idx = max(0, int(round(start_sec * fs)))
    end_idx = min(len(values), int(round(end_sec * fs)))
    sliced = values[start_idx:end_idx]
    times = np.arange(start_idx, end_idx, dtype=np.float32) / fs
    return sliced, times


def cough_spans(labels: np.ndarray, fs: int, start_sec: float) -> list[tuple[float, float]]:
    active = np.asarray(labels) > 0
    edges = np.diff(np.concatenate([[False], active, [False]]).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(start_sec + s / fs, start_sec + e / fs) for s, e in zip(starts, ends)]


def add_cough_spans(ax: plt.Axes, spans: list[tuple[float, float]]) -> None:
    for start, end in spans:
        ax.axvspan(start, end, color="crimson", alpha=0.12, linewidth=0)


def make_spectrogram(
    values: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("N_FFT and HOP_LENGTH must be positive.")
    if hop_length > n_fft:
        raise ValueError("HOP_LENGTH must be smaller than or equal to N_FFT.")
    if len(values) < n_fft:
        raise ValueError("Selected audio segment is shorter than N_FFT.")

    freqs, times, magnitude = signal.spectrogram(
        values,
        fs=fs,
        window=WINDOW,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    keep = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(keep):
        raise ValueError("No spectrogram bins left after applying F_MIN/F_MAX.")
    spec_db = 20.0 * np.log10(magnitude[keep] + LOG_EPS)
    return freqs[keep], times, spec_db


def plot_record(args: argparse.Namespace) -> Path | None:
    metadata_path = project_or_absolute(args.metadata)
    data_root = project_or_absolute(args.data_root) if args.data_root else metadata_path.parent
    metadata = load_metadata(metadata_path)
    record = load_record_preprocessed(args.record_id, metadata=metadata, data_root=data_root)
    row = metadata.loc[metadata["record_id"] == args.record_id].iloc[0]

    fs_audio = int(record["fs_audio"])
    fs_motion = int(record["fs_motion"])
    duration = float(record["duration_sec"])
    start_sec = 0.0 if args.start_sec is None else max(0.0, float(args.start_sec))
    end_sec = duration if args.end_sec is None else min(duration, float(args.end_sec))
    if end_sec <= start_sec:
        raise ValueError("end-sec must be greater than start-sec.")

    pulm, t_audio = slice_signal(record["pulm_bp"], fs_audio, start_sec, end_sec)
    amb, _ = slice_signal(record["amb_bp"], fs_audio, start_sec, end_sec)
    labels, t_label = slice_signal(record["cough_label"], fs_audio, start_sec, end_sec)
    stretch, t_motion = slice_signal(record["stretch_lp"], fs_motion, start_sec, end_sec)
    accz, _ = slice_signal(record["accz_lp"], fs_motion, start_sec, end_sec)

    freqs, spec_t, pulm_spec = make_spectrogram(
        pulm, fs_audio, args.n_fft, args.hop_length, args.f_min, args.f_max
    )
    _, _, amb_spec = make_spectrogram(
        amb, fs_audio, args.n_fft, args.hop_length, args.f_min, args.f_max
    )
    spec_t = spec_t + start_sec
    vmin, vmax = np.percentile(np.concatenate([pulm_spec.ravel(), amb_spec.ravel()]), [2, 98])
    spans = cough_spans(labels, fs_audio, start_sec)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(16, 13),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.55, 1.55, 0.75]},
    )

    sensor_rows = [
        ("Pulmonary mic BP", t_audio, robust_scale(pulm), "#1f77b4"),
        ("Ambient mic BP", t_audio, robust_scale(amb), "#ff7f0e"),
        ("Stretch LP", t_motion, robust_scale(stretch), "tab:green"),
        ("Accel Z LP", t_motion, robust_scale(accz), "tab:brown"),
    ]
    for ax, (name, times, values, color) in zip(axes[:4], sensor_rows):
        add_cough_spans(ax, spans)
        ax.plot(times, values, color=color, linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    im0 = axes[4].imshow(
        pulm_spec,
        origin="lower",
        aspect="auto",
        extent=[spec_t[0], spec_t[-1], freqs[0], freqs[-1]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    axes[4].set_ylabel("Pulm spec\nHz")
    add_cough_spans(axes[4], spans)

    axes[5].imshow(
        amb_spec,
        origin="lower",
        aspect="auto",
        extent=[spec_t[0], spec_t[-1], freqs[0], freqs[-1]],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    axes[5].set_ylabel("Ambient spec\nHz")
    add_cough_spans(axes[5], spans)
    fig.colorbar(im0, ax=axes[4:6], label="Magnitude (dB)", pad=0.01)

    axes[6].fill_between(t_label, 0, labels, step="pre", color="silver", alpha=0.9)
    axes[6].step(t_label, labels, where="pre", color="black", linewidth=0.8)
    axes[6].set_yticks([0, 1])
    axes[6].set_ylim(-0.1, 1.1)
    axes[6].set_ylabel("GT cough")
    axes[6].set_xlabel("Time (seconds)")
    axes[6].grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    axes[6].set_xlim(start_sec, end_sec)

    fig.suptitle(
        (
            f"Record {args.record_id}: {row['filename']} | "
            f"{row['subject']} / {row['activity']} / {row['context']} | "
            f"n_fft={args.n_fft}, hop={args.hop_length}, "
            f"hop={args.hop_length / fs_audio * 1000:.1f} ms"
        ),
        fontsize=14,
        fontweight="bold",
    )

    output_dir = project_or_absolute(args.output_dir)
    output_path = (
        output_dir
        / f"record_{args.record_id:03d}_fft{args.n_fft}_hop{args.hop_length}_{start_sec:.1f}-{end_sec:.1f}s.png"
    )
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    plt.show()
    return output_path if args.save else None


def main() -> int:
    plot_record(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
