from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import FS_AUDIO
from cough_analysis.v3 import build_record_dataset


CATEGORY_ORDER = [
    "cough",
    "clean_non_cough",
    "noise_non_cough",
    "walking_noise_non_cough",
]

REFINED_CATEGORY_ORDER = [
    "pure_cough_core",
    "boundary_cough",
    "pure_non_cough",
    "hard_noise_non_cough",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create visual audit montages for audio spectrogram and motion windows. "
            "This is an experiment diagnostic tool, not the stable V3 workflow."
        )
    )
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--max-records", type=int, default=85)
    parser.add_argument("--record-ids", default=None, help="Comma/range list, e.g. 0-84 or 87,94.")
    parser.add_argument("--samples-per-category", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/experiments/visual_audit/v3")
    parser.add_argument("--include-per-window-plots", action="store_true")
    parser.add_argument(
        "--category-mode",
        choices=["center", "refined"],
        default="center",
        help=(
            "center keeps the original center-label categories; refined separates "
            "cough cores, boundary cough windows, pure clean negatives, and hard noisy negatives."
        ),
    )
    parser.add_argument(
        "--core-cough-min-fraction",
        type=float,
        default=0.8,
        help="Minimum full-window cough fraction for a positive window to be a pure cough core.",
    )
    parser.add_argument(
        "--pure-noncough-max-fraction",
        type=float,
        default=0.0,
        help="Maximum full-window cough fraction allowed for pure non-cough categories.",
    )
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def parse_record_ids(value: str | None, metadata) -> list[int]:
    if value is None:
        return [int(x) for x in metadata["record_id"].tolist()]
    record_ids: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", maxsplit=1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                raise ValueError(f"Invalid record id range: {token}")
            record_ids.extend(range(start, end + 1))
        else:
            record_ids.append(int(token))
    return record_ids


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


def normalize_image(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def window_category(row: dict, label: int) -> str | None:
    context = str(row["context"])
    activity = str(row["activity"])
    if label == 1:
        return "cough"
    if context == "clean":
        return "clean_non_cough"
    if activity == "walking" and context != "clean":
        return "walking_noise_non_cough"
    if context != "clean":
        return "noise_non_cough"
    return None


def refined_window_category(
    row: dict,
    label: int,
    cough_fraction_full_window: float,
    core_cough_min_fraction: float,
    pure_noncough_max_fraction: float,
) -> str | None:
    context = str(row["context"])
    if label == 1:
        if cough_fraction_full_window >= core_cough_min_fraction:
            return "pure_cough_core"
        return "boundary_cough"

    if cough_fraction_full_window > pure_noncough_max_fraction:
        return None
    if context == "clean":
        return "pure_non_cough"
    return "hard_noise_non_cough"


def cough_fraction(record: dict, start_sec: float, end_sec: float) -> float:
    start = max(0, int(round(start_sec * FS_AUDIO)))
    end = min(len(record["cough_label"]), int(round(end_sec * FS_AUDIO)))
    if end <= start:
        return 0.0
    return float(np.mean(record["cough_label"][start:end] > 0))


def collect_candidates(
    metadata,
    record_ids: list[int],
    cfg: dict,
    category_mode: str,
    core_cough_min_fraction: float,
    pure_noncough_max_fraction: float,
) -> list[dict]:
    window_cfg = cfg["windowing"]
    spec_cfg = cfg["spectrogram"]
    rows = []

    for record_id in record_ids:
        record_data = build_record_dataset(
            int(record_id),
            metadata,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
            spectrogram_config=spec_cfg,
        )
        record = record_data["record"]
        meta = metadata.loc[metadata["record_id"] == int(record_id)].iloc[0].to_dict()
        for idx, ((start, end), label) in enumerate(
            zip(record_data["spans"], record_data["labels"])
        ):
            frac = cough_fraction(record, float(start), float(end))
            center_category = window_category(meta, int(label))
            if category_mode == "refined":
                category = refined_window_category(
                    meta,
                    int(label),
                    frac,
                    core_cough_min_fraction=core_cough_min_fraction,
                    pure_noncough_max_fraction=pure_noncough_max_fraction,
                )
            else:
                category = center_category
            if category is None:
                continue
            rows.append(
                {
                    "record_id": int(record_id),
                    "window_index": int(idx),
                    "filename": str(meta["filename"]),
                    "date": str(meta["date"]),
                    "subject": str(meta["subject"]),
                    "activity": str(meta["activity"]),
                    "context": str(meta["context"]),
                    "category": category,
                    "center_category": center_category,
                    "label": int(label),
                    "start_sec": float(start),
                    "end_sec": float(end),
                    "cough_fraction_full_window": frac,
                }
            )
    return rows


def sample_candidates(
    rows: list[dict],
    category_order: list[str],
    samples_per_category: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    sampled = []
    for category in category_order:
        candidates = [row for row in rows if row["category"] == category]
        if not candidates:
            continue
        indices = np.arange(len(candidates))
        rng.shuffle(indices)
        for idx in indices[:samples_per_category]:
            sampled.append(candidates[int(idx)])
    return sampled


def load_window_payload(row: dict, metadata, cfg: dict) -> dict:
    window_cfg = cfg["windowing"]
    spec_cfg = cfg["spectrogram"]
    record_data = build_record_dataset(
        int(row["record_id"]),
        metadata,
        window_sec=float(window_cfg["window_sec"]),
        hop_sec=float(window_cfg["hop_sec"]),
        center_fraction=float(window_cfg["center_fraction"]),
        spectrogram_config=spec_cfg,
    )
    idx = int(row["window_index"])
    return {
        "spec": record_data["spec"][idx],
        "motion": record_data["motion"][idx],
        "span": record_data["spans"][idx],
        "label": int(record_data["labels"][idx]),
    }


def plot_montage(
    rows: list[dict],
    metadata,
    cfg: dict,
    category: str,
    output_path: Path,
) -> None:
    selected = [row for row in rows if row["category"] == category]
    if not selected:
        return

    cols = 4
    rows_count = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 4.2, rows_count * 3.2))
    axes_arr = np.asarray(axes).reshape(rows_count, cols)

    for ax in axes_arr.ravel():
        ax.axis("off")

    for ax, row in zip(axes_arr.ravel(), selected):
        payload = load_window_payload(row, metadata, cfg)
        spec = payload["spec"]
        motion = payload["motion"]
        pulmonary = normalize_image(spec[0])
        ambient = normalize_image(spec[1])
        image = np.vstack([pulmonary, ambient])
        ax.imshow(image, origin="lower", aspect="auto", cmap="magma")

        motion_ax = ax.inset_axes([0.0, -0.33, 1.0, 0.28])
        t = np.linspace(payload["span"][0], payload["span"][1], motion.shape[1], endpoint=False)
        motion_ax.plot(t, robust_scale(motion[0]), color="tab:green", linewidth=0.8)
        motion_ax.plot(t, robust_scale(motion[1]), color="tab:brown", linewidth=0.8)
        motion_ax.set_xlim(payload["span"][0], payload["span"][1])
        motion_ax.set_yticks([])
        motion_ax.tick_params(axis="x", labelsize=6)
        motion_ax.grid(True, linewidth=0.3, alpha=0.25)

        title = (
            f"r{row['record_id']} w{row['window_index']} "
            f"{row['activity']}/{row['context']}\n"
            f"{row['start_sec']:.2f}-{row['end_sec']:.2f}s "
            f"cough_frac={row['cough_fraction_full_window']:.2f}"
        )
        ax.set_title(title, fontsize=8)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{category}: pulmonary+ambient log-mel with stretch/accel overlay",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_single_window(row: dict, metadata, cfg: dict, output_path: Path) -> None:
    payload = load_window_payload(row, metadata, cfg)
    spec = payload["spec"]
    motion = payload["motion"]
    start, end = payload["span"]

    fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=False)
    axes[0].imshow(spec[0], origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Pulmonary log-mel")
    axes[1].imshow(spec[1], origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Ambient log-mel")

    t = np.linspace(start, end, motion.shape[1], endpoint=False)
    axes[2].plot(t, robust_scale(motion[0]), color="tab:green")
    axes[2].set_title("Stretch robust-scaled")
    axes[3].plot(t, robust_scale(motion[1]), color="tab:brown")
    axes[3].set_title("Accel Z robust-scaled")
    for ax in axes[2:]:
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.set_xlim(start, end)
    fig.suptitle(
        f"record={row['record_id']} window={row['window_index']} "
        f"{row['activity']}/{row['context']} label={row['label']} "
        f"{start:.2f}-{end:.2f}s"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def category_counts(rows: list[dict], category_order: list[str]) -> dict[str, int]:
    return {
        category: int(sum(1 for row in rows if row["category"] == category))
        for category in category_order
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    metadata_path = project_or_absolute(args.metadata or cfg["data"]["metadata"])
    metadata = load_metadata(metadata_path)
    if args.max_records is not None:
        metadata = metadata.head(args.max_records).copy()

    record_ids = parse_record_ids(args.record_ids, metadata)
    output_dir = project_or_absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_order = REFINED_CATEGORY_ORDER if args.category_mode == "refined" else CATEGORY_ORDER
    candidates = collect_candidates(
        metadata,
        record_ids,
        cfg,
        category_mode=args.category_mode,
        core_cough_min_fraction=args.core_cough_min_fraction,
        pure_noncough_max_fraction=args.pure_noncough_max_fraction,
    )
    sampled = sample_candidates(candidates, category_order, args.samples_per_category, args.seed)
    write_csv(output_dir / "candidate_windows.csv", candidates)
    write_csv(output_dir / "sampled_windows.csv", sampled)

    for category in category_order:
        plot_montage(
            sampled,
            metadata,
            cfg,
            category=category,
            output_path=output_dir / f"{category}_montage.png",
        )

    if args.include_per_window_plots:
        for row in sampled:
            plot_single_window(
                row,
                metadata,
                cfg,
                output_path=(
                    output_dir
                    / "windows"
                    / row["category"]
                    / f"record_{row['record_id']:03d}_window_{row['window_index']:04d}.png"
                ),
            )

    summary = {
        "config": args.config,
        "metadata": str(metadata_path),
        "record_count": int(len(record_ids)),
        "window_sec": float(cfg["windowing"]["window_sec"]),
        "hop_sec": float(cfg["windowing"]["hop_sec"]),
        "center_fraction": float(cfg["windowing"]["center_fraction"]),
        "n_fft": int(cfg["spectrogram"]["n_fft"]),
        "mel_hop_length": int(cfg["spectrogram"]["hop_length"]),
        "n_mels": int(cfg["spectrogram"]["n_mels"]),
        "category_mode": args.category_mode,
        "core_cough_min_fraction": float(args.core_cough_min_fraction),
        "pure_noncough_max_fraction": float(args.pure_noncough_max_fraction),
        "candidate_counts": category_counts(candidates, category_order),
        "sampled_counts": category_counts(sampled, category_order),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved visual audit: {output_dir}")
    print(f"Candidates: {summary['candidate_counts']}")
    print(f"Sampled: {summary['sampled_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
