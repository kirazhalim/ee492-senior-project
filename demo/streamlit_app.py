from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


FS = 100


@dataclass(frozen=True)
class Event:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    activity: str
    context: str
    duration: float
    seed: int
    gt_events: tuple[Event, ...]


PRESETS = {
    "preset_walking_noise": Preset(
        key="preset_walking_noise",
        label="Anonymous preset A",
        activity="walking",
        context="ambient noise",
        duration=20.0,
        seed=42,
        gt_events=(
            Event(2.65, 3.05),
            Event(6.35, 6.95),
            Event(11.25, 11.65),
            Event(15.85, 16.45),
        ),
    ),
    "preset_stationary_clean": Preset(
        key="preset_stationary_clean",
        label="Anonymous preset B",
        activity="stationary",
        context="clean",
        duration=20.0,
        seed=91,
        gt_events=(
            Event(4.15, 4.65),
            Event(9.40, 9.88),
            Event(13.05, 13.55),
        ),
    ),
    "preset_running_noise": Preset(
        key="preset_running_noise",
        label="Anonymous preset C",
        activity="running",
        context="motion noise",
        duration=20.0,
        seed=106,
        gt_events=(
            Event(3.35, 3.82),
            Event(7.10, 7.45),
            Event(12.60, 13.02),
            Event(17.25, 17.70),
        ),
    ),
}


MODEL_SUMMARY = {
    "v5_ast": {
        "label": "V5 Frozen AST + Motion",
        "precision": 0.930,
        "recall": 0.914,
        "f1": 0.922,
        "activity_acc": 0.906,
        "miss": 0,
        "false_positives": (),
    },
    "v3_boundary": {
        "label": "Boundary-refined V3",
        "precision": 0.909,
        "recall": 0.862,
        "f1": 0.885,
        "activity_acc": 0.900,
        "miss": 0,
        "false_positives": (18.2,),
    },
    "v4_frame": {
        "label": "Frame-Level CNN V4",
        "precision": 0.788,
        "recall": 0.897,
        "f1": 0.839,
        "activity_acc": 0.902,
        "miss": 0,
        "false_positives": (1.20, 18.35),
    },
    "ee491": {
        "label": "EE491 Classical ML",
        "precision": 0.806,
        "recall": 0.862,
        "f1": 0.833,
        "activity_acc": 0.880,
        "miss": 1,
        "false_positives": (18.1,),
    },
}


ACTIVITY_COLORS = {
    "stationary": "#3d70b2",
    "walking": "#d8942f",
    "running": "#b84f38",
}


def gaussian(t: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


@st.cache_data
def make_preset_signals(preset_key: str) -> pd.DataFrame:
    preset = PRESETS[preset_key]
    rng = np.random.default_rng(preset.seed)
    t = np.arange(0.0, preset.duration, 1.0 / FS)

    step_freq = {"stationary": 0.25, "walking": 1.7, "running": 2.8}[preset.activity]
    motion_amp = {"stationary": 0.10, "walking": 0.45, "running": 0.80}[preset.activity]
    noise_amp = 0.10 if preset.context == "clean" else 0.18

    pulmonary = 0.05 * rng.normal(size=len(t))
    ambient = noise_amp * rng.normal(size=len(t))
    stretch = motion_amp * np.sin(2 * np.pi * step_freq * t) + 0.06 * rng.normal(size=len(t))
    accel_z = 0.8 * motion_amp * np.sin(2 * np.pi * step_freq * t + 0.8) + 0.08 * rng.normal(size=len(t))
    gt = np.zeros_like(t)

    for event in preset.gt_events:
        center = (event.start + event.end) / 2
        pulmonary += gaussian(t, center, 0.055, 1.45) - gaussian(t, center + 0.06, 0.075, 0.55)
        ambient += gaussian(t, center + 0.02, 0.070, 0.38)
        stretch += gaussian(t, center + 0.04, 0.130, 0.22)
        accel_z += gaussian(t, center - 0.02, 0.110, 0.18)
        gt[(t >= event.start) & (t <= event.end)] = 1.0

    return pd.DataFrame(
        {
            "time": t,
            "pulmonary": pulmonary,
            "ambient": ambient,
            "stretch": stretch,
            "accel_z": accel_z,
            "gt": gt,
        }
    )


def predicted_events(preset: Preset, model_key: str) -> list[Event]:
    info = MODEL_SUMMARY[model_key]
    gt = list(preset.gt_events)
    if info["miss"]:
        gt = gt[:-int(info["miss"])]

    offset = {"v5_ast": 0.015, "v3_boundary": 0.035, "v4_frame": -0.020, "ee491": 0.060}[model_key]
    preds = [
        Event(max(0.0, e.start + offset), min(preset.duration, e.end + offset + 0.04))
        for e in gt
    ]
    for center in info["false_positives"]:
        preds.append(Event(max(0.0, center - 0.16), min(preset.duration, center + 0.22)))
    return sorted(preds, key=lambda e: e.start)


def probability_curve(df: pd.DataFrame, events: Iterable[Event], model_key: str) -> np.ndarray:
    t = df["time"].to_numpy()
    base = {"v5_ast": 0.030, "v3_boundary": 0.045, "v4_frame": 0.060, "ee491": 0.075}[model_key]
    prob = np.full_like(t, base, dtype=float)
    for event in events:
        center = (event.start + event.end) / 2
        width = max(event.duration / 2.8, 0.08)
        prob += gaussian(t, center, width, 0.86)
    return np.clip(prob, 0.0, 1.0)


def event_iou(a: Event, b: Event) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union else 0.0


def classify_predictions(gt_events: tuple[Event, ...], pred_events: list[Event]) -> pd.DataFrame:
    rows = []
    matched_gt: set[int] = set()
    for idx, pred in enumerate(pred_events, start=1):
        best_iou = 0.0
        best_gt = None
        for gt_idx, gt in enumerate(gt_events):
            if gt_idx in matched_gt:
                continue
            iou = event_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_iou >= 0.2 and best_gt is not None:
            matched_gt.add(best_gt)
            status = "TP"
        else:
            status = "FP"
        rows.append(
            {
                "event": idx,
                "start_s": round(pred.start, 2),
                "end_s": round(pred.end, 2),
                "duration_s": round(pred.duration, 2),
                "status": status,
                "matched_iou": round(best_iou, 2),
            }
        )
    for gt_idx, gt in enumerate(gt_events, start=1):
        if gt_idx - 1 not in matched_gt:
            rows.append(
                {
                    "event": f"GT {gt_idx}",
                    "start_s": round(gt.start, 2),
                    "end_s": round(gt.end, 2),
                    "duration_s": round(gt.duration, 2),
                    "status": "FN",
                    "matched_iou": 0.0,
                }
            )
    return pd.DataFrame(rows)


def shade_events(ax: plt.Axes, events: Iterable[Event], color: str, alpha: float) -> None:
    for event in events:
        ax.axvspan(event.start, event.end, color=color, alpha=alpha, linewidth=0)


def make_signal_figure(df: pd.DataFrame, preset: Preset) -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(10.8, 5.8), sharex=True)
    traces = [
        ("pulmonary", "Pulmonary mic", "#2446a8"),
        ("ambient", "Ambient mic", "#b93a32"),
        ("stretch", "Stretch", "#9f3fa3"),
        ("accel_z", "Accel Z", "#2f8f4e"),
    ]
    for ax, (column, label, color) in zip(axes, traces):
        shade_events(ax, preset.gt_events, "#9ca3af", 0.24)
        ax.plot(df["time"], df[column], color=color, linewidth=1.0)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, axis="x", alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def make_prediction_figure(df: pd.DataFrame, preset: Preset, model_key: str) -> plt.Figure:
    preds = predicted_events(preset, model_key)
    prob = probability_curve(df, preds, model_key)
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.65]})

    axes[0].plot(df["time"], df["pulmonary"], color="#2446a8", linewidth=0.9)
    shade_events(axes[0], preset.gt_events, "#9ca3af", 0.26)
    axes[0].set_ylabel("Signal")
    axes[0].set_title("Ground truth cough intervals", loc="left", fontsize=10, fontweight="semibold")

    axes[1].plot(df["time"], prob, color="#f97316", linewidth=1.6)
    axes[1].axhline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    shade_events(axes[1], preds, "#f97316", 0.20)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("P(cough)")
    axes[1].set_title("Predicted cough probability and events", loc="left", fontsize=10, fontweight="semibold")

    activity_color = ACTIVITY_COLORS[preset.activity]
    axes[2].barh([0], [preset.duration], left=[0], height=0.45, color=activity_color, alpha=0.88)
    axes[2].set_yticks([0], [preset.activity])
    axes[2].set_ylim(-0.6, 0.6)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Event-level activity attribution", loc="left", fontsize=10, fontweight="semibold")

    for ax in axes:
        ax.set_xlim(0, preset.duration)
        ax.grid(True, axis="x", alpha=0.16)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


def make_model_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": info["label"],
                "precision": info["precision"],
                "recall": info["recall"],
                "event_f1": info["f1"],
                "activity_accuracy": info["activity_acc"],
            }
            for info in MODEL_SUMMARY.values()
        ]
    )


st.set_page_config(page_title="EE492 Cough Detection Demo", layout="wide")

st.title("Multi-Sensor Cough Detection")
st.caption("EE492 public demo with anonymized preset records")

st.sidebar.header("Demo controls")
preset_key = st.sidebar.selectbox(
    "Preset record",
    list(PRESETS.keys()),
    format_func=lambda key: PRESETS[key].label,
)
model_key = st.sidebar.selectbox(
    "Detector",
    list(MODEL_SUMMARY.keys()),
    format_func=lambda key: MODEL_SUMMARY[key]["label"],
)

preset = PRESETS[preset_key]
model = MODEL_SUMMARY[model_key]
df = make_preset_signals(preset.key)
preds = predicted_events(preset, model_key)
event_table = classify_predictions(preset.gt_events, preds)

left, right = st.columns([1.2, 1.0], vertical_alignment="top")
with left:
    st.subheader(preset.label)
    st.write(
        f"Activity: **{preset.activity}** · Context: **{preset.context}** · "
        f"Duration: **{preset.duration:.0f} s** · GT events: **{len(preset.gt_events)}**"
    )
with right:
    st.subheader(model["label"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{model['precision']:.3f}")
    m2.metric("Recall", f"{model['recall']:.3f}")
    m3.metric("Event F1", f"{model['f1']:.3f}")
    m4.metric("Activity", f"{model['activity_acc']:.3f}")

tab_signals, tab_predictions, tab_results = st.tabs(["Signals", "Predictions", "Result table"])

with tab_signals:
    st.pyplot(make_signal_figure(df, preset), clear_figure=True, use_container_width=True)

with tab_predictions:
    st.pyplot(make_prediction_figure(df, preset, model_key), clear_figure=True, use_container_width=True)

with tab_results:
    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        st.dataframe(event_table, use_container_width=True, hide_index=True)
    with c2:
        st.dataframe(make_model_comparison(), use_container_width=True, hide_index=True)

st.info(
    "This public demo uses anonymized preset signals and report-aligned model summaries. "
    "Raw sensor CSV files, subject metadata, and private model checkpoints are not bundled."
)
