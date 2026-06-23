"""Streamlit demo for the EE492 cough-detection project.

You can launch this app any of these ways and they all "just work":

  * IDE "Run File" button on this script
  * ``python app/streamlit_app.py``   (system Python is fine)
  * ``./run_demo.sh``                  (convenience wrapper)
  * ``PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py``  (canonical)

The first three are handled by the self-relaunch block below: if streamlit
isn't importable in the active interpreter, or if the script was invoked as a
plain Python file instead of through the ``streamlit run`` launcher, we re-exec
ourselves using ``.venv/bin/python -m streamlit run ...``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------- #
# Self-relaunch shim. Must run before any third-party import so the system
# Python case (no streamlit installed) doesn't crash on ``import streamlit``.
# ---------------------------------------------------------------------------- #
import os as _os
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent


def _running_under_streamlit_runtime() -> bool:
    """True only when invoked via ``streamlit run`` (script context exists)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _relaunch_under_streamlit() -> "None":
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit(
            "Could not find .venv/bin/python next to this script.\n"
            "Either create the project venv or launch the demo manually:\n"
            "    PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py"
        )
    env = _os.environ.copy()
    src_path = str(_REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + _os.pathsep + env.get("PYTHONPATH", "")
    # Pass through any extra args (e.g. ``--server.port 8770``) so the user can
    # still override defaults when invoking the script directly.
    extra_args = list(_sys.argv[1:])
    args = [
        str(venv_python), "-m", "streamlit", "run",
        str(_Path(__file__).resolve()),
        "--server.fileWatcherType", "none",   # watchdog throttles inference
        "--browser.gatherUsageStats", "false",
        *extra_args,
    ]
    _os.execvpe(args[0], args, env)


if __name__ == "__main__" and not _running_under_streamlit_runtime():
    _relaunch_under_streamlit()

# ---------------------------------------------------------------------------- #
# Normal app code below. We are now guaranteed to be inside ``streamlit run``.
# ---------------------------------------------------------------------------- #

import io
import sys
import tempfile
import time
from pathlib import Path

# ``streamlit run app/streamlit_app.py`` sets sys.path[0] to ``app/``, so
# ``from app import inference`` would fail without the repo root on the path.
for _extra in (_REPO_ROOT, _REPO_ROOT / "src"):
    _path_str = str(_extra)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

# IMPORTANT: keep xgboost imported first via app.inference (see comment there).
from app import inference  # noqa: F401, E402  re-exported for side-effect import order
from app.inference import (
    MODEL_REGISTRY,
    preprocess_raw_csv,
    run_inference,
)
from app.plotting import (
    make_model_input_figure,
    make_predictions_figure,
    make_preprocessed_figure,
    make_raw_figure,
)
from app.presets import PRESETS, PRESETS_BY_KEY

from cough_analysis.event_metrics import binary_labels_to_events

import streamlit as st


st.set_page_config(
    page_title="Cough Detection — EE492 Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Cached wrappers ------------------------------------------------ #


@st.cache_resource(show_spinner="Loading model checkpoint…")
def cached_load_model(model_id: str):
    return inference.load_model_bundle(model_id)


@st.cache_resource(show_spinner="Loading V4 activity head…")
def cached_load_activity():
    return inference.load_activity_bundle()


@st.cache_data(show_spinner="Preprocessing CSV…")
def cached_preprocess(path_str: str, mtime_ns: int):
    # mtime_ns is part of the cache key so re-uploads invalidate correctly.
    return preprocess_raw_csv(path_str)


# ---------- Sidebar ------------------------------------------------------- #


st.sidebar.title("Cough Detection Demo")
st.sidebar.caption("EE492 final project · live inference")

model_choice = st.sidebar.selectbox(
    "Model",
    options=list(MODEL_REGISTRY.keys()),
    format_func=lambda key: MODEL_REGISTRY[key]["display_name"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Input record**")

source_options = ("Preset record", "Upload own CSV") if PRESETS else ("Upload own CSV",)
source = st.sidebar.radio(
    "Source",
    options=source_options,
    horizontal=False,
)

selected_path: Path | None = None
upload_buffer: io.BytesIO | None = None
upload_name: str | None = None

if source == "Preset record":
    preset_key = st.sidebar.selectbox(
        "Preset",
        options=[p.key for p in PRESETS],
        format_func=lambda k: PRESETS_BY_KEY[k].label,
    )
    preset = PRESETS_BY_KEY[preset_key]
    selected_path = preset.absolute_path
    if not selected_path.exists():
        st.sidebar.error(f"Preset file missing: {selected_path}")
        selected_path = None
else:
    if not PRESETS:
        st.sidebar.caption("Preset records require the private dataset under `data/clean_v4/`.")
    uploaded = st.sidebar.file_uploader(
        "Raw CSV (4 columns, no header, 20 s @ 4800 Hz)",
        type=["csv"],
    )
    if uploaded is not None:
        upload_buffer = uploaded
        upload_name = uploaded.name

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("Run inference", type="primary", use_container_width=True)

with st.sidebar.expander("Model details"):
    entry = MODEL_REGISTRY[model_choice]
    st.write(f"**Checkpoint**: `{Path(entry['checkpoint']).name}`")
    st.write(f"**Config**: `{Path(entry['config']).name}`")
    st.write(f"**Kind**: `{entry['kind']}`")


# ---------- Main panel ---------------------------------------------------- #


if not run_clicked:
    st.info("Choose a model and record on the left, then press *Run inference*.")
    st.stop()

# Resolve path: either preset (real path on disk) or upload (write to temp file).
if upload_buffer is not None and upload_name is not None:
    tmpdir = Path(tempfile.gettempdir()) / "cough_demo_uploads"
    tmpdir.mkdir(parents=True, exist_ok=True)
    target = tmpdir / upload_name
    target.write_bytes(upload_buffer.getvalue())
    csv_path = target
elif selected_path is not None and selected_path.exists():
    csv_path = selected_path
else:
    st.error("No input record selected.")
    st.stop()

mtime_ns = csv_path.stat().st_mtime_ns

with st.spinner("Preprocessing record…"):
    record = cached_preprocess(str(csv_path), mtime_ns)

# Compute GT events once now so we can shade them in the preprocessed plot
# *before* inference runs. This is identical to what the model evaluators do.
gt_events_preview = binary_labels_to_events(
    record.cough_label,
    sample_rate=record.fs_audio,
    min_duration_sec=0.1,
    merge_gap_sec=0.1,
)

# Compact CSS — Streamlit defaults add lots of vertical padding around every
# block. Tightening it here means the four pipeline stages fit on a single
# 1080p screen with minimal scrolling during the live demo.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 1rem; }
      .stMarkdown h5    { margin: 0.35rem 0 0.10rem 0; }
      .stPlotlyChart, .element-container { margin-bottom: 0.20rem !important; }
      div[data-testid="stVerticalBlock"] { gap: 0.30rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _stage_header(text: str) -> None:
    # ``##### …`` is the smallest standard heading. With the CSS above it
    # occupies only ~24 px instead of subheader's ~52 px.
    st.markdown(f"##### {text}")


# --- Stage 1: raw signals ------------------------------------------------- #
_stage_header("1 · Raw signals")
st.pyplot(make_raw_figure(record), clear_figure=True, use_container_width=True)

# --- Stage 2: preprocessed signals --------------------------------------- #
_stage_header("2 · Preprocessed signals")
st.pyplot(
    make_preprocessed_figure(record, gt_events_preview),
    clear_figure=True, use_container_width=True,
)

# --- Stage 3: what the model sees --------------------------------------- #
_stage_header(f"3 · What {MODEL_REGISTRY[model_choice]['display_name']} sees")
st.pyplot(
    make_model_input_figure(record, gt_events_preview, model_choice),
    clear_figure=True, use_container_width=True,
)

# Warm the caches so the first run from a fresh worker still feels snappy.
_ = cached_load_model(model_choice)
_ = cached_load_activity()

# --- Stage 4: model output ----------------------------------------------- #
_stage_header("4 · Model output")

display_name = MODEL_REGISTRY[model_choice]["display_name"]
# V5 is single-threaded (OMP=1) on CPU so a 20 s record runs in ~75 s.
# Make the wait expectation explicit instead of letting the user think it hung.
spinner_msg = f"Running {display_name}…"
if MODEL_REGISTRY[model_choice]["kind"] == "v5_ast":
    spinner_msg += "  (CPU inference, expect ~75 s)"
with st.spinner(spinner_msg):
    t0 = time.perf_counter()
    result = run_inference(model_choice, record)
    wall = time.perf_counter() - t0

# One-line summary instead of four metric cards — same info, ~80 px less.
st.caption(
    f"GT cough events: **{len(result['gt_cough_events'])}** · "
    f"predicted: **{len(result['cough']['events'])}** · "
    f"cough inference: **{result['timings']['cough_sec']:.2f} s** · "
    f"activity inference: **{result['timings']['activity_sec']:.2f} s**"
)

st.pyplot(
    make_predictions_figure(record, result),
    clear_figure=True, use_container_width=True,
)

# --- Optional: per-event detail table ------------------------------------ #

with st.expander("Per-event details"):
    pred_events = result["cough"]["events"]
    gt_events = result["gt_cough_events"]
    st.write(f"**Predicted ({len(pred_events)})**")
    if pred_events:
        st.dataframe(
            {
                "start (s)": [round(e.start, 3) for e in pred_events],
                "end (s)":   [round(e.end, 3)   for e in pred_events],
                "duration (s)": [round(e.duration, 3) for e in pred_events],
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("_no predicted events_")
    st.write(f"**Ground truth ({len(gt_events)})**")
    if gt_events:
        st.dataframe(
            {
                "start (s)": [round(e.start, 3) for e in gt_events],
                "end (s)":   [round(e.end, 3)   for e in gt_events],
                "duration (s)": [round(e.duration, 3) for e in gt_events],
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("_no GT events_")

with st.expander("Post-processing settings used"):
    st.json(result["cough"]["post"])
