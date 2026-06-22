"""Select raw CSVs, preview preprocessed signals, label, copy to curated_csv/, append metadata."""

import glob
import os
import shutil
import subprocess
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, simpledialog

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

# --- constants ---
METADATA_COLUMNS = [
    "record_id", "filename", "date", "subject", "activity",
    "context", "clothing", "relative_path",
]
ACTIVITY_OPTIONS = ["sitting", "walking", "running", "standing"]
CONTEXT_OPTIONS = [
    "clean", "coughnoise", "musicnoise", "sneezenoise", "snoozenoise",
    "doornoise", "falsepositive", "noise",
]
CLOTHING_OPTIONS = ["overclothes", "underclothes"]
FS_AUDIO, FS_MOTION = 4800, 100
DEFAULT_SUBJECT = "subject01"


def to_iso_date(s):
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def sync_metadata(curated_dir, metadata_path):
    if not os.path.isdir(curated_dir):
        return
    existing = set()
    if os.path.isfile(metadata_path):
        try:
            df = pd.read_csv(metadata_path)
            if "filename" in df.columns:
                existing = set(df["filename"])
        except Exception as e:
            print(f"Warning: metadata read: {e}")
    rows = []
    for path in glob.glob(os.path.join(curated_dir, "*.csv")):
        fn = os.path.basename(path)
        if fn in existing:
            continue
        p = fn.replace(".csv", "").split("_")
        if len(p) < 5:
            continue
        try:
            rows.append({
                "record_id": int(p[0]),
                "filename": fn,
                "date": to_iso_date(p[1]),
                "subject": p[2],
                "activity": p[3],
                "context": "_".join(p[4:]),
                "clothing": "underclothes" if "underclothes" in fn else "overclothes",
                "relative_path": f"curated_csv/{fn}",
            })
        except Exception as e:
            print(f"Sync skip {fn}: {e}")
    if not rows:
        print("SYNC: metadata up to date.")
        return
    out = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    out.to_csv(metadata_path, mode="a" if os.path.isfile(metadata_path) else "w",
               header=not os.path.isfile(metadata_path), index=False)
    print(f"SYNC: added {len(rows)} rows.")


def next_record_id(curated_dir):
    os.makedirs(curated_dir, exist_ok=True)
    m = -1
    for path in glob.glob(os.path.join(curated_dir, "*.csv")):
        p = os.path.basename(path).split("_")
        if p and p[0].isdigit():
            m = max(m, int(p[0]))
    return m + 1


def append_metadata(path, row):
    df = pd.DataFrame([row], columns=METADATA_COLUMNS)
    df.to_csv(path, mode="a" if os.path.isfile(path) else "w",
              header=not os.path.isfile(path), index=False)


def read_sensor_csv(path):
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 4:
        raise ValueError("Need at least 4 columns.")
    df = df.iloc[:, :4].copy()
    df.columns = ["pulmonary_mic", "ambient_mic", "stretch_raw", "accel_z"]
    df["stretch_raw"] = pd.to_numeric(df["stretch_raw"], errors="coerce")
    v = df["stretch_raw"].fillna(0).astype(np.int64).to_numpy()
    df["stretch_signal"] = np.right_shift(v, 1)
    df["cough_label"] = np.bitwise_and(v, 1)
    return df


def _norm(sig):
    c = sig - np.median(sig)
    m = np.max(np.abs(c))
    return c / m if m else c


def _for_plot_xy(t, y, max_points=120_000):
    """
    Preview plotting: keep full resolution up to max_points.
    Beyond that, min/max per bucket (preserves peaks; naive stride looks bad on audio).
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y)
    n = y.shape[0]
    if n <= max_points:
        return t, y
    n_buckets = max(max_points // 2, 1)
    w = int(np.ceil(n / n_buckets))
    ts, ys = [], []
    for i in range(0, n, w):
        j = min(i + w, n)
        sl = slice(i, j)
        tb, yb = t[sl], y[sl]
        k_lo = int(np.argmin(yb))
        k_hi = int(np.argmax(yb))
        i_lo, i_hi = i + k_lo, i + k_hi
        if i_lo == i_hi:
            ts.append(t[i_lo])
            ys.append(y[i_lo])
        elif i_lo < i_hi:
            ts.extend((t[i_lo], t[i_hi]))
            ys.extend((y[i_lo], y[i_hi]))
        else:
            ts.extend((t[i_hi], t[i_lo]))
            ys.extend((y[i_hi], y[i_lo]))
    return np.asarray(ts), np.asarray(ys)


def _butter_bp(lo, hi, fs, order=4):
    ny = 0.5 * fs
    return signal.butter(order, [lo / ny, hi / ny], btype="band")


def _butter_lp(cut, fs, order=4):
    ny = 0.5 * fs
    return signal.butter(order, cut / ny, btype="low", analog=False)


def preprocess(prep_dict, df):
    pulm = df["pulmonary_mic"].to_numpy(dtype=np.float64)
    amb = df["ambient_mic"].to_numpy(dtype=np.float64)
    stretch = df["stretch_signal"].to_numpy(dtype=np.float64)
    accz = df["accel_z"].to_numpy(dtype=np.float64)
    label = df["cough_label"].to_numpy(dtype=np.int64)

    pc, ac = pulm - np.median(pulm), amb - np.median(amb)
    b, a = _butter_bp(60, 2200, FS_AUDIO)
    pulm_bp = signal.filtfilt(b, a, pc)
    amb_bp = signal.filtfilt(b, a, ac)

    sc = stretch - np.median(stretch)
    n_m = int(len(sc) * (FS_MOTION / FS_AUDIO))
    sr = signal.resample(sc, n_m)
    ar = signal.resample(accz, n_m)
    b2, a2 = _butter_lp(20, FS_MOTION)
    stretch_lp = signal.filtfilt(b2, a2, sr)
    accz_lp = signal.filtfilt(b2, a2, ar)

    dur = len(pulm) / FS_AUDIO
    t_a = np.linspace(0, dur, len(pulm))
    t_m = np.linspace(0, dur, len(stretch_lp))
    prep_dict.update(
        t_audio=t_a, t_motion=t_m,
        pulm_raw=pulm, amb_raw=amb, stretch_raw=stretch, accz_raw=accz,
        pulm_bp=pulm_bp, amb_bp=amb_bp, stretch_lp=stretch_lp, accz_lp=accz_lp,
        cough_label=label,
    )


def show_plot(_root, df, title):
    """Native matplotlib Figure window (pyplot), same as pre-embedded preview."""
    p = {}
    preprocess(p, df)
    rc = {
        "path.simplify": False,
        "path.simplify_threshold": 0,
        "lines.antialiased": True,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(5, 1, figsize=(14, 10.5), dpi=130, sharex=True)
        ta, pr = _for_plot_xy(p["t_audio"], p["pulm_raw"])
        _, pb = _for_plot_xy(p["t_audio"], p["pulm_bp"])
        axes[0].plot(ta, _norm(pr), color="#c8c8c8", lw=0.85, zorder=1)
        axes[0].plot(ta, _norm(pb), color="blue", lw=1.05, zorder=2)
        axes[0].set_title("Pulmonary (Overlay)")

        _, ar = _for_plot_xy(p["t_audio"], p["amb_raw"])
        _, ab = _for_plot_xy(p["t_audio"], p["amb_bp"])
        axes[1].plot(ta, _norm(ar), color="#c8c8c8", lw=0.85, zorder=1)
        axes[1].plot(ta, _norm(ab), color="red", lw=1.05, zorder=2)
        axes[1].set_title("Ambient (Overlay)")

        _, sr = _for_plot_xy(p["t_audio"], p["stretch_raw"])
        tm, sl = _for_plot_xy(p["t_motion"], p["stretch_lp"])
        axes[2].plot(ta, _norm(sr), color="#c8c8c8", lw=0.85, zorder=1)
        axes[2].plot(tm, _norm(sl), color="magenta", lw=1.35, zorder=2)
        axes[2].set_title("Stretch (Overlay)")

        _, azr = _for_plot_xy(p["t_audio"], p["accz_raw"])
        _, azp = _for_plot_xy(p["t_motion"], p["accz_lp"])
        axes[3].plot(ta, _norm(azr), color="#c8c8c8", lw=0.85, zorder=1)
        axes[3].plot(tm, _norm(azp), color="lime", lw=1.2, zorder=2)
        axes[3].set_title("Acc Z (Overlay)")

        tlb, lb = _for_plot_xy(p["t_audio"], p["cough_label"].astype(float))
        axes[4].fill_between(tlb, 0, lb, color="silver", step="pre", edgecolor="gray", linewidth=0.5)
        axes[4].set_title("Ground Truth Label")
        axes[4].set_xlabel("Time (s)")
        axes[4].set_ylim(0, 1.1)
        for ax in axes[:4]:
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([-1, 0, 1])
            ax.grid(True, linestyle="--", alpha=0.45)
        axes[4].set_yticks([0, 0.5, 1])
        axes[4].grid(True, linestyle="--", alpha=0.45)
        fig.suptitle(f"Preprocessed Sensor Overlay: {title}", fontsize=12)
        fig.tight_layout()
        plt.show(block=True)
        plt.close(fig)


def _ask_choice(root, title, prompt, options, default):
    """Text prompt; must match one of options (like before dropdown UI)."""
    opts = ", ".join(options)
    while True:
        v = simpledialog.askstring(
            title,
            f"{prompt}\nOptions: {opts}\nDefault: {default}",
            parent=root,
        )
        if v is None:
            return None
        v = v.strip().lower()
        if not v:
            return default
        if v in options:
            return v
        print(f"Invalid '{v}'. Allowed: {opts}")


def ask_metadata_labels(root, today_str, last_subj):
    """Sequential simpledialog prompts (pre-dropdown behavior)."""
    root.update_idletasks()

    d = simpledialog.askstring(
        "Date",
        f"Date YYYYMMDD [{today_str}]:",
        parent=root,
        initialvalue=today_str,
    )
    if d is None:
        return None
    d = d.strip() or today_str
    if len(d) != 8 or not d.isdigit():
        print("Invalid date; use YYYYMMDD.")
        return None

    subj_def = last_subj or DEFAULT_SUBJECT
    subj = simpledialog.askstring(
        "Subject",
        f"Subject [{subj_def}]:",
        parent=root,
        initialvalue=subj_def,
    )
    if subj is None:
        return None
    subj = (subj.strip() or subj_def).lower()

    act = _ask_choice(root, "Activity", "Activity", ACTIVITY_OPTIONS, "sitting")
    if act is None:
        return None
    ctx = _ask_choice(root, "Context", "Context", CONTEXT_OPTIONS, "clean")
    if ctx is None:
        return None
    clo = _ask_choice(root, "Clothing", "Clothing", CLOTHING_OPTIONS, "underclothes")
    if clo is None:
        return None

    return {
        "date_str": d,
        "date_iso": to_iso_date(d),
        "subject": subj,
        "activity": act,
        "context": ctx,
        "clothing": clo,
    }


def _pick_csv_windows_native(initialdir, title="Select raw CSV"):
    """Fallback: Explorer-style dialog via PowerShell WinForms (STA)."""
    initialdir = os.path.normpath(os.path.abspath(initialdir))
    if not os.path.isdir(initialdir):
        initialdir = os.path.expanduser("~")
    idir = initialdir.replace("'", "''")
    ttl = title.replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.OpenFileDialog; "
        "$d.Filter = 'CSV (*.csv)|*.csv|All (*.*)|*.*'; "
        f"$d.Title = '{ttl}'; "
        f"$d.InitialDirectory = '{idir}'; "
        "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $d.FileName }"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Sta", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=600,
        )
        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        if not lines:
            return None
        path = lines[-1]
        return path if os.path.isfile(path) else None
    except Exception:
        return None


def pick_csv_path(root, initialdir):
    """
    Windows: native OpenFileDialog first (Tk filedialog often fails with hidden root).
    Other OS: Tk with briefly visible root window.
    """
    initialdir = os.path.normpath(os.path.abspath(initialdir))
    if not os.path.isdir(initialdir):
        initialdir = os.path.expanduser("~")

    if os.name == "nt":
        path = _pick_csv_windows_native(initialdir)
        if path:
            return path

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = 280, 48
    root.deiconify()
    root.title("CSV label tool")
    root.geometry(f"{w}x{h}+{max(0, sw // 2 - w // 2)}+{max(0, sh // 2 - h // 2)}")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.update_idletasks()
    root.update()

    path = filedialog.askopenfilename(
        parent=root,
        title="Select raw CSV",
        filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        initialdir=initialdir,
    )
    root.withdraw()
    root.update()
    return path or None


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    curated = os.path.join(root_dir, "curated_csv")
    meta = os.path.join(root_dir, "metadata.csv")
    os.makedirs(curated, exist_ok=True)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    today = datetime.now().strftime("%Y%m%d")
    last_subj = DEFAULT_SUBJECT
    start_dir = root_dir if os.path.isdir(root_dir) else os.path.expanduser("~")

    print("--- Label tool ---")
    print(f"curated: {curated}\nmetadata: {meta}\n")
    sync_metadata(curated, meta)

    while True:
        path = pick_csv_path(root, start_dir)
        if not path:
            print("Exit.")
            break

        name = os.path.basename(path)
        print(f"File: {name}")
        try:
            df = read_sensor_csv(path)
            show_plot(root, df, name)
        except Exception as e:
            print(f"Load/plot error: {e}")
            continue

        labels = ask_metadata_labels(root, today, last_subj)
        if not labels:
            print("Cancelled.")
            continue
        last_subj = labels["subject"]

        rid = next_record_id(curated)
        new_name = f"{rid:03d}_{labels['date_str']}_{labels['subject']}_{labels['activity']}_{labels['context']}.csv"
        dst = os.path.join(curated, new_name)
        try:
            shutil.copy2(path, dst)
            append_metadata(meta, {
                "record_id": rid,
                "filename": new_name,
                "date": labels["date_iso"],
                "subject": labels["subject"],
                "activity": labels["activity"],
                "context": labels["context"],
                "clothing": labels["clothing"],
                "relative_path": f"curated_csv/{new_name}",
            })
            print(f"Saved: {new_name}")
        except Exception as e:
            print(f"Save error: {e}")

    root.destroy()


if __name__ == "__main__":
    main()
