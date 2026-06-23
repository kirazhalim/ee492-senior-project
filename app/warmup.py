"""Warm-up script for the live demo.

Run from the project root once before presenting so:

  * the AST backbone is downloaded from HuggingFace (one-time, ~400 MB)
  * every model checkpoint is loaded into the in-process lru_cache
  * the OS page cache is warm for the .pt files

What it deliberately does NOT do: pre-compute or cache any inference results.
Each "Run inference" press during the demo must actually run the model end to
end. The warm-up just removes the cold-start latency from library imports and
weight loading, not from the math.

Invocation:

    PYTHONPATH=src .venv/bin/python -m app.warmup
"""

from __future__ import annotations

import time

from app.inference import MODEL_REGISTRY, get_device, preprocess_raw_csv, run_inference
from app.presets import PRESETS


def main() -> int:
    if not PRESETS:
        raise RuntimeError("No presets registered in app.presets")

    preset = PRESETS[0]
    csv_path = preset.absolute_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Preset CSV missing: {csv_path}")

    print(f"[warmup] device={get_device()}")
    print(f"[warmup] preset: {preset.label} ({csv_path.name})")

    t0 = time.perf_counter()
    record = preprocess_raw_csv(csv_path)
    print(
        f"[warmup] preprocess: {time.perf_counter() - t0:.2f}s  "
        f"duration={record.duration_sec:.2f}s  "
        f"activity_gt={record.activity_gt}"
    )

    for model_id in MODEL_REGISTRY:
        label = MODEL_REGISTRY[model_id]["display_name"]
        t = time.perf_counter()
        try:
            result = run_inference(model_id, record)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"[warmup] FAIL  {model_id:9s}  {type(exc).__name__}: {exc}")
            continue
        total = time.perf_counter() - t
        ts = result["timings"]
        n_pred = len(result["cough"]["events"])
        n_gt = len(result["gt_cough_events"])
        print(
            f"[warmup] OK    {model_id:9s}  pred={n_pred} gt={n_gt}  "
            f"total={total:.2f}s (cough={ts['cough_sec']:.2f}, "
            f"activity={ts['activity_sec']:.2f})  · {label}"
        )

    print("[warmup] done — model caches are hot, inference results are NOT cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
