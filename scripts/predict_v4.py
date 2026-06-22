from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.models import V4ActivityCNN, V4CoughFrameCNN
from cough_analysis.paths import project_path
from cough_analysis.v4 import (
    assign_activity_to_event,
    frame_predictions_to_events,
    predict_activity_probabilities_for_record,
    predict_cough_probabilities_for_record,
    resolve_device,
)
from cough_analysis.preprocessing import load_record_preprocessed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict V4 cough events with activity labels.")
    parser.add_argument("--model-dir", default="artifacts/models/v4")
    parser.add_argument("--config", default="configs/v4.yaml")
    parser.add_argument("--record-id", type=int, required=True)
    parser.add_argument("--output-dir", default="artifacts/predictions/v4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_path(path)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    model_dir = project_or_absolute(args.model_dir)
    summary = json.loads((model_dir / "v4_summary.json").read_text(encoding="utf-8"))
    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    data_root = project_or_absolute(cfg["data"]["data_root"])
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])

    spec_name = summary["primary_cough_spec"]
    cough_ckpt = torch.load(model_dir / f"v4_cough_{spec_name}.pt", map_location=device)
    cough_model = V4CoughFrameCNN().to(device)
    cough_model.load_state_dict(cough_ckpt["model_state_dict"])
    cough_model.eval()

    activity_ckpt = torch.load(model_dir / "v4_activity.pt", map_location=device)
    activity_model = V4ActivityCNN(num_classes=len(cfg["activity"]["classes"])).to(device)
    activity_model.load_state_dict(activity_ckpt["model_state_dict"])
    activity_model.eval()

    record = load_record_preprocessed(
        args.record_id,
        metadata=metadata,
        data_root=data_root,
    )
    probs = predict_cough_probabilities_for_record(
        cough_model,
        record,
        cfg["cough"],
        cough_ckpt["spec_config"],
        device=device,
        batch_size=batch_size,
    )
    frame_rate = int(round(int(cfg["sampling"]["audio_hz"]) / int(cfg["cough"]["frame_hop_samples"])))
    post_cfg = cough_ckpt["selected_postprocessing"]
    cough_events = frame_predictions_to_events(
        probs,
        frame_rate=frame_rate,
        threshold=float(post_cfg["threshold"]),
        min_duration_sec=float(post_cfg["pred_min_duration_sec"]),
        merge_gap_sec=float(post_cfg["pred_merge_gap_sec"]),
        duration_sec=float(record["duration_sec"]),
    )

    centers, activity_probs = predict_activity_probabilities_for_record(
        activity_model,
        record,
        cfg["activity"],
        device=device,
        batch_size=batch_size,
    )
    events = []
    for event in cough_events:
        assigned = assign_activity_to_event(
            event,
            centers,
            activity_probs,
            cfg["activity"]["classes"],
            context_sec=float(cfg["activity"]["attribution_context_sec"]),
        )
        start_frame = max(0, int(round(event.start * frame_rate)))
        end_frame = min(len(probs), int(round(event.end * frame_rate)))
        confidence = float(probs[start_frame:end_frame].max()) if end_frame > start_frame else 0.0
        events.append(
            {
                "cough_start": float(event.start),
                "cough_end": float(event.end),
                "cough_confidence": confidence,
                **assigned,
            }
        )

    output = {
        "model_id": "v4",
        "record_id": int(args.record_id),
        "filename": record["filename"],
        "primary_cough_spec": spec_name,
        "postprocessing": post_cfg,
        "events": events,
    }
    output_dir = project_or_absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"record_{args.record_id:03d}_v4_predictions.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Predicted events: {len(events)}")
    print(f"Saved predictions: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
