import numpy as np
import torch

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.models import V4ActivityCNN, V4CoughFrameCNN
from cough_analysis.v4 import (
    V4ActivityWindowDataset,
    V4CoughChunkDataset,
    activity_target_label,
    cough_frame_count,
    frame_labels_from_samples,
    remove_short_events,
    resize_frame_logits,
    split_records_v4,
)
from tests.conftest import require_private_dataset


def test_v4_split_has_no_record_overlap():
    require_private_dataset()
    cfg = load_config("configs/v4.yaml")
    metadata = load_metadata()

    split = split_records_v4(metadata, cfg["split"])

    train = set(split.train)
    val = set(split.val)
    test = set(split.test)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == len(metadata)


def test_remove_short_events_drops_tiny_runs():
    labels = np.zeros(1000, dtype=np.int64)
    labels[10:20] = 1
    labels[200:350] = 1

    filtered = remove_short_events(labels, sample_rate=1000, min_duration_sec=0.1)

    assert filtered[10:20].sum() == 0
    assert filtered[200:350].sum() == 150


def test_frame_labels_use_fixed_bins():
    labels = np.zeros(100, dtype=np.int64)
    labels[12:14] = 1
    labels[40:50] = 1

    frames = frame_labels_from_samples(labels, frame_hop_samples=10, frame_count=10)

    assert np.array_equal(frames, np.asarray([0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32))


def test_v4_single_record_dataset_shapes():
    require_private_dataset()
    cfg = load_config("configs/v4.yaml")
    metadata = load_metadata()
    spec_cfg = cfg["cough"]["specs"]["spec128"]

    cough_ds = V4CoughChunkDataset([0], metadata, cfg["cough"], spec_cfg)
    activity_ds = V4ActivityWindowDataset([0], metadata, cfg["activity"])

    assert len(cough_ds) == 16
    assert cough_ds[0]["spec"].shape[0] == 2
    assert cough_ds[0]["spec"].shape[1] == spec_cfg["n_mels"]
    assert cough_ds[0]["motion"].shape == (2, 500)
    assert cough_ds[0]["label"].shape == (cough_frame_count(cfg["cough"]),)
    assert len(activity_ds) == 35
    assert activity_ds[0]["motion"].shape == (2, 300)


def test_v4_model_forward_shapes():
    cough_model = V4CoughFrameCNN()
    logits = cough_model(
        torch.randn(2, 2, 64, 501),
        torch.randn(2, 2, 500),
    )
    resized = resize_frame_logits(logits, frame_count=500)

    activity_model = V4ActivityCNN(num_classes=4)
    activity_logits = activity_model(torch.randn(2, 2, 300))

    assert resized.shape == (2, 500)
    assert activity_logits.shape == (2, 4)


def test_activity_label_map_supports_stationary_class():
    cfg = load_config("configs/final/v4_clean.yaml")

    assert activity_target_label("sitting", cfg["activity"]) == "stationary"
    assert activity_target_label("standing", cfg["activity"]) == "stationary"
    assert activity_target_label("walking", cfg["activity"]) == "walking"

    require_private_dataset("data/clean_v4/metadata.csv")
    metadata = load_metadata("data/clean_v4/metadata.csv")
    activity_ds = V4ActivityWindowDataset([0, 3], metadata, cfg["activity"])
    labels = set(activity_ds.labels.numpy().tolist())

    assert labels == {0, 1}
