import torch

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.models import RawWaveformCoughCNN
from cough_analysis.raw_baselines import RawWaveformDataset, build_raw_dataset
from tests.conftest import require_private_dataset


def test_raw_waveform_dataset_shapes_v1():
    require_private_dataset("data/clean_v4/metadata.csv")
    cfg = load_config("configs/final/v1_clean_raw_waveform.yaml")
    metadata = load_metadata("data/clean_v4/metadata.csv")
    audio, motion, labels = build_raw_dataset(
        [0],
        metadata,
        data_root=cfg["data"]["data_root"],
        window_sec=cfg["windowing"]["window_sec"],
        hop_sec=cfg["windowing"]["hop_sec"],
        label_rule=cfg["windowing"]["label_rule"],
        center_fraction=cfg["windowing"]["center_fraction"],
    )

    assert audio.shape[1:] == (2, 4800)
    assert motion.shape[1:] == (2, 100)
    assert labels.shape[0] == audio.shape[0] == motion.shape[0]


def test_raw_waveform_model_forward_shape():
    require_private_dataset("data/clean_v4/metadata.csv")
    cfg = load_config("configs/final/v2_clean_raw_waveform.yaml")
    metadata = load_metadata("data/clean_v4/metadata.csv")
    audio, motion, labels = build_raw_dataset(
        [0],
        metadata,
        data_root=cfg["data"]["data_root"],
        window_sec=cfg["windowing"]["window_sec"],
        hop_sec=cfg["windowing"]["hop_sec"],
        label_rule=cfg["windowing"]["label_rule"],
        center_fraction=cfg["windowing"]["center_fraction"],
    )
    dataset = RawWaveformDataset(audio[:2], motion[:2], labels[:2])
    batch = [dataset[idx] for idx in range(2)]
    model = RawWaveformCoughCNN()

    out = model(
        torch.stack([item["audio"] for item in batch]),
        torch.stack([item["motion"] for item in batch]),
    )

    assert tuple(out.shape) == (2,)
