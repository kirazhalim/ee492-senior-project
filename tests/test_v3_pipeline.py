import torch

from cough_analysis.data import load_metadata
from cough_analysis.models import Spec2DCoughCNN
from cough_analysis.config import load_config
from cough_analysis.v3 import build_dataset, split_records_from_config
from cough_analysis.v4 import split_records_v4
from tests.conftest import require_private_dataset


def test_v3_single_record_dataset_shapes():
    require_private_dataset()
    metadata = load_metadata()
    X_spec, X_motion, y = build_dataset([0], metadata)

    assert X_spec.shape == (77, 2, 64, 38)
    assert X_motion.shape == (77, 2, 100)
    assert y.shape == (77,)


def test_v3_model_forward_shape():
    require_private_dataset()
    metadata = load_metadata()
    X_spec, X_motion, _ = build_dataset([0], metadata)
    model = Spec2DCoughCNN(num_classes=1)

    out = model(
        torch.tensor(X_spec[:2]),
        torch.tensor(X_motion[:2]),
    )

    assert tuple(out.shape) == (2,)


def test_v3_all_records_split_matches_v4():
    require_private_dataset()
    metadata = load_metadata()
    cfg_v3 = load_config("configs/v3_all_records.yaml")
    cfg_v4 = load_config("configs/v4.yaml")

    train_ids, val_ids, test_ids = split_records_from_config(metadata, cfg_v3["split"])
    split_v4 = split_records_v4(metadata, cfg_v4["split"])

    assert train_ids.tolist() == split_v4.train
    assert val_ids.tolist() == split_v4.val
    assert test_ids.tolist() == split_v4.test
