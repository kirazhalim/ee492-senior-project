from cough_analysis.classical_ml import FEATURE_COLUMNS, build_classical_feature_table
from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from tests.conftest import require_private_dataset


def test_classical_feature_table_clean_v4_shapes():
    require_private_dataset("data/clean_v4/metadata.csv")
    cfg = load_config("configs/final/ee491_classical_clean.yaml")
    metadata = load_metadata("data/clean_v4/metadata.csv")
    table = build_classical_feature_table(
        [0],
        metadata,
        data_root=cfg["data"]["data_root"],
        window_sec=cfg["windowing"]["window_sec"],
        hop_sec=cfg["windowing"]["hop_sec"],
        label_overlap_tau=cfg["windowing"]["label_overlap_tau"],
    )

    assert len(table) > 0
    assert set(FEATURE_COLUMNS).issubset(table.columns)
    assert {"record_id", "t0", "t1", "y_cough", "best_overlap_ratio"}.issubset(table.columns)
