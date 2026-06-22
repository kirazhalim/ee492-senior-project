import pytest

from cough_analysis.config import load_config
from cough_analysis.paths import find_project_root, project_path


def test_find_project_root_contains_metadata():
    root = find_project_root()
    assert (root / "data").exists()


def test_project_path_builds_absolute_path():
    metadata_path = project_path("data", "metadata.csv")
    assert metadata_path.is_absolute()
    assert metadata_path.parent.name == "data"


def test_load_paths_config():
    pytest.importorskip("yaml")
    config = load_config("configs/paths.yaml")
    assert config["data"]["metadata"] == "data/metadata.csv"
