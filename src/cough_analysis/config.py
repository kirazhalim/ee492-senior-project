from __future__ import annotations

from pathlib import Path
from typing import Any

from cough_analysis.paths import project_path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_path(str(config_path))

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {config_path}")

    return data
