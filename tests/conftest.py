from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_private_dataset(relative_path: str = "data/metadata.csv") -> Path:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        pytest.skip(
            f"Private dataset file is not included in the public repository: {relative_path}"
        )
    return path
