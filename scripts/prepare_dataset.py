from __future__ import annotations

import importlib.util
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "data" / "prepate_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load dataset preparation script: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()

