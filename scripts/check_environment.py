from __future__ import annotations

import csv
import importlib
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)


REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("PyYAML", "yaml"),
]


def module_version(import_name: str) -> str:
    module = importlib.import_module(import_name)
    return str(getattr(module, "__version__", "available"))


def check_python_version() -> bool:
    version = sys.version_info
    ok = (3, 11) <= (version.major, version.minor) < (3, 13)
    status = "OK" if ok else "FAIL"
    print(f"{status} Python: {version.major}.{version.minor}.{version.micro}")
    return ok


def check_modules() -> bool:
    ok = True
    for package_name, import_name in REQUIRED_MODULES:
        try:
            version = module_version(import_name)
            print(f"OK {package_name}: {version}")
        except Exception as exc:
            ok = False
            print(f"FAIL {package_name}: {type(exc).__name__}: {exc}")
    return ok


def check_project_files() -> bool:
    ok = True
    paths = [
        PROJECT_ROOT / "data" / "metadata.csv",
        PROJECT_ROOT / "configs" / "paths.yaml",
        PROJECT_ROOT / "configs" / "v3.yaml",
        PROJECT_ROOT / "src" / "cough_analysis",
    ]

    for path in paths:
        exists = path.exists()
        print(f"{'OK' if exists else 'FAIL'} path: {path.relative_to(PROJECT_ROOT)}")
        ok = ok and exists

    return ok


def check_metadata() -> bool:
    metadata_path = PROJECT_ROOT / "data" / "metadata.csv"
    if not metadata_path.exists():
        print("FAIL metadata: data/metadata.csv not found")
        return False

    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = {
        "record_id",
        "filename",
        "date",
        "subject",
        "activity",
        "context",
        "relative_path",
    }
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        print(f"FAIL metadata columns: {sorted(missing)}")
        return False

    missing_files = []
    for row in rows:
        record_path = PROJECT_ROOT / "data" / row["relative_path"]
        if not record_path.exists():
            missing_files.append(row["relative_path"])

    if missing_files:
        print(f"FAIL metadata paths: {len(missing_files)} missing files")
        for rel_path in missing_files[:5]:
            print(f"  missing: {rel_path}")
        return False

    print(f"OK metadata: {len(rows)} records")
    return True


def main() -> int:
    checks = [
        check_python_version(),
        check_modules(),
        check_project_files(),
        check_metadata(),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
