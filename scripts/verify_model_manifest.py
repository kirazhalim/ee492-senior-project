from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from cough_analysis.config import load_config
from cough_analysis.paths import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify model manifest checkpoint and config hashes."
    )
    parser.add_argument(
        "manifests",
        nargs="*",
        default=["configs/models/v3_cough_current.yaml"],
        help="Model manifest YAML files to verify.",
    )
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(project_or_absolute(path).read_bytes()).hexdigest()


def verify_manifest(path: str | Path) -> bool:
    manifest_path = project_or_absolute(path)
    manifest = load_config(manifest_path)
    checks = {
        "checkpoint_sha256": file_sha256(manifest["checkpoint_path"]),
        "config_sha256": file_sha256(manifest["config_path"]),
    }

    ok = True
    for key, actual in checks.items():
        expected = manifest[key]
        if actual != expected:
            ok = False
            print(
                f"[FAIL] {manifest_path} {key}: "
                f"expected {expected}, got {actual}"
            )
    if ok:
        print(
            f"[OK] {manifest['model_id']} "
            f"checkpoint={checks['checkpoint_sha256'][:12]}"
        )
    return ok


def main() -> int:
    args = parse_args()
    results = [verify_manifest(path) for path in args.manifests]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
