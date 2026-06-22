from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from cough_analysis.config import load_config
from cough_analysis.paths import project_path


MANIFEST_FIELDS = [
    "record_id",
    "filename",
    "date",
    "subject",
    "activity",
    "context",
    "clothing",
    "relative_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify dataset manifest record counts and content hashes."
    )
    parser.add_argument(
        "manifests",
        nargs="*",
        default=[
            "configs/datasets/dataset_v1_085_records.yaml",
            "configs/datasets/dataset_v2_096_records.yaml",
        ],
        help="Dataset manifest YAML files to verify.",
    )
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def load_metadata_rows(metadata_path: Path) -> list[dict]:
    with metadata_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["record_id"] = int(row["record_id"])
    return rows


def calculate_hashes(manifest: dict) -> dict[str, str | int]:
    metadata_path = project_or_absolute(manifest["metadata_path"])
    data_root = project_or_absolute(manifest.get("data_root", "data"))
    row_range = manifest["record_id_range"]
    start = int(row_range["start"])
    end = int(row_range["end"])

    rows = [
        row
        for row in load_metadata_rows(metadata_path)
        if start <= int(row["record_id"]) <= end
    ]
    rows.sort(key=lambda row: int(row["record_id"]))

    metadata_hash = hashlib.sha256()
    data_files_hash = hashlib.sha256()
    for row in rows:
        normalized = "|".join(str(row[field]) for field in MANIFEST_FIELDS)
        metadata_hash.update(f"{normalized}\n".encode("utf-8"))

        record_path = data_root / row["relative_path"]
        file_hash = hashlib.sha256(record_path.read_bytes()).hexdigest()
        data_files_hash.update(
            f"{row['relative_path']}|{file_hash}\n".encode("utf-8")
        )

    metadata_digest = metadata_hash.hexdigest()
    files_digest = data_files_hash.hexdigest()
    combined_digest = hashlib.sha256(
        f"{metadata_digest}{files_digest}".encode("utf-8")
    ).hexdigest()
    return {
        "record_count": len(rows),
        "metadata_rows_sha256": metadata_digest,
        "data_files_sha256": files_digest,
        "combined_sha256": combined_digest,
    }


def verify_manifest(path: str | Path) -> bool:
    manifest_path = project_or_absolute(path)
    manifest = load_config(manifest_path)
    expected_hashes = manifest["hashes"]
    actual = calculate_hashes(manifest)

    checks = {
        "record_count": int(manifest["record_count"]),
        "metadata_rows_sha256": expected_hashes["metadata_rows_sha256"],
        "data_files_sha256": expected_hashes["data_files_sha256"],
        "combined_sha256": expected_hashes["combined_sha256"],
    }

    ok = True
    for key, expected in checks.items():
        if actual[key] != expected:
            ok = False
            print(
                f"[FAIL] {manifest_path} {key}: "
                f"expected {expected}, got {actual[key]}"
            )
    if ok:
        print(
            f"[OK] {manifest['dataset_id']} "
            f"records={actual['record_count']} "
            f"hash={actual['combined_sha256'][:12]}"
        )
    return ok


def main() -> int:
    args = parse_args()
    results = [verify_manifest(path) for path in args.manifests]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
