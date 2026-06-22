from pathlib import Path
import argparse
import shutil
import re

# =========================
# USER SETTINGS
# =========================
DATA_ROOT = Path(__file__).resolve().parent
RAW_ROOT = DATA_ROOT / "raw_csv"
CURATED_ROOT = RAW_ROOT.parent / "curated_csv"
METADATA_PATH = RAW_ROOT.parent / "metadata.csv"

# Force subject and set defaults. Use anonymized subject IDs in shared configs.
SUBJECT = "subject01"
DEFAULT_CONTEXT = "clean"
DEFAULT_CLOTHING = "overclothes"

# Activity map
ACTIVITY_MAP = {
    "sittingup": "sitting",
    "sitting": "sitting",
    "standingup": "standing",
    "standup": "standing",
    "standing": "standing",
    "walking": "walking",
    "running": "running",
}

# Context map
CONTEXT_MAP = {
    "coughnoise": "coughnoise",
    "musicnoise": "musicnoise",
    "sneezenoise": "sneezenoise",
    "snoozenoise": "snoozenoise",
    "doornoise": "doornoise",
    "clean": "clean",
    "unspecified": "clean", # Replaced unspecified with clean
}

def normalize_token(x: str) -> str:
    return x.strip().lower().replace(" ", "").replace("-", "")

def parse_old_filename(filename: str):
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Cannot parse filename: {filename}")

    date_raw = parts[0]
    local_record_id = parts[1]

    if not re.fullmatch(r"\d{8}", date_raw):
        raise ValueError(f"Invalid date format: {filename}")

    tail = parts[2:]
    tail_norm = [normalize_token(x) for x in tail]

    # Drop subject name if exists in filename
    if tail_norm and tail_norm[0] == SUBJECT:
        tail_norm.pop(0)

    if len(tail_norm) == 0:
        raise ValueError(f"No activity found: {filename}")

    activity_key = tail_norm.pop(0)
    if activity_key not in ACTIVITY_MAP:
        raise ValueError(f"Unknown activity '{activity_key}': {filename}")

    activity = ACTIVITY_MAP[activity_key]

    # Parse clothing and context
    clothing = DEFAULT_CLOTHING
    context_parts = []
    
    for token in tail_norm:
        if token == "underclothes":
            clothing = "underclothes"
        elif token == "overclothes":
            clothing = "overclothes"
        else:
            mapped_token = CONTEXT_MAP.get(token, token)
            context_parts.append(mapped_token)

    context = "_".join(context_parts) if context_parts else DEFAULT_CONTEXT
    date_iso = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"

    return {
        "date_raw": date_raw,
        "date": date_iso,
        "local_record_id": int(local_record_id),
        "activity": activity,
        "context": context,
        "clothing": clothing,
    }

def build_new_filename(global_record_id: int, parsed: dict):
    # Keep filename short: omitted clothing, stored only in metadata
    rid = f"{global_record_id:03d}"
    return f"{rid}_{parsed['date_raw']}_{SUBJECT}_{parsed['activity']}_{parsed['context']}.csv"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--curated-root", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--default-context", default=DEFAULT_CONTEXT)
    parser.add_argument("--default-clothing", default=DEFAULT_CLOTHING)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main(args=None):
    global SUBJECT, DEFAULT_CONTEXT, DEFAULT_CLOTHING

    args = parse_args() if args is None else args

    raw_root = Path(args.raw_root).expanduser().resolve()
    curated_root = (
        raw_root.parent / "curated_csv"
        if args.curated_root is None
        else Path(args.curated_root).expanduser().resolve()
    )
    metadata_path = (
        raw_root.parent / "metadata.csv"
        if args.metadata_path is None
        else Path(args.metadata_path).expanduser().resolve()
    )

    SUBJECT = args.subject
    DEFAULT_CONTEXT = args.default_context
    DEFAULT_CLOTHING = args.default_clothing

    if not raw_root.exists():
        raise FileNotFoundError(f"Folder not found: {raw_root}")

    curated_root.mkdir(parents=True, exist_ok=True)

    existing_curated = list(curated_root.glob("*.csv"))
    if not args.overwrite and (existing_curated or metadata_path.exists()):
        raise RuntimeError(
            "Refusing to overwrite existing curated data or metadata. "
            "Pass --overwrite to rebuild them."
        )

    import pandas as pd

    discovered = []
    errors = []

    # 1. Parse all files
    for csv_file in raw_root.rglob("*.csv"):
        try:
            parsed = parse_old_filename(csv_file.name)
            discovered.append({
                "source_path": csv_file,
                "source_filename": csv_file.name,
                **parsed
            })
        except Exception as e:
            errors.append({"source_file": str(csv_file), "error": str(e)})

    # 2. Sort chronologically
    discovered = sorted(
        discovered,
        key=lambda x: (x["date_raw"], x["local_record_id"], x["source_filename"])
    )

    metadata_rows = []

    # 3. Clear curated folder
    for f in curated_root.glob("*.csv"):
        f.unlink()

    # 4. Copy files and build metadata
    for global_id, item in enumerate(discovered):
        new_filename = build_new_filename(global_id, item)
        dst = curated_root / new_filename
        shutil.copy2(item["source_path"], dst)

        metadata_rows.append({
            "record_id": global_id,
            "filename": new_filename,
            "date": item["date"],
            "subject": SUBJECT,
            "activity": item["activity"],
            "context": item["context"],
            "clothing": item["clothing"],
            "relative_path": str(dst.relative_to(raw_root.parent)).replace("\\", "/"),
        })

    # 5. Save metadata
    df = pd.DataFrame(metadata_rows)
    df.to_csv(metadata_path, index=False)

    print(f"Curated files created: {len(df)}")
    print(f"Metadata saved to: {metadata_path}")

    # 6. Save errors
    if errors:
        err_path = raw_root.parent / "prepare_dataset_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Errors saved to: {err_path}")

if __name__ == "__main__":
    main()
