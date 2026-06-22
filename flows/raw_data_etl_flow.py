from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Literal

from prefect import flow, get_run_logger, task
from prefect.flow_runs import pause_flow_run
from pydantic import BaseModel, Field

from scripts.process_raw_data import (
    ACTIVITY_OPTIONS,
    CLOTHING_OPTIONS,
    CONTEXT_OPTIONS,
    RecordingInfo,
    append_metadata,
    build_preview,
    choose_csv_files,
    get_pyplot,
    load_metadata,
    next_record_id,
    read_raw_csv,
    sanitize_token,
    to_iso_date,
)


Activity = Literal["sitting", "standing", "walking", "running"]
Context = Literal[
    "clean",
    "coughnoise",
    "musicnoise",
    "sneezenoise",
    "snoozenoise",
    "doornoise",
    "falsepositive",
    "noise",
]
Clothing = Literal["underclothes", "overclothes"]

METADATA_PATH = "data/metadata.csv"
CURATED_DIR = "data/curated_csv"
PREVIEW_DIR = "artifacts/raw_previews/prefect"


class PreviewApproval(BaseModel):
    approve_to_curate: bool = Field(
        ...,
        description=(
            "Set true only after checking the generated preview plot and "
            "confirming that the metadata is correct."
        ),
    )
    notes: str = Field(
        "",
        description="Optional review note. It is logged but not written to metadata.",
    )


def project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / ".git").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root.")


def parse_paths(raw_csv_paths: str) -> list[Path]:
    return [
        Path(item.strip()).expanduser().resolve()
        for item in raw_csv_paths.replace("\n", ",").split(",")
        if item.strip()
    ]


@task
def select_raw_csv_files_with_finder() -> list[str]:
    paths = choose_csv_files()
    return [str(path) for path in paths]


@task
def preview_raw_csv(raw_csv_path: str, preview_dir: str) -> dict:
    logger = get_run_logger()
    path = Path(raw_csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {path}")

    df = read_raw_csv(path)
    output_dir = Path(preview_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / f"{path.stem}_preview.png"

    pyplot = get_pyplot(no_show=True)
    fig = build_preview(df, path.name, pyplot)
    fig.savefig(preview_path, bbox_inches="tight")
    pyplot.close(fig)

    cough_windows = int(df["cough_label"].sum())
    logger.info("Preview saved: %s", preview_path)
    return {
        "raw_csv_path": str(path),
        "preview_path": str(preview_path),
        "rows": int(len(df)),
        "cough_samples": cough_windows,
    }


@task
def curate_raw_csv(
    raw_csv_path: str,
    subject: str,
    activity: Activity,
    context: Context,
    clothing: Clothing,
    date_yyyymmdd: str,
    approve_to_curate: bool,
) -> dict:
    logger = get_run_logger()
    root = project_root()
    path = Path(raw_csv_path).expanduser().resolve()
    metadata_file = root / METADATA_PATH
    output_dir = root / CURATED_DIR

    if not approve_to_curate:
        logger.warning("approve_to_curate is false; skipping curated copy and metadata append.")
        return {"raw_csv_path": str(path), "status": "preview_only"}

    subject = sanitize_token(subject)
    if activity not in ACTIVITY_OPTIONS:
        raise ValueError(f"Invalid activity: {activity}")
    if context not in CONTEXT_OPTIONS:
        raise ValueError(f"Invalid context: {context}")
    if clothing not in CLOTHING_OPTIONS:
        raise ValueError(f"Invalid clothing: {clothing}")

    date_str = date_yyyymmdd.strip()
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError("date_yyyymmdd must use YYYYMMDD format.")

    metadata = load_metadata(metadata_file)
    record_id = next_record_id(output_dir, metadata)
    info = RecordingInfo(
        record_id=record_id,
        date_str=date_str,
        date_iso=to_iso_date(date_str),
        subject=subject,
        activity=activity,
        context=context,
        clothing=clothing,
    )
    destination = output_dir / info.filename
    if destination.exists():
        raise FileExistsError(destination)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    append_metadata(
        metadata_file,
        {
            "record_id": info.record_id,
            "filename": info.filename,
            "date": info.date_iso,
            "subject": info.subject,
            "activity": info.activity,
            "context": info.context,
            "clothing": info.clothing,
            "relative_path": f"curated_csv/{info.filename}",
        },
    )

    logger.info("Added record %s: %s", record_id, destination)
    return {
        "raw_csv_path": str(path),
        "status": "curated",
        "record_id": record_id,
        "curated_path": str(destination),
    }


@flow(name="raw-data-etl")
def raw_data_etl_flow(
    subject: Annotated[str, Field(description="Anonymized subject ID, for example subject01 or subject02.")],
    date_yyyymmdd: Annotated[str, Field(description="Recording date in YYYYMMDD format.")],
    activity: Annotated[Activity, Field(description="Activity label for metadata.")],
    context: Annotated[Context, Field(description="Recording condition/context label.")],
    clothing: Annotated[Clothing, Field(description="Sensor clothing placement condition.")],
    use_file_picker: Annotated[
        bool,
        Field(description="Open a local macOS file picker from the runner process."),
    ] = True,
    raw_csv_paths: Annotated[
        str,
        Field(
            description=(
                "Optional fallback. Paste one or more absolute CSV paths, separated "
                "by commas or new lines. Leave empty when use_file_picker is true."
            )
        ),
    ] = "",
    pause_for_approval: Annotated[
        bool,
        Field(description="Pause after preview and wait for approval before curating."),
    ] = True,
) -> list[dict]:
    logger = get_run_logger()
    root = project_root()
    paths = parse_paths(raw_csv_paths)
    if use_file_picker:
        selected_paths = select_raw_csv_files_with_finder()
        paths.extend(Path(path) for path in selected_paths)

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    paths = unique_paths

    if not paths:
        raise ValueError("No raw CSV selected. Enable use_file_picker or provide raw_csv_paths.")

    results = []
    for path in paths:
        preview = preview_raw_csv(str(path), str(root / PREVIEW_DIR))
        approve_to_curate = False
        if pause_for_approval:
            logger.info(
                "Preview ready: %s. Resume this flow run with approval input.",
                preview["preview_path"],
            )
            approval = pause_flow_run(
                wait_for_input=PreviewApproval,
                timeout=24 * 60 * 60,
                key=f"approve-{Path(path).stem}",
            )
            approve_to_curate = bool(approval.approve_to_curate)
            if approval.notes:
                logger.info("Approval note: %s", approval.notes)

        curated = curate_raw_csv(
            raw_csv_path=str(path),
            subject=subject,
            activity=activity,
            context=context,
            clothing=clothing,
            date_yyyymmdd=date_yyyymmdd,
            approve_to_curate=approve_to_curate,
        )
        results.append({**preview, **curated})
    return results


if __name__ == "__main__":
    raw_data_etl_flow()
