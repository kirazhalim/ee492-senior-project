# Prefect Usage Notes

This project can process new raw CSV files in two ways:

1. Prefect UI workflow
2. Direct script workflow with `scripts/process_raw_data.py`

Prefect is optional. The direct script still works.

## Install Prefect

Run once inside the project environment:

```bash
.venv/bin/python -m pip install -r requirements-orchestration.txt
```

## Start the Prefect UI

Use two terminals.

Terminal 1:

```bash
make prefect-ui PYTHON=.venv/bin/python PREFECT_PORT=4210
```

Open:

```text
http://127.0.0.1:4210
```

Terminal 2:

```bash
make prefect-serve-flows \
  PYTHON=.venv/bin/python \
  PREFECT_PORT=4210 \
  PREFECT_API_URL=http://127.0.0.1:4210/api
```

Keep both terminals open while using the UI.

## Raw Data ETL from UI

In the Prefect UI:

```text
Deployments -> raw-data-etl-ui -> Custom Run
```

Fill the metadata fields:

```text
subject
date_yyyymmdd
activity
context
clothing
```

Recommended settings:

```text
use_file_picker: true
raw_csv_paths: leave empty
pause_for_approval: true
```

With `use_file_picker=true`, the local runner process should open a macOS file
picker. If it does not, set `use_file_picker=false` and paste one or more raw
CSV paths into `raw_csv_paths`.

The flow first generates preview plots under:

```text
artifacts/raw_previews/prefect/
```

Then it pauses. Check the preview image. If the signals and labels look correct,
resume the flow from the Prefect UI and set:

```text
approve_to_curate: true
```

Only after this approval does the flow copy the CSV into:

```text
data/curated_csv/
```

and append a row to:

```text
data/metadata.csv
```

If the preview looks wrong, resume with:

```text
approve_to_curate: false
```

The file will not be added to the curated dataset.

## Validation and Summary Flow

In the Prefect UI:

```text
Deployments -> new-data-validation-summary-ui -> Custom Run
```

This flow runs:

```text
verify dataset manifest
verify model manifest
validate metadata and curated CSV files
generate dataset summary
```

You can also run it from terminal:

```bash
make prefect-flow-summary \
  PYTHON=.venv/bin/python \
  PREFECT_PORT=4210 \
  PREFECT_API_URL=http://127.0.0.1:4210/api
```

## Direct Script Fallback

Prefect is optional. You can still use the original raw-data script directly:

```bash
PYTHONPATH=src .venv/bin/python scripts/process_raw_data.py --select
```

or pass files directly:

```bash
PYTHONPATH=src .venv/bin/python scripts/process_raw_data.py /absolute/path/to/raw.csv
```

This script shows/saves previews, asks for approval and metadata in the terminal,
then writes to `data/curated_csv/` and `data/metadata.csv`.

## After Adding Data

Run these checks:

```bash
make validate-data PYTHON=.venv/bin/python
make verify-manifests PYTHON=.venv/bin/python
```

If the dataset intentionally changed, the dataset manifest hash may need to be
updated or a new dataset manifest should be created.
