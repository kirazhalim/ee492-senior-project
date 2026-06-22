# Prefect Flows

This folder contains lightweight Prefect workflows for local orchestration.

## Install

```bash
.venv/bin/python -m pip install -r requirements-orchestration.txt
```

## Start Prefect UI

```bash
PREFECT_HOME=.cache/prefect .venv/bin/prefect server start --host 127.0.0.1 --port 4200
```

Open the URL printed by Prefect, usually:

```text
http://127.0.0.1:4200
```

If port 4200 is busy, use another port such as 4210 and update `PREFECT_API_URL`
accordingly.

In another terminal, point local runs at the UI server:

```bash
export PREFECT_HOME=.cache/prefect
export PREFECT_API_URL=http://127.0.0.1:4200/api
```

Then serve the local deployments:

```bash
PYTHONPATH=.:src .venv/bin/python flows/serve_flows.py
```

Keep this process running. In the Prefect UI, open **Deployments** and use
**Custom run** to enter parameters.

## Raw Data ETL

Recommended UI parameters:

```text
subject: subject01
date_yyyymmdd: 20260511
activity: sitting | standing | walking | running
context: clean | coughnoise | musicnoise | sneezenoise | snoozenoise | doornoise | falsepositive | noise
clothing: underclothes | overclothes
use_file_picker: true
raw_csv_paths: leave empty when using the file picker
pause_for_approval: true
```

With `use_file_picker=true`, the local runner process opens a macOS file picker.
If that does not work in your terminal session, set `use_file_picker=false` and
paste one or more absolute CSV paths into `raw_csv_paths`.

The flow always writes previews first and then pauses. Check the preview image in:

```text
artifacts/raw_previews/prefect/
```

Then resume the paused flow run from the Prefect UI and set:

```text
approve_to_curate: true
```

Only after this approval does the flow copy the file into `data/curated_csv/` and
append a row to `data/metadata.csv`.

For a direct Python call without the Prefect UI:

```bash
PYTHONPATH=.:src .venv/bin/python - <<'PY'
from flows.raw_data_etl_flow import raw_data_etl_flow

raw_data_etl_flow(
    subject="subject01",
    date_yyyymmdd="20260511",
    activity="sitting",
    context="clean",
    clothing="underclothes",
    use_file_picker=False,
    raw_csv_paths="/absolute/path/to/raw.csv",
    pause_for_approval=True,
)
PY
```

## Dataset Validation and Summary

```bash
PYTHONPATH=.:src .venv/bin/python flows/new_data_eval_flow.py
```

This verifies dataset/model manifests, validates curated CSV files, and regenerates a dataset summary.
