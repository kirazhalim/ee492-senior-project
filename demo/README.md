# EE492 Public Streamlit Demo

This folder contains the public, deployable Streamlit demo for the EE492 final
project. It is intentionally self-contained:

- no raw sensor CSV files,
- no subject metadata,
- no model checkpoints,
- no private artifact paths.

The app uses anonymized synthetic preset signals and report-aligned model
summaries to demonstrate the cough-detection and activity-attribution workflow.
The heavier local app under `app/` is reserved for private checkpoint-backed
inference.

## Run Locally

```bash
streamlit run demo/streamlit_app.py
```

## Deploy

On Streamlit Community Cloud, select:

- repository: the clean/public submission repository,
- branch: the public submission branch,
- main file path: `demo/streamlit_app.py`.

The dependency file is `demo/requirements.txt`, placed next to the Streamlit
entrypoint so the public demo does not install the full training stack.
