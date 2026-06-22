from __future__ import annotations

import os
import subprocess
from pathlib import Path

from prefect import flow, get_run_logger, task


def project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / ".git").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root.")


@task
def run_command(command: list[str]) -> str:
    logger = get_run_logger()
    root = project_root()
    env = {**os.environ, "PYTHONPATH": "src"}
    logger.info("Running: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")
    return result.stdout


@flow(name="new-data-validation-summary")
def new_data_eval_flow(
    dataset_manifest: str = "configs/datasets/dataset_v2_096_records.yaml",
    model_manifest: str = "configs/models/v3_cough_current.yaml",
    summary_output_dir: str = "artifacts/dataset_summary/dataset_v2_096_records",
    python: str = ".venv/bin/python",
) -> dict:
    run_command([python, "scripts/verify_dataset_manifest.py", dataset_manifest])
    run_command([python, "scripts/verify_model_manifest.py", model_manifest])
    run_command([python, "scripts/validate_dataset.py"])
    run_command([python, "scripts/dataset_summary.py", "--output-dir", summary_output_dir])
    return {
        "dataset_manifest": dataset_manifest,
        "model_manifest": model_manifest,
        "summary_output_dir": summary_output_dir,
    }


if __name__ == "__main__":
    new_data_eval_flow()
