from __future__ import annotations

from prefect import serve

from new_data_eval_flow import new_data_eval_flow
from raw_data_etl_flow import raw_data_etl_flow


if __name__ == "__main__":
    serve(
        raw_data_etl_flow.to_deployment(
            name="raw-data-etl-ui",
            tags=["etl", "raw-data", "local"],
            description=(
                "Preview raw sensor CSV files and optionally add approved "
                "records to the curated dataset."
            ),
        ),
        new_data_eval_flow.to_deployment(
            name="new-data-validation-summary-ui",
            tags=["validation", "summary", "local"],
            description=(
                "Verify manifests, validate curated data, and regenerate "
                "the dataset summary."
            ),
        ),
    )
