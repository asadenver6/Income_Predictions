"""Run the local BigQuery-to-model workflow."""

import argparse
from pathlib import Path

from google.cloud import bigquery

from income_prediction.evaluation import evaluate_model, save_evaluation_report
from income_prediction.ingestion.query import load_dataframe
from income_prediction.ingestion.validate import validate_dataframe
from income_prediction.training.train import save_model, train_from_dataframe

DEFAULT_PROJECT_ID = "rta-genai-explorations-406d"
DEFAULT_SOURCE_TABLE = "bigquery-public-data.ml_datasets.census_adult_income"


def main() -> None:
    """Load, validate, train, evaluate, and save local artifacts."""
    parser = argparse.ArgumentParser(description="Run the local income prediction pipeline")
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument(
        "--metrics-output", type=Path, default=Path("artifacts/metrics.json")
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    client = bigquery.Client(project=args.project_id)
    dataframe = load_dataframe(client, args.source_table, limit=args.limit)
    quality_report = validate_dataframe(dataframe)
    model, splits = train_from_dataframe(dataframe, random_state=args.random_state)
    metrics = evaluate_model(model, splits.x_test, splits.y_test)

    save_model(model, args.model_output)
    save_evaluation_report(metrics, args.metrics_output)

    print(f"Loaded rows: {quality_report.row_count}")
    print(f"Train rows: {len(splits.x_train)}")
    print(f"Validation rows: {len(splits.x_validation)}")
    print(f"Test rows: {len(splits.x_test)}")
    print(f"accuracy={metrics.accuracy:.4f}")
    print(f"precision={metrics.precision:.4f}")
    print(f"recall={metrics.recall:.4f}")
    print(f"f1={metrics.f1:.4f}")
    print(f"roc_auc={metrics.roc_auc:.4f}")
    print(f"Model artifact: {args.model_output}")
    print(f"Metrics artifact: {args.metrics_output}")


if __name__ == "__main__":
    main()
