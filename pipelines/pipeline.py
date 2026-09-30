"""Vertex AI Pipelines definition for Census Adult income classification."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output, Metrics, Artifact


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "google-cloud-bigquery>=3.25,<4",
        "db-dtypes>=1.3,<2",
        "pandas>=2.2,<3",
        "pyarrow>=15,<20",
    ],
)
def ingest(
    project_id: str,
    source_table: str,
    row_limit: int,
    dataset: Output[Dataset],
) -> None:
    """Read a BigQuery table and materialize a Parquet pipeline artifact."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{source_table}`"
    if row_limit > 0:
        query += f" LIMIT {row_limit}"
    dataframe = client.query(query).to_dataframe()
    dataframe.to_parquet(dataset.path, index=False)
    dataset.metadata["source_table"] = source_table
    dataset.metadata["row_count"] = len(dataframe)


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas>=2.2,<3", "pyarrow>=15,<20"],
)
def validate(
    dataset: Input[Dataset],
    validation_report: Output[Artifact],
) -> None:
    """Validate basic dataset shape and persist a quality report."""
    import json
    import pandas as pd

    dataframe = pd.read_parquet(dataset.path)
    required = {
        "age", "workclass", "functional_weight", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income_bracket",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    labels = {str(value).strip() for value in dataframe["income_bracket"].dropna()}
    if labels != {"<=50K", ">50K"}:
        raise ValueError(f"Unexpected income labels: {sorted(labels)}")
    report = {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "duplicate_row_count": int(dataframe.duplicated().sum()),
        "null_fraction_by_column": dataframe.isna().mean().to_dict(),
    }
    with open(validation_report.path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "pandas>=2.2,<3",
        "pyarrow>=15,<20",
        "scikit-learn>=1.5,<2",
        "joblib>=1.4,<2",
    ],
)
def train_and_evaluate(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    random_state: int = 42,
) -> None:
    """Train and evaluate the baseline model inside a pipeline component."""
    import json
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    dataframe = pd.read_parquet(dataset.path)
    feature_columns = [column for column in dataframe.columns if column != "income_bracket"]
    numeric_columns = [
        "age", "functional_weight", "education_num", "capital_gain",
        "capital_loss", "hours_per_week",
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    features = dataframe[feature_columns]
    labels = dataframe["income_bracket"].astype("string").str.strip().map({"<=50K": 0, ">50K": 1})
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels.astype("int64"), test_size=0.2, random_state=random_state, stratify=labels
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
        ]
    )
    fitted_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)),
    ])
    fitted_model.fit(x_train, y_train)
    predictions = fitted_model.predict(x_test)
    probabilities = fitted_model.predict_proba(x_test)[:, 1]
    values = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    joblib.dump(fitted_model, model.path)
    model.metadata["feature_columns"] = feature_columns
    model.metadata["random_state"] = random_state
    with open(metrics.path, "w", encoding="utf-8") as file:
        json.dump(values, file, indent=2)
    for name, value in values.items():
        metrics.log_metric(name, value)


@dsl.pipeline(name="census-adult-income-training")
def income_training_pipeline(
    project_id: str = "rta-genai-explorations-406d",
    source_table: str = "bigquery-public-data.ml_datasets.census_adult_income",
    row_limit: int = 0,
    random_state: int = 42,
) -> None:
    """Ingest, validate, train, and evaluate the income model."""
    ingestion_task = ingest(
        project_id=project_id,
        source_table=source_table,
        row_limit=row_limit,
    )
    validation_task = validate(dataset=ingestion_task.outputs["dataset"])
    validation_task.set_caching_options(False)
    train_task = train_and_evaluate(
        dataset=ingestion_task.outputs["dataset"],
        random_state=random_state,
    )
    train_task.after(validation_task)
    train_task.set_caching_options(False)
