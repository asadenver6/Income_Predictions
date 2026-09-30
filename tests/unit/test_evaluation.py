import pandas as pd
import pytest

from income_prediction.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from income_prediction.evaluation import enforce_quality_gate, evaluate_model
from income_prediction.preprocessing.pipeline import split_dataset
from income_prediction.training.train import train_baseline


def sample_dataframe(row_count: int = 40) -> pd.DataFrame:
    categorical_columns = {
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    }
    dataframe = pd.DataFrame(
        {
            column: [f"value-{index % 4}" for index in range(row_count)]
            if column in categorical_columns
            else list(range(1, row_count + 1))
            for column in FEATURE_COLUMNS
        }
    )
    dataframe[TARGET_COLUMN] = ["<=50K", ">50K"] * (row_count // 2)
    return dataframe


def test_evaluate_model_returns_metrics_and_confusion_matrix() -> None:
    splits = split_dataset(sample_dataframe())
    model = train_baseline(splits)

    report = evaluate_model(model, splits.x_test, splits.y_test)

    print(
        f"accuracy={report.accuracy:.4f}, "
        f"precision={report.precision:.4f}, "
        f"recall={report.recall:.4f}, "
        f"f1={report.f1:.4f}, "
        f"roc_auc={report.roc_auc:.4f}"
    )
    assert 0 <= report.accuracy <= 1
    assert 0 <= report.precision <= 1
    assert 0 <= report.recall <= 1
    assert 0 <= report.f1 <= 1
    assert 0 <= report.roc_auc <= 1
    assert sum(sum(row) for row in report.confusion_matrix) == len(splits.y_test)


def test_quality_gate_rejects_metrics_below_threshold() -> None:
    splits = split_dataset(sample_dataframe())
    report = evaluate_model(train_baseline(splits), splits.x_test, splits.y_test)

    with pytest.raises(ValueError, match="failed quality gate"):
        enforce_quality_gate(report, {"accuracy": 1.01})


def test_quality_gate_rejects_unknown_metric() -> None:
    splits = split_dataset(sample_dataframe())
    report = evaluate_model(train_baseline(splits), splits.x_test, splits.y_test)

    with pytest.raises(ValueError, match="Unknown evaluation metrics"):
        enforce_quality_gate(report, {"loss": 0.1})
