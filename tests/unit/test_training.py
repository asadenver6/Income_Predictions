import pandas as pd

from income_prediction.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from income_prediction.preprocessing.pipeline import split_dataset
from income_prediction.training.train import load_model, save_model, train_baseline


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


def test_train_baseline_predicts_binary_labels() -> None:
    splits = split_dataset(sample_dataframe())

    model = train_baseline(splits)
    predictions = model.predict(splits.x_test)

    assert len(predictions) == len(splits.x_test)
    assert set(predictions).issubset({0, 1})


def test_trained_model_can_be_serialized(tmp_path) -> None:
    model = train_baseline(split_dataset(sample_dataframe()))
    model_path = save_model(model, tmp_path / "model.joblib")

    restored_model = load_model(model_path)

    assert model_path.exists()
    assert restored_model.predict(sample_dataframe().drop(columns=TARGET_COLUMN).head(2)).shape == (2,)
