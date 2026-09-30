import pandas as pd

from income_prediction.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from income_prediction.preprocessing.pipeline import create_preprocessor, split_dataset


def sample_dataframe(row_count: int = 20) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        {
            column: [f"value-{index}" for index in range(row_count)]
            for column in FEATURE_COLUMNS
            if column not in {"age", "functional_weight", "education_num", "capital_gain", "capital_loss", "hours_per_week"}
        }
    )
    for column in ("age", "functional_weight", "education_num", "capital_gain", "capital_loss", "hours_per_week"):
        dataframe[column] = list(range(1, row_count + 1))
    dataframe[TARGET_COLUMN] = ["<=50K", ">50K"] * (row_count // 2)
    return dataframe


def test_split_dataset_is_reproducible_and_preserves_all_rows() -> None:
    dataframe = sample_dataframe()

    first = split_dataset(dataframe, random_state=7)
    second = split_dataset(dataframe, random_state=7)

    assert len(first.x_train) + len(first.x_validation) + len(first.x_test) == len(dataframe)
    assert first.x_train.equals(second.x_train)
    assert first.y_test.equals(second.y_test)
    assert set(first.y_train) == {0, 1}


def test_preprocessor_handles_missing_and_unseen_categories() -> None:
    dataframe = sample_dataframe()
    splits = split_dataset(dataframe)
    preprocessor = create_preprocessor()

    preprocessor.fit(splits.x_train)
    transformed = preprocessor.transform(splits.x_test)

    assert transformed.shape[0] == len(splits.x_test)
    assert transformed.shape[1] > len(FEATURE_COLUMNS)
