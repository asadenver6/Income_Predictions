import pandas as pd
import pytest

from income_prediction.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from income_prediction.ingestion.validate import validate_dataframe


def valid_dataframe() -> pd.DataFrame:
    values = {column: [1, 2] for column in FEATURE_COLUMNS}
    values[TARGET_COLUMN] = ["<=50K", ">50K"]
    return pd.DataFrame(values)


def test_validate_dataframe_returns_quality_report() -> None:
    dataframe = valid_dataframe()

    report = validate_dataframe(dataframe)

    assert report.row_count == 2
    assert report.column_count == len(FEATURE_COLUMNS) + 1
    assert report.duplicate_row_count == 0
    assert report.null_fraction_by_column[TARGET_COLUMN] == 0


def test_validate_dataframe_rejects_empty_dataset() -> None:
    dataframe = valid_dataframe().iloc[0:0]

    with pytest.raises(ValueError, match="no rows"):
        validate_dataframe(dataframe)


def test_validate_dataframe_rejects_invalid_labels() -> None:
    dataframe = valid_dataframe()
    dataframe.loc[0, TARGET_COLUMN] = "unknown"

    with pytest.raises(ValueError, match="invalid income labels"):
        validate_dataframe(dataframe)


def test_validate_dataframe_enforces_null_fraction() -> None:
    dataframe = valid_dataframe()
    dataframe.loc[0, "age"] = None

    with pytest.raises(ValueError, match="maximum null fraction"):
        validate_dataframe(dataframe, max_null_fraction=0.25)
