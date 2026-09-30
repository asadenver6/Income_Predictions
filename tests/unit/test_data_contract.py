import pytest

from income_prediction.data_contract import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_columns,
    validate_labels,
    validate_schema,
)


VALID_COLUMNS = list(FEATURE_COLUMNS) + [TARGET_COLUMN]


def test_valid_schema_is_accepted() -> None:
    validate_schema(VALID_COLUMNS)


def test_missing_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(VALID_COLUMNS[:-1])


def test_invalid_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid income labels"):
        validate_labels(["<=50K", "unknown"])


def test_empty_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="no labels"):
        validate_labels([])
