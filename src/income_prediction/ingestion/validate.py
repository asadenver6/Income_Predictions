"""Data-quality validation for ingested Census Adult income data."""

from dataclasses import dataclass

import pandas as pd

from income_prediction.data_contract import TARGET_COLUMN, validate_columns, validate_labels


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of the validated input data."""

    row_count: int
    column_count: int
    duplicate_row_count: int
    null_fraction_by_column: dict[str, float]


def validate_dataframe(
    dataframe: pd.DataFrame,
    max_null_fraction: float | None = None,
) -> DataQualityReport:
    """Validate an ingested DataFrame and return a quality summary."""
    if dataframe.empty:
        raise ValueError("Dataset contains no rows")
    if max_null_fraction is not None and not 0 <= max_null_fraction <= 1:
        raise ValueError("max_null_fraction must be between zero and one")

    validate_columns(dataframe.columns)
    validate_labels(dataframe[TARGET_COLUMN].dropna().tolist())

    null_fraction_by_column = dataframe.isna().mean().to_dict()
    if max_null_fraction is not None:
        invalid_nulls = sorted(
            column
            for column, fraction in null_fraction_by_column.items()
            if fraction > max_null_fraction
        )
        if invalid_nulls:
            raise ValueError(
                "Columns exceed the maximum null fraction: " + ", ".join(invalid_nulls)
            )

    return DataQualityReport(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        duplicate_row_count=int(dataframe.duplicated().sum()),
        null_fraction_by_column=null_fraction_by_column,
    )
