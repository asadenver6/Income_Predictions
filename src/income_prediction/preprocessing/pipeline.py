"""Feature preprocessing and reproducible dataset splitting."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from income_prediction.data_contract import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_columns,
    validate_labels,
)


NUMERIC_COLUMNS = (
    "age",
    "functional_weight",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
)
CATEGORICAL_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS
)


@dataclass(frozen=True)
class DatasetSplits:
    """Raw feature and target partitions."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def split_dataset(
    dataframe: pd.DataFrame,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """Split a labeled DataFrame into train, validation, and test partitions."""
    validate_columns(dataframe.columns)
    validate_labels(dataframe[TARGET_COLUMN].dropna().tolist())
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between zero and one")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size plus test_size must be less than one")

    features = dataframe.loc[:, FEATURE_COLUMNS]
    target = (
        dataframe[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .map({"<=50K": 0, ">50K": 1})
    )
    if target.isna().any():
        raise ValueError("Target contains labels that cannot be encoded")

    x_remaining, x_test, y_remaining, y_test = train_test_split(
        features,
        target.astype("int64"),
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    validation_fraction_of_remaining = validation_size / (1 - test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_remaining,
        y_remaining,
        test_size=validation_fraction_of_remaining,
        random_state=random_state,
        stratify=y_remaining,
    )
    return DatasetSplits(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def create_preprocessor() -> ColumnTransformer:
    """Create the unfitted feature transformation pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
