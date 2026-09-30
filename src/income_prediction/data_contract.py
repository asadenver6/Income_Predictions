"""Data contract for the BigQuery Census Adult income dataset."""

from collections.abc import Collection, Mapping

TARGET_COLUMN = "income_bracket"
VALID_LABELS = frozenset({"<=50K", ">50K"})

FEATURE_COLUMNS = (
    "age",
    "workclass",
    "functional_weight",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
)


def validate_columns(columns: Collection[str]) -> None:
    """Raise an error when required feature or target columns are absent."""
    available = set(columns)
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = sorted(required - available)
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))


def validate_labels(labels: Collection[object]) -> None:
    """Raise an error when labels are empty or outside the contract."""
    observed = {str(label).strip() for label in labels}
    if not observed:
        raise ValueError("Target column contains no labels")
    invalid = sorted(observed - VALID_LABELS)
    if invalid:
        raise ValueError("Dataset contains invalid income labels: " + ", ".join(invalid))


def validate_schema(schema: Mapping[str, str] | Collection[str]) -> None:
    """Validate either a column-to-type mapping or a collection of names."""
    columns = schema.keys() if isinstance(schema, Mapping) else schema
    validate_columns(columns)
