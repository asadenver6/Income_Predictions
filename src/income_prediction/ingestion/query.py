"""BigQuery ingestion helpers for the Census Adult income dataset."""

import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

from income_prediction.data_contract import FEATURE_COLUMNS, TARGET_COLUMN


DEFAULT_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
_TABLE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")


def build_select_query(
    source_table: str,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    limit: int | None = None,
) -> str:
    """Build a bounded BigQuery SELECT statement for a fully qualified table."""
    if not _TABLE_PATTERN.fullmatch(source_table):
        raise ValueError(
            "source_table must use the fully qualified format project.dataset.table"
        )
    if not columns:
        raise ValueError("At least one column must be selected")
    if any(not column or not column.replace("_", "").isalnum() for column in columns):
        raise ValueError("Column names must contain only letters, numbers, and underscores")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    selected_columns = ", ".join(f"`{column}`" for column in columns)
    query = f"SELECT {selected_columns} FROM `{source_table}`"
    if limit is not None:
        query += f" LIMIT {limit}"
    return query


def load_dataframe(
    client: Any,
    source_table: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Execute the dataset query using an injected BigQuery-compatible client."""
    query_job = client.query(build_select_query(source_table, limit=limit))
    return query_job.to_dataframe()
