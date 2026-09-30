import pandas as pd
import pytest

from income_prediction.data_contract import FEATURE_COLUMNS, TARGET_COLUMN
from income_prediction.ingestion.query import (
    build_select_query,
    load_dataframe,
)


SOURCE_TABLE = "bigquery-public-data.ml_datasets.census_adult_income"


def test_build_select_query_uses_contract_columns_and_limit() -> None:
    query = build_select_query(SOURCE_TABLE, limit=25)

    assert query.startswith("SELECT ")
    assert all(f"`{column}`" in query for column in (*FEATURE_COLUMNS, TARGET_COLUMN))
    assert f"FROM `{SOURCE_TABLE}` LIMIT 25" in query


def test_build_select_query_rejects_invalid_table_name() -> None:
    with pytest.raises(ValueError, match="fully qualified"):
        build_select_query("census_adult_income")


def test_build_select_query_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        build_select_query(SOURCE_TABLE, limit=0)


def test_load_dataframe_executes_query_and_returns_dataframe() -> None:
    expected = pd.DataFrame({"income_bracket": ["<=50K"]})

    class QueryJob:
        def to_dataframe(self) -> pd.DataFrame:
            return expected

    class Client:
        def __init__(self) -> None:
            self.query_text = None

        def query(self, query: str) -> QueryJob:
            self.query_text = query
            return QueryJob()

    client = Client()
    result = load_dataframe(client, SOURCE_TABLE, limit=1)

    assert result.equals(expected)
    assert "LIMIT 1" in client.query_text
