import pandas as pd

from income_prediction.data_contract import FEATURE_COLUMNS
from income_prediction.monitoring.prediction_logging import (
    attach_ground_truth,
    create_prediction_record,
)
from serving.app import create_app


def sample_instance() -> dict[str, object]:
    categorical = {
        "workclass": "Private",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States",
    }
    numeric = {
        "age": 35,
        "functional_weight": 100000,
        "education_num": 13,
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
    }
    return {**numeric, **categorical}


class StubModel:
    def predict(self, features: pd.DataFrame) -> list[int]:
        assert list(features.columns) == list(FEATURE_COLUMNS)
        return [1] * len(features)

    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        return [[0.25, 0.75] for _ in range(len(features))]


def test_health_endpoint() -> None:
    client = create_app(StubModel()).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_endpoint_returns_predictions() -> None:
    client = create_app(StubModel()).test_client()

    response = client.post(
        "/predict",
        json={"instances": [sample_instance()]},
        headers={"X-Request-ID": "test-request-1"},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "request_id": "test-request-1",
        "predictions": [{"prediction": ">50K", "probability": 0.75}],
    }


def test_predict_endpoint_rejects_missing_features() -> None:
    client = create_app(StubModel()).test_client()

    response = client.post("/predict", json={"instances": [{}]})

    assert response.status_code == 400
    assert "missing required features" in response.get_json()["error"]


def test_ground_truth_can_be_attached_to_prediction_record() -> None:
    record = create_prediction_record("request-1", ">50K", 0.75)

    enriched = attach_ground_truth(record, " <=50K")

    assert enriched.request_id == "request-1"
    assert enriched.ground_truth == "<=50K"
