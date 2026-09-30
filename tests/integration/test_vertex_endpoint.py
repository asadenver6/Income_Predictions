"""Integration test for the deployed Vertex AI prediction endpoint."""

import os

import pytest
from google.cloud import aiplatform


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_GCP_INTEGRATION_TESTS") != "1",
    reason="Set RUN_GCP_INTEGRATION_TESTS=1 to run GCP integration tests",
)

PROJECT_ID = os.getenv("GCP_PROJECT_ID") or "rta-genai-explorations-406d"
REGION = os.getenv("GCP_REGION") or "europe-west1"
ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID") or "5090292434882002944"


@pytest.fixture
def endpoint():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    return aiplatform.Endpoint(
        f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
    )


def test_vertex_endpoint_prediction_contract(endpoint) -> None:
    instance = {
        "age": 45,
        "workclass": "Private",
        "functional_weight": 150000,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States",
    }

    response = endpoint.predict(instances=[instance])
    prediction = response.predictions[0]

    assert prediction["prediction"] in {"<=50K", ">50K"}
    assert 0 <= prediction["probability"] <= 1
