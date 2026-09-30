"""Vertex AI custom prediction server for the income model."""

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from flask import Flask, jsonify, request

from income_prediction.data_contract import FEATURE_COLUMNS
from income_prediction.monitoring.prediction_logging import (
    create_prediction_record,
    record_prediction,
)
from income_prediction.training.train import load_model

DEFAULT_MODEL_PATH = Path("/app/model/model.joblib")


def _predict(model: Any, instances: Any) -> list[dict[str, Any]]:
    """Validate instances and return JSON-compatible predictions."""
    if not isinstance(instances, list) or not instances:
        raise ValueError("Request field 'instances' must be a non-empty list")
    if not all(isinstance(instance, dict) for instance in instances):
        raise ValueError("Each prediction instance must be an object")

    dataframe = pd.DataFrame(instances)
    missing = sorted(set(FEATURE_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError("Prediction is missing required features: " + ", ".join(missing))
    features = dataframe.loc[:, FEATURE_COLUMNS]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return [
        {
            "prediction": ">50K" if int(prediction) == 1 else "<=50K",
            "probability": float(probability[int(prediction)]),
        }
        for prediction, probability in zip(predictions, probabilities)
    ]


def create_app(model: Any | None = None) -> Flask:
    """Create the prediction HTTP application."""
    model_holder = {"model": model}

    application = Flask(__name__)

    @application.get("/health")
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok"}), 200

    @application.post("/predict")
    def predict() -> tuple[Any, int]:
        try:
            if model_holder["model"] is None:
                model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
                model_holder["model"] = load_model(model_path)
            payload = request.get_json(silent=True) or {}
            request_id = request.headers.get("X-Request-ID", str(uuid4()))
            predictions = _predict(model_holder["model"], payload.get("instances"))
            for prediction in predictions:
                record = create_prediction_record(
                    request_id=request_id,
                    prediction=prediction["prediction"],
                    probability=prediction["probability"],
                )
                record_prediction(record)
            return jsonify({"request_id": request_id, "predictions": predictions}), 200
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

    return application


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
