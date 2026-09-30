"""Structured prediction and delayed ground-truth logging."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from typing import Any

LOGGER = logging.getLogger("income_prediction.predictions")


@dataclass(frozen=True)
class PredictionRecord:
    """A prediction event that can later be joined to its true label."""

    request_id: str
    timestamp: str
    prediction: str
    probability: float
    ground_truth: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible event payload."""
        return asdict(self)


def create_prediction_record(
    request_id: str,
    prediction: str,
    probability: float,
) -> PredictionRecord:
    """Create a timestamped prediction record without logging raw features."""
    if not request_id:
        raise ValueError("request_id must not be empty")
    if prediction not in {"<=50K", ">50K"}:
        raise ValueError("prediction must be <=50K or >50K")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    return PredictionRecord(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        prediction=prediction,
        probability=probability,
    )


def record_prediction(record: PredictionRecord) -> None:
    """Emit a structured prediction event for Cloud Logging ingestion."""
    LOGGER.info("prediction_event=%s", json.dumps(record.to_dict(), sort_keys=True))


def attach_ground_truth(record: PredictionRecord, ground_truth: str) -> PredictionRecord:
    """Return a prediction record enriched with its later observed label."""
    normalized = ground_truth.strip()
    if normalized not in {"<=50K", ">50K"}:
        raise ValueError("ground_truth must be <=50K or >50K")
    enriched = PredictionRecord(
        request_id=record.request_id,
        timestamp=record.timestamp,
        prediction=record.prediction,
        probability=record.probability,
        ground_truth=normalized,
    )
    LOGGER.info("ground_truth_event=%s", json.dumps(enriched.to_dict(), sort_keys=True))
    return enriched
