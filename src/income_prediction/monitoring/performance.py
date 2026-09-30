"""Production model-performance metrics from delayed ground-truth labels."""

from dataclasses import dataclass
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class PerformanceReport:
    """Performance metrics for predictions matched with observed labels."""

    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    mean_absolute_calibration_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""
        return {
            "sample_count": self.sample_count,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_absolute_calibration_error": self.mean_absolute_calibration_error,
        }


def calculate_performance(
    predictions: list[str],
    probabilities: list[float],
    ground_truth: list[str],
) -> PerformanceReport:
    """Calculate performance metrics for aligned production observations."""
    if not predictions or len(predictions) != len(probabilities) or len(predictions) != len(ground_truth):
        raise ValueError("predictions, probabilities, and ground_truth must have equal nonzero length")
    if any(not 0 <= probability <= 1 for probability in probabilities):
        raise ValueError("probabilities must be between zero and one")

    label_map = {"<=50K": 0, ">50K": 1}
    predicted_values = [label_map.get(label.strip()) for label in predictions]
    actual_values = [label_map.get(label.strip()) for label in ground_truth]
    if any(value is None for value in (*predicted_values, *actual_values)):
        raise ValueError("predictions and ground_truth contain invalid income labels")

    predicted_binary = [int(value) for value in predicted_values]
    actual_binary = [int(value) for value in actual_values]
    calibration_error = sum(
        abs(probability - actual)
        for probability, actual in zip(probabilities, actual_binary)
    ) / len(actual_binary)
    return PerformanceReport(
        sample_count=len(actual_binary),
        accuracy=float(accuracy_score(actual_binary, predicted_binary)),
        precision=float(precision_score(actual_binary, predicted_binary, zero_division=0)),
        recall=float(recall_score(actual_binary, predicted_binary, zero_division=0)),
        f1=float(f1_score(actual_binary, predicted_binary, zero_division=0)),
        mean_absolute_calibration_error=float(calibration_error),
    )
