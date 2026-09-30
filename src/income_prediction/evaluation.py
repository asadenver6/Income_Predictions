"""Model evaluation metrics and promotion quality gates."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationReport:
    """Classification metrics produced for one dataset split."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


def save_evaluation_report(report: EvaluationReport, path: str | Path) -> Path:
    """Write evaluation metrics to a JSON artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output_path


def evaluate_model(model: Any, features: Any, labels: Any) -> EvaluationReport:
    """Calculate binary classification metrics for a fitted model."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    return EvaluationReport(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
    )


def enforce_quality_gate(
    report: EvaluationReport,
    minimums: dict[str, float],
) -> None:
    """Raise an error when any requested metric is below its minimum."""
    unknown_metrics = set(minimums) - {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }
    if unknown_metrics:
        raise ValueError("Unknown evaluation metrics: " + ", ".join(sorted(unknown_metrics)))

    failures = [
        f"{metric}={getattr(report, metric):.4f} < {minimum:.4f}"
        for metric, minimum in minimums.items()
        if getattr(report, metric) < minimum
    ]
    if failures:
        raise ValueError("Model failed quality gate: " + "; ".join(failures))
