from income_prediction.monitoring.performance import PerformanceReport
from income_prediction.monitoring.promotion import decide_promotion


def report(accuracy: float, precision: float, recall: float, f1: float) -> PerformanceReport:
    return PerformanceReport(10, accuracy, precision, recall, f1, 0.1)


def test_candidate_is_approved_when_it_passes_gates() -> None:
    decision = decide_promotion(
        report(0.8, 0.7, 0.7, 0.7),
        report(0.85, 0.75, 0.75, 0.75),
        {"accuracy": 0.8, "f1": 0.7},
    )

    assert decision.approved is True


def test_candidate_is_rejected_when_it_regresses() -> None:
    decision = decide_promotion(
        report(0.8, 0.7, 0.7, 0.7),
        report(0.85, 0.65, 0.75, 0.7),
        {"accuracy": 0.8},
    )

    assert decision.approved is False
    assert "regressed" in decision.reason
