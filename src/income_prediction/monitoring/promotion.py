"""Candidate-model promotion and rollback decisions."""

from dataclasses import dataclass

from income_prediction.monitoring.performance import PerformanceReport


@dataclass(frozen=True)
class PromotionDecision:
    """Decision produced before changing production traffic."""

    approved: bool
    reason: str


def decide_promotion(
    champion: PerformanceReport,
    candidate: PerformanceReport,
    minimums: dict[str, float],
) -> PromotionDecision:
    """Approve a candidate only when it meets gates and does not regress."""
    metric_names = ("accuracy", "precision", "recall", "f1")
    failures = [
        f"{metric} below minimum ({getattr(candidate, metric):.4f} < {minimum:.4f})"
        for metric, minimum in minimums.items()
        if metric not in metric_names or getattr(candidate, metric) < minimum
    ]
    regressions = [
        f"{metric} regressed ({getattr(candidate, metric):.4f} < {getattr(champion, metric):.4f})"
        for metric in metric_names
        if getattr(candidate, metric) < getattr(champion, metric)
    ]
    reasons = failures + regressions
    if reasons:
        return PromotionDecision(approved=False, reason="; ".join(reasons))
    return PromotionDecision(approved=True, reason="Candidate passed promotion gates")
