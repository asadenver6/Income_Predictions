import pytest

from income_prediction.monitoring.performance import calculate_performance


def test_calculate_performance_returns_production_metrics() -> None:
    report = calculate_performance(
        predictions=[">50K", "<=50K", ">50K", "<=50K"],
        probabilities=[0.9, 0.2, 0.7, 0.4],
        ground_truth=[">50K", "<=50K", "<=50K", "<=50K"],
    )

    assert report.sample_count == 4
    assert report.accuracy == 0.75
    assert report.precision == 0.5
    assert report.recall == 1.0
    assert report.f1 == pytest.approx(2 / 3)
    assert report.mean_absolute_calibration_error == pytest.approx(0.35)


def test_calculate_performance_requires_aligned_observations() -> None:
    with pytest.raises(ValueError, match="equal nonzero length"):
        calculate_performance([">50K"], [0.8], [])


def test_calculate_performance_rejects_invalid_labels() -> None:
    with pytest.raises(ValueError, match="invalid income labels"):
        calculate_performance(["unknown"], [0.8], [">50K"])
