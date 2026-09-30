import json

from scripts.promote_model import load_report


def test_load_report_reads_performance_artifact(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "sample_count": 4,
                "accuracy": 0.75,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.67,
                "mean_absolute_calibration_error": 0.2,
            }
        ),
        encoding="utf-8",
    )

    report = load_report(path)

    assert report.sample_count == 4
    assert report.f1 == 0.67
