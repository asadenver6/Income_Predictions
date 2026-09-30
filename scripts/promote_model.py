"""Promote an approved model through a canary deployment."""

import argparse
import json
from pathlib import Path

from google.cloud import aiplatform

from income_prediction.monitoring.performance import PerformanceReport
from income_prediction.monitoring.promotion import decide_promotion

PROJECT_ID = "rta-genai-explorations-406d"
REGION = "europe-west1"
ENDPOINT_ID = "5090292434882002944"
SERVING_IMAGE = (
    "europe-west1-docker.pkg.dev/"
    "rta-genai-explorations-406d/income-prediction/serving:latest"
)


def load_report(path: Path) -> PerformanceReport:
    """Load a performance report from JSON."""
    values = json.loads(path.read_text(encoding="utf-8"))
    return PerformanceReport(
        sample_count=values["sample_count"],
        accuracy=values["accuracy"],
        precision=values["precision"],
        recall=values["recall"],
        f1=values["f1"],
        mean_absolute_calibration_error=values["mean_absolute_calibration_error"],
    )


def main() -> None:
    """Approve and optionally canary-deploy a candidate model."""
    parser = argparse.ArgumentParser(description="Promote an income model candidate")
    parser.add_argument("--champion-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-artifact-uri", required=True)
    parser.add_argument("--model-name", default="census-adult-income-candidate")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--minimum-f1", type=float, default=0.30)
    args = parser.parse_args()

    decision = decide_promotion(
        load_report(args.champion_report),
        load_report(args.candidate_report),
        {"f1": args.minimum_f1},
    )
    print(f"Approved: {decision.approved}")
    print(f"Reason: {decision.reason}")
    if not decision.approved or not args.promote:
        print("No production change made.")
        return

    aiplatform.init(project=PROJECT_ID, location=REGION)
    model = aiplatform.Model.upload(
        display_name=args.model_name,
        artifact_uri=args.candidate_artifact_uri,
        serving_container_image_uri=SERVING_IMAGE,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        sync=True,
    )
    endpoint = aiplatform.Endpoint(
        f"projects/856095474659/locations/{REGION}/endpoints/{ENDPOINT_ID}"
    )
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{args.model_name}-canary",
        machine_type="e2-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=0,
        sync=True,
    )
    print("Candidate deployed with 0% traffic for smoke testing.")
    print("Promote manually after verification with Vertex traffic split controls.")


if __name__ == "__main__":
    main()
