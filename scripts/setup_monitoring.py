"""Create Vertex AI monitoring for the deployed income model."""

from google.cloud import aiplatform
from google.cloud.aiplatform.jobs import ModelDeploymentMonitoringJob
from google.cloud.aiplatform.model_monitoring import alert, objective, sampling, schedule

PROJECT_ID = "rta-genai-explorations-406d"
REGION = "europe-west1"
ENDPOINT_ID = "5090292434882002944"
DEPLOYED_MODEL_ID = "7057274756508155904"

ENDPOINT_RESOURCE = (
    f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
)
FEATURE_DRIFT_THRESHOLDS = {
    "age": 0.3,
    "workclass": 0.3,
    "functional_weight": 0.3,
    "education": 0.3,
    "education_num": 0.3,
    "marital_status": 0.3,
    "occupation": 0.3,
    "relationship": 0.3,
    "race": 0.3,
    "sex": 0.3,
    "capital_gain": 0.3,
    "capital_loss": 0.3,
    "hours_per_week": 0.3,
    "native_country": 0.3,
}


def main() -> None:
    """Create a scheduled Vertex AI model monitoring job."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    objective_config = objective.ObjectiveConfig(
        drift_detection_config=objective.DriftDetectionConfig(
            drift_thresholds=FEATURE_DRIFT_THRESHOLDS
        )
    )
    monitoring_job = ModelDeploymentMonitoringJob.create(
        endpoint=ENDPOINT_RESOURCE,
        objective_configs={DEPLOYED_MODEL_ID: objective_config},
        logging_sampling_strategy=sampling.RandomSampleConfig(sample_rate=0.1),
        schedule_config=schedule.ScheduleConfig(monitor_interval=3600),
        display_name="census-adult-income-drift-monitor",
        deployed_model_ids=[DEPLOYED_MODEL_ID],
        alert_config=alert.EmailAlertConfig(user_emails=[], enable_logging=True),
        enable_monitoring_pipeline_logs=True,
        bigquery_tables_log_ttl=30,
        labels={"application": "income-prediction", "environment": "dev"},
        project=PROJECT_ID,
        location=REGION,
    )
    print("MONITORING_JOB:", monitoring_job.resource_name)


if __name__ == "__main__":
    main()
