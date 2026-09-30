"""Runtime configuration for the income prediction pipeline."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class PipelineConfig:
    """Environment-specific settings required to run the pipeline."""

    project_id: str
    region: str
    staging_bucket: str
    bigquery_source_table: str
    artifact_registry_repository: str
    model_name: str
    endpoint_name: str

    @classmethod
    def from_environment(cls) -> "PipelineConfig":
        """Load required settings from environment variables."""
        values = {
            "project_id": os.getenv("PROJECT_ID"),
            "region": os.getenv("REGION"),
            "staging_bucket": os.getenv("STAGING_BUCKET"),
            "bigquery_source_table": os.getenv("BIGQUERY_SOURCE_TABLE"),
            "artifact_registry_repository": os.getenv("ARTIFACT_REGISTRY_REPOSITORY"),
            "model_name": os.getenv("MODEL_NAME"),
            "endpoint_name": os.getenv("ENDPOINT_NAME"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(
                "Missing required pipeline configuration: "
                + ", ".join(missing)
            )
        return cls(**values)
