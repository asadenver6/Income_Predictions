"""Compile the Vertex AI/Kubeflow pipeline package."""

from pathlib import Path

from kfp import compiler

from pipelines.pipeline import income_training_pipeline


OUTPUT_PATH = Path("artifacts/income_training_pipeline.json")


def main() -> None:
    """Compile the pipeline to a Vertex AI-compatible JSON package."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=income_training_pipeline,
        package_path=str(OUTPUT_PATH),
    )
    print(f"Compiled pipeline: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
