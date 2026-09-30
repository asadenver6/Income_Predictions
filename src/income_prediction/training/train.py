"""Local baseline model training and serialization."""

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from income_prediction.preprocessing.pipeline import DatasetSplits, create_preprocessor


def train_baseline(
    splits: DatasetSplits,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """Fit a logistic-regression baseline using training data only."""
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than zero")

    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(splits.x_train, splits.y_train)
    return model


def train_from_dataframe(
    dataframe: pd.DataFrame,
    random_state: int = 42,
    max_iter: int = 1000,
) -> tuple[Pipeline, DatasetSplits]:
    """Split a labeled DataFrame and train the baseline model."""
    from income_prediction.preprocessing.pipeline import split_dataset

    splits = split_dataset(dataframe, random_state=random_state)
    return train_baseline(splits, random_state=random_state, max_iter=max_iter), splits


def save_model(model: Any, path: str | Path) -> Path:
    """Serialize a trained model and return its output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_model(path: str | Path) -> Any:
    """Load a serialized model from disk."""
    return joblib.load(path)


def main() -> None:
    """Train a baseline model from a local CSV file."""
    parser = argparse.ArgumentParser(description="Train the income prediction baseline")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--random-state", default=42, type=int)
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input_csv)
    model, _ = train_from_dataframe(dataframe, random_state=args.random_state)
    save_model(model, args.model_output)


if __name__ == "__main__":
    main()
