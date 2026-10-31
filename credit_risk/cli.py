"""Command-line entry points for training and scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from credit_risk.data import TARGET, prepare_prediction_data, prepare_training_data
from credit_risk.model import build_pipeline, coefficient_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_model(data_path: Path, output_dir: Path) -> dict[str, float]:
    """Train and evaluate a model, saving its pipeline and reports."""
    frame = pd.read_csv(data_path)
    features, labels = prepare_training_data(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    pipeline = build_pipeline(x_train)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype("int64")

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "threshold": 0.5,
        "test_size": 0.3,
        "random_state": 42,
        "positive_class": 1,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "feature_columns": list(features.columns),
        "target": TARGET,
    }
    joblib.dump(bundle, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    coefficient_summary(pipeline).to_csv(
        output_dir / "feature_coefficients.csv", index=False
    )

    return metrics


def predict_file(
    data_path: Path, model_path: Path, output_path: Path, threshold: float
) -> int:
    """Score unlabeled records and write default predictions."""
    if not 0 <= threshold <= 1:
        raise ValueError("The prediction threshold must be between 0 and 1.")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or not {
        "pipeline",
        "feature_columns",
    }.issubset(bundle):
        raise ValueError(f"{model_path} is not a supported credit-risk model.")

    frame = pd.read_csv(data_path)
    features = prepare_prediction_data(frame, bundle["feature_columns"])
    probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    results = pd.DataFrame()
    if "customer_id" in frame.columns:
        results["customer_id"] = frame["customer_id"]
    results["predicted_default"] = (probabilities >= threshold).astype("int64")
    results["default_probability"] = probabilities

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return len(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-risk",
        description="Train and score an explainable credit-default model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and evaluate the model")
    train_parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "dataset" / "train.csv"
    )
    train_parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts"
    )

    predict_parser = commands.add_parser(
        "predict", help="score records with a trained model"
    )
    predict_parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "dataset" / "test.csv"
    )
    predict_parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "artifacts" / "model.joblib"
    )
    predict_parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "predictions.csv"
    )
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        metrics = train_model(args.data, args.output_dir)
        print(json.dumps(metrics, indent=2))
        print(f"Saved model and reports to {args.output_dir}")
    elif args.command == "predict":
        count = predict_file(args.data, args.model, args.output, args.threshold)
        print(f"Scored {count} records; predictions saved to {args.output}")


if __name__ == "__main__":
    main()
