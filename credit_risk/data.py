"""Loading and validating the credit-risk dataset."""

from __future__ import annotations

import pandas as pd

TARGET = "credit_card_default"
IDENTIFIER_COLUMNS = ("customer_id", "name")


def prepare_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target and remove identifiers from labeled records."""
    if TARGET not in frame.columns:
        raise ValueError(f"Training data must include the {TARGET!r} target column.")

    labels = pd.to_numeric(frame[TARGET], errors="raise")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError(f"The {TARGET!r} target must contain only 0 and 1.")
    if labels.nunique() != 2:
        raise ValueError("Training data must contain both target classes (0 and 1).")

    features = prepare_prediction_data(frame, expected_columns=None)
    return features, labels.astype("int64")


def prepare_prediction_data(
    frame: pd.DataFrame, expected_columns: list[str] | None
) -> pd.DataFrame:
    """Remove labels and identifiers, then align input to model feature columns."""
    excluded = [TARGET, *IDENTIFIER_COLUMNS]
    features = frame.drop(columns=excluded, errors="ignore").copy()

    if "gender" in features.columns:
        features["gender"] = features["gender"].replace("XNA", "M")

    if expected_columns is not None:
        missing = sorted(set(expected_columns) - set(features.columns))
        if missing:
            raise ValueError(
                "Input data is missing required model features: "
                + ", ".join(missing)
            )
        features = features.loc[:, expected_columns]

    if features.empty or features.shape[1] == 0:
        raise ValueError("No model features were found in the input data.")
    return features
