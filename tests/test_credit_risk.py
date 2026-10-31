import pandas as pd
import pytest

from credit_risk.data import prepare_training_data
from credit_risk.model import build_pipeline, coefficient_summary


def test_training_data_excludes_identifiers_and_normalizes_gender():
    frame = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "name": ["A", "B", "C", "D"],
            "gender": ["F", "XNA", "M", "F"],
            "income": [100.0, 200.0, None, 400.0],
            "credit_card_default": [0, 1, 0, 1],
        }
    )

    features, labels = prepare_training_data(frame)

    assert list(features.columns) == ["gender", "income"]
    assert features["gender"].tolist() == ["F", "M", "M", "F"]
    assert labels.tolist() == [0, 1, 0, 1]


def test_pipeline_handles_missing_and_unseen_values_and_exposes_coefficients():
    features = pd.DataFrame(
        {
            "income": [10, 20, None, 40, 50, 60, 70, 80],
            "occupation": ["A", "B", "A", None, "B", "A", "B", "A"],
        }
    )
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    pipeline = build_pipeline(features)
    pipeline.fit(features, labels)

    unseen = pd.DataFrame({"income": [25.0], "occupation": ["new-job"]})
    probabilities = pipeline.predict_proba(unseen)[:, 1]
    summary = coefficient_summary(pipeline)

    assert len(probabilities) == 1
    assert 0 <= probabilities[0] <= 1
    assert {"feature", "coefficient", "odds_ratio"} <= set(summary.columns)
    assert pipeline.named_steps["classifier"].class_weight == "balanced"


def test_training_data_rejects_non_binary_labels():
    frame = pd.DataFrame(
        {
            "income": [10, 20, 30],
            "credit_card_default": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="only 0 and 1"):
        prepare_training_data(frame)
