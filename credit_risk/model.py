"""An interpretable, imbalance-aware logistic-regression pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    """Build preprocessing and logistic-regression steps from feature dtypes."""
    categorical = list(
        features.select_dtypes(include=["object", "category", "string", "bool"]).columns
    )
    numeric = [column for column in features.columns if column not in categorical]
    transformers = []

    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))

    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required.")

    preprocessing = ColumnTransformer(transformers=transformers)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="liblinear",
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def coefficient_summary(pipeline: Pipeline) -> pd.DataFrame:
    """Return logistic coefficients and their odds-ratio interpretation."""
    preprocessing = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessing.get_feature_names_out()
    coefficients = classifier.coef_[0]

    summary = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(np.clip(coefficients, -700, 700)),
        }
    )
    return (
        summary.assign(absolute_coefficient=summary["coefficient"].abs())
        .sort_values("absolute_coefficient", ascending=False)
        .drop(columns="absolute_coefficient")
        .reset_index(drop=True)
    )
