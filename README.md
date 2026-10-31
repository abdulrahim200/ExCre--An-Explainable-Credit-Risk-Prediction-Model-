# ExCre: Explainable Credit Risk Prediction

ExCre predicts whether a customer will default on a credit card. The notebook
workflows are preserved as reference; the `credit_risk` package provides a
reproducible command-line project for training, evaluating, explaining, and
scoring records.

## Project layout

```text
credit_risk/          Python package and CLI
dataset/
  train.csv           Labeled training data
  test.csv            Unlabeled records to score
tests/                Project tests
artifacts/            Generated model and reports (created on training)
Credit Risk Analysis.ipynb
preprocessing.ipynb
```

The notebooks are kept unchanged. Training uses `credit_card_default` as the
target, omits customer identifiers, imputes missing values, standardizes
numeric features, one-hot encodes categorical features, and fits a
class-weighted logistic regression. The class weighting addresses the target
imbalance without creating synthetic one-hot category values.

## Set up

From the project directory, create and activate a virtual environment, then
install the package:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Train and evaluate

```powershell
python -m credit_risk train
```

By default this reads `dataset/train.csv`, makes a stratified 70/30
train/validation split (random seed 42), and writes:

- `artifacts/model.joblib` — fitted preprocessing and model pipeline
- `artifacts/metrics.json` — accuracy, precision, recall, F1, ROC-AUC, and
  confusion matrix for the holdout split
- `artifacts/feature_coefficients.csv` — logistic coefficients and odds ratios

Pass `--data` or `--output-dir` to use different paths. The coefficients
describe the fitted logistic model; they are not causal effects.

## Score records

After training, score the supplied unlabeled data:

```powershell
python -m credit_risk predict
```

This reads `dataset/test.csv` and writes `artifacts/predictions.csv`, with
`customer_id`, `predicted_default`, and `default_probability` columns. Change
the decision threshold with `--threshold`, or provide `--data`, `--model`, and
`--output` paths.

## Run tests

```powershell
python -m pytest
```

## Dataset fields

The target is `credit_card_default` (`1` = default, `0` = no default).
`customer_id` and `name` are excluded from model features. The supplied test
file has no target column.
