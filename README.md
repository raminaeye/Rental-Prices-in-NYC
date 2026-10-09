# ML Pipeline for Short-Term Rental Prices in NYC

An end-to-end, reusable ML pipeline that predicts the nightly price of short-term rental
listings in New York City from listing attributes (borough, room type, reviews, location,
and the listing title). Built with MLflow for step orchestration, Hydra for configuration,
and Weights & Biases for experiment tracking, artifact versioning, and model promotion.

The pipeline is retrainable: new weekly data samples flow through the same steps
(download, cleaning, data tests, splitting, training), and the best model is promoted
to production for testing on a held-out set.

## How it works

The pipeline is defined in `main.py` and driven by `config.yaml`:

1. **download** (`components/get_data`): uploads a sample CSV (`data/sample1.csv` or
   `sample2.csv`) as a raw data artifact in W&B.
2. **basic_cleaning** (`src/basic_cleaning`): drops price outliers outside the
   configured range ($10 to $350 per night), converts `last_review` to datetime, and
   removes listings outside the NYC geographic boundaries.
3. **data_check** (`src/data_check`): pytest data quality tests. Checks column names
   and order, borough names, NYC longitude/latitude boundaries, price range, row
   count, and KL divergence of the borough distribution against a tagged reference
   dataset.
4. **data_split** (`components/train_val_test_split`): stratified train/validation/test
   split, logged as `trainval_data.csv` and `test_data.csv` artifacts.
5. **train_random_forest** (`src/train_random_forest`): trains a
   `RandomForestRegressor` on a sklearn pipeline (ordinal/one-hot encoding of
   categoricals, zero imputation of numerics, days-since-last-review feature, and
   TF-IDF on the listing title). Logs MAE, R2, and a feature importance plot, and
   exports the model in MLflow format.
6. **test_regression_model** (`components/test_regression_model`): evaluates the
   model export tagged `prod` against the test set. Run explicitly; it is not part
   of the default training flow.

An exploratory analysis notebook lives at `src/eda/EDA.ipynb`: it profiles the raw
data with pandas-profiling and records the observations that drove the cleaning rules.

## Repository structure

```
.
├── main.py                      # Pipeline entrypoint: selects and runs the steps
├── config.yaml                  # All pipeline parameters (prices, splits, model config)
├── environment.yml              # Canonical conda environment (nyc_airbnb_dev)
├── requirements.txt             # Pip-installable mirror of environment.yml
├── MLproject / conda.yml        # MLflow project definition for `mlflow run .`
├── src/
│   ├── basic_cleaning/          # Cleaning step (run.py, MLproject, conda.yml)
│   ├── data_check/              # Data quality tests (test_data.py, conftest.py)
│   ├── eda/                     # Exploratory data analysis notebook
│   └── train_random_forest/     # Model training (run.py, feature_engineering.py)
├── components/
│   ├── get_data/                # Download/upload raw data artifact
│   ├── train_val_test_split/    # Train/validation/test split
│   ├── test_regression_model/   # Test the prod model on the held-out set
│   └── wandb_utils/             # Shared W&B artifact helpers
├── cookie-mlflow-step/          # Cookiecutter template for new pipeline steps
└── images/                      # Screenshots used by the docs
```

## Setup

1. Install conda (Miniconda or Anaconda), then create the environment:

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

Pip-only users can install the equivalent pinned dependencies instead:

```bash
pip install -r requirements.txt
```

Note: `environment.yml` is the canonical environment for this project.

2. Log in to Weights & Biases with your API key from https://wandb.ai/authorize:

```bash
wandb login
```

## Running the pipeline

Run the full pipeline from the repository root:

```bash
mlflow run .
```

Run selected steps only:

```bash
mlflow run . -P steps=download
mlflow run . -P steps=download,basic_cleaning
```

Override configuration values with Hydra syntax:

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.n_estimators=200 etl.min_price=50"
```

Multi-run hyperparameter search (vary two parameters at once):

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.5,1.0 -m"
```

Test the production model (requires a model export manually tagged `prod` in W&B first):

```bash
mlflow run . -P steps=test_regression_model
```

Train on a new data sample:

```bash
mlflow run . -P steps=download,basic_cleaning,data_check,data_split,train_random_forest \
  -P hydra_options="etl.sample='sample2.csv'"
```

Run the EDA notebook:

```bash
mlflow run src/eda
```

then open the `EDA.ipynb` notebook in the launched Jupyter instance.

## What this project demonstrates

* MLflow projects and multi-step pipelines with per-step conda environments
* Hydra configuration management, including multi-run hyperparameter sweeps
* Data validation as a pipeline stage (pytest data tests with KL-divergence drift checks)
* Artifact versioning and model promotion with Weights & Biases
* Baseline regression modeling: preprocessing, feature engineering, TF-IDF text features,
  and Random Forest evaluation (MAE/R2, feature importance)

The modeling is intentionally a baseline; the focus is on the MLOps workflow. With more
effort on cleaning and features, significantly better models are achievable on this dataset.

## License

[MIT License](LICENSE.txt)
