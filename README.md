# Short-Term Rental Prices in NYC

An ML pipeline that cleans, tests and splits NYC short-term rental data and trains a random forest price model. MLflow runs the steps, Hydra handles config, and Weights & Biases tracks data and models.

## What's inside

- `main.py` and `config.yaml` define the pipeline
- `src/` holds cleaning, data checks, EDA and training
- the download, split and test steps come from Udacity's starter components

## How to run

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
mlflow run .
```

You need a Weights & Biases account (`wandb login`).
