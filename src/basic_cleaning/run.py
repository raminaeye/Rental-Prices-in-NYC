#!/usr/bin/env python
"""Apply basic cleaning to the raw rental listings and export a new artifact.

Downloads the raw CSV artifact from Weights & Biases, drops price outliers,
converts ``last_review`` to datetime, removes listings outside the NYC
geographic boundaries, and logs the cleaned CSV back to W&B.
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """Run the basic cleaning step.

    :param args: parsed command line arguments with ``input_artifact``,
        ``output_artifact``, ``output_type``, ``output_description``,
        ``min_price`` and ``max_price``
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)

    # Drop outliers
    logger.info("Drop outlier")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Convert String to Datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Saving Clean Sample DF")
    # Drop rows outside the NYC area (seen as bad GPS coordinates in sample2.csv)
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact to fetch from W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="W&B artifact type for the output",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the cleaned output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum accepted nightly price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum accepted nightly price",
        required=True
    )

    args = parser.parse_args()

    go(args)
