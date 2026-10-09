#!/usr/bin/env python
"""Upload a local sample CSV as a Weights & Biases artifact.

Reads the requested sample from the ``data/`` directory next to this script
and logs it to W&B under the given artifact name, so downstream steps can
fetch it with ``run.use_artifact``.
"""
import argparse
import logging
import os

import wandb

from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """Run the get_data step.

    :param args: parsed command line arguments with ``sample``,
        ``artifact_name``, ``artifact_type`` and ``artifact_description``
    """
    run = wandb.init(job_type="download_file")
    run.config.update(args)

    logger.info(f"Returning sample {args.sample}")
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        os.path.join("data", args.sample),
        run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a sample CSV as a W&B artifact")

    parser.add_argument("sample", type=str, help="Name of the sample to download")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()

    go(args)
