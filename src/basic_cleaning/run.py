#!/usr/bin/env python
"""
Download raw dataset from W&B, apply basic data cleaning, and export the result as a new artifact.
"""
import argparse
import logging
import os

import pandas as pd
import wandb

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

def download_and_clean_data(args):
    # Initialize a W&B run
    run = wandb.init(project= "build-ml-pipeline-for-short-term-rental-prices-nyc_airbnb_dev_components_get_data",job_type="basic_cleaning")
    run.config.update(vars(args))

    # Download the input artifact
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Drop outliers based on price
    logger.info("Dropping outliers based on price")
    df = df[df['price'].between(args.min_price, args.max_price)].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Keep only the rows within the specified geolocation
    logger.info("Filtering rows by geolocation")
    df = df[df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)].copy()

    # Save the cleaned dataset
    cleaned_file_name = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {cleaned_file_name}")
    df.to_csv(cleaned_file_name, index=False)

    # Create a new artifact for the cleaned datas
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(cleaned_file_name)

    # Log the new artifact
    logger.info("Logging artifact")
    run.log_artifact(artifact)

    # Clean up the local file
    os.remove(cleaned_file_name)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Perform basic data cleaning on input dataset.")

    parser.add_argument(
        "--input_artifact", type=str, required=True,
        help="Fully-qualified name for the input artifact"
    )
    parser.add_argument(
        "--output_artifact", type=str, required=True,
        help="Name of the output artifact"
    )
    parser.add_argument(
        "--output_type", type=str, required=True,
        help="Type of the output artifact"
    )
    parser.add_argument(
        "--output_description", type=str, required=True,
        help="Description for the output artifact"
    )
    parser.add_argument(
        " ", type=float, required=True,
        help="Minimum price for cleaning outliers"
    )
    parser.add_argument(
        "--max_price", type=float, required=True,
        help="Maximum price for cleaning outliers"
    )

    args = parser.parse_args()

    download_and_clean_data(args)
