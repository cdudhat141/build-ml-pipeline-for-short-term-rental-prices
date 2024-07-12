#!/usr/bin/env python
"""
This script splits the provided dataframe into train/validation and test sets.
"""
import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(project="build-ml-pipeline-for-short-term-rental-prices-nyc_airbnb_dev_components_get_data", job_type="train_val_test_split")
    run.config.update(vars(args))

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting train/validation and test")
    stratify_col = df[args.stratify_by] if args.stratify_by != 'none' else None
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify_col,
    )

    # Save to output files
    for df_split, split_name in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {split_name}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            df_split.to_csv(fp.name, index=False)
            artifact = wandb.Artifact(
                name=f"{split_name}_data.csv",
                type=f"{split_name}_data",
                description=f"{split_name} split of dataset"
            )
            artifact.add_file(fp.name)
            run.log_artifact(artifact)
            # Close and remove the temporary file
            fp.close()
            os.remove(fp.name)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/validation and test sets")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none'
    )

    args = parser.parse_args()
    go(args)
