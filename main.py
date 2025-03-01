# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
import os
import sys
import logging
import argparse
import json

from pydantic import TypeAdapter, ValidationError

from src.config_schema import FullConfig, GPUOption, ModelType
from src.train_gnn_based_xgboost import run_sg_embedding_based_xgboost
from src.train_xgboost import run_sg_xgboost_training


def setup_logging(level=logging.INFO):
    """Configure logging format and level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


setup_logging()


def load_config(path: str) -> FullConfig:
    """
    Read training configuration file.
    """
    with open(path, "r") as f:
        data = json.load(f)
    adapter = TypeAdapter(FullConfig)
    validated = adapter.validate_python(data)
    return validated


if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser(
            description="This program requires a training configuration file in JSON format "
            "Provide the path to your training configuration file using the --config option.",
            usage="--config /path/to/training_config.json",
        )

        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to a training configuration JSON file",
        )

        args = parser.parse_args()

        if not args.config:
            logging.error("--config argument is required.")
            sys.exit(0)

        if not os.path.exists(args.config):
            logging.error(f" {args.config} file doesn't exist")
            sys.exit(0)

        logging.info(f"Using configuration file: {args.config}")

        configuration = load_config(args.config)

        # If we reach here, no ValidationError occurred
        logging.info(
            "The provided training configuration has been successfully validated."
        )

    except (ValueError, ValidationError) as e:
        logging.error("Validation of the input training configuration file failed.")
        logging.error(f"Full validation error message = {e}")
        exit(0)

    for idx, user_config in enumerate(configuration.models):
        if user_config.kind == ModelType.XGB.value:
            if user_config.gpu == GPUOption.SINGLE.value:
                run_sg_xgboost_training(
                    data_dir=configuration.paths.data_dir,
                    model_dir=configuration.paths.output_dir,
                    input_config=user_config,
                    model_index=idx,
                )
            else:
                assert user_config.gpu == GPUOption.MULTIGPU.value
                logging.info(
                    "------- Multi-GPU XGBoost traning is not yet ready.-------"
                )
        elif user_config.kind == ModelType.GRAPH_SAGE_XGB.value:
            if user_config.gpu == GPUOption.SINGLE.value:
                run_sg_embedding_based_xgboost(
                    configuration.paths.data_dir,
                    configuration.paths.output_dir,
                    user_config,
                    model_index=idx,
                )
            else:
                assert user_config.gpu == GPUOption.MULTIGPU.value
                logging.info("------- GraphSAGE MG training is not yet ready.-------")
