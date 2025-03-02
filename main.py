# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.

import logging
import argparse

from src.validate_and_launch import validate_config_and_run_training


def setup_logging(level=logging.INFO):
    """Configure logging format and level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


setup_logging()

if __name__ == "__main__":

    usage_msg = (
        'docker run --cap-add SYS_NICE -it --rm --gpus "device = 0" '
        "-v <path_to_data_dir_on_host>:/data "
        "-v <path_to_model_output_dir_on_host>:/trained_models "
        "financial-fraud-training --config <path_to_training_config_on_host.json>"
    )
    parser = argparse.ArgumentParser(
        description="This program requires a training configuration file in JSON format "
        "Provide the path to your training configuration file using the --config option.",
        usage=usage_msg,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a training configuration file in JSON format.",
    )

    args = parser.parse_args()

    validate_config_and_run_training(args.config)
