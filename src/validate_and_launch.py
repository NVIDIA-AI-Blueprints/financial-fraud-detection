# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.

import json
import logging
import os
import sys

from pydantic import TypeAdapter, ValidationError
from src.config_schema import FullConfig, GPUOption, ModelType

from src.train_gnn_based_xgboost import run_sg_embedding_based_xgboost
from src.train_xgboost import run_sg_xgboost_training


def load_config(path: str) -> FullConfig:
    """
    Read training configuration file.
    """
    with open(path, "r") as f:
        data = json.load(f)
    adapter = TypeAdapter(FullConfig)
    validated = adapter.validate_python(data)
    return validated


def validate_config_and_run_training(path_to_config_file):
    try:
        if path_to_config_file is None:
            logging.error(
                "Docker run is missing --config <path_to_training_config_on_host.json>")
            sys.exit(0)
        if not os.path.exists(path_to_config_file):
            logging.error(f" {path_to_config_file} file doesn't exist")
            sys.exit(0)

        logging.info(f"Using configuration file: {path_to_config_file}")

        configuration = load_config(path_to_config_file)
        logging.info(
            "The provided training configuration has been successfully validated."
        )

    except (ValueError, ValidationError) as e:
        logging.error(
            "Validation of the input training configuration file failed.")
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
                logging.info(
                    "------- GraphSAGE MG training is not yet ready.-------")
