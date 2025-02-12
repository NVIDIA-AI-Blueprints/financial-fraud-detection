# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json

from pydantic import TypeAdapter, ValidationError

from src.config_schema import FullConfig, GPUOption, ModelType
from src.train_gnn_based_xgboost import run_sg_embedding_based_xgboost
from src.train_xgboost import run_sg_xgboost_training


def load_config(path: str) -> FullConfig:
    with open(path, "r") as f:
        data = json.load(f)
    adapter = TypeAdapter(FullConfig)
    validated = adapter.validate_python(data)
    return validated


if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser(
            description="Path to JSON training configuration file."
        )
        parser.add_argument(
            "--config", help="Path to the JSON training configuration file."
        )

        args = parser.parse_args()
        config_file = args.config

        configuration = load_config(config_file)

        # If we reach here, no ValidationError occurred
        print("The provided training configuration has been successfully validated.")

    except (ValueError, ValidationError) as e:
        print("\nValidation error or incorrect format:")
        print(e)
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
                print("------- Multi-GPU XGBoost traning is not yet ready.-------")
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
                print("------- GraphSAGE MG training is not yet ready.-------")
