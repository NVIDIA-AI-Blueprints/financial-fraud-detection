import argparse
import json
import sys
import os


from pydantic import TypeAdapter, ValidationError

from config_schema import FullConfig, GPUOption, ModelType
from train_gnn_based_xgboost import run_sg_embedding_based_xgboost
from train_xgboost import run_sg_xgboost_training


def load_config(path: str) -> FullConfig:
    with open(path, 'r') as f:
        data = json.load(f)
    adapter = TypeAdapter(FullConfig)
    validated = adapter.validate_python(data)
    return validated

if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser(description="Path to JSON training configuration file.")
        parser.add_argument(
            "--config",
            help="Path to the JSON training configuration file."
        )
            
        args = parser.parse_args()
        config_file = args.config

        configuration = load_config(config_file)

        # If we reach here, no ValidationError occurred
        print("Validation succeeded!")
    
    
    except (ValueError, ValidationError) as e:
        print("\nValidation error or incorrect format:")
        print(e)
        exit(0)

    for idx, user_config in enumerate(configuration.models):
        if user_config.kind == ModelType.XGB.value:
            if user_config.gpu == GPUOption.SINGLE.value:
                run_sg_xgboost_training(
                    data_dir=configuration.paths.data_dir,
                    output_dir=configuration.paths.output_dir,
                    input_config=user_config,
                    model_index=idx)
            else:
                assert(user_config.gpu == GPUOption.MULTIGPU.value)
                print('------- Multi-GPU XGBoost traning is not yet ready.-------')
        elif user_config.kind == ModelType.GRAPH_SAGE_XGB.value:
            if user_config.gpu == GPUOption.SINGLE.value:
                #TODO: Read it from meta file

                path_to_gnn_data = os.path.join(configuration.paths.data_dir, "gnn")

                file_containing_nr_tx_nodes = 'info.json'
                if os.path.exists(path_to_gnn_data):
                    for file_name in ['edges.csv', 'labels.csv', 'features.csv', file_containing_nr_tx_nodes]:
                        if not os.path.exists(os.path.join(path_to_gnn_data, file_name)):
                            sys.exit(f'{file_name} does not exist in {path_to_gnn_data}')
                else:
                    sys.exit(f'{path_to_gnn_data} does not exist.')


                # Read number of transactions from info.json
                try:
                    with open(os.path.join(path_to_gnn_data, file_containing_nr_tx_nodes), 'r') as file:
                        json_data = json.load(file)
                except FileNotFoundError:
                    print(f'Could not find {file_containing_nr_tx_nodes}. Exiting...', file=sys.stderr)
                    sys.exit(1)
                except json.JSONDecodeError:
                    print(f'Invalid JSON in {file_containing_nr_tx_nodes} . Exiting...', file=sys.stderr)
                    sys.exit(1)

                if "NUM_TRANSACTION_NODES" not in json_data:
                    print(f'Key NUM_TRANSACTION_NODES not found in {file_containing_nr_tx_nodes}. Exiting...', file=sys.stderr)
                    sys.exit(1)

                num_transaction_nodes = json_data["NUM_TRANSACTION_NODES"]

                print(f' num_transaction_nodes {num_transaction_nodes}')

                run_sg_embedding_based_xgboost(
                    configuration.paths.data_dir,
                    configuration.paths.output_dir,
                    user_config,
                    num_transaction_nodes,
                    model_index=idx
                    )
            else:
                assert(user_config.gpu == GPUOption.MULTIGPU.value)
                print('------- GraphSAGE MG training is not yet ready.-------')
