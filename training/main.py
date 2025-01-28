from config_schema import FullConfig, ModelType, GPUOption
from train_xgboost import run_sg_xgboost_training
from train_gnn_based_xgboost import run_sg_embedding_based_xgboost
from pydantic import ValidationError, TypeAdapter
import argparse
import json

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
                num_transaction_nodes=281063

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
