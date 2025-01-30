from collections import defaultdict
import itertools
import os

import cupy
import cudf
import matplotlib.pyplot as plt
import xgboost as xgb

from cuml.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
from cuml.metrics.accuracy import accuracy_score
from cuml.model_selection import train_test_split

from sklearn.metrics import auc, f1_score, precision_score, recall_score

from typing import List, Literal, Tuple, Union

from config_schema import (
    XGBSingleConfig,
    XGBListConfig,
    XGBHyperparametersList,
    XGBHyperparametersSingle,
)


def f1_eval_gpu(preds, dMat):
    labels = dMat.get_label()
    preds = (preds > 0.5).astype(int)  # Binary threshold
    f1 = f1_score(labels, preds)
    print(f"f1 = {f1}")
    return "f1", f1


def tran_sg_xgboost(
    param_combinations: List[XGBHyperparametersSingle],
    data_dir: str,
    output_dir: str,
    model_file_name: str,
    random_state: int = 42,
    verbose: bool = False,
):

    dataset_dir = data_dir
    xgb_data_dir = os.path.join(dataset_dir, "xgb")
    models_dir = output_dir

    train_data_path = os.path.join(xgb_data_dir, "training.csv")

    df = cudf.read_csv(train_data_path)

    # Target column
    target_col_name = df.columns[-1]

    # Split the dataframe into features (X) and labels (y)
    y = df[target_col_name]
    X = df.drop(target_col_name, axis=1)

    # Split data into train and test sets

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Convert the training and test data to DMatrix
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    deval = xgb.DMatrix(data=X_val, label=y_val)

    # Grid search for the best hyperparameters
    best_score = float("inf")  # Initialize best score
    best_params = None  # To store best hyperparameters

    for hyperparameter in param_combinations:

        # Create a dictionary of parameters
        params = {
            "max_depth": hyperparameter.max_depth,
            "learning_rate": hyperparameter.learning_rate,
            "gamma": hyperparameter.gamma,
            "num_parallel_tree": hyperparameter.num_parallel_tree,
            "eval_metric": "logloss",
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cuda",
            "seed": random_state,
        }

        evals_result = {}
        evals = [(dtrain, "train"), (deval, "val")]
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=hyperparameter.num_boost_round,
            early_stopping_rounds=4,
            evals=evals,
            evals_result=evals_result,
            verbose_eval=False,
        )

        # Get the evaluation score (logloss) on the validation set
        score = bst.best_score

        if verbose:
            print(params)
            print(
                f"Trained for {bst.best_iteration} rounds, last few train losses are"
                f"{evals_result['train']['logloss'][-3:]} and validation losses are"
                f"{evals_result['val']['logloss'][-3:]}"
            )

        # Update the best parameters if the current model is better
        if score < best_score:
            best_score = score
            best_params = params
            best_num_boost_round = bst.best_iteration

    if len(param_combinations) > 1:
        print(f"Best hyperparameters {best_params}")

    y_val_pred = (bst.predict(deval) >= 0.5).astype(int)
    f1_eval_gpu(y_val_pred, deval)

    # Train the final model using the best parameters and best number of boosting rounds
    dtrain = xgb.DMatrix(data=X, label=y)
    final_model = xgb.train(best_params, dtrain, num_boost_round=best_num_boost_round)

    # Save the best model
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    final_model.save_model(os.path.join(models_dir, model_file_name))
    print(f"Saved XGBoost model to {os.path.join(models_dir, model_file_name)}")


def run_sg_xgboost_training(
    data_dir: str,
    output_dir: str,
    input_config: Union[XGBSingleConfig, XGBListConfig],
    model_index: int,
) -> None:
    """
    This function does something with the validated Pydantic object.
    """
    # Generate all combinations of hyperparameters

    hyperparameter_list = []
    if isinstance(input_config.hyperparameters, XGBHyperparametersList):
        param_combinations = list(
            itertools.product(*input_config.hyperparameters.dict().values())
        )
        for params in param_combinations:
            hyperparameter_list.append(
                XGBHyperparametersSingle(
                    max_depth=params[0],
                    num_parallel_tree=params[1],
                    num_boost_round=params[2],
                    learning_rate=params[3],
                    gamma=params[4],
                )
            )
    elif isinstance(input_config.hyperparameters, XGBHyperparametersSingle):
        hyperparameter_list.append(input_config.hyperparameters)

    tran_sg_xgboost(
        hyperparameter_list,
        data_dir,
        output_dir,
        model_file_name=f"{input_config.kind}_{model_index}.json",
    )
