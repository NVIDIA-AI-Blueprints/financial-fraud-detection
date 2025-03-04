# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.

import itertools
import os
import logging
import sys

import cupy
import cudf
import xgboost as xgb

from cuml.metrics import confusion_matrix
from cuml.metrics.accuracy import accuracy_score

from sklearn.metrics import f1_score, precision_score, recall_score

from typing import List, Union

from src.config_schema import (
    XGBSingleConfig,
    XGBListConfig,
    XGBHyperparametersList,
    XGBHyperparametersSingle,
)

from utils.triton_model_repo_generator import create_triton_repo_for_xgboost


def f1_eval_gpu(predictions, labels):
    f1 = f1_score(labels, predictions)
    return "f1", f1


def read_data(file_path: str):
    """
    Read a data file using cuDF based on the file extension.
    Supported extensions: .csv, .parquet, .orc
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return cudf.read_csv(file_path)
    elif ext == ".parquet":
        return cudf.read_parquet(file_path)
    elif ext == ".orc":
        df = cudf.read_orc(file_path)
        if "__index_level_0__" in df.columns:
            df = df.drop(columns="__index_level_0__")
        return df
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def evaluate_on_unseen_data(
    xgb_model:xgb.Booster, test_data_path:str, threshold: float = 0.5
):

    # Read the test data
    test_df = read_data(test_data_path)

    # Assume the target column is the last column
    target_col_name = test_df.columns[-1]

    # Prepare the DMatrix for XGBoost by dropping the target column
    dnew = xgb.DMatrix(test_df.drop(target_col_name, axis=1))

    # Make predictions
    y_pred_prob = xgb_model.predict(dnew)
    y_pred = (y_pred_prob >= threshold).astype(int)
    y_test = test_df[target_col_name].values.astype(int)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    y_test = cupy.asnumpy(y_test)

    # Precision
    precision = precision_score(y_test, y_pred)

    # Recall
    recall = recall_score(y_test, y_pred)

    # F1 Score
    f1 = f1_score(y_test, y_pred)

    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Confusion Matrix: {conf_mat}")


def tran_sg_xgboost(
    param_combinations: List[XGBHyperparametersSingle],
    data_dir: str,
    random_state: int = 42,
    decision_threshold: float = 0.5,
    verbose: bool = False,
):

    logging.info(f"-----Running XGBoost training-----")

    xgb_data_dir = os.path.join(data_dir, "xgb")

    # Get all file names in xgb_data_dir
    all_files = os.listdir(xgb_data_dir)
    valid_extensions = {".csv", ".parquet", ".orc"}

    # Look for files with the base names "training" and "validation"
    training_file = None
    validation_file = None

    for filename in all_files:
        base, ext = os.path.splitext(filename)
        if ext.lower() in valid_extensions:
            if base == "training":
                training_file = filename
            elif base == "validation":
                validation_file = filename

    # If training file is not found, log an error and exit
    if training_file is None:
        logging.error(f"Training file (training.<ext>) not found in the {xgb_data_dir} directory.")
        sys.exit(1)

    # Read the training data file
    training_path = os.path.join(xgb_data_dir, training_file)
    df_train_full = read_data(training_path)

    # If a validation file exists, read it; otherwise, split training data 80/20
    if validation_file is not None:
        validation_path = os.path.join(xgb_data_dir, validation_file)
        df_val = read_data(validation_path)
        df_train = df_train_full
    else:
        # Create an 80/20 train/validation split
        df_train = df_train_full.sample(frac=0.8, random_state=42)
        df_val = df_train_full.drop(df_train.index)

    # Assume that the target column is the last column in the training dataframe
    target_col_name = df_train_full.columns[-1]

    # Split training data into features (X_train) and labels (y_train)
    y_train = df_train[target_col_name].astype(int)
    X_train = df_train.drop(target_col_name, axis=1)
    nr_input_features = X_train.shape[1]

    # Split validation data into features (X_val) and labels (y_val)
    y_val = df_val[target_col_name].astype(int)
    X_val = df_val.drop(target_col_name, axis=1)

    logging.debug(f"Training data   shape:  X = {X_train.shape} y = {y_train.shape}")
    logging.debug(f"Validation data shape:  X = {X_val.shape} y = {y_val.shape}")

    # Convert the training and validation data to DMatrix
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
            logging.info(
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
        logging.info(f"Best hyperparameters {best_params}")

    y_val_pred = (bst.predict(deval) >= decision_threshold).astype(int)
    f1_eval_gpu(y_val_pred, deval.get_label())

    # Train the final model using the best parameters and best number of
    # boosting rounds
    dtrain = xgb.DMatrix(
        data=cudf.concat([X_train, X_val], axis=0),
        label=cudf.concat([y_train, y_val], axis=0),
    )
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=best_num_boost_round)
    

    # Look for the test file with base name 'test' and a valid extension
    test_file = None
    for filename in os.listdir(xgb_data_dir):
        base, ext = os.path.splitext(filename)
        if base == "test" and ext.lower() in valid_extensions:
            test_file = filename
            break
    
    # If test file is not found, log error and exit
    if test_file is None:
        logging.info("No test file (test.<ext>) is found in directory: %s", xgb_data_dir)
    else:
        logging.info("Evaluating the model on: %s", os.path.join(xgb_data_dir, test_file))

        evaluate_on_unseen_data(final_model, os.path.join(xgb_data_dir, test_file))

    return final_model, nr_input_features


def run_sg_xgboost_training(
    data_dir: str,
    model_dir: str,
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

    xgb_model, nr_input_features = tran_sg_xgboost(
        hyperparameter_list, data_dir)
    model_file_name = f"{input_config.kind}_{model_index}.json"
    create_triton_repo_for_xgboost(
        xgb_model,
        model_dir,
        model_file_name,
        nr_input_features,
        0.5,
        "xgb_model_repository",
    )
