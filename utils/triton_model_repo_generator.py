# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.

import os
import shutil
import logging
import torch
import xgboost as xgb

from src.gnn_models import GraphSAGE
from utils.proto_text_generator import (
    generate_xgb_pbtxt,
    generate_gnn_pbtxt,
    generate_python_backend_pbtxt,
)


def create_triton_repo_for_gnn_based_xgboost(
    model: GraphSAGE,
    xgb_model: xgb.Booster,
    output_dir: str,
    gnn_file_name: str,
    xgb_model_file_name: str,
    decision_threshold: float,
    model_repository_name: str = "model_repository",
):
    """
    Create a Triton Inference Server model repository containing both a GraphSAGE and an XGBoost model.

    This function generates a directory structure compatible with Triton Inference Server that includes
    the artifacts for two models: a GraphSAGE (GNN) model and an XGBoost model. This repository can then
    be deployed with Triton Inference Server for serving predictions.

    Args:
        model (GraphSAGE):
            The GraphSAGE model instance that will  produce embeddings of input transactions.
        xgb_model (xgb.Booster):
            An XGBoost model instance that will produce fraud scores based on embeddings produced
            by the GraphSAGE model
        output_dir (str):
            The path to the directory the model repository will be saved.
        gnn_file_name (str):
            The filename to save the GraphSAGE model.
        xgb_model_file_name (str):
            The filename to save the XGBoost model.
        decision_threshold (float):
            A threshold value applied during inference to determine the final decision.
        model_repository_name (str, optional):
            The name of the model repository directory. Defaults to "model_repository".
    """

    model.eval()

    # Generate random input tensors
    num_nodes = 64
    num_features = model.in_channels
    num_edges = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random node features: shape [num_nodes, num_features]
    x = torch.randn(num_nodes, num_features).to(device)

    # Random edge index: shape [2, num_edges], values in [0, num_nodes)
    edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)

    return_hidden = True

    # Prepare the example input as a tuple (or list) matching the model's forward signature
    example_input = (x, edge_index, return_hidden)

    gnn_repository_path = os.path.join(output_dir, model_repository_name, "model")
    xgb_repository_path = os.path.join(output_dir, model_repository_name, "xgboost")

    gnn_model_dir = os.path.join(gnn_repository_path, "1")
    xgb_model_dir = os.path.join(xgb_repository_path, "1")

    os.makedirs(gnn_model_dir, exist_ok=True)
    os.makedirs(xgb_model_dir, exist_ok=True)
    path_to_onnx_model = os.path.join(gnn_model_dir, gnn_file_name)
    path_to_xgboost_model = os.path.join(xgb_model_dir, xgb_model_file_name)

    torch.onnx.export(
        model,  # The scripted model with dynamic control flow
        example_input,  # Example input for tracing the model's graph
        path_to_onnx_model,
        export_params=True,  # Include model parameters in the ONNX file
        opset_version=11,  # ONNX opset version (11+ supports control flow)
        do_constant_folding=True,  # Perform constant folding for optimization
        input_names=["x", "edge_index"],
        output_names=["output"],  # (Optional) Name for the output tensor
        dynamic_axes={
            "x": {0: "batch_size"},
            "edge_index": {1: "num_edges"},
        },
    )

    logging.info(
        f"------Saving ONNX model repository in {os.path.join(output_dir, model_repository_name)}-----"
    )

    logging.info(f"Saved GraphSAGE model to {path_to_onnx_model}")

    xgb_model.save_model(path_to_xgboost_model)

    logging.info(f"Saved XGBoost model to {path_to_xgboost_model}")

    generate_gnn_pbtxt(
        gnn_file_name,
        model.in_channels,
        model.hidden_channels,
        os.path.join(gnn_repository_path, "config.pbtxt"),
    )

    generate_xgb_pbtxt(
        xgb_model_file_name,
        model.hidden_channels + model.in_channels,
        1,
        decision_threshold,
        os.path.join(xgb_repository_path, "config.pbtxt"),
    )


def create_triton_repo_for_xgboost(
    xgb_model: xgb.Booster,
    output_dir: str,
    xgb_model_file_name: str,
    nr_input_features: int,
    decision_threshold: float,
    model_repository_name: str = "model_repository",
):
    """
    Create a Triton Inference Server model repository containing an XGBoost model.

    This function generates a directory structure compatible with Triton Inference Server that includes
    a model file and config protocol buffer text for an XGBoost model. This repository can then
    be deployed with Triton Inference Server for serving predictions.

    Args:

        xgb_model (xgb.Booster):
            An XGBoost model instance that will produce fraud scores based on embeddings produced
            by the GraphSAGE model
        output_dir (str):
            The path to the directory the model repository will be saved.
        xgb_model_file_name (str):
            The filename to save the XGBoost model.
        decision_threshold (float):
            A threshold value applied during inference to determine the final decision.
        model_repository_name (str, optional):
            The name of the model repository directory. Defaults to "model_repository".
    """

    xgb_repository_path = os.path.join(output_dir, model_repository_name, "xgboost")
    xgb_model_dir = os.path.join(xgb_repository_path, "1")
    os.makedirs(xgb_model_dir, exist_ok=True)
    path_to_xgboost_model = os.path.join(xgb_model_dir, xgb_model_file_name)

    logging.info(
        f"------Saving model repository in {os.path.join(output_dir, model_repository_name)}-----"
    )

    xgb_model.save_model(path_to_xgboost_model)

    logging.info(f"Saved XGBoost model to {path_to_xgboost_model}")

    generate_xgb_pbtxt(
        xgb_model_file_name,
        nr_input_features,
        1,
        decision_threshold,
        os.path.join(xgb_repository_path, "config.pbtxt"),
    )


def create_repo_for_python_backend(
    gnn_model: GraphSAGE,
    xgb_model: xgb.Booster,
    output_dir: str,
    gnn_sate_dict_filename: str,
    xgb_model_filename: str,
    model_repository_name: str = "python_backend_model_repository",
    model_name: str = "prediction_and_shapley",
):
    """
    Create a Triton Inference Server model repository for python backed, containing a GraphSAGE and an XGBoost model.

    Args:

        xgb_model (xgb.Booster):
            An XGBoost model instance that will produce fraud scores based on embeddings produced
            by the GraphSAGE model
        output_dir (str):
            The path to the directory the model repository will be saved.
        xgb_model_file_name (str):
            The filename to save the XGBoost model.
        decision_threshold (float):
            A threshold value applied during inference to determine the final decision.
        model_repository_name (str, optional):
            The name of the model repository directory. Defaults to "model_repository".
    """

    python_backend_repository_path = os.path.join(
        output_dir, model_repository_name, model_name
    )
    model_dir = os.path.join(python_backend_repository_path, "1")
    os.makedirs(model_dir, exist_ok=True)

    path_to_state_dict = os.path.join(model_dir, gnn_sate_dict_filename)
    path_to_xgboost_model = os.path.join(model_dir, xgb_model_filename)

    logging.info(
        f"------Saving model repository for python backend in {os.path.join(output_dir, model_repository_name)}-----"
    )

    torch.save(gnn_model.state_dict(), path_to_state_dict)
    logging.info(f"Saved GraphSAGE model state dict to {path_to_state_dict}")

    xgb_model.save_model(path_to_xgboost_model)
    logging.info(f"Saved XGBoost model to {path_to_xgboost_model}")

    current_directory = os.path.dirname(os.path.abspath(__file__))

    shutil.copy(
        os.path.join(current_directory, "python_backend_model.py"),
        os.path.join(model_dir, "model.py"),
    )

    generate_python_backend_pbtxt(
        model_name,
        gnn_sate_dict_filename,
        xgb_model_filename,
        gnn_model.in_channels,
        gnn_model.hidden_channels,
        gnn_model.out_channels,
        gnn_model.n_hops,
        1,
        os.path.join(python_backend_repository_path, "config.pbtxt"),
    )
