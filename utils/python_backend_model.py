# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.

import triton_python_backend_utils as pb_utils
import xgboost as xgb
from captum.attr import ShapleyValueSampling
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import torch
import json
import os
import sys
import subprocess


def install_package(package, version=None):
    """
    Attempts to import `package`. If not installed, installs it using pip.
    If `version` is provided, installs `package==version`.
    """
    try:
        __import__(package)
        print(f"'{package}' is already installed.")
    except ImportError:
        # Build the package spec depending on whether a specific version is
        # supplied
        package_spec = f"{package}=={version}" if version else package
        print(f"Installing {package_spec} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_spec])
        print(f"Installation of {package_spec} finished!")


install_package("torch", "2.6.0")
install_package("torch-geometric", "2.6.0")
install_package("xgboost", "2.1.4")
install_package("captum", "0.7.0")


# Triton Python backend utilities.


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for graph-based learning.

    This model learns node embeddings by aggregating information from a node's
    neighborhood using multiple graph convolutional layers.

    Parameters:
    ----------
    in_channels : int
        The number of input features for each node.
    hidden_channels : int
        The number of hidden units in each layer, controlling
        the embedding dimension.
    out_channels : int
        The number of output features (or classes) for the final layer.
    n_hops : int
        The number of GraphSAGE layers (or hops) used to aggregate information
        from neighboring nodes.
    dropout_prob : float, optional (default=0.25)
        The probability of dropping out nodes during training for
        regularization.
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, n_hops, dropout_prob=0.25
    ):
        super(GraphSAGE, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_hops = n_hops

        # list of conv layers
        self.convs = nn.ModuleList()
        # add first conv layer to the list
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # add the remaining conv layers to the list
        for _ in range(n_hops - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # output layer
        self.fc = nn.Linear(hidden_channels + in_channels, out_channels)
        self.dropout_prob = dropout_prob

    def forward(self, x, edge_index, return_hidden: bool = False):

        for conv in self.convs:
            embeddings = conv(x, edge_index)
            embeddings = F.relu(embeddings)
            embeddings = F.dropout(
                embeddings, p=self.dropout_prob, training=self.training
            )
        if return_hidden:
            return torch.cat((x, embeddings), dim=1)
        else:
            return self.fc(torch.cat((x, embeddings), dim=1))


class TritonPythonModel:
    def initialize(self, args):
        # Parse the model configuration passed in from Triton.
        model_config = json.loads(args["model_config"])
        parameters = model_config["parameters"]

        # Directly index the parameters (no default values). A KeyError will be
        # raised if any key is missing.
        self.in_channels = int(parameters["in_channels"]["string_value"])
        self.hidden_channels = int(
            parameters["hidden_channels"]["string_value"])
        self.out_channels = int(parameters["out_channels"]["string_value"])
        self.n_hops = int(parameters["n_hops"]["string_value"])

        self.embedder_state_dict_filename = parameters[
            "embedding_generator_model_state_dict"
        ]["string_value"]
        self.xgb_model_filename = parameters["embeddings_based_xgboost_model"][
            "string_value"
        ]

        feature_mask_config = pb_utils.get_input_config_by_name(
            model_config, "FEATURE_MASK"
        )

        prediction_config = pb_utils.get_output_config_by_name(
            model_config, "PREDICTION"
        )
        shap_config = pb_utils.get_output_config_by_name(
            model_config, "SHAP_VALUES")

        self.feature_mask_dtype = pb_utils.triton_string_to_numpy(
            feature_mask_config["data_type"]
        )

        self.prediction_dtype = pb_utils.triton_string_to_numpy(
            prediction_config["data_type"]
        )
        self.shap_dtype = pb_utils.triton_string_to_numpy(
            shap_config["data_type"])

        self.device = torch.device("cuda")
        self.model = GraphSAGE(
            self.in_channels, self.hidden_channels, self.out_channels, self.n_hops
        )
        # Load the pre-trained state dict.
        current_directory = os.path.dirname(os.path.abspath(__file__))
        state_dict = torch.load(
            os.path.join(current_directory, self.embedder_state_dict_filename),
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.bst = xgb.Booster()
        self.bst.load_model(
            os.path.join(
                current_directory,
                self.xgb_model_filename))
        self.bst.set_param({"tree_method": "hist", "device": "cuda"})

    def execute(self, requests):
        responses = []
        for request in requests:

            node_features_numpy = pb_utils.get_input_tensor_by_name(
                request, "NODE_FEATURES"
            ).as_numpy()
            edge_index_numpy = pb_utils.get_input_tensor_by_name(
                request, "EDGE_INDEX"
            ).as_numpy()
            compute_shap_numpy = pb_utils.get_input_tensor_by_name(
                request, "COMPUTE_SHAP"
            ).as_numpy()
            feature_mask_numpy = pb_utils.get_input_tensor_by_name(
                request, "FEATURE_MASK"
            ).as_numpy()

            embeddings = self.model(
                torch.as_tensor(node_features_numpy, device=self.device),
                torch.as_tensor(edge_index_numpy, device=self.device),
                True,
            )
            y_pred_prob = self.bst.predict(
                xgb.DMatrix(embeddings.detach()))[:, None]
            if compute_shap_numpy[0]:

                def forward_function(node_x_tensor):
                    embeddings = self.model(
                        node_x_tensor.to(self.device),
                        torch.as_tensor(edge_index_numpy, device=self.device),
                        True,
                    )
                    return torch.tensor(
                        self.bst.predict(xgb.DMatrix(embeddings.detach()))
                    )

                shapley_sampler = ShapleyValueSampling(forward_function)
                x_input = torch.as_tensor(
                    node_features_numpy[:1]).to(torch.float32)
                baseline = torch.zeros_like(x_input)

                # Compute Shapley attributions
                attributions = shapley_sampler.attribute(
                    x_input,
                    baselines=baseline,
                    feature_mask=torch.tensor(
                        feature_mask_numpy).to(torch.int32),
                    n_samples=128,
                )
            else:
                attributions = torch.zeros((1, self.in_channels))

            # Prepare response

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "PREDICTION",
                        y_pred_prob.astype(self.prediction_dtype),
                    ),
                    pb_utils.Tensor(
                        "SHAP_VALUES",
                        attributions.numpy().astype(self.shap_dtype),
                    ),
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        pass
