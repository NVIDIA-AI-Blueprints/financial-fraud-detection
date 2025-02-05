# General-purpose libraries and OS handling
import os
import sys
import json
import pickle


from collections import namedtuple
from typing import List, Tuple, Dict, Union
from enum import Enum

from config_schema import (
    GraphSAGEHyperparametersSingle,
    GraphSAGEHyperparametersList,
    GraphSAGEAndXGB,
    GraphSAGEGridAndXGB,
    GraphSAGEAndXGBConfig,
    GraphSAGEGridAndXGBConfig,
    XGBHyperparametersSingle,
)

# GPU-accelerated libraries (torch, cupy, cudf, rmm)
import torch
import cupy
import cudf
import rmm
from cuml.metrics import confusion_matrix

from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.torch import rmm_torch_allocator

# Reinitialize RMM and set allocators to manage memory efficiently on GPU
rmm.reinitialize(devices=[0], pool_allocator=True, managed_memory=True)
cupy.cuda.set_allocator(rmm_cupy_allocator)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

# PyTorch and related libraries
import torch.nn.functional as F
import torch.nn as nn

# PyTorch Geometric and cuGraph libraries for GNNs and graph handling
import cugraph_pyg
from cugraph_pyg.loader import NeighborLoader
import torch_geometric
from torch_geometric.nn import SAGEConv

# Enable GPU memory spilling to CPU with cuDF to handle larger datasets
from cugraph.testing.mg_utils import enable_spilling  # noqa: E402

enable_spilling()

# XGBoost for machine learning model building
import xgboost as xgb

# Numerical operations with cupy and numpy
import cupy as cp
import numpy as np
import random

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from torch.utils.dlpack import to_dlpack

GraphSAGEModelConfig = namedtuple(
    "GraphSAGEModelConfig", ["in_channels", "out_channels"]
)


HyperParams = namedtuple(
    "HyperParams",
    [
        "n_folds",
        "n_hops",
        "fan_out",
        "batch_size",
        "metric",
        "learning_rate",
        "dropout_prob",
        "hidden_channels",
        "num_epochs",
        "weight_decay",
    ],
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        # list of conv layers
        self.convs = nn.ModuleList()
        # add first conv layer to the list
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # add the remaining conv layers to the list
        for _ in range(n_hops - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # output layer
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout_prob = dropout_prob

    def forward(self, x, edge_index, return_hidden: bool = False):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if return_hidden:
            return x
        else:
            return self.fc(x)


def train_gnn(model, loader, optimizer, criterion) -> float:
    """
    Trains the GraphSAGE model for one epoch.

    Parameters:
    ----------
    model : torch.nn.Module
        The GNN model to be trained.
    loader : tcugraph_pyg.loader.NeighborLoader
        DataLoader that provides batches of graph data for training.
    optimizer : torch.optim.Optimizer
        Optimizer used to update the model's parameters.
    criterion : torch.nn.Module
        Loss function used to calculate the difference between predictions and targets.

    Returns:
    -------
    float
        The average training loss over all batches for this epoch.
    """
    model.train()
    total_loss = 0
    batch_count = 0
    for batch in loader:
        batch_count += 1
        optimizer.zero_grad()

        batch_size = batch.batch_size
        out = model(batch.x[:, :].to(torch.float32), batch.edge_index)[:batch_size]
        y = batch.y[:batch_size].view(-1).to(torch.long)
        loss = criterion(out, y)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    return total_loss / batch_count


def extract_embeddings(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts node embeddings produced by the GraphSAGE model.

    Parameters:
    ----------
    model : torch.nn.Module
        The model used to generate embeddings, typically a pre-trained neural network.
    loader : cugraph_pyg.loader.NeighborLoader
        NeighborLoader that provides batches of data for embedding extraction.

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing two tensors:
        - embeddings: A tensor containing embeddings for each input sample in the dataset.
        - labels: A tensor containing the corresponding labels for each sample.
    """
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch_size = batch.batch_size
            hidden = model(
                batch.x.to(torch.float32), batch.edge_index, return_hidden=True
            )[:batch_size]
            embeddings.append(hidden)  # Keep embeddings on GPU
            labels.append(batch.y[:batch_size].view(-1).to(torch.long))
    embeddings = torch.cat(embeddings, dim=0)  # Concatenate embeddings on GPU
    labels = torch.cat(labels, dim=0)  # Concatenate labels on GPU
    return embeddings, labels


class Metric(Enum):
    RECALL = "recall"
    F1 = "f1"
    PRECISION = "precision"


def maximize_f1(model, loader):
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for batch in loader:

            batch_size = batch.batch_size
            out = model(batch.x.to(torch.float32), batch.edge_index)[:batch_size]
            pred_probs = torch.softmax(out, dim=1)[:, 1]
            y = batch.y[:batch_size].view(-1).to(torch.long)
            probs.append(pred_probs.cpu())
            targets.append(y.cpu())
    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy()

    precision, recall, thresholds = precision_recall_curve(targets, probs)

    # Compute F1 Score for Each Threshold
    f1_scores = []
    for threshold in thresholds:
        # Convert probabilities to binary predictions based on the current threshold
        y_pred = (probs >= threshold).astype(int)
        # Compute the F1 score for these predictions
        f1 = f1_score(targets, y_pred)
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)

    # Select the Best Threshold (that maximizes the F1 score)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]
    return best_threshold, best_f1


def tune_threshold(model, loader, desired_recall=0.90):
    """
    Evaluates the performance of the GraphSAGE model.

    Parameters:
    ----------
    model : torch.nn.Module
        The GNN model to be evaluated.
    loader : cugraph_pyg.loader.NeighborLoader
        NeighborLoader that provides batches of data for evaluation.

    Returns:
    -------
    float
        The average f1-score computed over all batches.
    """

    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for batch in loader:

            batch_size = batch.batch_size
            out = model(batch.x.to(torch.float32), batch.edge_index)[:batch_size]
            pred_probs = torch.softmax(out, dim=1)[:, 1]
            y = batch.y[:batch_size].view(-1).to(torch.long)
            probs.append(pred_probs.cpu())
            targets.append(y.cpu())
    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy()

    precision_vals, recall_vals, thresholds = precision_recall_curve(targets, probs)

    # Find the threshold closest to the desired recall
    idx = np.where(recall_vals >= desired_recall)[0]
    if len(idx) == 0:
        threshold = 0.5  # Default threshold if desired recall not achievable
    else:
        threshold = thresholds[idx[-1]]

    # Make predictions with the new threshold
    predictions = (probs > threshold).astype(int)

    # Recompute metrics
    final_recall = recall_score(targets, predictions)
    final_precision = precision_score(targets, predictions, zero_division=0)
    final_f1 = f1_score(targets, predictions, zero_division=0)
    final_roc_auc = roc_auc_score(targets, probs)
    final_auc_pr = average_precision_score(targets, probs)
    final_cm = confusion_matrix(targets, predictions)

    print(
        f"\n Need to set the threshold to {threshold:.4f} to achieve "
        f"recall of {desired_recall} where the precision would be {final_precision}."
    )

    return {
        "threshold": threshold,
        "recall": final_recall,
        "precision": final_precision,
        "f1_score": final_f1,
        "roc_auc": final_roc_auc,
        "auc_pr": final_auc_pr,
        "confusion_matrix": final_cm,
    }


def evaluate_gnn(model, loader, metric=Metric.F1.value) -> float:
    """
    Evaluates the performance of the GraphSAGE model.

    Parameters:
    ----------
    model : torch.nn.Module
        The GNN model to be evaluated.
    loader : cugraph_pyg.loader.NeighborLoader
        NeighborLoader that provides batches of data for evaluation.

    Returns:
    -------
    float
        The average f1-score computed over all batches.
    """

    model.eval()
    all_preds = []
    all_labels = []
    total_pos_seen = 0
    with torch.no_grad():
        for batch in loader:

            batch_size = batch.batch_size
            out = model(batch.x[:, :].to(torch.float32), batch.edge_index)[:batch_size]
            predictions = out.argmax(dim=1)
            y = batch.y[:batch_size].view(-1).to(torch.long)

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            total_pos_seen += (y.cpu().numpy() == 1).sum()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # accuracy = accuracy_score(all_labels, all_preds)
    # precision = precision_score(all_labels, all_preds, zero_division=0)
    # recall = recall_score(all_labels, all_preds, zero_division=0)
    # f1 = f1_score(all_labels, all_preds, zero_division=0)

    if metric == Metric.PRECISION.value:
        return precision_score(all_labels, all_preds, zero_division=0)
    elif metric == Metric.RECALL.value:
        return recall_score(all_labels, all_preds, zero_division=0)
    elif metric == Metric.F1.value:
        return f1_score(all_labels, all_preds, zero_division=0)


def validation_loss(model, loader, criterion) -> float:
    """
    Computes the average validation loss for the GraphSAGE model.

    Parameters:
    ----------
    model : torch.nn.Module
        The model for which the validation loss is calculated.
    loader : cugraph_pyg.loader.NeighborLoader
        NeighborLoader that provides batches of validation data.
    criterion : torch.nn.Module
        Loss function used to compute the loss between predictions and targets.

    Returns:
    -------
    float
        The average validation loss over all batches.
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        batch_count = 0
        for batch in loader:
            batch_count += 1
            batch_size = batch.batch_size
            out = model(batch.x[:, :].to(torch.float32), batch.edge_index)[:batch_size]
            y = batch.y[:batch_size].view(-1).to(torch.long)
            loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / batch_count


def train_xgboost(
    embeddings,
    labels,
    hyper_params_xgb: XGBHyperparametersSingle,
    random_state: int = 42,
) -> xgb.Booster:
    """
    Trains an XGBoost classifier on the provided embeddings and labels.

    Parameters:
    ----------
    embeddings : torch.Tensor
        The input feature embeddings for transaction nodes.
    labels : torch.Tensor
        The target labels (Fraud or Non-fraud) transaction, with the same length as the number of
        rows in `embeddings`.

    Returns:
    -------
    xgboost.Booster
        A trained XGBoost model fitted on the provided data.
    """

    labels_cudf = cudf.Series(cp.from_dlpack(to_dlpack(labels)))
    embeddings_cudf = cudf.DataFrame(cp.from_dlpack(to_dlpack(embeddings)))

    assert isinstance(hyper_params_xgb, XGBHyperparametersSingle)

    # Convert data to DMatrix format for XGBoost on GPU
    dtrain = xgb.DMatrix(embeddings_cudf, label=labels_cudf)

    # Set XGBoost parameters for GPU usage
    param = {
        "max_depth": hyper_params_xgb.max_depth,
        "learning_rate": hyper_params_xgb.learning_rate,
        "gamma": hyper_params_xgb.gamma,
        "num_parallel_tree": hyper_params_xgb.num_parallel_tree,
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "seed": random_state,
    }

    # Train the XGBoost model
    bst = xgb.train(param, dtrain, num_boost_round=hyper_params_xgb.num_boost_round)

    return bst


def evaluate_xgboost(bst, embeddings, labels, decision_threshold=0.5):
    """
    Evaluates the performance of a XGBoost model by calculating different metrics.

    Parameters:
    ----------
    bst : xgboost.Booster
        The trained XGBoost model to be evaluated.
    embeddings : torch.Tensor
        The input feature embeddings for transaction nodes.
    labels : torch.Tensor
        The target labels (Fraud or Non-fraud) transaction, with the same length as the number of
        rows in `embeddings`.
    Returns:
    -------
    A tuple containing f1-score, recall, precision, accuracy and the confusion matrix
    """

    # Convert embeddings to cuDF DataFrame
    embeddings_cudf = cudf.DataFrame(cp.from_dlpack(to_dlpack(embeddings)))

    # Create DMatrix for the test embeddings
    dtest = xgb.DMatrix(embeddings_cudf)

    # Predict using XGBoost on GPU
    predictions = bst.predict(dtest)
    pred_labels = (predictions > decision_threshold).astype(int)

    # Move labels to CPU for evaluation
    labels_cpu = labels.cpu().numpy()

    # Compute evaluation metrics
    accuracy = accuracy_score(labels_cpu, pred_labels)
    precision = precision_score(labels_cpu, pred_labels, zero_division=0)
    recall = recall_score(labels_cpu, pred_labels, zero_division=0)
    f1 = f1_score(labels_cpu, pred_labels, zero_division=0)
    # roc_auc = roc_auc_score(labels_cpu, predictions)
    conf_mat = confusion_matrix(labels.cpu().numpy(), pred_labels)

    return f1, recall, precision, accuracy, conf_mat


def load_data(
    dataset_root: str,
    edge_filename: str = "edges.csv",
    label_filename: str = "labels.csv",
    node_feature_filename: str = "features.csv",
    has_edge_feature: bool = False,
    use_cross_weights: bool = False,
    cross_weights: List[float] = None,
    edge_src_col: str = "src",
    edge_dst_col: str = "dst",
    edge_att_col: str = "type",
) -> Tuple[
    Tuple[torch_geometric.data.FeatureStore, torch_geometric.data.GraphStore],
    Dict[str, torch.Tensor],
    int,
    int,
]:
    """
    Create a graph from edge data and reads in features associated with the graph nodes.
    """

    # Load the Graph data
    edge_path = os.path.join(dataset_root, edge_filename)
    edge_data = cudf.read_csv(
        edge_path,
        header=None,
        names=[edge_src_col, edge_dst_col, edge_att_col],
        dtype=["int32", "int32", "float"],
    )

    num_nodes = max(edge_data[edge_src_col].max(), edge_data[edge_dst_col].max()) + 1
    src_tensor = torch.as_tensor(edge_data[edge_src_col], device="cuda")
    dst_tensor = torch.as_tensor(edge_data[edge_dst_col], device="cuda")

    graph_store = cugraph_pyg.data.GraphStore()
    graph_store[("n", "e", "n"), "coo", False, (num_nodes, num_nodes)] = [
        src_tensor,
        dst_tensor,
    ]

    edge_feature_store = None
    if has_edge_feature:
        from cugraph_pyg.data import TensorDictFeatureStore

        edge_feature_store = TensorDictFeatureStore()
        edge_attr = torch.as_tensor(edge_data[edge_att_col], device="cuda")
        edge_feature_store[("n", "e", "n"), "rel"] = edge_attr.unsqueeze(1)

    del edge_data

    # load the label
    label_path = os.path.join(dataset_root, label_filename)
    label_data = cudf.read_csv(label_path, header=None, dtype=["int32"])
    y_label_tensor = torch.as_tensor(label_data["0"], device="cuda")
    num_classes = label_data["0"].unique().count()

    wt_data = None
    if use_cross_weights:
        if cross_weights is None:
            counts = label_data.value_counts()
            wt_data = torch.as_tensor(
                counts.sum() / counts, device="cuda", dtype=torch.float32
            )
            wt_data = wt_data / wt_data.sum()

            if num_classes > 2:
                wt_data = wt_data.T
        else:
            wt_data = torch.as_tensor(cross_weights, device="cuda")

    del label_data

    # load the features
    feature_path = os.path.join(dataset_root, node_feature_filename)
    feature_data = cudf.read_csv(feature_path)

    feature_columns = feature_data.columns

    col_tensors = []
    for c in feature_columns:
        t = torch.as_tensor(feature_data[c].values, device="cuda")
        col_tensors.append(t)

    x_feature_tensor = torch.stack(col_tensors).T

    feature_store = cugraph_pyg.data.TensorDictFeatureStore()
    feature_store["node", "x", None] = x_feature_tensor
    feature_store["node", "y", None] = y_label_tensor

    num_features = len(feature_columns)

    return (
        (feature_store, graph_store),
        edge_feature_store,
        num_nodes,
        num_features,
        num_classes,
        wt_data,
    )


def k_fold_validation(
    data,
    num_transaction_nodes: int,
    model_config: GraphSAGEModelConfig,
    h_params: HyperParams,
    loss_function,
    early_stopping,
    random_state=42,
    verbose=False,
):
    """
    Run k-fold cross validation.
    """

    fold_size = num_transaction_nodes // h_params.n_folds

    # Perform cross-validation
    metric_scores = []

    for k in range(h_params.n_folds):
        training_nodes = torch.cat(
            (
                torch.arange(0, k * fold_size).unsqueeze(dim=0),
                torch.arange((k + 1) * fold_size, num_transaction_nodes).unsqueeze(
                    dim=0
                ),
            ),
            dim=1,
        ).squeeze(0)

        validation_nodes = torch.arange(k * fold_size, (k + 1) * fold_size)

        # Create NeighborLoader for both training and testing (using cuGraph NeighborLoader)
        train_loader = NeighborLoader(
            data,
            num_neighbors=[h_params.fan_out, h_params.fan_out],
            batch_size=h_params.batch_size,
            input_nodes=training_nodes,
            shuffle=True,
            random_state=random_state,
        )

        # Use same graph but different seed nodes
        val_loader = NeighborLoader(
            data,
            num_neighbors=[h_params.fan_out, h_params.fan_out],
            batch_size=h_params.batch_size,
            input_nodes=validation_nodes,
            shuffle=False,
            random_state=random_state,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the model
        model = GraphSAGE(
            in_channels=model_config.in_channels,
            hidden_channels=h_params.hidden_channels,
            out_channels=model_config.out_channels,
            n_hops=h_params.n_hops,
            dropout_prob=h_params.dropout_prob,
        ).to(device)

        check = EarlyStopping(
            patience=early_stopping.patience,
            path=early_stopping.path,
            verbose=verbose,
            delta=early_stopping.delta,
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=h_params.learning_rate,
            weight_decay=h_params.weight_decay,
        )

        # Train the GNN model
        for epoch in range(h_params.num_epochs):
            train_loss = train_gnn(model, train_loader, optimizer, loss_function)
            metric_value = evaluate_gnn(model, val_loader, metric=h_params.metric)

            if verbose:
                print(
                    f"Epoch {epoch}, training loss : {train_loss}, validation {h_params.metric} : {metric_value}"
                )
            if check:
                check(metric_value, model)
                if check.early_stop:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if check:
            # Load the Best Model
            model.load_state_dict(torch.load(check.path, weights_only=True))

        metric_value = evaluate_gnn(model, val_loader, metric=h_params.metric)
        metric_scores.append(metric_value)

    return np.mean(metric_scores)


class EarlyStopping:
    """
    Early stops the training if validation recall doesn't improve after a given patience.
    """

    def __init__(self, patience=10, verbose=False, delta=0.0, path="best_model.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation recall improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation recall improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0.0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_recall = -np.Inf
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, recall, model):
        if recall > self.best_recall + self.delta:
            self.best_recall = recall
            self.counter = 0
            self.best_model_state = model.state_dict()
            if self.verbose:
                print(
                    f"Validation recall increased to {self.best_recall:.4f}. Saving model."
                )
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation recall for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")


def generate_gnn_pbtx(
    model_file_name: str, input_dim: int, hidden_dim: int, path_to_gnn_pbtx: str
):
    # Use an f-string to insert the parameter values into the PBtx content.
    pbtx_content = f"""\
default_model_filename: "{model_file_name}"
platform: "onnxruntime_onnx"
input [                                 
 {{  
    name: "x"
    data_type: TYPE_FP32
    dims: [-1, {input_dim} ]                    
  }},
  {{
    name: "edge_index"
    data_type: TYPE_INT64
    dims: [ 2, -1]
  }}
]
output [
 {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, {hidden_dim} ]
  }}
]
instance_group [{{ kind: KIND_GPU }}]

"""

    # Write the content to the specified file.
    with open(path_to_gnn_pbtx, "w") as file:
        file.write(pbtx_content)

    print(f"Saved embedder model config to {path_to_gnn_pbtx}")


def generate_xgb_pbtx(
    model_file_name, input_dim: int, decision_threshold: float, path_to_xgb_pbtx: str
):
    # Use an f-string with escaped braces to insert the variable input dimension.
    storage_type = "AUTO"
    pbtx_content = f"""\
default_model_filename: "{model_file_name}"
backend: "fil"
input [
 {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ -1, {input_dim} ]
 }}
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
 }}
]
instance_group [{{ kind: KIND_GPU }}]
parameters [
 {{
    key: "model_type"
    value: {{ string_value: "xgboost_json" }}
 }},
 {{
    key: "output_class"
    value: {{ string_value: "false" }}
 }}
]
"""

    # Write the PBtx content to the specified file.
    with open(path_to_xgb_pbtx, "w") as file:
        file.write(pbtx_content)
    print(f"Saved XGBoost model config to {path_to_xgb_pbtx}")


def create_triton_model_repo(
    model: GraphSAGE,
    xgb_model: xgb.Booster,
    output_dir: str,
    gnn_file_name: str,
    xgb_model_file_name: str,
    decision_threshold: float,
    model_repository_name: str = "model_repository",
):

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

    print(
        f"Saving configs and models for Triton sever in {os.path.join(output_dir, model_repository_name)}"
    )

    print(f"Saved GraphSAGE node embedder model to {path_to_onnx_model}")

    xgb_model.save_model(path_to_xgboost_model)

    print(f"Saved XGBoost model to {path_to_xgboost_model}")

    generate_gnn_pbtx(
        gnn_file_name,
        model.in_channels,
        model.hidden_channels,
        os.path.join(gnn_repository_path, "config.pbtxt"),
    )

    generate_xgb_pbtx(
        xgb_model_file_name,
        model.hidden_channels,
        decision_threshold,
        os.path.join(xgb_repository_path, "config.pbtxt"),
    )


def train_with_specific_hyper_params(
    data,
    num_transaction_nodes: int,
    model_config: GraphSAGEModelConfig,
    hyper_params_gnn: GraphSAGEHyperparametersSingle,
    hyper_params_xgb: XGBHyperparametersSingle,
    loss_function,
    dir_to_save_models: str,
    train_val_ratio: float = 0.8,
    early_stopping: EarlyStopping = None,
    validation_metric=Metric.RECALL.value,
    embedding_model_name: str = "node_embedder.pth",
    xgboost_model_name: str = "embedding_based_xgb_model.json",
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[GraphSAGE, xgb.Booster]:

    print(f"Running GraphSAGE based XGBoost training....")

    # Set the device to GPU if available; otherwise, default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    model = GraphSAGE(
        in_channels=model_config.in_channels,
        hidden_channels=hyper_params_gnn.hidden_channels,
        out_channels=model_config.out_channels,
        n_hops=hyper_params_gnn.n_hops,
        dropout_prob=hyper_params_gnn.dropout_prob,
    ).to(device)

    num_training_tx = int(train_val_ratio * num_transaction_nodes)
    tx_nodes_perm = torch.randperm(num_transaction_nodes)
    train_txs = tx_nodes_perm[:num_training_tx]
    val_txs = tx_nodes_perm[num_training_tx:num_transaction_nodes]

    train_loader = NeighborLoader(
        data,
        num_neighbors=[hyper_params_gnn.fan_out, hyper_params_gnn.fan_out],
        batch_size=hyper_params_gnn.batch_size,
        input_nodes=train_txs,
        # input_nodes=torch.arange(int(train_val_ratio * num_transaction_nodes)),
        shuffle=True,
        random_state=random_state,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=[hyper_params_gnn.fan_out, hyper_params_gnn.fan_out],
        batch_size=hyper_params_gnn.batch_size,
        input_nodes=val_txs,
        # input_nodes=torch.arange(
        #     int(train_val_ratio * num_transaction_nodes), num_transaction_nodes
        # ),
        shuffle=False,
        random_state=random_state,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyper_params_gnn.learning_rate,
        weight_decay=hyper_params_gnn.weight_decay,
    )

    for epoch in range(hyper_params_gnn.num_epochs):
        train_loss = train_gnn(model, train_loader, optimizer, loss_function)
        metric_value = evaluate_gnn(model, val_loader, metric=validation_metric)

        if verbose:
            print(
                f"Epoch {epoch}, training loss : {train_loss}, validation {validation_metric} : {metric_value}"
            )
        if early_stopping:
            early_stopping(metric_value, model)
            if early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    if early_stopping:
        # Load the Best Model
        model.load_state_dict(torch.load(early_stopping.path, weights_only=True))

    # Finally train on validation data well
    for epoch in range(hyper_params_gnn.num_epochs):
        train_loss = train_gnn(model, val_loader, optimizer, loss_function)

    # And, save the final model
    if not os.path.exists(dir_to_save_models):
        os.makedirs(dir_to_save_models)

    # Train the XGBoost model based on embeddings produced by the GraphSAGE model
    data_loader = NeighborLoader(
        data,
        num_neighbors=[hyper_params_gnn.fan_out, hyper_params_gnn.fan_out],
        batch_size=hyper_params_gnn.batch_size,
        input_nodes=torch.arange(num_transaction_nodes),
        shuffle=True,
        random_state=random_state,
    )

    # Extract embeddings from the second-to-last layer and keep them on GPU
    embeddings, labels = extract_embeddings(model, data_loader)

    # Train an XGBoost model on the extracted embeddings (on GPU)
    bst = train_xgboost(embeddings.to(device), labels.to(device), hyper_params_xgb)

    xgb_model_path = os.path.join(dir_to_save_models, xgboost_model_name)

    if not os.path.exists(os.path.dirname(xgb_model_path)):
        os.makedirs(os.path.dirname(xgb_model_path))
    bst.save_model(xgb_model_path)

    print(f"Saved GraphSAGE based XGBoost model to {xgb_model_path}")
    return model, bst


def find_best_params(
    data,
    num_transaction_nodes: int,
    model_config: GraphSAGEModelConfig,
    params: GraphSAGEHyperparametersList,
    loss_function,
    early_stopping,
    random_state: int = 42,
) -> HyperParams:

    # Hyperparameter grid
    from sklearn.model_selection import ParameterGrid

    param_grid = {
        "n_folds": params.n_folds,
        "n_hops": params.n_hops,
        "fan_out": params.fan_out,
        "batch_size": params.batch_size,
        "learning_rate": params.learning_rate,
        "metric": params.metric,
        "dropout_prob": params.dropout_prob,
        "hidden_channels": params.hidden_channels,
        "num_epochs": params.num_epochs,
        "weight_decay": params.weight_decay,
    }

    grid = list(ParameterGrid(param_grid))

    # Find the best hyperparameters in the parameter grid

    best_metric_value = -float("inf")
    best_h_params = grid[0]

    print(
        "-----------Running cross-validation to find best set of hyperparameters---------"
    )

    for param_dict in grid:
        h_params = HyperParams(**param_dict)

        metric_value = k_fold_validation(
            data,
            num_transaction_nodes,
            model_config,
            h_params,
            loss_function,
            early_stopping,
            random_state=random_state,
            verbose=False,
        )
        print(f"{h_params}, {h_params.metric}: {metric_value}")
        if metric_value > best_metric_value:
            best_h_params = h_params
            best_metric_value = metric_value

    return best_h_params


def evaluate_on_unseen_data(
    embedder_model: GraphSAGE, xgb_model: xgb.Booster, dataset_root: str
):

    # Load and prepare test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = os.path.join(dataset_root, "xgb/test.csv")
    test_data = cudf.read_csv(test_path)
    X = torch.tensor(test_data.iloc[:, :-1].values).to(torch.float32)
    y = torch.tensor(test_data.iloc[:, -1].values).to(torch.long)

    # Extract the embeddings of the transactions using the GraphSAGE model
    embedder_model.eval()
    with torch.no_grad():
        test_embeddings = embedder_model(
            X.to(device),
            torch.tensor([[], []], dtype=torch.int).to(device),
            return_hidden=True,
        )

    # Evaluate the XGBoost model

    f1, recall, precision, accuracy, conf_mat = evaluate_xgboost(
        xgb_model, test_embeddings, y
    )

    print("XGBoost Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:", conf_mat)


def read_number_of_transaction_node(dataset_root: str):
    path_to_gnn_data = os.path.join(dataset_root, "gnn")
    file_containing_nr_tx_nodes = "info.json"
    if os.path.exists(path_to_gnn_data):
        for file_name in [
            "edges.csv",
            "labels.csv",
            "features.csv",
            file_containing_nr_tx_nodes,
        ]:
            if not os.path.exists(os.path.join(path_to_gnn_data, file_name)):
                sys.exit(f"{file_name} does not exist in {path_to_gnn_data}")
    else:
        sys.exit(f"{path_to_gnn_data} does not exist.")

    # Read number of transactions from info.json
    try:
        with open(
            os.path.join(path_to_gnn_data, file_containing_nr_tx_nodes), "r"
        ) as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(
            f"Could not find {file_containing_nr_tx_nodes}. Exiting...",
            file=sys.stderr,
        )
        sys.exit(1)
    except json.JSONDecodeError:
        print(
            f"Invalid JSON in {file_containing_nr_tx_nodes} . Exiting...",
            file=sys.stderr,
        )
        sys.exit(1)

    if "NUM_TRANSACTION_NODES" not in json_data:
        print(
            f"Key NUM_TRANSACTION_NODES not found in {file_containing_nr_tx_nodes}. Exiting...",
            file=sys.stderr,
        )
        sys.exit(1)

    return json_data["NUM_TRANSACTION_NODES"]


def run_sg_embedding_based_xgboost(
    dataset_root: str,
    output_dir: str,
    input_config: Union[GraphSAGEAndXGB, GraphSAGEGridAndXGB],
    model_index,
):

    num_transaction_nodes = read_number_of_transaction_node(dataset_root)
    random_state = 42
    set_seed(random_state)

    assert isinstance(
        input_config.hyperparameters, GraphSAGEGridAndXGBConfig
    ) or isinstance(input_config.hyperparameters, GraphSAGEAndXGBConfig)

    path_to_gnn_data = os.path.join(dataset_root, "gnn")
    use_cross_weights = True

    # Create graph and read node features
    data, ef_store, num_nodes, num_features, num_classes, cross_wt_data = load_data(
        path_to_gnn_data, use_cross_weights=use_cross_weights
    )

    model_config = GraphSAGEModelConfig(
        in_channels=num_features, out_channels=num_classes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Consider exposing loss_function and early_stopping as well

    if use_cross_weights:
        loss_function = torch.nn.CrossEntropyLoss(weight=cross_wt_data).to(device)
    else:
        loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([0.05, 0.95], dtype=torch.float32)
        ).to(device)

    early_stopping = EarlyStopping(
        patience=3,
        verbose=False,
        delta=0.0,
        path=os.path.join(output_dir, "saved_checkpoint.pt"),
    )

    if isinstance(input_config.hyperparameters, GraphSAGEAndXGBConfig):
        embedder_model, xgb_model = train_with_specific_hyper_params(
            data,
            num_transaction_nodes,
            model_config,
            input_config.hyperparameters.gnn,
            input_config.hyperparameters.xgb,
            loss_function,
            output_dir,
            train_val_ratio=0.8,
            early_stopping=early_stopping,
            validation_metric=input_config.hyperparameters.gnn.metric,
            embedding_model_name=f"node_embedder_{model_index}.pth",
            xgboost_model_name=f"embedding_based_xgb_model_{model_index}.json",
            random_state=42,
        )

    elif isinstance(input_config.hyperparameters, GraphSAGEGridAndXGBConfig):
        best_h_params = find_best_params(
            data,
            num_transaction_nodes,
            model_config,
            input_config.hyperparameters.gnn,
            loss_function,
            early_stopping,
            random_state=random_state,
        )

        embedder_model, xgb_model = train_with_specific_hyper_params(
            data,
            num_transaction_nodes,
            model_config,
            GraphSAGEHyperparametersSingle(
                hidden_channels=best_h_params.hidden_channels,
                n_hops=best_h_params.n_hops,
                dropout_prob=best_h_params.dropout_prob,
                batch_size=best_h_params.batch_size,
                fan_out=best_h_params.fan_out,
                metric=best_h_params.metric,
                num_epochs=best_h_params.num_epochs,
                learning_rate=best_h_params.learning_rate,
                weight_decay=best_h_params.weight_decay,
            ),
            input_config.hyperparameters.xgb,
            loss_function,
            output_dir,
            train_val_ratio=0.8,
            early_stopping=early_stopping,
            validation_metric=best_h_params.metric,
            embedding_model_name="graph_sage_node_embedder.pth",
            xgboost_model_name="xgboost_on_embeddings.json",
            random_state=42,
        )

    # evaluate_on_unseen_data(embedder_model, xgb_model, dataset_root)
    create_triton_model_repo(
        embedder_model,
        xgb_model,
        output_dir,
        "graph_sage_node_embedder.onnx",
        "xgboost_on_embeddings.json",
        decision_threshold=0.5,
    )
