# ## Train a GNN-based XGBoost Model
# #### Goals
# * Train a GNN (GraphSAGE) model that produces node (transaction) embeddings.
# * Use these node embeddings to train an XGBoost model.
# * Save the trained GNN and XGBoost models for inference.
# 

# General-purpose libraries and OS handling
import os
from typing import Tuple, Dict
import json
from collections import defaultdict

# GPU-accelerated libraries (torch, cupy, cudf, rmm)
import torch
import cupy
import cudf
import rmm

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

# Machine learning metrics from sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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
        The number of hidden units in each layer, controlling the embedding dimension.
    out_channels : int
        The number of output features (or classes) for the final layer.
    n_hops : int
        The number of GraphSAGE layers (or hops) used to aggregate information 
        from neighboring nodes.
    dropout_prob : float, optional (default=0.25)
        The probability of dropping out nodes during training for regularization.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, n_hops, dropout_prob=0.25):
        super(GraphSAGE, self).__init__()

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

    def forward(self, x, edge_index, return_hidden=False):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
            
        if return_hidden:
            return x
        else:
            return self.fc(x)


def train_gnn(model, loader, optimizer, criterion)->float:
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
        out = model(batch.x[:,:].to(torch.float32), batch.edge_index)[:batch_size]
        y = batch.y[:batch_size].view(-1).to(torch.long)
        loss = criterion(out, y)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    return total_loss / batch_count


def extract_embeddings(model, loader)->Tuple[torch.Tensor, torch.Tensor]:
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
            hidden = model(batch.x[:,:].to(torch.float32), batch.edge_index, return_hidden=True)[:batch_size]
            embeddings.append(hidden)  # Keep embeddings on GPU
            labels.append(batch.y[:batch_size].view(-1).to(torch.long))
    embeddings = torch.cat(embeddings, dim=0)  # Concatenate embeddings on GPU
    labels = torch.cat(labels, dim=0)  # Concatenate labels on GPU
    return embeddings, labels


def evaluate_gnn(model, loader) -> float:
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
            out = model(batch.x[:,:].to(torch.float32), batch.edge_index)[:batch_size]
            preds = out.argmax(dim=1)
            y = batch.y[:batch_size].view(-1).to(torch.long)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            total_pos_seen += (y.cpu().numpy()==1).sum()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    # roc_auc = roc_auc_score(all_labels, all_preds)

    print(f"\nGNN Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # print(f"ROC AUC: {roc_auc:.4f}")
    return f1


def validation_loss(model, loader, criterion)->float:
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
            out = model(batch.x[:,:].to(torch.float32), batch.edge_index)[:batch_size]
            y = batch.y[:batch_size].view(-1).to(torch.long)
            loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / batch_count



from torch.utils.dlpack import to_dlpack

def train_xgboost(embeddings, labels)->xgb.Booster:
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

    # Convert data to DMatrix format for XGBoost on GPU
    dtrain = xgb.DMatrix(embeddings_cudf, label=labels_cudf)

    # Set XGBoost parameters for GPU usage
    param = {
        'max_depth': 6,
        'learning_rate': 0.2,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        # 'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'device': 'cuda'
    }

    # Train the XGBoost model
    bst = xgb.train(param, dtrain, num_boost_round=100)
    
    return bst


from cuml.metrics import confusion_matrix

def evaluate_xgboost(bst, embeddings, labels):
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
    preds = bst.predict(dtest)
    
    print(f'pred_shape: {preds.shape}')
    print(f'pred: {preds}')

    pred_labels = (preds > 0.5).astype(int)

    # Move labels to CPU for evaluation
    labels_cpu = labels.cpu().numpy()

    # Compute evaluation metrics
    accuracy = accuracy_score(labels_cpu, pred_labels)
    precision = precision_score(labels_cpu, pred_labels, zero_division=0)
    recall = recall_score(labels_cpu, pred_labels, zero_division=0)
    f1 = f1_score(labels_cpu, pred_labels, zero_division=0)
    roc_auc = roc_auc_score(labels_cpu, preds)
    conf_mat = confusion_matrix(labels.cpu().numpy(), pred_labels)
    
    return f1, recall, precision, accuracy, conf_mat


class EarlyStopping:
    """
    EarlyStopping class to halt training when a monitored metric stops improving.
    
    Parameters:
    ----------
    patience : int, optional (default=10)
        The number of epochs with no improvement after which training will be stopped.
    min_delta : float, optional (default=0)
        The minimum change in the monitored metric to qualify as an improvement. 
        If the change is smaller than `min_delta`, it is considered as no improvement.
    """
    def __init__(self, patience=10, min_delta=0):
        
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def check_early_stopping(self, val_loss):

        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if there's an improvement
        else:
            self.counter += 1  # Increment counter if no improvement
            
        if self.counter >= self.patience:
            return True
        return False



from typing import List

def load_data(
    dataset_root : str,
    edge_filename: str = 'edges.csv',
    label_filename: str = 'labels.csv',
    node_feature_filename: str = 'features.csv',
    has_edge_feature: bool = False,
    use_cross_weights: bool = False,
    cross_weights: List[float] = None,
    edge_src_col: str = 'src',
    edge_dst_col: str = 'dst',
    edge_att_col: str = 'type'
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
    edge_path   = os.path.join(dataset_root, edge_filename)
    edge_data = cudf.read_csv(edge_path, header=None, names=[edge_src_col, edge_dst_col, edge_att_col], dtype=['int32','int32','float'])
    
    num_nodes = max(edge_data[edge_src_col].max(), edge_data[ edge_dst_col].max()) + 1 
    src_tensor = torch.as_tensor(edge_data[edge_src_col], device='cuda')
    dst_tensor = torch.as_tensor(edge_data[edge_dst_col], device='cuda')

    graph_store = cugraph_pyg.data.GraphStore()
    graph_store[("n", "e", "n"), "coo", False, (num_nodes, num_nodes)] = [src_tensor, dst_tensor]

    edge_feature_store = None
    if has_edge_feature:
        from cugraph_pyg.data import  TensorDictFeatureStore
        edge_feature_store = TensorDictFeatureStore()
        edge_attr = torch.as_tensor(edge_data[edge_att_col], device='cuda')
        edge_feature_store[("n", "e", "n"), "rel"] = edge_attr.unsqueeze(1)

    del(edge_data)
    
    # load the label
    label_path   = os.path.join (dataset_root, label_filename)
    label_data = cudf.read_csv(label_path, header=None, dtype=['int32'])
    y_label_tensor = torch.as_tensor(label_data['0'], device='cuda')
    num_classes = label_data['0'].unique().count()

    wt_data = None
    if (use_cross_weights):
        if (cross_weights is None):
            num_labels_rows = label_data.size
            counts = label_data.value_counts()
            wt_data = torch.as_tensor(counts.sum()/counts, device='cuda', dtype=torch.float32)
            wt_data = wt_data/wt_data.sum()

            if (num_classes > 2):
                wt_data = wt_data.T
        else:
            wt_data = torch.as_tensor(cross_weights, device='cuda')

    del(label_data)
    
    # load the features
    feature_path   = os.path.join(dataset_root, node_feature_filename)
    feature_data = cudf.read_csv(feature_path)
    
    feature_columns = feature_data.columns
    
    col_tensors = []
    for c in feature_columns:
        t = torch.as_tensor(feature_data[c].values, device='cuda')
        col_tensors.append(t)

    x_feature_tensor = torch.stack(col_tensors).T

    
    feature_store = cugraph_pyg.data.TensorDictFeatureStore()
    feature_store["node", "x"] = x_feature_tensor
    feature_store["node", "y"] = y_label_tensor

    num_features = len(feature_columns)
    
    return (
        (feature_store, graph_store),
        edge_feature_store,
        num_nodes,
        num_features,
        num_classes,
        wt_data,
    )


from collections import namedtuple


GraphSAGEModelConfig = namedtuple('GraphSAGEModelConfig', ['in_channels',  'out_channels'])


HyperParams = namedtuple(
    'HyperParams', 
    [
        'n_folds',
        'n_hops',
        'fan_out',
        'batch_size',
        'learning_rate',
        'dropout_prob',
        'hidden_channels',
        'num_epochs',
        'weight_decay'
    ]
)


def train_model_with_config(data, num_transaction_nodes: int, model_config: GraphSAGEModelConfig, h_params: HyperParams, loss_function, random_state=42, verbose=False):
    """
    Train the GraphSAGE model for particular values of hyper-parameters.
    """

    print(f'Training GraphSAGE with {h_params}')

    fold_size = num_transaction_nodes // h_params.n_folds

    # Perform cross-validation
    validation_losses = []
    for k in range(h_params.n_folds):
        training_nodes = torch.cat(
            (
                torch.arange(0, k * fold_size).unsqueeze(dim=0),
                torch.arange((k+1) * fold_size, num_transaction_nodes).unsqueeze(dim=0)
            ),
            dim=1
        ).squeeze(0)

        validation_nodes = torch.arange(k * fold_size, (k+1) * fold_size)
        
        # Create NeighborLoader for both training and testing (using cuGraph NeighborLoader)
        train_loader = NeighborLoader(
            data,
            num_neighbors=[h_params.fan_out, h_params.fan_out],
            batch_size=h_params.batch_size,
            input_nodes= training_nodes,
            shuffle=True,
            random_state=random_state
        )

        # Use same graph but different seed nodes
        validation_loader = NeighborLoader(
            data,
            num_neighbors=[h_params.fan_out, h_params.fan_out],
            batch_size=h_params.batch_size,
            input_nodes= validation_nodes,
            shuffle=False,
            random_state=random_state
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define the model
        model = GraphSAGE(
            in_channels=model_config.in_channels,
            hidden_channels=h_params.hidden_channels,
            out_channels=model_config.out_channels,
            n_hops=h_params.n_hops,
            dropout_prob=h_params.dropout_prob).to(device)


        # Define optimizer and loss function for GNN
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=h_params.learning_rate,
                                    weight_decay=h_params.weight_decay)

        # criterion = torch.nn.CrossEntropyLoss(
        #     weight=cross_wt_data).to(device)  # Weighted loss function
        


        # Set up the early stopping object
        early_stopping = EarlyStopping(patience=4, min_delta=0.005)
        
        best_val_loss = float('inf')
        num_epoch_for_best_loss = 0

        # Train the GNN model
        for epoch in range(h_params.num_epochs):
            train_loss = train_gnn(model, train_loader, optimizer, loss_function)
            val_loss = validation_loss(model, validation_loader, loss_function)
            if verbose:
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Check early stopping criteria
            if early_stopping.check_early_stopping(val_loss):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                break

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_epoch_for_best_loss = epoch
        # Save validation loss for the current fold
        validation_losses.append(best_val_loss)
    return np.mean(validation_losses), model, num_epoch_for_best_loss



def sg_build_gnn_based_xgboost(dataset_root, output_dir, num_transaction_nodes):
    
    seed=42
    set_seed(seed)

    # #### Some config parameters for neighborhood sampler and training

    args = type('', (), {})()

    args.out_channels = 2
    args.batch_size = 1024
    args.fan_out = 10
    args.use_cross_weights = True
    args.cross_weights = None


    # ##### Path to pre-processed data and directory to save models
    args.dataset_base_path = dataset_root

    args.dataset_root = os.path.join(args.dataset_base_path, 'gnn')
    args.model_root_dir = output_dir
    # args.model_root_dir =  os.path.join(args.dataset_base_path, 'models')


    # Create graph and read node features
    data, ef_store, num_nodes, num_features, num_classes, cross_wt_data = load_data(args.dataset_root)

    model_config = GraphSAGEModelConfig(
        in_channels=num_features,
        out_channels=num_classes
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([0.01, 0.99], dtype=torch.float32)).to(device)
    
    # Hyperparameter grid
    from sklearn.model_selection import ParameterGrid

    param_grid = {
        'n_folds': [5],
        'n_hops': [1, 2],
        'fan_out': [8],
        'batch_size': [1024],
        'learning_rate': [0.005],
        'dropout_prob': [0.1],
        'hidden_channels': [32],
        'num_epochs': [4],
        'weight_decay': [1e-5],
    }

    grid = list(ParameterGrid(param_grid))

    # Find the best hyperparameters in the parameter grid
    best_val_loss = float('inf')
    best_h_params = None
    n_epoch_for_best_model = 0

    for param_dict in grid:
        h_params = HyperParams(**param_dict)
        val_loss, _, n_epoch_for_best_model = train_model_with_config(
            data,
            num_transaction_nodes,
            model_config,
            h_params, 
            loss_function,
            random_state=seed,
            verbose=True)
        print(f'h_params: {h_params} val_loss {val_loss}')
        if val_loss < best_val_loss:
            best_h_params = h_params
            best_val_loss = val_loss

    # Train and save best GraphSAGE model on entire dataset

    # Define the model
    model = GraphSAGE(
        in_channels=model_config.in_channels,
        hidden_channels=best_h_params.hidden_channels,
        out_channels=model_config.out_channels,
        n_hops=best_h_params.n_hops,
        dropout_prob=best_h_params.dropout_prob).to(device)


    train_loader = NeighborLoader(
        data,
        num_neighbors=[best_h_params.fan_out, best_h_params.fan_out],
        batch_size=best_h_params.batch_size,
        input_nodes= torch.arange(num_transaction_nodes),
        shuffle=True,
        random_state=seed
    )



    # Define optimizer and loss function for GNN
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=best_h_params.learning_rate,
                                weight_decay=best_h_params.weight_decay)
    
    


    # Set up the early stopping object
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    best_train_loss = float('inf')

    # Train the GNN model

    for epoch in range(n_epoch_for_best_model):
        train_loss = train_gnn(model, train_loader, optimizer, loss_function)
        
        # Check early stopping criteria
        if early_stopping.check_early_stopping(train_loss):
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

        # Save the best model based on validation loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            if not os.path.exists(args.model_root_dir):
                os.makedirs(args.model_root_dir)
            torch.save(model, os.path.join(args.model_root_dir, 'node_embedder.pth'))

            print(f"Model saved at epoch {epoch+1} with training loss {best_train_loss:.4f}.")


    
    # Train the XGBoost model based on embeddings produced by the GraphSAGE model


    # Set the device to GPU if available; otherwise, default to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract embeddings from the second-to-last layer and keep them on GPU
    embeddings, labels = extract_embeddings(model, train_loader)

    # Train an XGBoost model on the extracted embeddings (on GPU)
    bst = train_xgboost(embeddings.to(device), labels.to(device))
                
    xgb_model_path = os.path.join(args.model_root_dir, 'embedding_based_xgb_model.json')

    if not os.path.exists(os.path.dirname(xgb_model_path)):
        os.makedirs(os.path.dirname(xgb_model_path))

    bst.save_model(xgb_model_path)


    # Evaluation the model on unseen data

    # Load and prepare test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_path = os.path.join(args.dataset_base_path, 'xgb/test.csv')
    test_data = cudf.read_csv(test_path)
    X = torch.tensor(test_data.iloc[:, :-1].values).to(torch.float32)
    y = torch.tensor(test_data.iloc[:, -1].values).to(torch.long)


    #Extract the embeddings of the transactions using the GraphSAGE model
    model.eval()
    f1_value = 0.0
    with torch.no_grad():
        test_embeddings = model(
            X.to(device), torch.tensor([[], []], dtype=torch.int).to(device), return_hidden=True)


    # Evaluate the XGBoost model

    f1, recall, precision, accuracy, conf_mat = evaluate_xgboost(bst, test_embeddings, y)

    print(f"\nXGBoost Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print('Confusion Matrix:', conf_mat)
