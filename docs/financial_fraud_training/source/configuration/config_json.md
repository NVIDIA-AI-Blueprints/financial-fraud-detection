# Model Training Options


## Data Organization for Standalone XGBoost Training

Your dataset is expected to be organized in a specific structure under the `xgb/` folder. Follow the guidelines below to ensure your data is correctly formatted and named.

### Directory Structure

Place your data files under the `xgb/` directory within your main data folder. The expected files are:

- **`training.<ext>`** (Required): Contains the training data.
- **`validation.<ext>`** (Optional): Contains the validation data.
  - If this file is missing, the training data will be automatically split into 80% training and 20% validation.
- **`test.<ext>`** (Optional): Contains the test data.
  - If this file is missing *and* the validation file is also missing, the training data will be split into 70% training, 15% validation, and 15% test.

## XGBoost on the embeddings produced by GNN (GraphSAGE) model

To train an XGBoost model on the embeddings produced by a GNN model, the model configuration should have the following schema.

Note that the `kind` field must be set to `GraphSAGE_XGBoost`, and it needs `hyperparameters` for a GNN (GraphSAGE) model and an XGBoost model. The values passed to the different fields are for example purposes only. The description next to the fields are to describe the fields.

```sh
    {
      "kind": "GraphSAGE_XGBoost", //Train an XGBoost on embeddings produces by a GraphSAGE model
      "gpu": "single",             // Indicates whether to use single-gpu or multi-gpu
      "hyperparameters": {
        "gnn": {                   // Hyper-parameters for a GraphSAGE that will produce embeddings
          "hidden_channels": 16,   // Number of hidden channels in the GraphSAGE model
          "n_hops": 1,             // Number of hops/layers
          "dropout_prob": 0.1,     // Dropout probability for regularization
          "batch_size": 1024,      // Batch size for training the model
          "fan_out": 16,           // Number of neighbors to sample per node
          "num_epochs": 16         // Number of training epochs
        },
        "xgb": {  // Hyper-parameters of an XGBoost model that will predict fraud score using embeddings as input
          "max_depth": 6,          // Maximum depth of the tree
          "learning_rate": 0.2,    // Learning rate for boosting
          "num_parallel_tree": 3,  // Number of trees built in parallel
          "num_boost_round": 512,  // Number of boosting rounds
          "gamma": 0.0             // Minimum loss reduction required to make a further partition on a leaf node
        }
      }
```

Here is example of a full training configuration file for training an XGBoost model on the embeddings produced by a GNN model.


```sh
{
  "paths": {
    "data_dir": "/data",                   // Directory path within the container where training data is mounted.
    "output_dir": "/trained_models"   // Directory path within the container where trained models will be saved.
  },

  "models": [
    {
      "kind": "GraphSAGE_XGBoost", //Train an XGBoost on embeddings produces by a GraphSAGE model
      "gpu": "single",             // Indicates whether to use single-gpu or multi-gpu
      "hyperparameters": {
        "gnn": {                   // Hyper-parameters for a GraphSAGE that will produce embeddings
          "hidden_channels": 16,   // Number of hidden channels in the GraphSAGE model
          "n_hops": 1,             // Number of hops/layers
          "dropout_prob": 0.1,     // Dropout probability for regularization
          "batch_size": 1024,      // Batch size for training the model
          "fan_out": 16,           // Number of neighbors to sample per node
          "num_epochs": 16         // Number of training epochs
        },
        "xgb": {  // Hyper-parameters of an XGBoost model that will predict fraud score using embeddings as input
          "max_depth": 6,          // Maximum depth of the tree
          "learning_rate": 0.2,    // Learning rate for boosting
          "num_parallel_tree": 3,  // Number of trees built in parallel
          "num_boost_round": 512,  // Number of boosting rounds
          "gamma": 0.0             // Minimum loss reduction required to make a further partition on a leaf node
        }
      }
    }
  ]
}
```




The `gamma`, `dropout_prob` and `learning_rate` fields take floating point values and the rest of the fields take integer values.

```
GNN and Xgb(oost) parameters can either be single value or a list of the data type specified. The list must be of the same size as the n_hops value. The list values allow different hyperparameters for each hop(layer).


- "paths"  path/on/machine that Contains needed paths for input and output
  - "data_dir" - Path to the input data directory.
  - "output_dir" - Path to the output models directory.
- "gpu" - Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'. 
- "hyperparameters"
  - "gnn" - 
    - "hidden_channels" - Number of hidden channels (e.g., 32).
    - "n_hops" - Number of hops (e.g., 2). Same as number of layers.
    - "dropout_prob" - Dropout probability (e.g., 0.2). Chance to eliminate any given feature.
    - "batch_size" - Batch size (e.g., 1024). How many nodes to predict to grab at a time. Parallelizes the run to fit into gpu memory. More batches increases run time.
    - "fan_out" - Number of neighbors to sample for each node (e.g., 16). Random number of neighbors chosen from complete incident edgelist.
    - "num_epochs" - Number of epochs to train the model.
  - "xgb" -
    - "max_depth" - A list of possible max_depth values. e.g., [3, 6, 9]
    - "learning_rate" - A list of possible learning rates in range 0 to 1. e.g., [0.01, 0.1]
    - "num_parallel_tree" - A list of possible numbers of trees. e.g., [50, 100, 200]
    - "num_boost_round" - integer, A list of possible number of boosting rounds. e.g., [1, 2, 4]
    - "gamma" - float representing minimum loss reduction required to make a partition.


Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

This source code and/or documentation ("Licensed Deliverables") are
subject to NVIDIA intellectual property rights under U.S. and
international Copyright laws.