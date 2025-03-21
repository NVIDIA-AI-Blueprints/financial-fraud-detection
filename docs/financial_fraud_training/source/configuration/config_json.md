
# Write your training configuration

A training configuration file must be a JSON file with the following schema:

```bash
{
  "paths": {
    "data_dir":   // Directory path within the container where training data is mounted.
    "output_dir": // Directory path within the container where trained models will be saved.
  },

  "models": [
    //Provide your model configuration schema here (described below).
  ]
}
```
__Field Description__

`data_dir`: It must a path withing the container where the training data is mounted.

`output_dir`: It must a path withing the container where the models will be saved after the training.


As of now, the Training NIM supports two kinds of model trainings:
  - Train XGBoost on the embeddings produced by GNN (GraphSAGE) model
  - Train XGBoost directly on input features

The following two subsections describe the `Model Configuration Schema` for two types of trainings, and provide examples of full training configuration files.

## Train an XGBoost on the embeddings produced by a GNN model


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

The `gamma`, `dropout_prob` and `learning_rate` fields take floating point values and the rest of the fields take integer values.


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

## Train an XGBoost directly on input features

To train an XGBoost model directly on input features, the model configuration should have the following schema, and the `kind` field must be set to `XGBoost`.

**Note: This is not preferred but might be necessary if one or more of these characteristics is not acceptable.**
 - **performance - training takes too long on the gnn or resulting embeddings.**
 - **precision - fewer predicted positives were actual positives**
 - **accuracy - the percent of accurate predictions ( positive plus negative ) was too low.**
 - **recall - too few actual positives were predicted**


```sh
    {
      "kind": "XGBoost",          // Train XGBoost directly on input features
      "gpu": "single",            // Indicates whether to use single-gpu or multi-gpu
      "hyperparameters": {
        "max_depth": 6,           // Maximum tree depth
        "learning_rate": 0.2,     // Learning rate for the boosting process
        "num_parallel_tree": 3,   // Number of trees built in parallel
        "num_boost_round": 512,   // Total number of boosting rounds
        "gamma": 0.0              // Minimum loss reduction required to make a further partition on a leaf node
      }
    }
```

The `learning_rate` and `gamma`  fields take floating point values and the rest of the fields take integer values.


Here is example of a full training configuration file for training an XGBoost model on directly on input features.


```sh
{
  "paths": {
    "data_dir": "/data",                   // Directory path within the container where input data is stored.
    "output_dir": "/trained_models"   // Directory path within the container where trained models will be saved.
  },

  "models": [
    {
      "kind": "XGBoost",          // Train XGBoost directly on input features
      "gpu": "single",            // Indicates whether to use single-gpu or multi-gpu
      "hyperparameters": {
        "max_depth": 6,           // Maximum tree depth
        "learning_rate": 0.2,     // Learning rate for the boosting process
        "num_parallel_tree": 3,   // Number of trees built in parallel
        "num_boost_round": 512,   // Total number of boosting rounds
        "gamma": 0.0              // Minimum loss reduction required to make a further partition on a leaf node
      }
    }
  ]
}
```
- "paths"  path/on/machine that contains needed paths for input and output
  - "data_dir" - A path to a directory inside in container where the training data is mounted.
  - "output_dir" - A path to a mounted directory inside in container where the output models and the artifacts will be saved.

  - "gpu" - Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'. currently the NIM only supports single GPU.
  - "hyperparameters"
    - "gnn" -
      - "hidden_channels" - dimensionality of the feature representations in the hidden layers (e.g., 32).
        - A larger number here allows more complexity in the model but may also cause overfitting.
      - "n_hops" - Number of hops (e.g., 2). How many links messages are passed over.
        - When this value is one, messages are only passed to directly connected nodes. A value of 2 adds neighbors of neighbors. The higher this number, the more nearby but unconnected nodes effect each other. A higher number also results in more nodes being similar which can hinder classification. It also slows down training and requires more memory.
      - "dropout_prob" - Dropout probability (typically in the 0.1-0.5 range). Chance to eliminate any given feature.
        - Lower the value if the model overfits.
        - Raise the value if losses are too high (underfitting).
        - lower dropout rates enable faster convergence and less randomness.
      - "batch_size" - Batch size (typically 256-4096). How many nodes to predict to grab at a time.
        - A larger batch size increases memory use.
        - A smaller batch size has the reverse effects but increases variance, often used to parallelize the run to fit into gpu memory. More batches, the result of smaller ones, increase run time.
      - "fan_out" - Maximum number of direct neighbors to sample for each node (e.g., 16).
        - "Good" values depend on the graph structure and computation capability.
        - Higher values create more accurate models
        - Lower values train faster.
      - "num_epochs" - maximum number of epochs model will be trained (e.g., 16).
        - Training will finish earlier if no improvement occurs sooner than this number.
      - "metric" - The metric to be used to determine if a model is improving during the training. It can be Either "recall", "f1" or "precision". The default metric is "f1".
    - "xgb"
      - "max_depth" - The max_depth determines the maximum depth of the decision tree (e.g., 6).
        - A lower value can prevent a model from revealing complex patterns in the data. A higher value can cause overfitting, and slows down training.
      - "learning_rate" - The learning rate for each algorithm iteration in the range 0 to 1 (e.g., 0.2).
        - A lower value will slow down learning and prevent overfitting. A higher value increase the contribution of each iteration at the expense of overfitting.
      - "num_boost_round" - integer number of boosting rounds (e.g., 4).
        - During each boosting round, the algorithm fits a new model to the residual errors (or gradients) from the previous ensemble of trees, progressively improving the model's predictions. Too few rounds can lead to underfitting, where the model doesn’t capture the underlying data patterns. Conversely, too many rounds can result in overfitting, where the model learns the noise in the training data.
      - "num_parallel_tree" - A list of possible numbers of trees (e.g., 200).
        - This parameter specifies how many trees are built in parallel during each boosting round. Higher numbers will increase performance and memory usage.
      - "gamma" - float representing minimum loss reduction required to make a split in a decision tree (e.g., 0.05).
        - Increasing this value creates a simpler more conservative model.

The GNN hyper-parameters can either be a single value or a list of values, which will result in training a separate model for each possible combination of hyper-parameters. In that case, only the model with the best metric value is retained.







Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

This source code and/or documentation ("Licensed Deliverables") are
subject to NVIDIA intellectual property rights under U.S. and
international Copyright laws.