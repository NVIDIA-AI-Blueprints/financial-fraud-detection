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

The `gamma`, `dropout_prob` and `learning_rate` fields take floating point values and the rest of the fields take integer values.