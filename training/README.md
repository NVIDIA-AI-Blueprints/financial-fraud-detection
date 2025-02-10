# Model Training NIM

This document provides instructions to build the Docker image and run the training using the docker image.

## How to run locally

### 1. Build the docker image
Clone the repository and build the Docker image by running the following commands:
```sh
 git clone https://gitlab-master.nvidia.com/RAPIDS/graph-nims.git
 https://gitlab-master.nvidia.com/RAPIDS/graph-nims
 cd graph-nims
 docker build --no-cache -t training_container .
 ```


### 2. Make sure that data is organized correctly
To Train a GNN model (e.g. GraphSAGE), a dataset needs to be organized in the following structure-

```sh

  ├── gnn
  │   ├── edges.csv
  │   ├── features.csv
  │   ├── info.json
  │   └── labels.csv
```
`edges.csv` must contain  the graph topology in COO format. Each line contains source and destination vertex ids in the following format. NOTE that, the vertex ids must be zero based.
`src,dst,optional-attribute-value`

`info.json` must contain the number of transaction nodes, with key `NUM_TRANSACTION_NODES`, in the JSON file.
```sh
  {
      "NUM_TRANSACTION_NODES": 280945
  }
```

`features.csv` must contain the features for each of the graph nodes, indexed by the vertex id. !Important: The first row of `features.csv` must contain name of the features.

`labels.csv` must contain 0 (non-fraud) or 1(fraud) on each line


### 3. Write your training configuration on your host machine

A Training configuration must conform the schema defined in [configuration schema file](./config_schema.py). Note that, a valid JSON file can not have comments. Comments in the following example configuration file are for the purpose of explanation.

A training configuration must conform to the schema defined in the [configuration schema file](./config_schema.py). Note that a valid JSON file cannot contain comments. The comments in the following example training configuration are for explanation purposes only.

The example training configuration file is located [here](./example_training_config.json).

```bash
{
  // Configuration for file paths
  "paths": {
    "data_dir": "/data",                   // Directory path within the container where input data is stored.
    "output_dir": "/data/trained_models"   // Directory path within the container where trained models will be saved.
  },

  // List of models to train
  "models": [
    {
      "kind": "XGBoost",                   // Type of model: standard XGBoost
      "gpu": "single",                     // GPU usage: 'single' indicates using a single GPU
      "hyperparameters": {
        "max_depth": 6,                    // Maximum tree depth
        "learning_rate": 0.2,              // Learning rate for the boosting process
        "num_parallel_tree": 3,            // Number of trees built in parallel
        "num_boost_round": 512,            // Total number of boosting rounds
        "gamma": 0.0                       // Minimum loss reduction required to make a further partition on a leaf node
      }
    },
    {
      "kind": "GraphSAGE_XGBoost",         // Hybrid model combining GraphSAGE with XGBoost
      "gpu": "single",                     // GPU usage: 'single' indicates using a single GPU
      "hyperparameters": {
        "gnn": {                           // Hyper-parameters for the GraphSAGE model
          "hidden_channels": 16,           // Number of hidden channels in the the GraphSAGE model
          "n_hops": 1,                     // Number of hops/layers to aggregate
          "dropout_prob": 0.1,             // Dropout probability for regularization
          "batch_size": 1024,              // Batch size for training the model
          "fan_out": 16,                   // Number of neighbors to sample per node during message passing
          "num_epochs": 16                 // Number of training epochs for the GNN
        },
        "xgb": {                           // Hyper-parameters for the XGBoost component within the hybrid model
          "max_depth": 6,                  // Maximum depth of the tree
          "learning_rate": 0.2,            // Learning rate for boosting
          "num_parallel_tree": 3,          // Number of trees built in parallel
          "num_boost_round": 512,          // Number of boosting rounds
          "gamma": 0.0                     // Minimum loss reduction required to make a further partition on a leaf node
        }
      }
    }
  ]
}
```

### 3. Finally run the training

Execute the following command to run training. Make sure to `replace path_to_data_dir` and `path_to_train_config_json_file with` the actual paths on data directory and configuration file, respectively.

 ```sh
 docker run --cap-add SYS_NICE -it --rm --gpus "device=0" -v path_to_data_dir:/data  -v path_to_train_config_json_file:/app/config.json training_container --config /app/config.json
```

#### Command Explanation

    --cap-add SYS_NICE: Grants the container the capability to adjust process scheduling priorities.
    --it: Runs the container in interactive mode with a TTY.
    --rm: Automatically removes the container after it exits.
    --gpus "device=0": Limits the container's GPU access to GPU 0 only.
    -v path_to_data_dir:/data: Mounts your local data directory into the container at /data.
    -v path_to_train_config_json_file:/app/config.json: Mounts your local training configuration file into the container at /app/config.json.
    training_container: Specifies the Docker image to use.
    --config /app/config.json: Passes the configuration file path to the training command inside the container.

According the example training configuration show above, training-data must be mounted under `/data` within the container.

For example, `-v /home/user/data/TabFormer:/data` will mount your host directory `/home/user/data/TabFormer`  under `/data` within the container, and `-v ./training_configuration.json:/app/config.json` will mount `training_configuration.json`, from the current directory of your host machine, to `/app/config.json` within the container.
