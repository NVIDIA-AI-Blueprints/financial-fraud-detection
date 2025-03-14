
# Graph Data Organization for GNN based XGBoost training


Your data should be organized under a parent directory (for example, `data_root`) with the following subdirectories:


```sh
    data_root/
    ├── nodes/
    └── edges/
```




- **nodes/**: Contains the files for node features and labels.
- **edges/**: Contains the file for edges.


---


## Nodes Directory


### File Naming Convention

- **Node Feature File:**
  Name the file simply as `node.<ext>`, where `<ext>` is one of the supported file formats (CSV, Parquet, or ORC).
  Examples:
  - `node.csv`


- **Node Label File:**
  Provide the labels for your nodes in a file named `node_label.<ext>`.
  Examples:
  - `node_label.csv`


- **Optional JSON file containing the start and end offset of the target nodes in node.<ext> file**
  - `offset_range_of_training_node.json`


### File Contents


- **Feature File (`node.<ext>`):**
  This file should contain a table of node attributes (features).
  - Each row corresponds to one node.
  - The number of columns is the feature dimension.


  NOTE: If you have nodes of different types, it's up to you to decide how to prepare and align their features; however, each row must contain the same number of feature entries.




- **Label File (`node_label.<ext>`, Optional):**
  This file should contain the ground-truth labels for the nodes.
  - It is expected to have one column with any name, with the number of rows matching the feature file.




- **Optional JSON file containing the start and end offset of the target nodes in node.<ext> (`offset_range_of_training_node.json`)**
  This file should specify the start and end offsets for the training node—a contiguous range—in node.<ext>. In other words, it defines the range of node offsets that the model will be trained on.


  For example, for credit card fraud detection, if your `node.<ext>` file includes features for User, Merchant, and Transaction nodes, and your goal is to predict whether a transaction is fraudulent, you might be interested to train on transaction nodes only.


  ```sh
    {
      "start": ST #start offset of training nodes
      "end": ET #end offset of training nodes
    }
  ```


  If no file is provided, the models will be trained using the entire range.


---


## Edges Directory


### File Naming Convention

For edges connecting nodes of the single node type, you should have:


- **Main Edge File:**
  Name the file as `node_to_node.<ext>`, where `<ext>` is one of the supported file formats (CSV, Parquet, or ORC).
  Examples:
  - `node_to_node.csv`


#### File Contents


- **Main Edge File (`node_to_node.<ext>`):**
  This file must contain at least two columns:
  - `src`: The column containing indices of the source nodes
  - `dst`: The column containing indices of the destination nodes.
  - The index of a node is determined by its position in the node feature file `node.<ext>` and is zero-based.
  - The number of rows corresponds to the number of edges.
  NOTE: The column containing source and destination nodes must be named `src` and `dst`, respectively.


---


### Data types


#### `node.<ext>`
- **Description:** Contains node features.
- **Data Type:** All columns must contain floating point numbers in `float32` format.
- **Notes:** Each row represents a node and each column corresponds to a specific feature.


#### `node_label.<ext>`
- **Description:** Contains node labels.
- **Data Type:** All values are integers (0 or 1).
- **Column Name:** The file contains a single column named `target`.
- **Notes:** Each row corresponds to the label of a node.


#### `node_to_node.<ext>`
- **Description:** Contains edge indices.
- **Data Type:** All columns must contain integers.
- **Columns:**
  - `src`: Source node index.
  - `dst`: Destination node index.
- **Notes:** The indices must be zero-based (i.e., node numbering starts at 0).




### Example Data Layout


```sh
credit_card_transaction_data/
├── nodes/
│ ├── node.csv
│ └── node_label.csv
│ └── offset_range_of_training_node.json
└── edges/
  └── node_to_node.csv
```




###  Data Organization for Standalone XGBoost Training


Your dataset is expected to be organized in a specific structure under the `xgb/` folder. Follow the guidelines below to ensure your data is correctly formatted and named.


### Directory Structure


Place your data files under the `xgb/` directory within your main data folder. The expected files are:


- **`training.<ext>`** (Required): Contains the training data.
- **`validation.<ext>`** (Optional): Contains the validation data.
  - If this file is missing, the training data will be automatically split into 80% training and 20% validation.
- **`test.<ext>`** (Optional): Contains the test data.
  - If this file is missing *and* the validation file is also missing, the training data will be split into 70% training, 15% validation, and 15% test.


### Supported File Formats


The `<ext>` in the file names represents the file format. The supported file formats are:


- **CSV**: Files with the `.csv` extension (comma-separated values).
- **Parquet**: Files with the `.parquet` extension (columnar storage).
- **ORC**: Files with the `.orc` extension (Optimized Row Columnar format).


The code automatically detects the file extension and uses the appropriate reader.


#### Important Data Format Note


- **Target Column:**
  The last column in every file is assumed to be the target variable (`y`). All splits (training, validation, and test) are performed with this assumption in mind.




#### Example Directory Layout


```sh
data_root/
└── xgb/
    ├── training.csv # or training.parquet, training.orc
    ├── validation.csv # (optional) or validation.parquet, validation.orc
    └── test.csv # (optional) or test.parquet, test.orc
```




#### Data Splitting Behavior


- **When only `training.<ext>` is provided:**
  - If **only the training.<ext> file exists**, the code will split the data into 80% training and 20% validation.
- **When `validation.<ext>` is provided:**
  - The provided training and validation files are used directly without splitting.
- **When `test.<ext>` is present:**
  - The test file is used to evaluate the model after training.


#### Model Testing


If a `test.<ext>` file is present in the `xgb/` directory, it will be used for evaluation after training.


---




Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

This source code and/or documentation ("Licensed Deliverables") are
subject to NVIDIA intellectual property rights under U.S. and
international Copyright laws.