
## CSV Files

Your data should be organized under a parent directory (for example, `data_root`) with the following subdirectories:

```sh
    data_root/
    ├── nodes/
    └── edges/
```

```sh
merchant_features.csv
user_features.csv
edges.csv
edge_labels.csv
edge_features.csv
```


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

  ```

## Parquet Files

```sh
credit_card_transaction_data/
├── nodes/
│   ├── User.parquet
│   └── Merchant.parquet
└── edges/
    ├── User__transaction__Merchant.parquet
    ├── User__transaction__Merchant_attr.parquet
    └── User__transaction__Merchant_label.parquet
```

- **nodes/**: Contains Parquet files for different node types.
- **edges/**: Contains Parquet files for different edge types along with optional attributes and labels.

- **Main Edge File (`node_to_node.<ext>`):**
  This file must contain at least two columns:
  - `src`: The column containing indices of the source nodes
  - `dst`: The column containing indices of the destination nodes.
  - The index of a node is determined by its position in the node feature file `node.<ext>` and is zero-based and contiguous.
  - The number of rows corresponds to the number of edges.
  NOTE: The column containing source and destination nodes must be named `src` and `dst`, respectively.



- **Node Feature Files:**  
  Each node type must have its own Parquet file. The file name (without the `.parquet` extension) defines the node type. For 

- **Optional Node Label Files:**  
  If you have labels for a node type, create a label file by appending `_label` to the node type name.

  - **Feature Files:**  
  - These files should contain a table of node attributes (features).
  - The number of rows corresponds to the number of nodes for that type.
  - The number of columns is the feature dimension (which may vary between node types).

  - **Label Files (Optional):**  
  - These files should contain the ground-truth labels for nodes.
  - They are expected to have a single column with a number of rows equal to that of the corresponding feature file.

  Each edge type is represented by up to three files:

1. **Main Edge File:**  
   Named using the format:  

`{src_node_type}{relation}{dst_node_type}.parquet`

Example:
- `User__transaction__Merchant.parquet`

2. **Edge Attribute File (Optional):**  
Contains additional features for each edge. Name it by appending `_attr` to the main edge file name:

`{src_node_type}{relation}{dst_node_type}_attr.parquet`

Example:
- `User__transaction__Merchant_attr.parquet`

3. **Edge Label File (Optional):**  
Contains the ground-truth labels for the edges. Name it by appending `_label` to the main edge file name:

`{src_node_type}{relation}{dst_node_type}_label.parquet`


  **Main Edge Files:**  
- Must contain at least two columns: `src` and `dst`.
 - `src`: Index of the source node (refers to the row index in the corresponding `Node Feature` file).
 - `dst`: Index of the destination node (refers to the row index in the corresponding `Node Feature` file).
- The number of rows corresponds to the number of edges.

 **Edge Attribute Files (Optional):**  
- These files provide additional attributes (features) for each edge.
- They should have the same number of rows as the corresponding main edge file.
- Different edge types may have different attribute dimensions.

 **Edge Label Files (Optional):**  
- These files contain the labels for each edge (e.g., a fraud indicator).
- They should have a single column with a number of rows equal to that of the main edge file.

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

This source code and/or documentation ("Licensed Deliverables") are
subject to NVIDIA intellectual property rights under U.S. and
international Copyright laws.