# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.

import sys
import os
import json
import logging
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data


def read_data(file_path: str) -> pd.DataFrame:
    """
    Read a file (CSV, Parquet, or ORC) and return a DataFrame.

    For ORC files, it uses pyarrow.orc.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".parquet", ".parq"]:
        return pd.read_parquet(file_path)
    elif ext == ".orc":
        try:
            import pyarrow.orc as orc
        except ImportError:
            raise ImportError(
                "pyarrow is required to read ORC files. Please install pyarrow."
            )
        orc_file = orc.ORCFile(file_path)
        table = orc_file.read()
        return table.to_pandas()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def load_nodes(node_dir: str, data: HeteroData) -> dict:
    """
    Load node data from files in the given directory and update the HeteroData object.

    The file name (without extension) defines the node type.
    Supported file formats: CSV, Parquet, ORC.
    Optionally, if a file named {node}_label.<ext> exists, its labels are loaded.

    Returns:
        node_meta (dict): Mapping from node type to a dict with its id, number of nodes, and feature dimension.
    """
    if not os.path.isdir(node_dir):
        logging.error(f"Nodes directory '{node_dir}' does not exist.")
        raise ValueError(f"Nodes directory '{node_dir}' does not exist.")

    node_files = sorted(
        [
            f
            for f in os.listdir(node_dir)
            if (f.endswith(".csv") or f.endswith(".parquet") or f.endswith(".orc"))
            and "_label" not in f
        ],
        key=str.lower,
    )
    if len(node_files) == 0:
        raise ValueError(f"'{node_dir}' does not contain any node.<ext> file.")

    node_meta = {}
    for i, file in enumerate(node_files):
        node_type = os.path.splitext(file)[0]
        file_path = os.path.join(node_dir, file)
        try:
            df = read_data(file_path)
        except Exception as e:
            logging.error(f"Error reading node file '{file_path}': {e}")
            raise e

        num_nodes = df.shape[0]
        try:
            x = torch.tensor(df.values, dtype=torch.float)
        except Exception as e:
            logging.error(
                f"Error converting node data from '{file_path}' to tensor: {e}"
            )
            raise e

        data[node_type].x = x
        node_meta[node_type] = {
            "id": i,
            "num_nodes": num_nodes,
            "feat_dim": x.shape[1]}
        logging.info(
            f"Loaded node type '{node_type}': {num_nodes} nodes, feature dim {x.shape[1]} (id: {i})."
        )

        # Check for a corresponding label file.
        base, _ = os.path.splitext(file)
        for ext in [".csv", ".parquet", ".orc"]:
            label_file = os.path.join(node_dir, f"{base}_label{ext}")
            if os.path.exists(label_file):
                try:
                    df_label = read_data(label_file)
                    labels = torch.tensor(
                        df_label.values.squeeze(), dtype=torch.long)
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                    data[node_type].y = labels
                    logging.info(
                        f"Loaded labels for node type '{node_type}' from '{label_file}'."
                    )
                    break
                except Exception as e:
                    logging.error(
                        f"Error reading label file '{label_file}': {e}")
                    raise e

    logging.info(f"Node meta mapping: {node_meta}")
    return node_meta


def load_edges(edge_dir: str, data: HeteroData) -> dict:
    """
    Load edge data from files in the given directory and update the HeteroData object.

    Expected files:
      - Main edge files: {src}_{relation}_{dst}.<ext> (with 'src' and 'dst' columns).
      - Optional attribute files: {src}_{relation}_{dst}_attr.<ext>
      - Optional label files: {src}_{relation}_{dst}_label.<ext>

    Supported file formats: CSV, Parquet, ORC.

    Returns:
        edge_meta (dict): Mapping from edge type tuple to a dict with number of edges and attribute dimension.
    """
    if not os.path.isdir(edge_dir):
        logging.error(f"Edges directory '{edge_dir}' does not exist.")
        raise ValueError(f"Edges directory '{edge_dir}' does not exist.")

    edge_files = sorted(
        [
            f
            for f in os.listdir(edge_dir)
            if f.endswith(".csv") or f.endswith(".parquet") or f.endswith(".orc")
        ],
        key=str.lower,
    )
    edge_meta = {}

    # Process main edge files (exclude files with '_attr' or '_label').
    main_edge_files = [
        f for f in edge_files if "_attr" not in f and "_label" not in f]

    for file in main_edge_files:
        base = os.path.splitext(file)[0]
        parts = base.split("_")
        if len(parts) != 3:
            logging.error(
                f"Edge file '{file}' does not follow the 'src_relation_dst' naming convention."
            )
            raise ValueError(
                f"Edge file '{file}' does not follow the 'src_relation_dst' naming convention."
            )
        src_type, rel_type, dst_type = parts
        file_path = os.path.join(edge_dir, file)
        try:
            df = read_data(file_path)
        except Exception as e:
            logging.error(f"Error reading edge file '{file_path}': {e}")
            raise e

        if "src" not in df.columns or "dst" not in df.columns:
            logging.error(
                f"Edge file '{file_path}' must contain 'src' and 'dst' columns."
            )
            raise ValueError(
                f"Edge file '{file_path}' must contain 'src' and 'dst' columns."
            )
        try:
            edge_index = torch.tensor(
                df[["src", "dst"]].values.T, dtype=torch.long)
        except Exception as e:
            logging.error(
                f"Error converting edge index from '{file_path}' to tensor: {e}"
            )
            raise e

        key = (src_type, rel_type, dst_type)
        data[key].edge_index = edge_index
        edge_meta[key] = {"num_edges": edge_index.shape[1], "attr_dim": 0}
        logging.info(f"Loaded edge type {key}: {edge_index.shape[1]} edges.")

        extra_cols = set(df.columns) - {"src", "dst"}
        for col in extra_cols:
            try:
                attr_tensor = torch.tensor(df[col].values, dtype=torch.float)
            except Exception as e:
                logging.error(
                    f"Error converting extra edge attribute '{col}' in '{file_path}' to tensor: {e}"
                )
                raise e
            data[key][col] = attr_tensor

        # Check for a corresponding edge label file.
        for ext in [".csv", ".parquet", ".orc"]:
            label_file = os.path.join(edge_dir, f"{base}_label{ext}")
            if os.path.exists(label_file):
                try:
                    df_label = read_data(label_file)
                    labels = torch.tensor(
                        df_label.values.squeeze(), dtype=torch.long)
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                    data[key].y = labels
                    logging.info(
                        f"Loaded labels for edge type {key} from '{label_file}'."
                    )
                    break
                except Exception as e:
                    logging.error(
                        f"Error reading edge label file '{label_file}': {e}")
                    raise e

    # Process attribute edge files.
    attr_edge_files = [f for f in edge_files if "_attr" in f]
    for file in attr_edge_files:
        base = os.path.splitext(file)[0]
        if not base.endswith("_attr"):
            logging.error(
                f"Attribute edge file '{file}' does not follow naming convention (should contain '_attr')."
            )
            raise ValueError(
                f"Attribute edge file '{file}' does not follow naming convention."
            )
        edge_key_base = base[:-5]  # Remove trailing '_attr'
        parts = edge_key_base.split("_")
        if len(parts) != 3:
            logging.error(
                f"Attribute edge file '{file}' does not follow the 'src_relation_dst' naming convention."
            )
            raise ValueError(
                f"Attribute edge file '{file}' does not follow the 'src_relation_dst' naming convention."
            )
        src_type, rel_type, dst_type = parts
        file_path = os.path.join(edge_dir, file)
        try:
            df = read_data(file_path)
        except Exception as e:
            logging.error(
                f"Error reading attribute edge file '{file_path}': {e}")
            raise e
        try:
            edge_attr = torch.tensor(df.values, dtype=torch.float)
        except Exception as e:
            logging.error(
                f"Error converting edge attributes from '{file_path}' to tensor: {e}"
            )
            raise e

        key = (src_type, rel_type, dst_type)
        data[key].edge_attr = edge_attr
        edge_meta[key]["attr_dim"] = edge_attr.shape[1]
        logging.info(
            f"Loaded edge attributes for edge type {key} with shape {edge_attr.shape}."
        )

    logging.info(f"Edge meta mapping: {edge_meta}")
    return edge_meta


def build_heterodata(root_dir: str) -> (HeteroData, dict):
    """
    Construct a PyTorch Geometric HeteroData object from a root directory.

    The root directory must contain 'nodes/' and 'edges/' subdirectories.

    Returns:
        (HeteroData, meta) where meta is a dictionary with keys "nodes" and "edges".
    """
    if not os.path.isdir(root_dir):
        logging.error(f"Root directory '{root_dir}' does not exist.")
        raise ValueError(f"Root directory '{root_dir}' does not exist.")

    data = HeteroData()
    nodes_dir = os.path.join(root_dir, "nodes")
    edges_dir = os.path.join(root_dir, "edges")
    node_meta = load_nodes(nodes_dir, data)
    edge_meta = load_edges(edges_dir, data)
    meta = {"nodes": node_meta, "edges": edge_meta}
    return data, meta


def write_meta(meta: dict, meta_file: str):
    """
    Write the meta information to a JSON file.

    Edge meta keys (tuples) are converted to strings by joining with "_".
    """
    meta_for_json = {
        "nodes": meta["nodes"],
        "edges": {
            "_".join(map(str, key)): value for key, value in meta["edges"].items()
        },
    }
    try:
        with open(meta_file, "w") as f:
            json.dump(meta_for_json, f, indent=4)
        logging.info(f"Meta file written to '{meta_file}'.")
    except Exception as e:
        logging.error(f"Failed to write meta file '{meta_file}': {e}")
        raise e


def hetero_to_homogeneous(data: HeteroData, meta: dict) -> torch.Tensor:
    """
    Convert heterogeneous node features to a homogeneous feature matrix.

    For each node type (ordered by its meta id), pad the feature tensor to the maximum feature dimension.
    If there is more than one node type, append an extra column with the node type id.
    If there is only one node type, no extra column is added.

    Returns:
        torch.Tensor: Homogeneous feature matrix of shape [total_nodes, max_feat] or [total_nodes, max_feat + 1]
                     depending on the number of node types.
    """
    sorted_node_types = sorted(
        meta["nodes"].items(),
        key=lambda item: item[1]["id"])
    max_feat = max(data[node_type].x.shape[1]
                   for node_type, _ in sorted_node_types)

    homogeneous_features = []
    for node_type, info in sorted_node_types:
        x = data[node_type].x  # shape [num_nodes, feat_dim]
        num_nodes, feat_dim = x.shape
        pad_amount = max_feat - feat_dim
        if pad_amount > 0:
            x = F.pad(x, (0, pad_amount), mode="constant", value=0)
        # Only append node type id column if more than one node type exists.
        if len(sorted_node_types) > 1:
            type_id_col = torch.full(
                (num_nodes, 1), fill_value=info["id"], dtype=x.dtype
            )
            x = torch.cat([x, type_id_col], dim=1)
        homogeneous_features.append(x)
        if len(sorted_node_types) > 1:
            logging.info(
                f"Processed node type '{node_type}': original dim {feat_dim}, padded to {max_feat}."
            )

    homogeneous_x = torch.cat(homogeneous_features, dim=0)
    if len(sorted_node_types) > 1:
        logging.info(
            f"Homogeneous node feature matrix shape: {homogeneous_x.shape}")
    return homogeneous_x


def hetero_edges_to_homogeneous(data: HeteroData, meta: dict) -> torch.Tensor:
    """
    Convert heterogeneous edge indices to a single homogeneous edge index tensor.

    For each edge type (using a sorted order of data.edge_types), adjust the local indices by adding an offset.
    The offset for a node type is the cumulative number of nodes for all node types with lower meta id.

    Returns:
        torch.Tensor: Homogeneous edge index tensor of shape [2, total_edges]
    """
    sorted_node_types = sorted(
        meta["nodes"].items(),
        key=lambda item: item[1]["id"])
    offsets = {}
    cumulative = 0
    for node_type, info in sorted_node_types:
        offsets[node_type] = cumulative
        cumulative += info["num_nodes"]
        logging.info(
            f"Offset for node type '{node_type}' (id {info['id']}): {offsets[node_type]}"
        )

    edge_list = []
    sorted_edge_types = sorted(
        data.edge_types, key=lambda x: (
            x[0].lower(), x[1].lower(), x[2].lower())
    )
    for edge_type in sorted_edge_types:
        if hasattr(data[edge_type], "edge_index"):
            edge_index = data[edge_type].edge_index.clone()
            src_type, _, dst_type = edge_type
            edge_index[0] += offsets[src_type]
            edge_index[1] += offsets[dst_type]
            edge_list.append(edge_index)
            logging.info(
                f"Processed edge type {edge_type} with {edge_index.shape[1]} edges."
            )
        else:
            logging.warning(
                f"Edge type {edge_type} does not have an 'edge_index' attribute."
            )

    if edge_list:
        homogeneous_edge_index = torch.cat(edge_list, dim=1)
        logging.info(
            f"Homogeneous edge index shape: {homogeneous_edge_index.shape}")
        return homogeneous_edge_index
    else:
        logging.warning("No edges found in the heterogeneous data.")
        return torch.empty((2, 0), dtype=torch.long)


def hetero_edge_attrs_to_homogeneous(data: HeteroData) -> torch.Tensor:
    """
    Convert heterogeneous edge attributes to a single homogeneous edge attribute matrix.

    Iterates over edge types (using a sorted order) in the same order as the edge index conversion.
    Determines the maximum attribute dimension, pads each edge attribute tensor if needed,
    and concatenates them in the same order.

    Returns:
        torch.Tensor: Homogeneous edge attribute matrix of shape [total_edges, max_attr_dim]
    """
    max_attr_dim = 0
    sorted_edge_types = sorted(
        data.edge_types, key=lambda x: (
            x[0].lower(), x[1].lower(), x[2].lower())
    )
    for edge_type in sorted_edge_types:
        if hasattr(data[edge_type], "edge_attr"):
            attr = data[edge_type].edge_attr
            max_attr_dim = max(max_attr_dim, attr.shape[1])

    edge_attr_list = []
    for edge_type in sorted_edge_types:
        if hasattr(data[edge_type], "edge_attr"):
            attr = data[edge_type].edge_attr
            if attr.shape[1] < max_attr_dim:
                attr = F.pad(
                    attr, (0, max_attr_dim - attr.shape[1]), mode="constant", value=0
                )
            edge_attr_list.append(attr)
        else:
            if hasattr(data[edge_type], "edge_index"):
                num_edges = data[edge_type].edge_index.shape[1]
            else:
                num_edges = 0
            attr = torch.zeros((num_edges, max_attr_dim), dtype=torch.float)
            edge_attr_list.append(attr)

    if edge_attr_list:
        homogeneous_edge_attrs = torch.cat(edge_attr_list, dim=0)
        logging.info(
            f"Homogeneous edge attribute matrix shape: {homogeneous_edge_attrs.shape}"
        )
        return homogeneous_edge_attrs
    else:
        logging.warning("No edge attributes found in the heterogeneous data.")
        return torch.empty((0, max_attr_dim), dtype=torch.float)


def hetero_to_homogeneous_labels(
    data: HeteroData, meta: dict, default_label: int
) -> torch.Tensor:
    """
    Convert heterogeneous node labels to a single homogeneous node label tensor.

    For each node type (ordered by its meta id), if a label file was loaded (data[node_type].y exists),
    use it; otherwise, create a label tensor filled with the default_label.
    Returns None if no node label file was loaded for any node type.

    Returns:
        torch.Tensor: Homogeneous node label tensor of shape [total_nodes, 1] with integer labels, or None.
    """
    if not any(hasattr(data[node_type], "y") for node_type in meta["nodes"]):
        logging.info(
            "No node label file found; skipping homogeneous node label creation."
        )
        return None

    sorted_node_types = sorted(
        meta["nodes"].items(),
        key=lambda item: item[1]["id"])
    label_list = []
    for node_type, info in sorted_node_types:
        num_nodes = info["num_nodes"]
        if hasattr(data[node_type], "y"):
            labels = data[node_type].y
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
        else:
            labels = torch.full(
                (num_nodes, 1), default_label, dtype=torch.long)
        label_list.append(labels)
        logging.info(
            f"Processed labels for node type '{node_type}' with {num_nodes} entries."
        )
    homogeneous_labels = torch.cat(label_list, dim=0)
    logging.info(
        f"Homogeneous node label tensor shape: {homogeneous_labels.shape}")
    return homogeneous_labels


def hetero_edges_to_homogeneous_labels(
    data: HeteroData, default_label: int
) -> torch.Tensor:
    """
    Convert heterogeneous edge labels to a single homogeneous edge label tensor.

    Iterates over edge types (using a sorted order) in the same order as the homogeneous edge index.
    For each edge type, if a label file was loaded (data[edge_type].y exists), use it; otherwise,
    create a label tensor filled with the default_label.
    Returns None if no edge label file was loaded for any edge type.

    Returns:
        torch.Tensor: Homogeneous edge label tensor of shape [total_edges, 1] with integer labels, or None.
    """
    if not any(hasattr(data[edge_type], "y") for edge_type in data.edge_types):
        logging.info(
            "No edge label file found; skipping homogeneous edge label creation."
        )
        return None

    sorted_edge_types = sorted(
        data.edge_types, key=lambda x: (
            x[0].lower(), x[1].lower(), x[2].lower())
    )
    label_list = []
    for edge_type in sorted_edge_types:
        if hasattr(data[edge_type], "edge_index"):
            num_edges = data[edge_type].edge_index.shape[1]
        else:
            num_edges = 0
        if hasattr(data[edge_type], "y"):
            labels = data[edge_type].y
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
        else:
            labels = torch.full(
                (num_edges, 1), default_label, dtype=torch.long)
        label_list.append(labels)
        logging.info(
            f"Processed labels for edge type {edge_type} with {num_edges} entries."
        )
    if label_list:
        homogeneous_edge_labels = torch.cat(label_list, dim=0)
    else:
        homogeneous_edge_labels = torch.empty((0, 1), dtype=torch.long)
    logging.info(
        f"Homogeneous edge label tensor shape: {homogeneous_edge_labels.shape}"
    )
    return homogeneous_edge_labels


def read_offset_range_of_training_node(
    nodes_dir: str = "nodes", filename: str = "offset_range_of_training_node.json"
) -> dict:
    """
    Reads a JSON file containing the offset range for training nodes from the specified nodes directory.

    The JSON file is expected to have the following format:
    {
        "start": ST,  # start offset of the transaction node
        "end": ET     # end offset of the transaction node
    }

    If the file does not exist, returns None.
    If either the "start" or "end" key is missing, logs an error and returns None.

    Args:
        nodes_dir (str): The directory where the JSON file is located (default is "nodes").
        filename (str): The name of the JSON file (default is "offset_range_of_training_node.json").

    Returns:
        dict or None: A dictionary with keys "start" and "end" if present; otherwise, None.
    """
    file_path = os.path.join(nodes_dir, filename)
    if not os.path.exists(file_path):
        logging.error(f"JSON file '{file_path}' does not exist.")
        return None

    try:
        with open(file_path, "r") as f:
            offset_dict = json.load(f)
    except Exception as e:
        logging.error(f"Error reading JSON file '{file_path}': {e}")
        return None

    if "start" not in offset_dict:
        logging.error(f"Key 'start' not found in JSON file '{file_path}'.")
        return None
    if "end" not in offset_dict:
        logging.error(f"Key 'end' not found in JSON file '{file_path}'.")
        return None

    return offset_dict


def read_graph_data(
    root_dir, meta_file="meta.json", default_node_label=0, default_edge_label=0
):
    """
    Convert heterogeneous PyTorch Geometric data to homogeneous format.

    Args:
      - root_dir (str): Root directory containing nodes/ and edges/ folders.
      - meta_file (str): Path to output the meta information as a JSON file.
      - default_node_label (int): Default label value for nodes without a label file.
      - default_edge_label (int): Default label value for edges without a label file.
    """

    try:

        hetero_data, meta = build_heterodata(root_dir)
        write_meta(meta, meta_file)

        # Convert nodes.
        homogeneous_x = hetero_to_homogeneous(hetero_data, meta)
        logging.info(f"Node feature matrix shape: {homogeneous_x.shape}")

        # Convert node labels if available.
        homogeneous_node_labels = hetero_to_homogeneous_labels(
            hetero_data, meta, default_node_label
        )
        if homogeneous_node_labels is not None:
            logging.info(
                f"Node label tensor shape: {homogeneous_node_labels.shape}")
        else:
            logging.info(
                "No node labels available; skipping homogeneous node label tensor."
            )

        # Convert edges.
        homogeneous_edge_index = hetero_edges_to_homogeneous(hetero_data, meta)
        logging.info(f"edge index shape: {homogeneous_edge_index.shape}")

        # Convert edge attributes.
        homogeneous_edge_attrs = hetero_edge_attrs_to_homogeneous(hetero_data)
        logging.info(
            f"edge attribute matrix shape: {homogeneous_edge_attrs.shape}")

        # Convert edge labels if available.
        homogeneous_edge_labels = hetero_edges_to_homogeneous_labels(
            hetero_data, default_edge_label
        )
        if homogeneous_edge_labels is not None:
            logging.info(
                f"edge label tensor shape: {homogeneous_edge_labels.shape}")
        else:
            logging.info(
                "No edge labels available; skipping homogeneous edge label tensor."
            )

        training_node_offset = read_offset_range_of_training_node(
            os.path.join(root_dir, "nodes")
        )
        if training_node_offset:
            logging.info(
                f"No edge labels available; skipping homogeneous edge label tensor. {training_node_offset}"
            )

        if training_node_offset:
            training_node_offset_range = (
                training_node_offset["start"],
                training_node_offset["end"],
            )
        else:
            training_node_offset_range = (0, len(homogeneous_x))

        return (
            Data(
                x=homogeneous_x,
                edge_index=homogeneous_edge_index,
                y=homogeneous_node_labels,
            ),
            training_node_offset_range,
        )

    except Exception as e:
        logging.error(f"Invalid data: {e}")
        sys.exit(1)
