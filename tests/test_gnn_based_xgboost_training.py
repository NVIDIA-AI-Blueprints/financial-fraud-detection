# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.

from pathlib import Path
import logging
import json
import os
import pytest
import pandas as pd

from src.validate_and_launch import validate_config_and_run_training
from ogb.nodeproppred import PygNodePropPredDataset


def ogbn_proteins_root_dir(root_dir: str = "tests/dataset/ogbn-proteins"):
    """
    A session-scoped fixture that:
      1) Loads the ogbn-proteins dataset (if needed).
      2) Creates a directory structure:
            <root_dir>/
                nodes/
                    node.parquet       (Nx1 'species')
                    node_label.parquet (Nx1 'label')
                edges/
                    node_to_not.parquet (src,dst)
      3) Ensures species & label each have N rows (Nx1).
      4) Returns the <root_dir> path (str).

    If the files exist at <root_dir>, it skips re-generation.
    Otherwise, it generates them.
    On any mismatch or error, it raises pytest.Fail (which fails the test session).
    """
    node_dir = os.path.join(root_dir, "nodes")
    edge_dir = os.path.join(root_dir, "edges")

    species_path = os.path.join(node_dir, "node.parquet")
    label_path = os.path.join(node_dir, "node_label.parquet")
    edge_path = os.path.join(edge_dir, "node_to_not.parquet")

    # Check if files already exist to skip re-generation
    if (
        os.path.exists(species_path)
        and os.path.exists(label_path)
        and os.path.exists(edge_path)
    ):
        logging.info(
            "OGBN-Proteins Parquet files already exist. Skipping generation.")
        return root_dir  # Just return the root directory

    # Otherwise, generate them
    logging.info("Generating OGBN-Proteins Parquet files in '%s'...", root_dir)

    os.makedirs(node_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)

    # Load dataset
    try:
        dataset = PygNodePropPredDataset(name="ogbn-proteins", root="data")
        data = dataset[0]  # single graph
        logging.info("Loaded ogbn-proteins dataset.")
    except Exception as e:
        logging.error(f"Could not load ogbn-proteins: {e}")
        pytest.fail(f"Dataset loading failed: {e}")

    # Species: Nx1
    if data.node_species is None:
        logging.error("data.node_species is None.")
        pytest.fail("Missing node_species in data.")

    species_arr = data.node_species.squeeze(-1).numpy()  # shape [N]
    species_arr = species_arr.reshape(-1, 1)  # Nx1
    df_species = pd.DataFrame(species_arr, columns=["species"])

    # Label: Nx1 (take the first column of y)
    if data.y is None:
        logging.error("data.y is None.")
        pytest.fail("Missing labels (y) in data.")

    if data.y.size(1) < 1:
        pytest.fail("data.y has 0 columns, cannot extract first label column.")

    label_arr = data.y[:, 0].numpy().reshape(-1, 1)  # Nx1
    df_label = pd.DataFrame(label_arr, columns=["label"])

    # Check shapes match
    if df_species.shape[0] != df_label.shape[0]:
        logging.error(
            f"Row mismatch: species={df_species.shape}, label={df_label.shape}"
        )
        pytest.fail("Species and label have different row counts.")

    # Edges
    if data.edge_index is None:
        logging.error("data.edge_index is None.")
        pytest.fail("Missing edge_index in data.")

    edge_index = data.edge_index.numpy()  # shape [2, E]
    src, dst = edge_index[0], edge_index[1]
    df_edges = pd.DataFrame({"src": src, "dst": dst})

    # Write to Parquet
    try:
        df_species.to_parquet(species_path)
        df_label.to_parquet(label_path)
        df_edges.to_parquet(edge_path)
    except Exception as e:
        logging.error(f"Failed to write Parquet files: {e}")
        pytest.fail(f"Failed to write Parquet files: {e}")

    logging.info("Wrote Nx1 species to:       %s", species_path)
    logging.info("Wrote Nx1 label   to:       %s", label_path)
    logging.info("Wrote edges (src,dst) to:    %s", edge_path)
    logging.info(
        "OGBN-Proteins Parquet generation complete under '%s'.",
        root_dir)

    return root_dir


def test_process_protein_data_dir(tmp_path: Path):
    """
    Tests the function 'process_protein_data_dir' which only takes a root directory.
    We rely on the ogbn_proteins_root_dir fixture to ensure the data is present and Nx1.
    """

    data_path = ogbn_proteins_root_dir("tests/dataset/ogbn-proteins")

    logging.info(
        "Testing process_protein_data_dir with root dir: %s",
        data_path)

    logging.info("Dataset dir %s", data_path)
    logging.info("Output_dir %s", str(tmp_path))

    config = {
        "paths": {"data_dir": "/data", "output_dir": "/trained_models"},
        "models": [
            {
                "kind": "GraphSAGE_XGBoost",
                "gpu": "single",
                "hyperparameters": {
                    "gnn": {
                        "hidden_channels": 16,
                        "n_hops": 1,
                        "dropout_prob": 0.1,
                        "batch_size": 1024,
                        "fan_out": 16,
                        "num_epochs": 16,
                    },
                    "xgb": {
                        "max_depth": 6,
                        "learning_rate": 0.2,
                        "num_parallel_tree": 3,
                        "num_boost_round": 512,
                        "gamma": 0.0,
                    },
                },
            }
        ],
    }

    filepath = tmp_path / "tmp_train_config.json"
    with filepath.open("w") as f:
        json.dump(config, f, indent=4)
    validate_config_and_run_training(str(filepath))
