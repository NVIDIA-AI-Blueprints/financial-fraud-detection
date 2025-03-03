from pathlib import Path
import os
import json
import logging
import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc

from src.validate_and_launch import validate_config_and_run_training
from ogb.nodeproppred import PygNodePropPredDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_graph_data_from_ogbn_proteins(
    root_dir: str = "tests/dataset/ogbn-proteins",
    ext: str = "parquet",
):
    """
    Loads the ogbn-proteins dataset and saves node and edge data in a subdirectory
    under the given root_dir, where the subdirectory is named after the file format.

    The resulting directory structure is:

        <root_dir>/<ext>/
            nodes/
                node.<ext>       (Nx1 'species')
                node_label.<ext> (Nx1 'label')
            edges/
                node_to_node.<ext> (src, dst)

    The function ensures that the species and label DataFrames have matching numbers of rows.
    It writes the files in the specified format (CSV, Parquet, or ORC) and returns the full
    directory path where the data was saved.

    Supported extensions: csv, parquet, orc.

    If the files already exist at the target directory, generation is skipped.
    On any error, pytest.fail is called to stop the test session.
    """
    ext = ext.lower()
    if ext not in {"csv", "parquet", "orc"}:
        logging.error("Unsupported extension: %s", ext)
        pytest.fail(f"Unsupported extension: {ext}")

    # Create a suffix path by joining the root directory with the extension
    output_dir = os.path.join(root_dir, ext)

    node_dir = os.path.join(output_dir, "nodes")
    edge_dir = os.path.join(output_dir, "edges")

    species_path = os.path.join(node_dir, f"node.{ext}")
    label_path = os.path.join(node_dir, f"node_label.{ext}")
    edge_path = os.path.join(edge_dir, f"node_to_node.{ext}")

    # Check if files already exist to skip re-generation
    if (
        os.path.exists(species_path)
        and os.path.exists(label_path)
        and os.path.exists(edge_path)
    ):
        logging.info(
            "OGBN-Proteins %s files already exist under '%s'. Skipping generation.",
            ext.upper(),
            output_dir,
        )
        return output_dir

    logging.info(
        "Generating OGBN-Proteins %s files in '%s'...", ext.upper(), output_dir
    )

    os.makedirs(node_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)

    # Load the ogbn-proteins dataset
    try:
        dataset = PygNodePropPredDataset(name="ogbn-proteins", root="data")
        data = dataset[0]  # Single graph dataset
        logging.info("Loaded ogbn-proteins dataset.")
    except Exception as e:
        logging.error("Could not load ogbn-proteins: %s", e)
        pytest.fail(f"Dataset loading failed: {e}")

    # Process species data (Nx1)
    if data.node_species is None:
        logging.error("data.node_species is None.")
        pytest.fail("Missing node_species in data.")
    species_arr = data.node_species.squeeze(-1).numpy().reshape(-1, 1)
    df_species = pd.DataFrame(species_arr, columns=["species"])

    # Process label data (Nx1, using the first column of y)
    if data.y is None:
        logging.error("data.y is None.")
        pytest.fail("Missing labels (y) in data.")
    if data.y.size(1) < 1:
        pytest.fail("data.y has 0 columns, cannot extract first label column.")
    label_arr = data.y[:, 0].numpy().reshape(-1, 1)
    df_label = pd.DataFrame(label_arr, columns=["label"])

    # Ensure species and label have the same number of rows
    if df_species.shape[0] != df_label.shape[0]:
        logging.error(
            "Row mismatch: species=%s, label=%s", df_species.shape, df_label.shape
        )
        pytest.fail("Species and label have different row counts.")

    # Process edge data
    if data.edge_index is None:
        logging.error("data.edge_index is None.")
        pytest.fail("Missing edge_index in data.")
    edge_index = data.edge_index.numpy()
    src, dst = edge_index[0], edge_index[1]
    df_edges = pd.DataFrame({"src": src, "dst": dst})

    # Write output files using the specified extension
    try:
        if ext == "parquet":
            df_species.to_parquet(species_path, index=False)
            df_label.to_parquet(label_path, index=False)
            df_edges.to_parquet(edge_path, index=False)
        elif ext == "csv":
            df_species.to_csv(species_path, index=False)
            df_label.to_csv(label_path, index=False)
            df_edges.to_csv(edge_path, index=False)
        elif ext == "orc":
            # Convert DataFrames to PyArrow Tables and write ORC files
            table_species = pa.Table.from_pandas(df_species)
            orc.write_table(table_species, species_path)
            table_label = pa.Table.from_pandas(df_label)
            orc.write_table(table_label, label_path)
            table_edges = pa.Table.from_pandas(df_edges)
            orc.write_table(table_edges, edge_path)
    except Exception as e:
        logging.error("Failed to write %s files: %s", ext.upper(), e)
        pytest.fail(f"Failed to write {ext.upper()} files: {e}")

    logging.info("Wrote Nx1 species to: %s", species_path)
    logging.info("Wrote Nx1 label   to: %s", label_path)
    logging.info("Wrote edges (src, dst) to: %s", edge_path)
    logging.info(
        "OGBN-Proteins %s generation complete under '%s'.", ext.upper(), output_dir
    )

    return output_dir


@pytest.mark.parametrize("fmt", ["csv", "parquet", "orc"])
def test_with_three_data_format(tmp_path: Path, fmt):

    data_path = get_graph_data_from_ogbn_proteins(
        "tests/dataset/ogbn-proteins", fmt)

    logging.info("Dataset dir %s", data_path)
    logging.info(
        f"Trained models and protobuf config files for deploying "
        f"on Triton Inference Server will be saved under %s",
        str(tmp_path),
    )

    output_dir = tmp_path / "output" / fmt
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "paths": {"data_dir": data_path, "output_dir": str(output_dir)},
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
