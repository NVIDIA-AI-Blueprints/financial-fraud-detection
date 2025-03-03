from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
import json
import logging

from src.validate_and_launch import validate_config_and_run_training


def generate_random_xgb_data(root_data_dir: str, ext: str) -> str:
    """
    Generates random data for XGBoost model training and saves it under:

        <root_data_dir>/<ext>/xgb/
            training.<ext>
            validation.<ext>
            test.<ext>

    Data: 100 rows with 9 feature columns and 1 target column (last column).
    Target is binary. Data is split into 70% training, 15% validation, and 15% test.

    Args:
        root_data_dir (str): The root data folder.
        ext (str): The file format to use ("csv", "parquet", or "orc").

    Returns:
        str: The path to the directory where the data is saved.
    """
    ext = ext.lower()
    if ext not in {"csv", "parquet", "orc"}:
        raise ValueError("Unsupported extension: " + ext)

    # Create output directory: <root_data_dir>/<ext>/xgb/
    output_dir = Path(root_data_dir) / ext / "xgb"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random data: 100 rows, 9 features and 1 target (binary)
    n_rows = 100
    n_features = 9
    X = np.random.rand(n_rows, n_features).astype(np.float32)
    y = np.random.randint(0, 2, size=(n_rows, 1)).astype(np.int32)

    data = np.hstack([X, y])
    columns = [f"feature_{i}" for i in range(1, n_features + 1)] + ["target"]
    df = pd.DataFrame(data, columns=columns)

    # Shuffle and split indices: 70% train, 15% validation, 15% test.
    indices = np.random.permutation(n_rows)
    n_train = int(n_rows * 0.70)
    n_val = int(n_rows * 0.15)
    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]

    # Build file paths.
    training_path = output_dir / f"training.{ext}"
    validation_path = output_dir / f"validation.{ext}"
    test_path = output_dir / f"test.{ext}"

    # Write files in the specified format.
    if ext == "csv":
        df_train.to_csv(training_path, index=False)
        df_val.to_csv(validation_path, index=False)
        df_test.to_csv(test_path, index=False)
    elif ext == "parquet":
        df_train.to_parquet(training_path, index=False)
        df_val.to_parquet(validation_path, index=False)
        df_test.to_parquet(test_path, index=False)
    elif ext == "orc":
        table_train = pa.Table.from_pandas(df_train)
        orc.write_table(table_train, str(training_path))
        table_val = pa.Table.from_pandas(df_val)
        orc.write_table(table_val, str(validation_path))
        table_test = pa.Table.from_pandas(df_test)
        orc.write_table(table_test, str(test_path))

    logging.info(f"Random data generated in {output_dir}")
    return str(Path(root_data_dir) / ext)


@pytest.mark.parametrize("fmt", ["csv", "parquet", "orc"])
def test_with_three_data_format(tmp_path: Path, fmt):

    data_path = generate_random_xgb_data(str(tmp_path), fmt)
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
                "kind": "XGBoost",
                "gpu": "single",
                "hyperparameters": {
                    "max_depth": 6,
                    "learning_rate": 0.2,
                    "num_parallel_tree": 3,
                    "num_boost_round": 512,
                    "gamma": 0.0,
                },
            }
        ],
    }

    filepath = tmp_path / "tmp_train_config.json"
    with filepath.open("w") as f:
        json.dump(config, f, indent=4)
    validate_config_and_run_training(str(filepath))
