# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.


from typing import List, Literal, Union, Optional
from pydantic import BaseModel, ConfigDict, Field, conint, conlist, confloat
from enum import Enum


class ModelType(Enum):
    XGB = "XGBoost"
    GRAPH_SAGE_XGB = "GraphSAGE_XGBoost"


class GPUOption(str, Enum):
    SINGLE = "single"
    MULTIGPU = "multi"


class StrictBaseModel(BaseModel):
    """
    A custom base model that forbids extra/unknown fields.
    """

    model_config = ConfigDict(extra="forbid")


class XGBHyperparametersSingle(StrictBaseModel):
    """
    Hyperparameters for XGB when each parameter is a single numeric value.
    """

    max_depth: int = Field(...,
                           description="The maximum depth of each tree. e.g., 3")
    num_parallel_tree: int = Field(...,
                                   description="Size of the forest being trained.")
    num_boost_round: int = Field(...,
                                 description="Number of boosting rounds. e.g., 1")
    learning_rate: float = Field(...,
                                 description="Boosting learning rate. e.g., 0.1")
    gamma: float = Field(
        ...,
        description="Minimum loss reduction required to make a partition. e.g., 0.0",
    )


class XGBHyperparametersList(StrictBaseModel):
    """
    Hyperparameters for XGB when each parameter is a list of possible numeric values.
    Often used for hyperparameter search or tuning.
    """

    max_depth: List[int] = Field(
        ..., description="A list of possible max_depth values. e.g., [3, 6, 9]"
    )
    num_parallel_tree: List[int] = Field(
        ..., description="A list of possible numbers of trees. e.g., [50, 100, 200]"
    )
    num_boost_round: List[int] = Field(
        ..., description="A list of possible number of boosting rounds. e.g., [1, 2, 4]"
    )
    learning_rate: List[float] = Field(
        ..., description="A list of possible learning rates. e.g., [0.01, 0.1]"
    )
    gamma: List[float] = Field(
        ..., description="A list of gamma values for regularization. e.g., [0.0, 0.1]"
    )


class XGBSingleConfig(StrictBaseModel):
    """
    XGB configuration where hyperparameters are single values.
    Discriminated by kind='xgb_single'.
    """

    kind: Literal[ModelType.XGB.value] = ModelType.XGB.value
    gpu: GPUOption = Field(
        ...,
        description="Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'.",
    )
    hyperparameters: XGBHyperparametersSingle = Field(
        ..., description="All hyperparameters in single-value form."
    )


class XGBListConfig(StrictBaseModel):
    """
    XGB configuration where hyperparameters are lists of values.
    Discriminated by kind='xgb_list'.
    """

    kind: Literal[ModelType.XGB.value] = ModelType.XGB.value
    gpu: GPUOption = Field(
        ...,
        description="Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'.",
    )
    hyperparameters: XGBHyperparametersList = Field(
        ..., description="All hyperparameters in list-of-values form."
    )


class GraphSAGEHyperparametersSingle(StrictBaseModel):
    """
    Hyperparameters for XGB when each parameter is a list of possible numeric values.
    Often used for hyperparameter search or tuning.
    """

    hidden_channels: int = Field(
        ..., description="Number of hidden channels (e.g., 32)."
    )
    n_hops: int = Field(...,
                        description="Number of hop values or layers (e.g., 2).")
    dropout_prob: float = Field(...,
                                description="Dropout probability (e.g., 0.2).")
    batch_size: int = Field(..., description="Batch size (e.g., 1024).")
    fan_out: int = Field(
        ..., description="Number of neighbors to sample for each node (e.g., 16)."
    )
    metric: Literal["recall", "f1", "precision"] = Field(
        default="f1",
        description="The metric to be used. Must be one of: recall, f1, or precision.",
    )
    num_epochs: int = Field(
        4, ge=1, description="Number of epochs to train the model.")
    learning_rate: float = Field(
        0.005, gt=0, description="Learning rate (e.g., 0.005)."
    )
    weight_decay: float = Field(
        1e-5, ge=0, description="Learning rate (e.g., 0.005).")


class GraphSAGEHyperparametersList(StrictBaseModel):
    """
    Hyperparameters for XGB when each parameter is a list of possible numeric values.
    Often used for hyperparameter search or tuning.
    """

    hidden_channels: conlist(item_type=conint(gt=0), min_length=1) = Field(
        ..., description="List of positive integers representing hidden channel sizes."
    )

    n_hops: conlist(item_type=conint(gt=0), min_length=1) = Field(
        ...,
        description="List of positive integers representing the number of hops (or GNN layers).",
    )

    batch_size: conlist(item_type=conint(gt=0), min_length=1) = Field(
        ..., description="List of positive integers representing different batch sizes."
    )

    fan_out: conlist(item_type=conint(gt=0), min_length=1) = Field(
        ..., description="List of positive integers representing fan-outs."
    )

    metric: conlist(
        Literal["recall", "f1", "precision"], min_length=1, max_length=1
    ) = Field(
        default=["f1"],
        description="A list of metrics. Each must be one of: recall, f1, or precision.",
    )

    num_epochs: conlist(item_type=conint(gt=0), min_length=1) = Field(
        [16], description="List of positive integers representing num_epochs."
    )

    n_folds: Optional[conlist(item_type=conint(gt=0), min_length=1)] = Field(
        [5], description="List of positive integers representing number folds."
    )

    dropout_prob: Optional[conlist(item_type=confloat(ge=0, le=1), min_length=1)] = (
        Field(
            [0.1],
            description="List of floats in [0, 1], each representing dropout probabilities.",
        )
    )

    learning_rate: Optional[conlist(item_type=confloat(ge=0, le=1), min_length=1)] = (
        Field(
            [0.005],
            description="List of floats in [0, 1], each representing learning rates.",
        )
    )

    weight_decay: Optional[conlist(item_type=confloat(ge=0, le=0.1), min_length=1)] = (
        Field(
            [1e-5],
            description="List of floats in [0, 0.1], each representing learning rates.",
        )
    )


class GraphSAGEAndXGBConfig(StrictBaseModel):
    """
    GraphSAGE configuration where each hyperparameter is a single value.
    Discriminated by kind='graphsage_single'.
    """

    gnn: GraphSAGEHyperparametersSingle = Field(
        ..., description="Hyperparameters for GraphSAGE model."
    )
    xgb: XGBHyperparametersSingle = Field(
        ..., description="Hyperparameters for xgboost model."
    )


class GraphSAGEGridAndXGBConfig(StrictBaseModel):
    """
    GraphSAGE configuration where each hyperparameter is a single value.
    Discriminated by kind='graphsage_single'.
    """

    gnn: GraphSAGEHyperparametersList = Field(
        ..., description="Hyperparameters for GraphSAGE model."
    )
    xgb: XGBHyperparametersSingle = Field(
        ..., description="Hyperparameters for xgboost model."
    )


class GraphSAGEAndXGB(StrictBaseModel):
    """
    GraphSAGE configuration where each hyperparameter is a single value.
    Discriminated by kind='graphsage_single'.
    """

    kind: Literal[ModelType.GRAPH_SAGE_XGB.value] = ModelType.GRAPH_SAGE_XGB.value
    gpu: GPUOption = Field(
        ...,
        description="Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'.",
    )
    hyperparameters: GraphSAGEAndXGBConfig = Field(
        ...,
        description="Hyperparameters of GraphSAGE and XGBoost model in single-value form.",
    )


class GraphSAGEGridAndXGB(StrictBaseModel):
    """
    GraphSAGE configuration where hyperparameters are lists of values.
    Discriminated by kind='graphsage_list'.
    """

    kind: Literal[ModelType.GRAPH_SAGE_XGB.value] = ModelType.GRAPH_SAGE_XGB.value
    gpu: GPUOption = Field(
        ...,
        description="Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'.",
    )

    hyperparameters: GraphSAGEGridAndXGBConfig = Field(
        ..., description="All hyperparameters in list-of-values form."
    )


ModelConfig = Union[
    XGBSingleConfig, XGBListConfig, GraphSAGEAndXGB, GraphSAGEGridAndXGB
]


class Paths(StrictBaseModel):
    """
    Holds directory paths for data and models.
    """

    data_dir: str = Field(..., description="Path to the input data directory.")
    output_dir: str = Field(...,
                            description="Path to the output models directory.")


class FullConfig(StrictBaseModel):
    """
    Wrapper model that includes `paths` and a list of `models`.
    """

    paths: Paths = Field(...,
                         description="Directory paths for data and models.")
    models: List[ModelConfig] = Field(...,
                                      description="List of model configurations.")
