from typing import List, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum

class ModelType(Enum):
    XGB = "XGBoost"
    GRAPH_SAGE_XGB = "GraphSAGE_XGBoost"

class GPUOption(str, Enum):
    SINGLE = "single"
    MULTIGPU = "multi"


# Strict Base Model

class StrictBaseModel(BaseModel):
    """
    A custom base model that forbids extra/unknown fields.
    """
    class Config:
        extra = "forbid"


# XGB Hyperparameters

class XGBHyperparametersSingle(StrictBaseModel):
    """
    Hyperparameters for XGB when each parameter is a single numeric value.
    """
    max_depth: int = Field(
        ...,
        description="The maximum depth of each tree. e.g., 3"
    )
    learning_rate: float = Field(
        ...,
        description="Boosting learning rate. e.g., 0.1"
    )
    n_estimators: int = Field(
        ...,
        description="Number of gradient-boosted trees. e.g., 100"
    )
    gamma: float = Field(
        ...,
        description="Minimum loss reduction required to make a partition. e.g., 0.0"
    )

class XGBHyperparametersList(StrictBaseModel):
    """
    Hyperparameters for XGB when each parameter is a list of possible numeric values.
    Often used for hyperparameter search or tuning.
    """
    max_depth: List[int] = Field(
        ...,
        description="A list of possible max_depth values. e.g., [3, 6, 9]"
    )
    learning_rate: List[float] = Field(
        ...,
        description="A list of possible learning rates. e.g., [0.01, 0.1]"
    )
    n_estimators: List[int] = Field(
        ...,
        description="A list of possible numbers of trees. e.g., [50, 100, 200]"
    )
    gamma: List[float] = Field(
        ...,
        description="A list of gamma values for regularization. e.g., [0.0, 0.1]"
    )


class XGBSingleConfig(StrictBaseModel):
    """
    XGB configuration where hyperparameters are single values.
    Discriminated by kind='xgb_single'.
    """
    kind: Literal[ModelType.XGB.value] = ModelType.XGB.value
    gpu: GPUOption = Field(
        ...,
        description="Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'."
    )
    hyperparameters: XGBHyperparametersSingle = Field(
        ...,
        description="All hyperparameters in single-value form."
    )

class XGBListConfig(StrictBaseModel):
    """
    XGB configuration where hyperparameters are lists of values.
    Discriminated by kind='xgb_list'.
    """
    kind: Literal[ModelType.XGB.value] = ModelType.XGB.value
    gpu: GPUOption = Field(
        ...,
        description="Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'."    
    )
    hyperparameters: XGBHyperparametersList = Field(
        ...,
        description="All hyperparameters in list-of-values form."
    )


# GraphSAGE Models

class GraphSAGEHyperparametersSingle(StrictBaseModel):
    """
    Hyperparameters for XGB when each parameter is a list of possible numeric values.
    Often used for hyperparameter search or tuning.
    """
    in_channels: int = Field(
        ...,
        description="Number of input channels (e.g., 64)."
    )
    hidden_channels: int = Field(
        ...,
        description="Number of hidden channels (e.g., 128)."
    )
    out_channels: int = Field(
        ...,
        description="Number of output channels (e.g., 10)."
    )
    n_hops: int = Field(
        ...,
        description="Number of hop values or layers (e.g., 2)."
    )
    dropout_prob: float = Field(
        ...,
        description="Dropout probability (e.g., 0.2)."
    )
    batch_size: int = Field(
        ...,
        description="Batch size (e.g., 32)."
    )


class GraphSAGEHyperparametersList(StrictBaseModel):
    """
    Hyperparameters for XGB when each parameter is a list of possible numeric values.
    Often used for hyperparameter search or tuning.
    """
    in_channels: List[int] = Field(
        ...,
        description="List of possible input channels."
    )
    hidden_channels: List[int] = Field(
        ...,
        description="List of possible hidden channels."
    )
    out_channels: int = Field(
        ...,
        description="Number of output channels (usually a single integer)."
    )
    n_hops: List[int] = Field(
        ...,
        description="List of hop values for each variant or layer."
    )
    dropout_prob: List[float] = Field(
        ...,
        description="List of possible dropout probabilities."
    )
    batch_size: List[int] = Field(
        ...,
        description="List of possible batch sizes."
    )


class GraphSAGESingleConfig(StrictBaseModel):
    """
    GraphSAGE configuration where each hyperparameter is a single value.
    Discriminated by kind='graphsage_single'.
    """
    kind: Literal[ModelType.GRAPH_SAGE_XGB.value] = ModelType.GRAPH_SAGE_XGB.value
    gpu: GPUOption = Field(
        ...,
        description="Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'."    
    )
    hyperparameters: GraphSAGEHyperparametersSingle = Field(
        ...,
        description="All hyperparameters in single-value form."
    )


class GraphSAGEListConfig(StrictBaseModel):
    """
    GraphSAGE configuration where hyperparameters are lists of values.
    Discriminated by kind='graphsage_list'.
    """
    kind: Literal[ModelType.GRAPH_SAGE_XGB.value] = ModelType.GRAPH_SAGE_XGB.value
    gpu: GPUOption = Field(
        ...,
        description="Indicates whether to use a single GPU or multiple GPUs. Valid options are 'single' or 'multi'."    
    )

    hyperparameters: GraphSAGEHyperparametersList = Field(
        ...,
        description="All hyperparameters in list-of-values form."
    )


# Union of All Configs
ModelConfig = Union[
    XGBSingleConfig, 
    XGBListConfig, 
    GraphSAGESingleConfig,
    GraphSAGEListConfig
]


class Paths(StrictBaseModel):
    """
    Holds directory paths for data and models.
    """
    data_dir: str = Field(..., description="Path to the input data directory.")
    output_dir: str = Field(..., description="Path to the output models directory.")

class FullConfig(StrictBaseModel):
    """
    Wrapper model that includes `paths` and a list of `models`.
    """
    paths: Paths = Field(..., description="Directory paths for data and models.")
    models: List[ModelConfig] = Field(..., description="List of model configurations.")
