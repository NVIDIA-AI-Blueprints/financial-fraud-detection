# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.


def generate_xgb_pbtxt(
    model_file_name,
    input_dim: int,
    output_dim: int,
    decision_threshold: float,
    path_to_xgb_pbtx: str,
):
    """
    Write Protocol Buffers Text for XGBoost model.
    """
    # Use an f-string with escaped braces to insert the variable input dimension.
    pbtx_content = f"""\
default_model_filename: "{model_file_name}"
backend: "fil"
input [
 {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ -1, {input_dim} ]
 }}
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ -1, {output_dim} ]
 }}
]
instance_group [{{ kind: KIND_GPU }}]
parameters [
 {{
    key: "model_type"
    value: {{ string_value: "xgboost_json" }}
 }},
 {{
    key: "output_class"
    value: {{ string_value: "false" }}
 }}
]
"""
    # Write the PBtx content to the specified file.
    with open(path_to_xgb_pbtx, "w") as file:
        file.write(pbtx_content)
    print(f"\nSaved XGBoost model config to {path_to_xgb_pbtx}")


def generate_gnn_pbtxt(
    model_file_name: str, input_dim: int, hidden_dim: int, path_to_gnn_pbtx: str
):
    """
    Write Protocol Buffers Text for GNN model.
    """
    # Use an f-string to insert the parameter values into the PBtx content.
    pbtx_content = f"""\
default_model_filename: "{model_file_name}"
platform: "onnxruntime_onnx"
input [                                 
 {{  
    name: "x"
    data_type: TYPE_FP32
    dims: [-1, {input_dim} ]                    
  }},
  {{
    name: "edge_index"
    data_type: TYPE_INT64
    dims: [ 2, -1]
  }}
]
output [
 {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, {input_dim + hidden_dim } ]
  }}
]
instance_group [{{ kind: KIND_GPU }}]
"""
    # Write the content to the specified file.
    with open(path_to_gnn_pbtx, "w") as file:
        file.write(pbtx_content)

    print(f"\nSaved GraphSAGE model config to {path_to_gnn_pbtx}")


def generate_python_backend_pbtxt(
    model_name: str,
    gnn_state_dict_file_name: str,
    embedding_based_xgboost_model_filename: str,
    gnn_input_dim: int,
    gnn_hidden_dim: int,
    gnn_out_dim: int,
    gnn_n_hops: int,
    xgb_output_dim: int,
    path_to_gnn_pbtx: str,
):
    """
    Write Protocol Buffers Text for python backend.
    """
    # Use an f-string to insert the parameter values into the PBtx content.
    pbtx_content = f"""\
name: "{model_name}"
backend: "python"
input [
  {{
    name: "NODE_FEATURES"
    data_type: TYPE_FP32
    dims: [ -1, {gnn_input_dim} ]
  }},
  {{
    name: "EDGE_INDEX"
    data_type: TYPE_INT64
    dims: [ 2, -1 ]
  }},
  {{
    name: "COMPUTE_SHAP"
    data_type: TYPE_BOOL
    dims: [ 1 ]
  }},
  {{
    name: "FEATURE_MASK"
    data_type: TYPE_INT32
    dims: [ {gnn_input_dim} ]
  }}
]

output [
  {{
    name: "PREDICTION"
    data_type: TYPE_FP32
    dims: [ -1, {xgb_output_dim} ]
  }},
  {{
    name: "SHAP_VALUES"
    data_type: TYPE_FP32
    dims: [ -1, {gnn_input_dim} ]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
  }}
]


parameters {{
  key: "in_channels"
  value {{
    string_value: "{gnn_input_dim}"
  }}
}}
parameters {{
  key: "hidden_channels"
  value {{
    string_value: "{gnn_hidden_dim}"
  }}
}}
parameters {{
  key: "out_channels"
  value {{
    string_value: "{gnn_out_dim}"
  }}
}}
parameters {{
  key: "n_hops"
  value {{
    string_value: "{gnn_n_hops}"
  }}
}}
parameters {{
  key: "embedding_generator_model_state_dict"
  value {{
    string_value: "{gnn_state_dict_file_name}"
  }}
}}
parameters {{
  key: "embeddings_based_xgboost_model"
  value {{
    string_value: "{embedding_based_xgboost_model_filename}"
  }}
}}

"""
    # Write the content to the specified file.
    with open(path_to_gnn_pbtx, "w") as file:
        file.write(pbtx_content)

    print(f"\nSaved GraphSAGE model config to {path_to_gnn_pbtx}")
