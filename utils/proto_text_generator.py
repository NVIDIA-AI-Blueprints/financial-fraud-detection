def generate_xgb_pbtxt(
    model_file_name, input_dim: int, decision_threshold: float, path_to_xgb_pbtx: str
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
    dims: [ -1, 1 ]
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
    Write Protocol Buffers Text for GraphSAGE model.
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
    dims: [-1, {hidden_dim} ]
  }}
]
instance_group [{{ kind: KIND_GPU }}]
"""
    # Write the content to the specified file.
    with open(path_to_gnn_pbtx, "w") as file:
        file.write(pbtx_content)

    print(f"\nSaved GraphSAGE model config to {path_to_gnn_pbtx}")
