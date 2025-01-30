# Graph NIMs

This repo contains a collection of graph-based microservices that will be release as [NVIDIA NIMs](https://docs.nvidia.com/nim/index.html). The repo is organized by folders, where each folder (excluding docs) is a NIM.

<h1 align="center"; style="font-style: italic";>
  <img src="docs/imgs/nv-min.png" alt="NIM" width="300">
</h1>


The first set of NIMs focuses on GNN training and inference.  


## Training
This NIM handle building a XGBoost model from generated GNN embeddings. The NIM takes input data and produced the XGBoost model along with the GNN embedding translation model. The NIM encapsulate the complexity of creating the graph in cuGraph and building the attribute key-value store in WholeGraph. Once the graph is created, the GNN is run and used to produce the embeddings that are then feed to XGBoost.

<h1 align="center"; style="font-style: italic";>
  <img src="docs/imgs/training-nim.png" alt="Training NIM" width="500">
</h1>

For testing:
* Step 1: Build the Docker container
```sh
 cd graph-nims/training
 docker build --no-cache -t model_builder_container .
```
* Step 2: Run the container
```sh
 docker run --cap-add SYS_NICE -it --rm  --gpus all -v path_to_data_dir:/data  -v path_to_train_config_json_file:/app/config.json model_builder_container --config /app/config.json
````
