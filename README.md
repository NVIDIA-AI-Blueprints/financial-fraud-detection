# Graph NIMs

This repo contains a collection of graph-based microservices that will be release as [NVIDIA NIMs](https://docs.nvidia.com/nim/index.html). The repo is organized by folders, where each folder (excluding docs) is a NIM.

<h1 align="center"; style="font-style: italic";>
  <img src="docs/imgs/nv-min.png" alt="NIM" width="300">
</h1>


The first set of NIMs focuses on GNN training and inference.  


## Training
This NIM handle building a XGBoost model from generated GNN embeddings.  The NIM takes input data and produced the XGBoost model along with the GNN embedding translation model. The NIM encapsulate the complexity of creating the graph in cuGraph and building the attribute key-value store in WholeGraph. Once the geraph is created, the GNN is run and used to produce the embeddings that are then feed to XGBoost.  

<h1 align="center"; style="font-style: italic";>
  <img src="docs/imgs/training-nim.png" alt="Training NIM" width="500">
</h1>

For testing:
* Step 1: Build the Docker container
```
  cd training
  sudo docker build --tag "model_builder_container" .
```
* Step 2: Run the container


docker run -it --rm  --gpus all -e CUDA_VISIBLE_DEVICES=0,1  -v /home/mnaim/morpheus-experimental/ai-credit-fraud-workflow/data/TabFormer:/data  -v ~/training_configs/training_config.json:/app/config.json model_builder_container --config /app/config.json 

A) You only need to change 0,1 to 0 if you have only one GPU
B) Path to data, /home/mnaim/morpheus-experimental/ai-credit-fraud-workflow/data/TabFormer
C) path to json file  ~/training_configs/training_config.json

sudo docker build --tag "model_builder_container" .

