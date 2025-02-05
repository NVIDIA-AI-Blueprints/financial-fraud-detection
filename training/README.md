# Model training NIM

## How to run locally

Build docker image
```sh
 git clone ssh://git@gitlab-master.nvidia.com:12051/mnaim/graph-nims.git
 cd graph-nims
 docker build --no-cache -t model_builder_container .
 ```

Run training
 ```sh
 docker run --cap-add SYS_NICE -it --rm  --gpus all -e CUDA_VISIBLE_DEVICES=0,1  -v path_to_data_dir:/data  -v path_to_train_config_json_file:/app/config.json model_builder_container --config /app/config.json
```
