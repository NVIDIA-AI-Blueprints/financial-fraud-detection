# Model training NIM


## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

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
## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab-master.nvidia.com/RAPIDS/graph-nims.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab-master.nvidia.com/RAPIDS/graph-nims/-/settings/integrations)


## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
