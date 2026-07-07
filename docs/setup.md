
## Step 1: Clone the repo

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection
```

## Step 2: Create a new conda environment

You can get a minimum installation of Conda and Mamba using [Miniforge](https://github.com/conda-forge/miniforge).

And then create an environment using the following command.

Make sure that your shell or command prompt is pointing to `Financial-Fraud-Detection` before running `mamba env create`.

```bash
~/Financial-Fraud-Detection$ mamba env create -f conda/notebook_env.yaml
```


Alternatively, you can install [MiniConda](https://docs.anaconda.com/miniconda/miniconda-install) and run the following commands to create an environment to run the notebooks.

 Install `mamba` first with

```bash
conda install conda-forge::mamba
```
And, then run `mamba env create` from the right directory as shown below.

```bash
~/Financial-Fraud-Detection$ mamba env create -f conda/notebook_env.yaml
```

Finally, activate the environment.

```bash
conda activate fraud_blueprint_env
```


## Step 3: Download and organize the data

__TabFormer__</br>
1. Download the dataset: https://ibm.ent.box.com/v/tabformer-data/folder/130747715605
2. untar and uncompreess the file:  `tar -xvzf ./transactions.tgz`
3. Put `card_transaction.v1.csv` in in the ___"TabFormer/raw"___ folder 
```sh
TabFormer
    └── raw
        └── card_transaction.v1.csv
```

