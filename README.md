<h2><img align="center" src="https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f">NVIDIA AI Blueprint: Financial Fraud Detection
</h2>

## Table of Contents

- [Overview](#overview)
  - [Software Components](#software-components)
    - [Software Requirements](#software-requirements)
  - [Target Audience](#target-audience)
  - [Prerequisites](#prerequisites)
    - [Hardware Requirements](#hardware-requirements)
- [Getting Started](#getting-started)
  - [Installation System Requirements](#installation-system-requirements)
  - [Obtain API key](#obtain-api-key)
  - [Clone The Repository](#clone-the-repository)
  - [Set up the environment](#set-up-the-environment)
    - [API Key](#api-key)
    - [Conda Environment](#conda-environment)
    - [Authenticate Docker with NGC](#authenticate-docker-with-ngc)
  - [Running the workflow](#running-the-workflow)
- [License](#license)
- [Terms of Use](#terms-of-use)

<br>

__Notice__: This README is for users running the notebook locally and makes assumptions that the software can be installed on the hardware.


<br>


# Overview
Financial losses from worldwide credit card transaction fraud are [projected](https://www.paymentsdive.com/news/payments-fraud-losses-prevention-nilson-outlook/737440/) to reach more than $403 billion over the next decade. Transaction fraud poses a major challenge for financial institutions, which struggle to detect and prevent increasingly complicated fraudulent activities. Traditional fraud detection methods, which rely on rules-based systems or statistical methods, are reactive and increasingly ineffective in identifying sophisticated fraudulent activities. As data volumes grow and fraud tactics evolve, financial institutions need more proactive, intelligent approaches to detect and prevent fraudulent transactions.

This NVIDIA AI Blueprint provides a reference example to detect and prevent sophisticated fraudulent activities for financial services with high accuracy and reduced false positives. It shows developers how to build a financial fraud detection workflow using the NVIDIA container for fraud detection. For model building, the Financial Fraud Training container augments fraud detection using graph neural networks (GNNs)—a deep learning technique—for improved accuracy. Inference is done using [NVIDIA Dynamo-Triton (formerly Triton Inference Server)](https://developer.nvidia.com/dynamo) and produces fraud scores along with [Shapley values](https://en.wikipedia.org/wiki/Shapley_value) for explainability. Furthermore, to help simplify the workflow, the Financial Fraud Training container also produces all the needed configuration files required by Dynamo-Triton. 

![Architectural Diagram](https://assets.ngc.nvidia.com/products/api-catalog/financial-fraud-detection/diagram.jpg)

This NVIDIA AI blueprint is broken down into three steps, which map to processes within a typical payment processing environment, those steps are: (1) Data Preparation, (2) Model Building, and (3) Data Inference. For this example, the data is a collection of files containing synthetic data. Within a production system, the event data is often saved within a database or a data lake. The data is prepared and then fed into the [financial-fraud-training container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cugraph/containers/financial-fraud-training) model-building container. The output of the NIM folder with all the artifacts needs to be passed to Dynamo-Triton for inference.

This blueprint does not use any NVIDIA hosted services and runs fully in a locally hosted docker environment.

<br>

## Software Components
The following software components are used:
- [financial-fraud-training container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cugraph/containers/financial-fraud-training)
- [NVIDIA Dynamo-Triton](https://developer.nvidia.com/dynamo)

### Software Requirements
- Operating System: Ubuntu 20.04 or newer
- NVIDIA Driver version: 560.28.03 or newer
- NVIDIA CUDA version: 12.6 or newer
- NVIDIA Container Toolkit version: 1.15.0 or newer
- Docker version: Docker version 26 or newer

<br>

## Target Audience

This Blueprint targets users that:

- understand the financial fraud space
- understand how to deploy container-based microservices
- understand how to run a Jupyter notebook

This notebook is a simple example of how to orchestrate a financial fraud detection workflow that leverages the financial-fraud-training container. The notebook uses a synthetic dataset and produces the accuracy and a confusion matrix. Using real data in a production environment would not alter the workflow.

<br>

## Prerequisites

- [Obtain NVIDIA key](#obtain-api-key)
- [CUDA 12.6+ drivers](https://developer.nvidia.com/cuda-downloads) installed

<br>

### Hardware Requirements

- GPU: 1x A6000, A100, H100, or newer, minimum of 32 GB of memory 
- CPU: x86_64 architecture
- Storage: 10 GB
- System Memory: 16 GB


<br>

# Getting Started

## Installation System Requirements

- [git](https://git-scm.com/)
- [Jupyter Notebook or Jupyter Lab](https://jupyter.org/install)

Additional required Python packages are installed in the [conda environment](#conda-environment) step.

<br>

## Obtain API key

Here are two possible methods to generate an API key for NGC:

- Sign in to the [NVIDIA Build](https://build.nvidia.com/explore/discover?signin=true) portal with your email.
- Sign in to the [NVIDIA NGC](https://ngc.nvidia.com/) portal with your email.
  - Select your organization from the dropdown menu after logging in. You must select an organization which has NVIDIA AI Enterprise (NVAIE) enabled.
  - Click on your account in the top right, select "Setup" from the dropdown.
  - Click the "Generate Personal Key" option and then the "+ Generate Personal Key" button to create your API key.
    - This will be used as the `API_KEY` value in the notebook.
    - Click the "Generate API Key" option and then the "+ Generate API Key" button to create the API key.

IMPORTANT: This will be pasted into the `API_KEY` variable in the notebook.

- API catalog keys:
    NVIDIA [API catalog](https://build.nvidia.com/) or [NGC](https://org.ngc.nvidia.com/setup/personal-keys)

<br>

## Clone The Repository

   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection
   ```

<br>

## Set up the environment

### API Key

The notebooks read your NGC key from an in-notebook variable named `API_KEY`. In the environment-setup cell, replace the placeholder line with your key:

```python
API_KEY = "NGC API KEY"   # replace with your key, e.g. "nvapi-..."
```

This key is used to authenticate Docker with NGC (via `docker login`) so the required containers can be pulled. It is not read from an environment variable or a file, so it must be set inside the notebook.

### Conda Environment

The workflow uses Conda to create an environment with all the needed packages. You can get a minimum installation of Conda and Mamba using [Miniforge](https://github.com/conda-forge/miniforge).

Create an environment using the following command, making sure that you are in the `Financial-Fraud-Detection` folder.

```bash
 mamba env create -f conda/notebook_env.yaml
```

Finally, activate the environment.

```bash
conda activate fraud_blueprint_env
```



<br>

### Authenticate Docker with NGC

In order to pull images required by the Blueprint from NGC, Docker must be authenticated with NGC. The notebook does this for you using the `API_KEY` you set above. To authenticate manually from a shell, substitute your key from the [Obtain API key](#obtain-api-key) section:

```bash
echo "<your-ngc-api-key>" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

<br>

## Running the workflow

No command line option (CLI) is available without converting a Jupyter Notebook to a Python file.

Prerequisite: Start a browser

Starting Jupyter

__Option 1:__

```bash
   jupyter notebook
```

__Option 2:__

 Starting Jupyter Notebook with [notebooks/financial-fraud-usage-link-prediction.ipynb](notebooks/financial-fraud-usage-link-prediction.ipynb).

```bash
   cd notebooks
   jupyter notebook financial-fraud-usage-link-prediction.ipynb
```
NOTE: If you are interested in node prediction, use [notebooks/financial-fraud-usage-np.ipynb](notebooks/financial-fraud-usage-np.ipynb) instead.

__Option 3:__

Starting Jupyter Labs

```bash
jupyter-lab

or 

jupyter-lab --ip=* --no-browser
```

In either case above, Jupyter will output status information. One key line is the URL:

```bash
$ jupyter notebook
   ...
   The Jupyter Notebook is running at: http://localhost:8888/
```

Via the browser, connect to the specified URL and process the notebook.

<br>

# License

By using this software or microservice, you are agreeing to the [terms and conditions](https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/) of the [license](./LICENSE) and acceptable use policy.

# Terms of Use
GOVERNING TERMS: The NIM container is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [Product-Specific Terms for AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).
