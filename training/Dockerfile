FROM gitlab-master.nvidia.com:5005/dl/dgx/pyg:25.01-py3-devel

# Set working directory
WORKDIR /app

# Install Pydantic
RUN pip install pydantic

COPY config_schema.py config_schema.py
COPY train_xgboost.py train_xgboost.py
COPY train_gnn_based_xgboost.py train_gnn_based_xgboost.py
COPY main.py main.py
COPY training_config.json training_config.json

ENTRYPOINT ["python", "main.py"]
