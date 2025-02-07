# Use pyg base image
FROM gitlab-master.nvidia.com:5005/dl/dgx/pyg:25.01-py3-devel

# Set the working directory inside the container to /app.
# All subsequent commands (e.g., COPY, RUN) will be executed relative to this directory.
WORKDIR /app

# Install the Pydantic library using pip.
# Pydantic is used for data validation and settings management using Python type annotations.
RUN pip install pydantic

# Copy the config_schema.py file from the host machine into the container's /app directory.
COPY config_schema.py config_schema.py

# Copy the train_xgboost.py file into the container.
COPY train_xgboost.py train_xgboost.py

# Copy the train_gnn_based_xgboost.py file into the container.
COPY train_gnn_based_xgboost.py train_gnn_based_xgboost.py

# Copy the main.py file into the container.
COPY main.py main.py

# Copy the training_config.json file into the container.
COPY example_training_config.json example_training_config.json

# Set the container's entrypoint to run main.py with Python.
# This means that when the container starts, it will execute "python main.py".
ENTRYPOINT ["python", "main.py"]