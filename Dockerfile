# Use pyg base image
FROM gitlab-master.nvidia.com:5005/dl/dgx/pyg:25.01-py3-devel

# Set the working directory inside the container to /app.
# All subsequent commands (e.g., COPY, RUN) will be executed relative to this directory.
WORKDIR /app

# Copy src folder into the container's /app directory
COPY src src

# Copy main.py into the container's /app directory.
COPY main.py main.py

# Copy config_schema.py into the container's /app directory.
COPY example_training_config.json example_training_config.json

# Install pydantic, shap and captum using pip.
RUN pip install pydantic
RUN pip install shap
RUN pip install captum

# Set the container's entrypoint to run main.py with Python.
ENTRYPOINT ["python", "main.py"]