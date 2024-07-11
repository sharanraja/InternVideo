#!/bin/bash

# Name of the Docker image
IMAGE_NAME="internvideo:demo"

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container with all GPUs
docker run --gpus all -it -p 8888:8888 -v $(pwd):/workspace $IMAGE_NAME /bin/bash

