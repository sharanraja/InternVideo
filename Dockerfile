# Use the official NVIDIA CUDA image as a base
# Use the nvidia/cuda base image with Python 3.9 and CUDA12.0
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04

# Set environment variables to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    python3.9-dev \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Install the CUDA toolkit
RUN apt-get update && apt-get install -y \
    cuda-toolkit-12-0

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

COPY ./InternVideo2/multi_modality/requirements.txt requirements.txt

RUN pip install packaging
RUN pip install networkx==3.1
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# Install Jupyter Notebook
RUN pip install notebook

# Set working directory
WORKDIR /workspace

# Set Jupyter Notebook configuration
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py

# Expose the port for Jupyter Notebook
EXPOSE 8888