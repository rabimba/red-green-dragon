# Use NVIDIA CUDA Base Image with Ubuntu 24.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    vim \
    wget \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies
RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install --break-system-package tqdm datasets

RUN apt-get update && apt-get install -y libopencv-dev

RUN apt-get update && apt-get install -y libclang-18-dev curl libcurl4-openssl-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Rule-based CUDA→HIP (hipify-clang); matches hipify/run.sh flags
RUN git clone --depth 1 https://github.com/ROCm/HIPIFY.git /tmp/HIPIFY \
    && mkdir /tmp/HIPIFY/build && cd /tmp/HIPIFY/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hipify -DCMAKE_BUILD_TYPE=Release .. \
    && make -j"$(nproc)" install \
    && rm -rf /tmp/HIPIFY
ENV PATH="/opt/hipify/bin:${PATH}"

# # Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CPATH=$CUDA_HOME/targets/x86_64-linux/include/

# Set working directory
WORKDIR /workspace

# To add color to the terminal
ENV TERM=xterm-256color
RUN echo "PS1='\e[92m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc

# Default command
CMD ["/bin/bash"]