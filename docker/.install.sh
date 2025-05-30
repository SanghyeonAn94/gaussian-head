#!/bin/bash

echo "Starting installation process..."
apt-get update --fix-missing
apt-get install -y software-properties-common
apt-get install -y git

# Install graphics and OpenGL libraries
echo "Installing graphics libraries..."
apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libglu1-mesa-dev \
    mesa-utils

# Install Python 3.8 and pip
echo "Installing Python 3.8 and pip..."
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install -y python3.8 python3.8-dev python3.8-distutils python3-pip

# Set Python 3.8 as default
echo "Setting Python 3.8 as default..."
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Create pip symlink
echo "Setting up pip..."
ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install geoopt
echo "Installing geoopt..."
pip install geoopt

# Install PyTorch3D
echo "Installing PyTorch3D..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install basic requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Install simple-knn
echo "Installing simple-knn..."
pip install git+https://github.com/camenduru/simple-knn

# Force reinstall numpy to specific version for compatibility
echo "Installing numpy 1.19.5..."
pip install numpy==1.19.5 --force-reinstall\

echo "Installation completed!"
