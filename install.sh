#!/bin/bash

# Create and activate conda environment
conda create -n motor python=3.10 -y
conda activate motor

# Uncomment below to install Flash Attention for Torch 2.3.x (CUDA 11.8) - needed for Dragonfly-Med-v2
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.2/flash_attn-2.6.2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install flash_attn-2.6.2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install requirements
pip install -r requirements.txt

# For LLaVA-Med-v1.05
git clone https://github.com/Mai-CS/LLaVA-Med.git llavamed

# Uncomment below if needed for Dragonfly-Med-v2
# git clone https://github.com/togethercomputer/Dragonfly.git
# cd Dragonfly
# pip install --upgrade -e .

# Uncomment below to fix Torch version if you got error
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
