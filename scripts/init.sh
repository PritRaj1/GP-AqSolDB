#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up GP-KAN development environment...${NC}"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first:"
    echo "  - Anaconda: https://www.anaconda.com/products/distribution"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

CONDA_TYPE=""
if conda info --base | grep -q "miniconda"; then
    CONDA_TYPE="miniconda"
    echo -e "${GREEN}Detected: Miniconda${NC}"
elif conda info --base | grep -q "anaconda"; then
    CONDA_TYPE="anaconda"
    echo -e "${GREEN}Detected: Anaconda${NC}"
else
    echo -e "${YELLOW}Warning: Could not determine conda type, proceeding anyway${NC}"
fi

ENV_NAME="GP_sol"

if conda env list | grep -q "$ENV_NAME"; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Do you want to remove it and recreate? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${GREEN}Removing existing environment...${NC}"
        conda env remove -n "$ENV_NAME" -y
    else
        echo -e "${YELLOW}Using existing environment. Activating...${NC}"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
        echo -e "${GREEN}Environment activated successfully!${NC}"
        echo -e "${GREEN}Installing/updating dependencies...${NC}"
        pip install -e ".[dev]"
        echo -e "${GREEN}Setup completed!${NC}"
        exit 0
    fi
fi

echo -e "${GREEN}Creating new conda environment '$ENV_NAME'...${NC}"
conda create -n "$ENV_NAME" python=3.11 -y

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}Activating environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}Installing tmux for development sessions...${NC}"
conda install -c conda-forge tmux -y

echo -e "${GREEN}Installing project dependencies...${NC}"
pip install -e ".[dev]"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully!${NC}"
    
    echo -e "${GREEN}Testing installation...${NC}"
    python -c "
import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.gp_kan.fully_connected import GP_KAN
from src.gp_kan.dense_layer import DenseGPLayer
from src.gp_kan.normal_dist import NormalDist
from configparser import ConfigParser

print('✓ All imports successful!')
print(f'✓ JAX version: {jax.__version__}')
print(f'✓ NumPy version: {np.__version__}')

if jax.devices('gpu'):
    print(f'✓ GPU available: {jax.devices(\"gpu\")}')
else:
    print('✓ Using CPU (GPU not available)')
"
    
    echo -e "${GREEN}✓ Setup completed successfully!${NC}"
    echo -e "${GREEN}To activate the environment, run: conda activate $ENV_NAME${NC}"
    echo -e "${GREEN}To run tests: make test${NC}"
    echo -e "${GREEN}To start development session: make dev${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi 