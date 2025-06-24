#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda env list | grep -q 'GP_sol'
if [ $? -ne 0 ]; then
    echo "Creating conda environment GP_sol..."
    conda create -n GP_sol python=3.11 -y  
fi

conda activate GP_sol
conda install -c conda-forge tmux -y

# Install Python requirements
echo "Installing Python requirements..."
python setup/requirements.py
if [ $? -ne 0 ]; then
    echo "Failed to install Python requirements"
    exit 1
fi
