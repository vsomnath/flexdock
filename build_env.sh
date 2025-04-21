#!/bin/bash

source $(conda info --root)/etc/profile.d/conda.sh

ENV_NAME="flexdock"
ARCH=$(uname -m)
TORCH=2.1.0

if [[ "$ARCH" == "arm64" ]]; then
    CUDA=cpu
else
    CUDA=cu121
fi

conda create -y --name ${ENV_NAME} python=3.11
conda activate ${ENV_NAME}
conda install conda-forge::ambertools #TODO: Need to re-enable this

pip install torch==${TORCH} --index-url https://download.pytorch.org/whl/${CUDA}
pip install torch_geometric==2.6.1

pip install \
    numpy==1.23.5 \
    scipy==1.15.1 \
    wandb==0.18.7 \
    rdkit==2024.3.6 \
    pandas==2.2.2 \
    biopython==1.79 \
    geomstats==2.8.0 \
    networkx==3.2.1 \
    pandas==2.2.2 \
    jupyter==1.1.1 \
    pyyaml==6.0.2 \
    tqdm==4.67.0 \
    lightning==2.4.0 \
    ProDy==2.4.1 \
    e3nn==0.5.5 \
    omegaconf \
    parmed \
    spyrmsd \
    fair-esm

pip install --no-index \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html


conda install sqlite==3.45.*

# Only for evaluation / testing
pip install posebusters==0.3.1

python setup.py develop
python -m ipykernel install --name ${ENV_NAME} --user
