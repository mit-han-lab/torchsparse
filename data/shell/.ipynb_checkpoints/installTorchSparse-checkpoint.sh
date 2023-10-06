#!/bin/bash
#SBATCH --job-name "installTorchSparse"
#SBATCH --mem 16g
#SBATCH --gpus 1

source /opt/conda/bin/activate sparse

python install.py