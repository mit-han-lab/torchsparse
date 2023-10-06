#!/bin/bash
#SBATCH --job-name "exampleTorchSparse"
#SBATCH --mem 64g
#SBATCH --gpus 1

source /opt/conda/bin/activate sparse

python example.py