#!/bin/bash
#SBATCH --job-name "Training"
#SBATCH --mem 64g
#SBATCH --gpus 1

cd ../python
source /opt/conda/bin/activate sparse

python train_test.py