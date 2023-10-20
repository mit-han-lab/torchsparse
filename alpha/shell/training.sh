#!/bin/bash
#SBATCH --job-name "Training"
#SBATCH --mem 64g
#SBATCH --gpus 1

current_datetime="$(date "+%Y-%m-%d-%H:%M:%S")"
echo $current_datetime

cd ../python
source /opt/conda/bin/activate sparse

python training.py "$current_datetime"
python plotting.py "$current_datetime"