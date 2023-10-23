#!/bin/bash
#SBATCH --job-name "Training"
#SBATCH --mem 64g
#SBATCH --gpus 1

current_datetime="$(date "+%Y-%m-%d-%H:%M:%S")"
echo $current_datetime

loadfrom="../mg22simulated/" # where the data is stored
iso="Mg22" # isotope of the data

cd ../python
source /opt/conda/bin/activate sparse

python training.py $current_datetime $loadfrom $iso
python plotting.py $current_datetime