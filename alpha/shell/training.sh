#!/bin/bash
#SBATCH --job-name "Training"
#SBATCH --mem 64g
#SBATCH --gpus 1

current_datetime="$(date "+%Y-%m-%d-%H:%M:%S")"
echo $current_datetime

loadfrom="../mg22simulated/" # where the data is stored
iso="Mg22" # isotope of the data

learning_rate=0.001
epochs=100
batch_size=12

cd ../python
source /opt/conda/bin/activate sparse

python training.py $current_datetime $loadfrom $iso $learning_rate $epochs $batch_size
python plotting.py $current_datetime $epochs
python evaluate.py $current_datetime $loadfrom $iso $learning_rate $epochs $batch_size
python confusion_matrix.py $current_datetime 