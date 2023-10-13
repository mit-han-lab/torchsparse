#!/bin/bash
#SBATCH --job-name "Data Processing"
#SBATCH --mem 64g

cd ../python
source /opt/conda/bin/activate sparse

python data_proccessing.py