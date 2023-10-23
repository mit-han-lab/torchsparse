#!/bin/bash
#SBATCH --job-name "Data Processing"
#SBATCH --mem 64g


loadfrom="../mg22simulated/" #where the data is stored and where it will be stored
iso="Mg22" # isotope of the data
h5="output_digi_HDF_Mg22_Ne20pp_8MeV.h5" #name of h5 file

cd ../python
source /opt/conda/bin/activate sparse

python data_proccessing.py $loadfrom $iso $h5
python traintestsplit.py $loadfrom $iso