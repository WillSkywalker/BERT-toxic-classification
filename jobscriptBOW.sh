#!/bin/bash
#SBATCH –job-name=BOW2
#SBATCH –mail-type=BEGIN,END
#SBATCH –mail-user=j.van.arkel.1@student.rug.nl
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64000
module load scikit-learn
module load Python/3.6.4-foss-2018a
cd  ~/Desktop/
python3 BOW.py
