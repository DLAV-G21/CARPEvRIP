#!/bin/bash
#SBATCH --chdir /scratch/izar/plumey
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos dlav
#SBATCH --account civil-459-2023


source venvs/venv-g21/bin/activate
echo ici > ici.txt
cd ProjectRepository
python3 train.py
