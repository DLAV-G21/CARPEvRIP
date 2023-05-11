#!/bin/bash
#SBATCH --chdir /scratch/izar/plumey/ProjectRepository
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos dlav
#SBATCH --account civil-459-2023
#SBATCH --reservation civil-459

source venvs/venv-g21/bin/activate
echo ici > ici.txt
python3 train.py
