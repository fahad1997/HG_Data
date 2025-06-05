#!/bin/bash

#SBATCH --partition=gpu_h100

#SBATCH --gpus=1

#SBATCH --job-name=CheckEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

#SBATCH --time=24:00:00

#SBATCH --output=slurm_output_%A.txt
#SBATCH --error=slurm_error_%A.txt

#Load necessary modules (adjust based on your environment)
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source hg_venv/bin/activate #activate your virtual environment
# pip install --no-cache-dir -r requirements.txt

python train_pix2struct_base.py
