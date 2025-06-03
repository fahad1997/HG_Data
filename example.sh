#!/bin/bash

#SBATCH --partition=gpu_mig

#SBATCH --gpus=1

#SBATCH --job-name=CheckEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:05:00
#SBATCH --output=slurm_output_%A.txt
#SBATCH --error=slurm_error_%A.txt

#Load necessary modules (adjust based on your environment)
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source my_venv/bin/activate #activate your virtual environment
# pip install --no-cache-dir -r requirements.txt

srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"