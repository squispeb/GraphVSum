#!/bin/bash
#SBATCH --job-name=GraphVSum
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=shard:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.quispe.b@utec.edu.pe

# Load modules if needed (adjust based on cluster setup)
module load miniconda/3.0
module load python3/3.8.0
module load cuda/11.8

# Activate conda environment (adjust path/name as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate graphvsum

# Set dataset path
export DATASET_PATH=~/datasets
export DGLBACKEND=pytorch

# Run training
.venv/bin/python -m trainer wandb.logging=False dataset=bliss num-workers=0 ES.save_best_model=True
