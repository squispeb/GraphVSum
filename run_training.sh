#!/bin/bash
#SBATCH --job-name=GraphVSum
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=shard:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=10
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

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
.venv/bin/python -m trainer wandb.logging=False dataset=bliss epochs=1 batch_size=1 num-workers=0 ES.save_best_model=False
