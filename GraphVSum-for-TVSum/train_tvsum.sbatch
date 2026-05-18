#!/bin/bash
#SBATCH --job-name=GraphVSum_TVSum
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/home/edgar.chambilla/Tesis/GraphVSum-for-TVSum
#SBATCH --output=/home/edgar.chambilla/Tesis/GraphVSum-for-TVSum/slurm-%j.out

module purge
module load gnu12/12.4.0
module load python3/3.8.0
module load cuda/11.8

PROJECT_DIR="/home/edgar.chambilla/Tesis/GraphVSum-for-TVSum"
VENV_PYTHON="/home/edgar.chambilla/Tesis/GraphVSum/.venv/bin/python"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $VENV_PYTHON"

"$VENV_PYTHON" -u -m trainer \
  wandb.logging=False \
  gpus=[0] \
  wandb.model_name=TVSum-paper-graph-eval
