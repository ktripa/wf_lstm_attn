#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=hpcl-cli185
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32g
#SBATCH --time=01:30:00
#SBATCH --job-name=fwi-predictor-corr
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail
cd /projects/hpcl-cli185/users/ktk/lstm_attn
PYTHON=/projects/hpcl-cli185/users/ktk/envs/pgtm2/bin/python
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PYTHON" scripts/compute_predictor_correlation.py
