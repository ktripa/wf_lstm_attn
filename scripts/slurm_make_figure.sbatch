#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=hpcl-cli185
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --job-name=fwi-fig2
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail
cd /projects/hpcl-cli185/users/ktk/lstm_attn
PYTHON=/projects/hpcl-cli185/users/ktk/envs/pgtm2/bin/python
echo "host=$(hostname)"
"$PYTHON" figures/fig2_performance_error_structure.py
