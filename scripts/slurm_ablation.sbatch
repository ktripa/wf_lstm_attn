#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=hpcl-cli185
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24g
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%x_%j.out

# Usage: sbatch --job-name=abl-<tag>-s<seed> slurm_ablation.sbatch <seed> [--extra-precip] <branch letters...>
set -euo pipefail
cd /projects/hpcl-cli185/users/ktk/lstm_attn
PYTHON=/projects/hpcl-cli185/users/ktk/envs/pgtm2/bin/python
SEED="$1"
shift
echo "host=$(hostname) seed=$SEED args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PYTHON" scripts/run_ablation.py --seed "$SEED" "$@"
