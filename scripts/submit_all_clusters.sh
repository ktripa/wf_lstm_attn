#!/bin/bash
# Submits one SLURM GPU job per climate cluster (region), all in parallel.
set -euo pipefail
cd /projects/hpcl-cli185/users/ktk/lstm_attn

SEED="${1:-0}"
CLUSTERS=("Hot-Dry" "Hot-Humid" "Cold" "Mixed-Dry" "Mixed-Humid")

for c in "${CLUSTERS[@]}"; do
    jobname="fwi-${c}-s${SEED}"
    sbatch --job-name="$jobname" scripts/slurm_train_cluster.sbatch "$c" "$SEED"
done

squeue -u "$(whoami)" -o "%.10i %.20j %.8T %.10M %.6D %R"
