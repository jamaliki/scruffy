#!/bin/bash
#SBATCH --job-name=scruffy
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --mem=1024G
#SBATCH --time=24:00:00
#SBATCH --output=scruffy-%j.out
#SBATCH --error=scruffy-%j.err

set -euo pipefail

: "${SCRUFFY_ROOT:?set SCRUFFY_ROOT to a shared filesystem directory}"

# Install scruffy-gpu in this Python environment first. The controller remains
# the foreground payload so Slurm tears down every child step if it exits.
exec scruffy --root "$SCRUFFY_ROOT" serve \
  --gpus-per-node 8 \
  --cpus-per-node 112 \
  --memory-gb-per-node 1024
