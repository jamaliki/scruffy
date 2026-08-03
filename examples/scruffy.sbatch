#!/bin/bash
#SBATCH --job-name=scruffy
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --mem=1024G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=scruffy-%j.out
#SBATCH --error=scruffy-%j.err

set -euo pipefail
umask 077

: "${SCRUFFY_ROOT:?set SCRUFFY_ROOT to a shared filesystem directory}"

# SCRUFFY_ROOT and this Python environment must be visible at the same absolute
# paths on every node. Keeping the controller in the foreground lets Slurm tear
# down every child step if it exits; the early TERM gives it time to reconcile.
exec scruffy --root "$SCRUFFY_ROOT" serve \
  --gpus-per-node 8 \
  --cpus-per-node 112 \
  --memory-gb-per-node 1024
