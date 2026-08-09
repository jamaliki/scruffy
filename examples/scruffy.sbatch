#!/bin/bash
#SBATCH --job-name=scruffy
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=scruffy-%j.out
#SBATCH --error=scruffy-%j.err

set -euo pipefail
umask 077

: "${SCRUFFY_ROOT:?set SCRUFFY_ROOT to a shared filesystem directory}"

# SCRUFFY_ROOT and this Python environment must be visible at the same absolute
# paths on every node. Reuse that root for a successor allocation. Add
# --start-paused during a handover so the recovered queue can be inspected
# before it launches. The default controller drain begins 15 minutes before the
# allocation deadline.
exec scruffy --root "$SCRUFFY_ROOT" serve --drain-before-end-seconds 900
