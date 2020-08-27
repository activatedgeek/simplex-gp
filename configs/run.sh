#!/usr/bin/env bash

#SBATCH --job-name=BIGP
##SBATCH --output=
##SBATCH --error=
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
##SBATCH --gres=gpu:1

if [[ -z "${WANDB_SWEEP_ID}" ]]; then
  echo "Missing WANDB_SWEEP_ID"
  exit 1
fi

source "${HOME}/.bash_profile"

export WANDB_MODE=run
export WANDB_DIR="${LOGDIR}"
export WANDB_NAME="${SLURM_JOB_NAME}--${SLURM_JOB_ID}"

cd "${WORKDIR}/bilateral-gp"

export PYTHONPATH="$(pwd):${PYTHONPATH}"

conda deactivate
conda activate bilateral-gp

wandb agent --count=1 ${WANDB_SWEEP_ID}

