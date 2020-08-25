#!/usr/bin/env bash

#SBATCH --job-name=SKIP
##SBATCH --output=
##SBATCH --error=
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
##SBATCH --gres=gpu:1

source ~/.bash_profile

## Disable GPU for now.
export CUDA_VISIBLE_DEVICES=-1

export WANDB_SWEEP_ID=
export WANDB_MODE=run

pushd "${WORKDIR}/bilateral-gp"

conda activate bilateral-gp

wandb agent --count=1 ${WANDB_SWEEP_ID}

popd
