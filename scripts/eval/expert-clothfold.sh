#!/bin/bash
#SBATCH -A berzelius-2025-35
#SBATCH --gpus 1
#SBATCH -t 0-02:00:00

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate softgym

export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}/rlpyt_cloth:${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:${PWD}/rlkit/rlkit:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

python experiments/run_expert.py \
--env_name ClothFold \
--num_variations 1 \
--headless 1 \
--output_folder ./data/eval_expert/ClothFold/first_expert_policy

