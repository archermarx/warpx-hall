#!/bin/bash -l

# Copyright 2024 The WarpX Community
#
# Author: Axel Huebl
# License: BSD-3-Clause-LBNL

#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -J benchmark-2d
#SBATCH -A goroda1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=v100:1
#SBATCH --gpu-bind=single:1
#SBATCH --output output.log
#SBATCH --error error.log
#SBATCH --mail-type=START,END,FAIL

source ~/warpx.profile
source ${HOME}/sw/greatlakes/v100/venvs/warpx-v100/bin/activate

# executable & inputs file or python interpreter & PICMI script here
EXE=python3
INPUTS=picmi_hall.py
ARGS="--case 1 --numgpus 1 --resample 0 --resample_min 75 --resample_max 300 --seed 119899896"

# threads for OpenMP and threaded compressors per MPI rank
#   per node are 2x 2.4 GHz Intel Xeon Gold 6148
#   note: the system seems to only expose cores (20 per socket),
#         not hyperthreads (40 per socket)
export SRUN_CPUS_PER_TASK=10
export OMP_NUM_THREADS=${SRUN_CPUS_PER_TASK}

# run WarpX
srun --cpu-bind=cores \
  ${EXE} ${INPUTS} ${ARGS} \
  > output.log
