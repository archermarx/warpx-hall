#!/bin/bash
#SBATCH --job-name=case1
#SBATCH --account=(YOUR ACCOUNT HERE)
#SBATCH --partition=(YOUR PARTITION_HERE)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=h100:1
#SBATCH --gpu-bind=single:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=32g
#SBATCH -o output.log 
#SBATCH -e error.log
#SBATCH --mail-type=END,FAIL

# Load required modules
source ~/warpx.profile
source ${HOME}/sw/lighthouse/h100/venvs/warpx-h100/bin/activate

# Executable and input file (or python and picmi script)
EXE=python3
INPUTS=../warpx-hall/picmi_hall.py
ARGS="--grid_factor 1 --initial_ppc 75 --numgpus 1 --no-resample --resample_min 75 --resample_max 300 --sort_interval 500 --mlmg_precision 1e-5 --collision_interval 10000"

# CPU setup
export SRUN_CPUS_PER_TASK=16
export OMP_NUM_THREADS=${SRUN_CPUS_PER_TASK}

# Run simulation
srun --cpu-bind=cores ${EXE} ${INPUTS} ${ARGS}

# Run analysis
srun python3 ../warpx-hall/analysis.py
