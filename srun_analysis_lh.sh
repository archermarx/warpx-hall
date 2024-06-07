#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --account=ramanvr
#SBATCH --partition=ramanvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=0
#SBATCH --time=0-00:30:00
#SBATCH --mem=16g
#SBATCH -o output_analysis.log 
#SBATCH -e output_analysis.log
#SBATCH --mail-type=END,FAIL

# Load required modules
source ~/warpx.profile
source ${HOME}/sw/lighthouse/h100/venvs/warpx-h100/bin/activate

# Run simulation
srun python3 ../warpx-hall/analysis.py  > output_analysis.log
