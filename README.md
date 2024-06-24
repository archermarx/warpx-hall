# WarpX-Hall

Code used to run Hall thruster simulations in WarpX

## Contents
- `picmi_hall.py` - WarpX PICMI input file for Hall thruster simulations. Contains callbacks for particle injection and field adjustment.
- `analysis.py` - Code for analyzing benchmark results and producing plots
- `srun_gl.sh` and `srun_lh.sh` - Example SLURM job scripts for running the code, for UMich's Great Lakes (gl) and Lighthouse (lh) clusters
- `reference` - Extracted benchmark reference data for comparison
