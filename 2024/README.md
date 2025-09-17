# WarpX-Hall

Code used to run Hall thruster simulations in WarpX for the paper [GPU-acclerated kinetic Hall thruster simulations in WarpX](https://link.springer.com/article/10.1007/s44205-025-00133-1)

## Contents
- `picmi_hall.py` - WarpX PICMI input file for Hall thruster simulations. Contains callbacks for particle injection and field adjustment.
- `analysis.py` - Code for analyzing benchmark results and producing plots
- `analysis_utils.py` - Functions and utilities used by `analysis.py`
- `paper_plots.ipynb` - Notebook for generating the bulk of the plots in the paper
- `srun_gl.sh` and `srun_lh.sh` - Example SLURM job scripts for running the code, for UMich's Great Lakes (gl) and Lighthouse (lh) clusters
- `reference` - Extracted benchmark reference data for comparison
