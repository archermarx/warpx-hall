# WarpX-Hall

| Thomas Marks and Alex Gorodetsky, University of Michigan

Code for simulating Hall thrusters in WarpX, as well as for analyzing the results.
Better documentation to follow shortly.

The `2024` directory contains the code used for the 2024 IEPC paper "Hall Thruster Simulations in WarpX", as well as its Journal of Electric Propulsion follow-up "GPU-Accelerated Hall Thruster Simulations in WarpX".
The `2025` directory contains the code for our 2025 IEPC paper "Toward kinetic axial-azimuthal Hall thruster simulations including ionization".


## Usage

At present, the code is split into a library, `hallx.py`, which is used by a script, `run.py`.
The former contains all of the functionality for setting up benchmark simulations (and related), while the latter actually runs these simulations.
On a cluster, you will need a job management system script (SLURM or similar) to invoke `run.py` with the required args.
There are also some analysis functionalities.
The `analysis.py` script computes certain time-averaged properties and generates 2D colorplots at a selection of times. See that script for details.
The `generate_plots.ipynb` notebook contains the code used to perform the analysis and generate plots for the 2025 IEPC paper.
The `analysis_utils.py` file contains some of the functionality needed in `analysis.py`.
