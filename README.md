# WarpX-Hall

Authors: Thomas Marks and Alex Gorodetsky, University of Michigan

Licence: GNU GPL v3 

Copyright 2025 Regents of the University of Michigan
 
---

## Overview

This repository contains code for simulating Hall thrusters in [WarpX](https://github.com/BLAST-WarpX/warpx), as well as for analyzing the results.

The `jep_2025` directory contains the code used for the 2025 Journal of Electric Propulsion paper [_GPU-Accelerated Hall Thruster Simulations in WarpX_](https://link.springer.com/article/10.1007/s44205-025-00133-1).
The `iepc_2025` directory contains the code for the paper [_Toward kinetic axial-azimuthal Hall thruster simulations including ionization_](https://thomasmarks.space/p/iepc2025), presented at the 39th International Electric Propulsion Conference in London, United Kingdom.


## Usage

The newer version of the code is in the `iepc_2025` directory.
- At present, the code is split into a library, `hallx.py`, which is used by a script, `run.py`. The former contains all of the functionality for setting up benchmark simulations (and related), while the latter actually runs these simulations.
- The `analysis.py` script computes certain time-averaged properties and generates 2D colorplots at a selection of times, the `generate_plots.ipynb` notebook contains the code used to perform the analysis and generate plots for the 2025 IEPC paper, and the `analysis_utils.py` file contains some of the functionality needed in both.

On a cluster, we recommend using a job management system script (SLURM or similar) to invoke `run.py` with the required arguments.
See the inline documentation in that file for a list of options.
