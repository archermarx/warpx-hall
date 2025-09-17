import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def coord_name(axis: str) -> str:
    if axis == 'x':
        return 'Axial'
    elif axis == 'y':
        return 'Radial'
    elif axis == 'z':
        return 'Azimuthal'
    else:
        raise ValueError(f"'{axis}' is not a valid axis. Pick 'x', 'y', or 'z'")

def ensure_directory(file: Path | str) -> Path:
    os.makedirs(file, exist_ok=True)
    return Path(file)


def add_colorbar(fig, ax, im, **cbar_args):
    pos = ax.get_position()
    cax_dims = (pos.x1 + 0.01, pos.y0, 0.03, pos.height)
    cax = fig.add_axes(cax_dims)
    plt.colorbar(im, cax=cax, **cbar_args)


def make_ref_polygon(field: str, ref_dir: Path):
    """
    Takes two curves defined in csv files (`field`_lower_raw.csv) and (`field`_upper_raw.csv), and them together into a closed path.
    """
    low_points = np.genfromtxt(ref_dir / (field + "_lower_raw.csv"), delimiter=",")
    hi_points = np.genfromtxt(ref_dir / (field + "_upper_raw.csv"), delimiter=",")
    return (
        low_points,
        hi_points,
        np.vstack([low_points, np.flip(hi_points, axis=0), low_points[0, :]]),
    )
