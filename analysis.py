import yt
import numpy as np
from numpy import inf
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
from scipy.fft import fft, fftfreq
from pathlib import Path
import gc

# some common utility functions
import analysis_utils as utils

# some physical constants
from scipy.constants import c, m_e, e as q_e, atomic_mass

m_Xe = atomic_mass * 131.293

# === Module config ===
yt.funcs.mylog.setLevel(50)

# === Matplotlib config ===
font = {"family": "Helvetica", "weight": "normal", "size": 15}
plt.set_loglevel("error")
plt.rc("font", **font)
benchmark_color = "#ffb3b3"
fig_dpi = 200

# === Directories ===
ref_dir = Path("/home/marksta/src/warpx-hall/reference")
data_dir = Path("./diags")
particle_diag_dir = Path("./particle_diags")


class LandmarkBenchmark:
    """
    Define the LANDMARK 2-D PIC benchmark
    This can be scaled by setting different timesteps and dimensions
    """

    def __init__(self, dt=5e-12, x_max=2.5e-2, unit="m"):
        self.dt = dt
        self.x_max = x_max
        self.unit = unit
        self.x_bmax = 0.3 * self.x_max
        self.x_inj_0 = 0.1 * self.x_max
        self.x_inj_1 = 0.4 * self.x_max
        self.x_emit = 24 / 25 * self.x_max
        self.x_inj_m = 0.5 * (self.x_inj_0 + self.x_inj_1)

    def scaled(self, unit):
        scale_factors = {
            "m": 1.0,
            "cm": 100,
            "mm": 1000,
            "um": 1_000_000,
            "μm": 1_000_000,
            "nm": 1_000_000_000,
        }
        try:
            scale_factor = scale_factors[unit]
        except KeyError:
            raise KeyError(
                f"Invalid length unit '{unit}' specified. Please pick one of {'m', 'cm', 'mm', 'um', 'nm'}"
            )

        return LandmarkBenchmark(dt=self.dt, x_max=scale_factor * self.x_max, unit=unit)

    def grid(self, Nx):
        return np.linspace(0, self.x_max, Nx)


def benchmark_plot(
    yplots, xplots, benchmark=LandmarkBenchmark(), xscale=1.0, yscale=1.0, **kwargs
):
    base_fig_width = 6.4 * xscale
    base_fig_height = 4.8 * yscale
    fig, axes = plt.subplots(
        yplots,
        xplots,
        figsize=(xplots * base_fig_width, yplots * base_fig_height),
        **kwargs,
    )
    num_axes = yplots * xplots

    lightgrey = "#dddddd"
    if num_axes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, benchmark.x_max)
        ax.set_xlabel(f"x [{benchmark.unit}]")
        ax.axvspan(
            benchmark.x_inj_0,
            benchmark.x_inj_1,
            color=lightgrey,
            label="Injection region",
        )
        ax.axvline(
            [benchmark.x_emit],
            linestyle="--",
            color="grey",
            linewidth=1,
            label="Cathode plane",
        )
        ax.axvline(
            [benchmark.x_bmax],
            linestyle="-",
            color="grey",
            linewidth=1,
            label="Peak magnetic field",
        )
        ax.axhline([0], linestyle="-", color="grey")

    return fig, axes


class GriddedData:
    def __init__(self, path):
        ds = yt.load(path)
        self.dataset = ds
        self.gridded_data = self.dataset.covering_grid(
            level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
        )

    def grid1D(self, axis=0):
        x0 = self.dataset.domain_left_edge[axis]
        x1 = self.dataset.domain_right_edge[axis]
        return np.linspace(x0, x1, self.dataset.domain_dimensions[axis])

    def load1D(self, field, axis=1):
        """Load a 2-D field, averaged over axis `axis`"""
        data = self.gridded_data["boxlib", field].v[:, :, 0]
        return np.mean(data, axis)

    def load2D(self, field):
        """Load a 2-D field from the dataset"""
        return np.transpose(self.gridded_data["boxlib", field].v[:, :, 0])

    def load(self, field, dimension=1, axis=1):
        if dimension == 1:
            return self.load1D(field, axis)
        elif dimension == 2:
            return self.load2D(field)
        else:
            raise ValueError(f"Invalid dimension: {dimension}")

    def compute_temp(self, species, dimension=1, axis=1):
        ux = self.load("ux_" + species, dimension, axis)
        uy = self.load("uy_" + species, dimension, axis)
        uz = self.load("uz_" + species, dimension, axis)
        Pxx = self.load("P_xx_" + species, dimension, axis)
        Pyy = self.load("P_yy_" + species, dimension, axis)
        Pzz = self.load("P_zz_" + species, dimension, axis)
        E_factor = m_e * c**2 / q_e

        Ex = E_factor * (Pxx - ux**2)
        Ey = E_factor * (Pyy - uy**2)
        Ez = E_factor * (Pzz - uz**2)

        T = (Ex + Ey + Ez) / 3

        return T, Ex, Ey, Ez

    def temp_and_heat_flux(self, species, dimension=1, axis=1):
        ux = self.load("ux_" + species, dimension, axis)
        uy = self.load("uy_" + species, dimension, axis)
        uz = self.load("uz_" + species, dimension, axis)
        Pxx = self.load("P_xx_" + species, dimension, axis)
        Pxy = self.load("P_xy_" + species, dimension, axis)
        Pxz = self.load("P_xz_" + species, dimension, axis)
        Pyy = self.load("P_yy_" + species, dimension, axis)
        Pyz = self.load("P_yz_" + species, dimension, axis)
        Pzz = self.load("P_zz_" + species, dimension, axis)
        Qx = self.load("Q_x_" + species, dimension, axis)
        Qy = self.load("Q_y_" + species, dimension, axis)
        Qz = self.load("Q_z_" + species, dimension, axis)

        # Energy normalization factor
        E_factor = m_e * c**2 / q_e

        # Velocity vector
        u = np.array([ux, uy, uz])

        # Velocity squared magnitude
        u2 = np.einsum("ik,ik->k", u, u)

        # Energy flux vector
        Q = np.array([Qx, Qy, Qz])

        # Stress tensor
        P = np.array([[Pxx, Pxy, Pxz], [Pxy, Pyy, Pyz], [Pxz, Pyz, Pzz]])

        # Pressure tensor
        p = P - np.einsum("ik,jk->ijk", u, u)

        # Temperature
        T = np.trace(p, axis1=0, axis2=1) / 3 * E_factor

        # Directional temperatures
        Tx = p[0, 0, :] * E_factor
        Ty = p[1, 1, :] * E_factor
        Tz = p[2, 2, :] * E_factor

        # Total energy
        E_tot = 1.5 * T + 0.5 * u2

        # Heat flux
        # Q - pu - E_tot * u
        q = Q - np.einsum("ijk,jk->ik", p, u) - E_tot * u
        q = q * c * E_factor

        return T, q, Tx, Ty, Tz

    def extent(self):
        dimensionality = self.dataset.dimensionality
        return [
            self.dataset.domain_left_edge[0],
            self.dataset.domain_right_edge[0],
            self.dataset.domain_left_edge[dimensionality - 1],
            self.dataset.domain_right_edge[dimensionality - 1],
        ]


def read_plotfile_header(dir):
    header_file = os.path.join(dir, "Header")
    with open(header_file) as f:
        lines = f.readlines()

    num_fields = int(lines[1].strip())
    first_ind = 2
    last_ind = first_ind + num_fields
    fields = [line.strip() for line in lines[first_ind:last_ind]]

    time = float(lines[last_ind + 1].strip())
    iter = int(lines[len(lines) - 4].strip())

    return {"num_fields": num_fields, "fields": fields, "time": time, "iter": iter}


def dir_pred(start_time, end_time):
    def pred(dir):
        if os.path.split(dir)[-1] == "reducedfiles":
            return False
        header = read_plotfile_header(dir)
        header["time"]
        if start_time <= header["time"] <= end_time:
            return True
        return False

    return pred


def get_data(dir):
    header = read_plotfile_header(dir)
    return {"time": header["time"], "iter": header["iter"], "data": GriddedData(dir)}


def load_field(field, dimension, axis):
    def _load(data):
        return data.load(field, dimension, axis)

    return _load


class DataSeries:
    def __init__(self, dir, start_time=0, end_time=inf):
        # read data from selected time window
        subdirs = filter(
            dir_pred(start_time, end_time),
            [f.path for f in os.scandir(dir) if f.is_dir()],
        )
        data = [get_data(subdir) for subdir in subdirs]

        # Sort data by iteration
        perm = sorted(range(len(data)), key=lambda k: data[k]["iter"])

        # set member variables
        self.iterations = np.array([data[i]["iter"] for i in perm])
        self.time = np.array([data[i]["time"] for i in perm])
        self.data = np.array([data[i]["data"] for i in perm])
        self.num_iters = len(perm)
        dt = np.zeros(self.num_iters)
        dt[1:] = np.diff(self.time)
        dt[0] = dt[1]
        self.delta_time = dt
        self.time_interval = self.time[-1] - self.time[0]

    def load(self, field, dimension=1, axis=1, inds=None, start_time=0.0, verbose=False):
        closure = load_field(field, dimension, axis)
        if inds is None:
            inds = (n for n, t in enumerate(self.time) if t > start_time)
        values = np.array([closure(self.data[i]) for i in inds])

        if verbose:
            print(f"Data {field} loaded.")

        return values

    def load_averaged(self, field, dimension=1, axis=1):
        return (
            np.sum(self.delta_time * self.load(field, dimension, axis), 0)
            / self.time_interval
        )


def load_last_particle_diag(particle_diag_dir):
    # Find the final simulation output by checking which particle diagnostic dir has the largest number as a suffix
    diag_prefix = "hall_particles"
    diag_dirs = os.listdir(particle_diag_dir)
    diag_indices = [
        int(dir.removeprefix(diag_prefix)) for dir in os.listdir(particle_diag_dir)
    ]
    last_diag_ind = np.argmax(diag_indices)
    last_diag_dir = particle_diag_dir / diag_dirs[last_diag_ind]
    return yt.load(last_diag_dir)


def plot_vdf_1D_right(ds, species: str, output_dir=Path(".")):
    data = ds.all_data()
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    Lx, _, _ = (grid.right_edge - grid.left_edge).to_ndarray()
    nx, _, _ = grid.ActiveDimensions
    nbins = 50

    # Bin particles in the last grid cell before the cathode line
    x_cathode = 24 / 25 * Lx
    dx = Lx / nx
    x = data[(species, "particle_position_x")]
    inds = np.logical_and(x < x_cathode, x >= x_cathode - 3 * dx)

    if species == "electrons":
        mass = m_e
    else:
        mass = m_Xe

    fig, ax = plt.subplots(figsize=(5, 5), dpi=fig_dpi, tight_layout=True)
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("Frequency [arb.]")

    def get_freqs(coord):
        # Get velocity in km/s
        u = data[(species, f"particle_momentum_{coord}")] / mass / 1000

        # Plot histogram
        u_bins = np.linspace(np.min(u), np.max(u), nbins + 1)
        bin_centers = 0.5 * (u_bins[0:-1] + u_bins[1:])
        freq, _ = np.histogram(u[inds], bins=u_bins)
        return bin_centers, freq

    coords = ["x", "y", "z"]

    hists = [get_freqs(coord) for coord in coords]
    max_count = max(np.max(h[1]) for h in hists)

    header = ""
    for coord, hist in zip(coords, hists):
        ax.plot(
            hist[0], hist[1] / max_count, label=f"{utils.coord_name(coord)} velocity"
        )
        header += f"u{coord} [m/s],freq_{coord}"
        if coord != coords[-1]:
            header += ","

    ax.set_xlim(-5000, 5000)
    ax.axvline(0, color="lightgray", zorder=0)
    ax.axhline(0, color="lightgray", zorder=0)
    ax.legend()

    data = np.zeros((6, nbins))
    for i, (bins, freqs) in enumerate(hists):
        data[2 * i, :] = bins * 1000  # convert back to m/s
        data[2 * i + 1, :] = freqs

    np.savetxt(
        output_dir / f"vdf_{species}_right.csv",
        data.T,
        delimiter=",",
        header=header,
        comments="",
        fmt="%1.12e",
    )
    fig.savefig(output_dir / f"vdf_{species}_right.png")
    fig.savefig(output_dir / f"vdf_{species}_right.eps")


def plot_vdf_2D(ds, species: str, axis: str = "x", output_dir: Path = Path(".")):
    ax_descriptor = utils.coord_name(axis)
    data = ds.all_data()
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    Lx, _, _ = (grid.right_edge - grid.left_edge).to_ndarray()
    nx, _, _ = grid.ActiveDimensions

    nbins = 100
    hists = np.zeros((nbins, nx))

    # Bin particles based on axial position
    xbins = np.linspace(0, Lx, nx + 1)
    x = data[(species, "particle_position_x")]
    indices = np.digitize(x, bins=xbins, right=True) - 1

    if species == "electrons":
        mass = m_e
    else:
        mass = 131.293 * atomic_mass

    # Get velocity in km/s
    ux = data[(species, f"particle_momentum_{axis}")].to_ndarray() / mass / 1000

    # Set up velocity bins
    round_up = lambda x: np.sign(x) * np.ceil(abs(x))
    u_min = round_up(np.min(ux))
    u_max = round_up(np.max(ux))
    
    u_bins = np.linspace(u_min, u_max, nbins + 1)

    for i in range(nx):
        inds = indices == i
        freq, _ = np.histogram(ux[inds], bins=u_bins, density=True)
        hists[:, i] = freq

    fig, ax = plt.subplots(figsize=(5, 5), dpi=fig_dpi)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel(f"{ax_descriptor} velocity [km/s]")
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    hists /= np.max(hists)
    extent = (0, Lx*100, u_min, u_max)
    im = ax.imshow(
        hists,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )
    ax.axhline(0, color="white")
    utils.add_colorbar(fig, ax, im, label="Intensity (arb.)")

    # Save figure
    outfile = f"vdf_{species}_{axis}"
    fig.savefig(output_dir / (outfile + ".png"), bbox_inches="tight")
    fig.savefig(output_dir / (outfile + ".eps"), bbox_inches="tight")

    # Save histogram data
    header = f"{extent[0]},{extent[1]},{extent[2]},{extent[3]}"
    np.savetxt(output_dir / (outfile + ".csv"), hists, header=header, delimiter=",")
    
    plt.close(fig)


def frequency_spectrum(series, start_time):
    """
    Compute the frequency spectrum of the azimuthal electric field as a function of axial location.
    The amplitudes are averaged over the y-coordinate.
    Returns a dict containing the following keys:
    - "x_m": The axial coordinates at which the spectra were computed.
    - "frequencies": the frequencies at which the amplitudes are known
    - "amplitudes": a `list` of amplitude vectors, with each vector corresponding to a specific axial location
    - "peak_freqs": a vector containing the dominant frequency for each axial location
    """
    indices = [n for n, t in enumerate(series.time) if t >= start_time]
    x_m = series.data[0].grid1D(axis=0).v
    y_m = series.data[0].grid1D(axis=1).v

    Ey = series.load("Ez", dimension=2, inds=indices, verbose=False)
    M = len(Ey)
    dt = series.time[2] - series.time[1]

    frequency = fftfreq(M, dt)[1 : M // 2] * 2 * np.pi
    amplitudes = []
    peak_freqs = []

    # Compute the y-averaged frequency spectrum at each x location
    for i, _ in enumerate(x_m):
        _amplitude = np.zeros_like(frequency)
        for j, _ in enumerate(y_m):
            E = [_Ey[j, i] for _Ey in Ey]
            E_fft = np.abs(fft(E))[1 : M // 2]
            _amplitude += E_fft

        amplitudes.append(_amplitude / len(y_m))
        peak_freq = frequency[np.argmax(_amplitude)]
        peak_freqs.append(peak_freq)

    return {
        "x_m": x_m,
        "frequency": frequency,
        "amplitudes": amplitudes,
        "peak_freqs": peak_freqs,
    }


def wavenumber_spectrum(series, start_time):
    """
    Compute the wavenumber spectrum of the azimuthal electric field as a function of axial location.
    The amplitudes are averaged over time, starting at the provided `start_time`.
    Returns a dict containing the following keys:
    - "x_m": The axial coordinates at which the spectra were computed.
    - "frequencies": the wavenumbers at which the amplitudes are known
    - "amplitudes": a `list` of amplitude vectors, with each vector corresponding to a specific axial location
    - "peak_freqs": a vector containing the dominant wavenumber for each axial location
    """
    indices = [n for n, t in enumerate(series.time) if t >= start_time]
    x_m = series.data[0].grid1D(axis=0).v
    y_m = series.data[0].grid1D(axis=1).v

    Ey = series.load("Ez", dimension=2, inds=indices)
    M = len(y_m)
    dy = y_m[1] - y_m[0]

    frequency = fftfreq(M, dy)[1 : M // 2]
    amplitudes = []
    peak_freqs = []

    # Compute the y-averaged frequency spectrum at each x location
    for i, x in enumerate(x_m):
        _amplitude = np.zeros_like(frequency)
        for E in Ey:
            E_fft = np.abs(fft(E[:, i]))[1 : M // 2]
            _amplitude += E_fft

        amplitudes.append(_amplitude / len(Ey))
        peak_freq = frequency[np.argmax(_amplitude)]
        peak_freqs.append(peak_freq)

    return {
        "x_m": x_m,
        "frequency": frequency,
        "amplitudes": amplitudes,
        "peak_freqs": peak_freqs,
    }


def save_spectrum(outfile, spectrum, verbose=True):
    """
    Save a spectrum dictionary to a JSON file
    """
    out_dict = {
        "x_m": spectrum["x_m"].tolist(),
        "frequency": spectrum["frequency"].tolist(),
        "ampitudes": np.array(spectrum["amplitudes"]).tolist(),
        "peak_freqs": np.array(spectrum["peak_freqs"]).tolist(),
    }
    with open(outfile, "w") as fd:
        json.dump(out_dict, fd, indent=4)

    if verbose:
        print(f"Spectrum {outfile} saved.")


def ax_bbox(fig, ax):
    """Get the bounding box of an axis"""
    return ax.get_tightbbox(fig.canvas.renderer).transformed(
        fig.dpi_scale_trans.inverted()
    )


def ax_center(fig, ax):
    """Get the center coordinates of an axis"""
    bbox = ax_bbox(fig, ax)
    center = bbox.p0 + 0.5 * np.array(bbox.width, bbox.height)
    return center


def save_subfig(fig, ax, output_dir, filename):
    """Save an axis to an image file"""
    fig.savefig(output_dir / (filename + ".png"), bbox_inches=ax_bbox(fig, ax))
    fig.savefig(output_dir / (filename + ".eps"), bbox_inches=ax_bbox(fig, ax))


def plot_field_2D(field, figsize, title, im_args, outfile):
    fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)
    ax.set_title(title)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticks([0.0, 0.4, 0.8, 1.2])
    im = ax.imshow(field, **im_args)
    utils.add_colorbar(fig, ax, im)
    fig.savefig(outfile)
    plt.close(fig)


def plot_reduced_diags(
    diag_dir: Path,
    output_dir: Path,
    num_cells: int,
    start_time: float,
    verbose: bool = True,
):
    reduced_dir = diag_dir / "reducedfiles"
    particle_number_file = reduced_dir / "ParticleNumber.txt"
    particle_energy_file = reduced_dir / "ParticleEnergy.txt"
    field_energy_file = reduced_dir / "FieldEnergy.txt"
    if not os.path.exists(particle_number_file):
        particle_number_file = reduced_dir / "hall.txt "

    fig_args = {"dpi": fig_dpi, "figsize": (8, 6)}

    if os.path.exists(particle_number_file):
        # Plot particle counts
        particle_counts = np.genfromtxt(
            particle_number_file, skip_header=1, delimiter=" "
        )

        if particle_counts.size > 0:
            time = particle_counts[:, 1] * 1e6
            num_ion = particle_counts[:, 3]
            num_ele = particle_counts[:, 4]
            fig, ax = plt.subplots(**fig_args)
            ax.autoscale(enable=True, axis="x", tight=True)
            ax.set_xlabel("Time ($\\mu$s)")
            ax.set_ylabel("Macroparticles/cell")
            ax.axvline(start_time * 1e6, color="gray", linestyle="--", zorder=0)
            ax.plot(time, num_ion / num_cells, label="Ions")
            ax.plot(
                time,
                num_ele / num_cells,
                color="red",
                linestyle="--",
                label="Electrons",
            )
            ax.plot(time, (num_ion + num_ele) / num_cells, color="black", label="Total")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "particle_number.png")
            fig.savefig(output_dir / "particle_number.eps")
            if verbose:
                print("Macroparticle numbers plotted.")

    if os.path.exists(particle_energy_file) and os.path.exists(field_energy_file):
        # Plot particle, field, and total energy in system
        particle_energy = np.genfromtxt(
            particle_energy_file, skip_header=1, delimiter=" "
        )
        field_energy = np.genfromtxt(field_energy_file, skip_header=1, delimiter=" ")

        if particle_energy.size > 0:
            time = particle_energy[:, 1] * 1e6
            field_energy_J = field_energy[:, 2]
            ion_energy_J = particle_energy[:, 3]
            electron_energy_J = particle_energy[:, 4]
            total_energy_J = field_energy_J + ion_energy_J + electron_energy_J

            fig, ax = plt.subplots(**fig_args)
            ax.autoscale(enable=True, axis="x", tight=True)
            ax.set_yscale("log")
            ax.set_xlabel("Time ($\\mu$s)")
            ax.set_ylabel("Energy [J]")
            ax.axvline(start_time * 1e6, color="gray", linestyle="--", zorder=0)
            ax.plot(time, ion_energy_J, color="firebrick", label="Ions", linestyle="--")
            ax.plot(
                time,
                electron_energy_J,
                color="orange",
                label="Electrons",
                linestyle="-.",
            )
            ax.plot(time, field_energy_J, color="dodgerblue", label="Electric field")
            ax.plot(time, total_energy_J, color="black", linewidth=2, label="Total")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "energy.png")
            fig.savefig(output_dir / "energy.eps")
            if verbose:
                print("System energy plotted.")


if __name__ == "__main__":
    output_dir = utils.ensure_directory("./output")

    # parse args and log
    parser = argparse.ArgumentParser()
    parser.add_argument("--averaging-time", type=float, default=8e-6)
    parser.add_argument("--plot-2D", action="store_true", default=False)
    args, left = parser.parse_known_args()
    print(f"{args.averaging_time=}")
    print(f"{args.plot_2D=}")

    # Load data
    series = DataSeries(data_dir, start_time=0)
    max_time = np.max(series.time)
    start_time = max(max_time - args.averaging_time, 0.0)
    print("Data series loaded")

    # load n at first timestep to get dimensions
    n_ions_2D = series.load("rho_ions", dimension=2, inds=[0])[0] / q_e
    nz, nx = n_ions_2D.shape
    print(f"2D series loaded, shape = ({nx}, {nz})")

    # Create output directories
    ni_dir = utils.ensure_directory(output_dir / "ni")
    Ex_dir = utils.ensure_directory(output_dir / "Ex")
    Ez_dir = utils.ensure_directory(output_dir / "Ez")

    # Common figure setup
    fig_x = 13
    fig_y = nz / nx * fig_x
    figsize = (fig_x, fig_y)
    num_cells = nx * nz

    # Plot reduced diagnostics (particle number and energy)
    plot_reduced_diags(data_dir, output_dir, num_cells, start_time)

    # Plot and save VDFs
    if os.path.exists(particle_diag_dir):
        ds = load_last_particle_diag(particle_diag_dir)
        vdf_dir = utils.ensure_directory(output_dir / "vdf")

        # VDF on right boundary for electrons
        plot_vdf_1D_right(ds, "electrons", vdf_dir)

        # Axial and azimuthal VDFs vs x for both species
        for species in ["electrons", "ions"]:
            for coord in ["x", "z"]:
                plot_vdf_2D(ds, species, axis=coord, output_dir=vdf_dir)
        ds = None
        gc.collect()
        print("VDFs plotted")

    # plot fields every ten output intervals
    last_iter = series.iterations.size - 1
    inds = list(range(0, last_iter + 1, 10))
    if inds[-1] != last_iter:
        inds.append(last_iter)

    if args.plot_2D:
        print("Plotting 2D fields over time")
        for ind in inds:
            # common setup for this index
            t_us = series.time[ind] * 1e6
            series_str = f"ind={ind:04d}_time={t_us:2.2f}us"
            series_name = "_" + series_str + ".png"
            print(series_str)

            # Plot electric field - x
            E_2D = series.load("Ex", dimension=2, inds=[ind])[0] / 1000
            min_E, max_E = np.min(E_2D[:, 10:-10]), np.max(E_2D[:, 10:-10])
            max_E = max(abs(min_E), abs(max_E)) * 0.9
            plot_field_2D(
                E_2D,
                figsize,
                title=f"Axial electric field [kV/m] (t = {t_us:.2f} $\\mu$s)",
                outfile=Ex_dir / ("Ex" + series_name),
                im_args={"cmap": "RdBu_r", "vmin": -max_E, "vmax": max_E},
            )

            # Plot electric field - z
            E_2D = series.load("Ez", dimension=2, inds=[ind])[0] / 1000
            plot_field_2D(
                E_2D,
                figsize,
                title=f"Azimuthal electric field [kV/m] (t = {t_us:.2f} $\\mu$s)",
                outfile=Ez_dir / ("Ez" + series_name),
                im_args={
                    "cmap": "RdBu_r",
                    "vmin": -40,
                    "vmax": 40,
                    "extent": (0, 2.5, 0, 1.28),
                },
            )

            # Plot ion number density
            n_ions_2D = (
                series.load("rho_ions", dimension=2, inds=[ind])[0] / q_e
            )
            plot_field_2D(
                n_ions_2D,
                figsize,
                title="fIon number density [m$^{-3}$] (t = {t_us:.2f} $\\mu$s)",
                outfile=ni_dir / ("ni" + series_name),
                im_args={"vmin": 0},
            )

            # Collect garbage (help avoid running out of memory)
            gc.collect()

    # Check if we have enough times to average over
    inds = series.time >= start_time
    if not np.any(inds):
        quit()

    # Compute frequency spectrum of azimuthal electric field and save to JSON file
    if len(inds) > 10:
        freq_info = frequency_spectrum(series, start_time)
        save_spectrum(output_dir / "frequency.json", freq_info)

    # Compute wavenumber spectrum of azimuthal electric field and save to JSON file
    wavenumber_info = wavenumber_spectrum(series, start_time)
    save_spectrum(output_dir / "wavenumber.json", wavenumber_info)

    data = series.data[inds]

    # Load electron temperature and heat flux
    te_result = [ds.temp_and_heat_flux("electrons") for ds in data]
    Te = np.mean(np.array([t[0] for t in te_result]), 0)
    Q = np.mean(np.array([t[1] for t in te_result]), 0)
    print(f"{Te.size=}, {Te.shape=}")
    gc.collect()

    # Load ion density
    n_ions = np.mean(series.load("rho_ions", start_time=start_time, verbose=True), 0) / q_e
    print(f"{n_ions.size=}, {n_ions.shape=}")
    gc.collect()

    # Load electron density
    n_electrons = -np.mean(series.load("rho_electrons", start_time=start_time, verbose=True), 0) / q_e
    print(f"{n_electrons.size=}, {n_electrons.shape=}")
    gc.collect()

    # Load axial electric field
    E_x = np.mean(series.load("Ex", start_time=start_time, verbose=True), 0) / 1000
    print(f"{E_x.size=}, {E_x.shape=}")
    gc.collect()

    # Load ion and electron speeds
    ux_ions = np.mean(series.load("ux_ions", start_time=start_time, verbose=True), 0) * c
    ux_electrons = np.mean(series.load("ux_electrons", start_time=start_time, verbose=True), 0) * c
    gc.collect()

    # Begin making benchmark plots
    benchmark = LandmarkBenchmark().scaled("cm")
    print(f"Num checkpoints: {len(series.data)}")
    x = series.data[0].grid1D()
    xs = benchmark.grid(x.size)
    cath_ind = np.argmax(xs > benchmark.x_emit) - 1
    print(f"{cath_ind=}, {xs[cath_ind]=}")

    # Compute global metrics (avg temp, anode currents, cathode currents)
    with open(output_dir / "metrics.json", "w") as fd:
        metrics = {
            "Te_avg [eV]": np.mean(Te),
            "j_i_anode [A/m^2]": q_e * n_ions[0] * ux_ions[0],
            "j_e_anode [A/m^2]": -q_e * n_electrons[0] * ux_electrons[0],
            "j_i_cathode [A/m^2]": q_e * n_ions[cath_ind] * ux_ions[cath_ind],
            "j_e_cathode [A/m^2]": -q_e * n_electrons[cath_ind] * ux_electrons[cath_ind],
        }
        json.dump(metrics, fd, indent=4)

    _, _, ni_poly = utils.make_ref_polygon("ni", ref_dir)
    _, _, E_poly = utils.make_ref_polygon("E", ref_dir)
    E_poly[:, 1] /= 1000
    Te_low, Te_hi, Te_poly = utils.make_ref_polygon("Te", ref_dir)
    lw = 2

    fig, axes = benchmark_plot(1, 3, benchmark, xscale=0.63, yscale=1.1, dpi=fig_dpi)
    axes[0].add_patch(
        Polygon(E_poly, color=benchmark_color, label="Benchmark", zorder=2)
    )
    axes[0].plot(xs, E_x, color="black", linewidth=lw)
    axes[0].set_ylabel("Electric field [kV/m]")
    max_E = np.max(E_x[10:-10])
    axes[0].set_ylim(-5, max(max_E, 60))

    axes[1].add_patch(
        Polygon(ni_poly, color=benchmark_color, label="Benchmark", zorder=2)
    )
    axes[1].plot(xs, n_ions, color="black", linewidth=lw)
    axes[1].set_ylabel("Number density [m$^{-3}$]")

    axes[2].add_patch(
        Polygon(Te_poly, color=benchmark_color, label="Benchmark", zorder=2)
    )
    axes[2].plot(xs, Te, color="black", label="This work", linewidth=lw)
    axes[2].set_ylabel("Electron temperature [eV]")

    # Save individual subplots
    fig.tight_layout()
    save_subfig(fig, axes[0], output_dir, "E")
    save_subfig(fig, axes[1], output_dir, "ni")
    save_subfig(fig, axes[2], output_dir, "Te")

    # Add legend and save main plot
    lines_labels = axes[2].get_legend_handles_labels()
    lines = [ln[0] for ln in zip(*lines_labels)]
    labels = [ln[1] for ln in zip(*lines_labels)]
    fig.legend(reversed(lines), reversed(labels), loc="upper center", ncols=5)

    # Add plot numbers
    for i, ax in enumerate(axes):
        figsize = fig.get_size_inches()
        center = ax_center(fig, ax)
        text_x = center[0] / figsize[0]
        text_y = 0.02
        label = f"({chr(ord('a') + i)})"
        fig.text(text_x, text_y, label, size=20)

    # Adjust margins to avoid overlap
    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(bottom=0.2)

    # Save combined figure
    fig.savefig(output_dir / "output.png")
    fig.savefig(output_dir / "output.eps")

    # Save 1-D averaged fields to file
    data_fields = (xs, n_ions, n_electrons, E_x, Te, ux_ions, ux_electrons, Q[0, :])
    header = "x[cm],n_i[m^-3],n_e[m^-3],E_x[kV/m],T_e[eV],ux_ions[m/s],ux_electrons[m/s],q_x[eVm^-2s^-1]"
    np.savetxt(output_dir / "extracted.csv", data_fields, header=header, delimiter=",")
