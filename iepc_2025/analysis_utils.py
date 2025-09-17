import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import yt
from scipy.fft import fft, fftfreq

# === Constants ===
q_e = 1.60217663e-19
m_e = 9.1093837e-31
c = 299_792_458.0
cathode_frac = 24 / 25
inf = np.inf

# === Module config ===
yt.funcs.mylog.setLevel(50)


def coord_name(axis: str) -> str:
    if axis == "x":
        return "Axial"
    elif axis == "y":
        return "Radial"
    elif axis == "z":
        return "Azimuthal"
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


def frequency_spectrum(series, field):
    """
    Compute the frequency spectrum of a field as a function of axial location.
    The amplitudes are averaged over the y-coordinate.
    Returns a dict containing the following keys:
    - "x_m": The axial coordinates at which the spectra were computed.
    - "frequencies": the frequencies at which the amplitudes are known
    - "amplitudes": a `list` of amplitude vectors, with each vector corresponding to a specific axial location
    - "peak_freqs": a vector containing the dominant frequency for each axial location
    """
    x_m = series["x"]
    y_m = series["z"]
    frames = series[field]
    t = series["t"]
    M = len(frames)
    dt = t[2] - t[1]

    frequency = fftfreq(M, dt)[1 : M // 2] * 2 * np.pi
    amplitudes = []
    peak_freqs = []

    # Compute the y-averaged frequency spectrum at each x location
    for i, _ in enumerate(x_m):
        _amplitude = np.zeros_like(frequency)
        for j, _ in enumerate(y_m):
            frame = [_frame[j, i] for _frame in frames]
            frame_fft = np.abs(fft(frame))[1 : M // 2]
            _amplitude += frame_fft

        amplitudes.append(_amplitude / len(y_m))
        peak_freq = frequency[np.argmax(_amplitude)]
        peak_freqs.append(peak_freq)

    return {
        "x_m": x_m,
        "frequency": frequency,
        "amplitudes": amplitudes,
        "peak_freqs": peak_freqs,
    }


def wavenumber_spectrum(series, field):
    """
    Compute the wavenumber spectrum of a field as a function of axial location.
    The amplitudes are averaged over time, starting at the provided `start_time`.
    Returns a dict containing the following keys:
    - "x_m": The axial coordinates at which the spectra were computed.
    - "frequencies": the wavenumbers at which the amplitudes are known
    - "amplitudes": a `list` of amplitude vectors, with each vector corresponding to a specific axial location
    - "peak_freqs": a vector containing the dominant wavenumber for each axial location
    """
    x_m = series["x"]
    y_m = series["z"]
    frames = series[field]
    M = len(y_m)
    dy = y_m[1] - y_m[0]

    frequency = fftfreq(M, dy)[1 : M // 2]
    amplitudes = []
    peak_freqs = []

    # Compute the y-averaged frequency spectrum at each x location
    for i, x in enumerate(x_m):
        _amplitude = np.zeros_like(frequency)
        for _frame in frames:
            frame_fft = np.abs(fft(_frame[:, i]))[1 : M // 2]
            _amplitude += frame_fft

        amplitudes.append(_amplitude / len(frames))
        peak_freq = frequency[np.argmax(_amplitude)]
        peak_freqs.append(peak_freq)

    return {
        "x_m": x_m,
        "frequency": frequency,
        "amplitudes": amplitudes,
        "peak_freqs": peak_freqs,
    }


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

    def load(self, field, dimension=2, axis=1):
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

    def extent(self, scale=1.0):
        dimensionality = self.dataset.dimensionality
        return [
            self.dataset.domain_left_edge[0] * scale,
            self.dataset.domain_right_edge[0] * scale,
            self.dataset.domain_left_edge[dimensionality - 1] * scale,
            self.dataset.domain_right_edge[dimensionality - 1] * scale,
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

    def load(
        self, field, dimension=1, axis=1, inds=None, start_time=0.0, verbose=False
    ):
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
