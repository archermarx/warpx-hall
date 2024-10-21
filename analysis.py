import yt
yt.funcs.mylog.setLevel(50)

import numpy as np
from numpy import inf

# Matplotlib config
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
plt.set_loglevel("error")
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)

# Directories
import os
ref_dir = "/home/marksta/src/warpx-hall/reference"
data_dir = "./diags"
output_dir = "./output"
fig_dpi = 200
# create output directory
os.makedirs(output_dir, exist_ok = True)

# some physical constants
import scipy.constants as constants
c = constants.c
q_e = constants.e
m_e = constants.m_e
eps_0 = constants.epsilon_0
k_B = constants.Boltzmann

# Benchmark definition
class LandmarkBenchmark:
    def __init__(self, dt = 5e-12, x_max = 2.5e-2, unit = "m"):
        self.dt = dt
        self.x_max = x_max
        self.unit = unit
        self.x_bmax  = 0.3   * self.x_max
        self.x_inj_0 = 0.1   * self.x_max
        self.x_inj_1 = 0.4   * self.x_max
        self.x_emit  = 24/25 * self.x_max
        self.x_inj_m = 0.5 * (self.x_inj_0 + self.x_inj_1)

    def scaled(self, unit):
        scale_factors = {"m": 1.0, "cm": 100, "mm": 1000, "um": 1_000_000, "μm": 1_000_000, "nm": 1_000_000_000}
        try:
            scale_factor= scale_factors[unit]
        except KeyError:
            raise KeyError(f"Invalid length unit '{unit}' specified. Please pick one of {'m', 'cm', 'mm', 'um', 'nm'}")

        return LandmarkBenchmark(dt = self.dt, x_max = scale_factor * self.x_max, unit = unit)
    
    def grid(self, Nx):
        return np.linspace(0, self.x_max, Nx)
    

def make_ref_polygon(field, ref_dir):
    """
    Takes two curves defined in csv files (`field`_lower_raw.csv) and (`field`_upper_raw.csv), and them together into a closed path.
    """
    low_points = np.genfromtxt(os.path.join(ref_dir, field + '_lower_raw.csv'), delimiter = ',')
    hi_points = np.genfromtxt(os.path.join(ref_dir, field + '_upper_raw.csv'), delimiter = ',')
    return low_points, hi_points, np.vstack([low_points, np.flip(hi_points, axis = 0), low_points[0, :]])

def benchmark_plot(yplots, xplots, benchmark = LandmarkBenchmark(), xscale = 1.0, yscale = 1.0, **kwargs):
    base_fig_width = 6.4 * xscale
    base_fig_height = 4.8 * yscale
    fig, axes = plt.subplots(yplots, xplots, figsize = (xplots * base_fig_width, yplots * base_fig_height), **kwargs)
    num_axes = yplots * xplots

    lightgrey = "#dddddd"
    if (num_axes == 1):
        axes = [axes]
    for (i, ax) in enumerate(axes):
        ax.set_xlim(0, benchmark.x_max)
        ax.set_xlabel(f"x [{benchmark.unit}]")
        ax.axvspan(benchmark.x_inj_0, benchmark.x_inj_1, color = lightgrey, label = "Injection region")
        ax.axvline([benchmark.x_emit], linestyle = '--', color = 'grey', linewidth = 1, label = "Cathode plane")
        ax.axvline([benchmark.x_bmax], linestyle = '-', color = 'grey', linewidth = 1, label = "Peak magnetic field")
        ax.axhline([0], linestyle = '-', color = 'grey')

    return fig, axes

class GriddedData:
    def __init__(self, path):
        ds = yt.load(path)
        self.dataset = ds
        self.gridded_data = self.dataset.covering_grid(level=0,left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)

    def grid1D(self, axis = 0):
        x0 = self.dataset.domain_left_edge[axis]
        x1 = self.dataset.domain_right_edge[axis]
        return np.linspace(x0, x1, self.dataset.domain_dimensions[axis])

    def load1D(self, field, axis = 1):
        """Load a 2-D field, averaged over axis `axis`"""
        data = self.gridded_data['boxlib', field].v[:, :, 0]
        return np.mean(data, axis)

    def load2D(self, field):
        """Load a 2-D field from the dataset"""
        return np.transpose(self.gridded_data['boxlib', field].v[:, :, 0])
    
    def load(self, field, dimension = 1, axis = 1):
        if (dimension == 1):
            return self.load1D(field, axis)
        elif (dimension == 2):
            return self.load2D(field)
        else:
            raise ValueError(f"Invalid dimension: {dimension}")

    def compute_temp(self, species, dimension = 1, axis = 1):
        ux  = self.load("ux_" + species, dimension, axis)
        uy  = self.load("uy_" + species, dimension, axis)
        uz  = self.load("uz_" + species, dimension, axis)
        Pxx = self.load("P_xx_" + species, dimension, axis)
        Pyy = self.load("P_yy_" + species, dimension, axis)
        Pzz = self.load("P_zz_" + species, dimension, axis)
        E_factor = m_e * c**2 / q_e

        Ex = E_factor * (Pxx - ux**2)
        Ey = E_factor * (Pyy - uy**2)
        Ez = E_factor * (Pzz - uz**2)

        T = (Ex + Ey + Ez) / 3

        return T

    def temp_and_heat_flux(self, species, dimension = 1, axis = 1):
        ux  = self.load("ux_" + species, dimension, axis)
        uy  = self.load("uy_" + species, dimension, axis)
        uz  = self.load("uz_" + species, dimension, axis)
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
        P = np.array([
            [Pxx, Pxy, Pxz],
            [Pxy, Pyy, Pyz],
            [Pxz, Pyz, Pzz]
        ])

        # Pressure tensor
        # p = P - np.outer(u,u)
        p = P - np.einsum("ik,jk->ijk", u, u)

        # Temperature
        T = np.trace(p, axis1=0, axis2=1) / 3

        # Total energy
        E_tot = 1.5 * T + 0.5 * u2

        # Energy flux (E * u)
        E_flux = E_tot * u

        # Heat flux
        # Q - pu - E_flux
        q = Q - np.einsum("ijk,jk->ik", p, u) - E_tot * u
        
        return T * E_factor, q * c * E_factor

    def extent(self):
        dimensionality = self.dataset.dimensionality
        return [
            self.dataset.domain_left_edge[0], self.dataset.domain_right_edge[0],
            self.dataset.domain_left_edge[dimensionality-1], self.dataset.domain_right_edge[dimensionality-1]
        ]

def read_plotfile_header(dir):
    header_file = os.path.join(dir, "Header")
    with open(header_file) as f:
        lines = f.readlines()

    num_fields = int(lines[1].strip())
    fields = []
    first_ind = 2
    last_ind = first_ind + num_fields
    fields = map(lambda l: l.strip(), lines[first_ind:last_ind])

    time = float(lines[last_ind+1].strip())
    iter = int(lines[len(lines)-4].strip())

    return {"num_fields": num_fields, "fields": fields, "time": time, "iter": iter}

def sortperm(list):
    return sorted(range(len(list)), key = lambda i: list[i])

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
    def __init__(self, dir, start_time = 0, end_time = inf):
        # read data from selected time window
        subdirs = filter(dir_pred(start_time, end_time), [f.path for f in os.scandir(dir) if f.is_dir()])

        # get data
        data = [get_data(subdir) for subdir in subdirs]

        # sort data by iteration
        perm = sorted(range(len(data)), key = lambda k: data[k]["iter"])

        #set member variables
        self.iterations = [data[i]["iter"] for i in perm]
        self.time       = [data[i]["time"] for i in perm]
        self.data       = [data[i]["data"] for i in perm]
        self.num_iters = len(perm)
        dt = np.zeros(self.num_iters);
        dt[1:] = np.diff(self.time)
        dt[0] = dt[1]
        self.delta_time = dt
        self.time_interval = self.time[-1] - self.time[0]


    def load(self, field, dimension = 1, axis = 1, jobs = 1, inds = None):
        closure = load_field(field, dimension, axis)
        if inds is None:
            values = np.array([closure(ds) for ds in self.data])
        else:
            values = np.array([closure(self.data[i]) for i in inds])
        return values
    
    def load_averaged(self, field, dimension = 1, axis = 1):
        return np.sum(self.delta_time * self.load(field, dimension, axis), 0) / self.time_interval
    
# Load data
series = DataSeries(data_dir, start_time = 1.2e-5)
print("Data series loaded")

# Load 2D ion density and azimuthal electric field at last timestep
n_ions_2D = series.load("rho_ions", dimension = 2, inds = [-1])[0] / q_e
E_2D = series.load("Ez", dimension = 2, inds = [-1])[0]
nz, nx = n_ions_2D.shape
print(f"2D series loaded, shape = ({nx}, {nz})")
fig_x = 13
fig_y = nz/nx * fig_x
num_cells = nx * nz
fig, ax = plt.subplots(1,1, figsize = (fig_x, fig_y), dpi = fig_dpi)
ax.set_title("Azimuthal electric field")
im = ax.imshow(E_2D)
fig.colorbar(im, ax = ax)
fig.savefig(os.path.join(output_dir, "E_2D.png"))

fig, ax = plt.subplots(1,1, figsize = (fig_x, fig_y), dpi = fig_dpi)
ax.set_title("Ion number density")
im = ax.imshow(n_ions_2D)
fig.colorbar(im, ax = ax)
fig.savefig(os.path.join(output_dir, "n_2D.png"))

# Plot particle counts
particle_data = np.genfromtxt("diags/reducedfiles/hall.txt", skip_header = 1, delimiter = ' ')
time = particle_data[:, 1] * 1e6
num_cells = nx * nz
num_ion = particle_data[:, 3]
num_ele = particle_data[:, 4]
fig, ax = plt.subplots(1,1, dpi = fig_dpi)
ax.set_xlabel("Time (us)")
ax.set_ylabel("Particles/cell")
ax.set_title("Macroparticles per cell")
l_ion = plt.plot(time, num_ion / num_cells, label = 'Ions')
l_ele = plt.plot(time, num_ele / num_cells, color = 'red', linestyle = '--', label = 'Electrons')
l_tot = plt.plot(time, (num_ion + num_ele) / num_cells, color = 'black', label = 'Total')
plt.legend(loc = "upper left")
fig.tight_layout();
fig.savefig(os.path.join(output_dir, "particle_number.png"))
fig.savefig(os.path.join(output_dir, "particle_number.eps"))


te_result = [ds.temp_and_heat_flux("electrons") for ds in series.data]
Te = np.array([t[0] for t in te_result])
Q = np.array([t[1] for t in te_result])
print("Temperature and heat flux loaded")

n_ions = series.load("rho_ions") / q_e
print("Density loaded")

E_x = series.load("Ex") / 1000
print("Electric field loaded")

# compute averaged properties
ni_avg = np.mean(n_ions, 0)
Ex_avg = np.mean(E_x, 0)
Te_avg = np.mean(Te, 0)
Qe_avg = np.mean(Q, 0)

def ax_bbox(fig, ax):
    return ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())

def ax_center(fig, ax):
    bbox = ax_bbox(fig, ax)
    center = bbox.p0 + 0.5 * np.array(bbox.width, bbox.height)
    return center

def save_subfig(fig, ax, output_dir, filename):
    fig.savefig(os.path.join(output_dir, filename + ".png"),  bbox_inches = ax_bbox(fig, ax))
    fig.savefig(os.path.join(output_dir, filename + ".eps"),  bbox_inches = ax_bbox(fig, ax))

benchmark = LandmarkBenchmark().scaled("cm")
print(f"Num checkpoints: {len(series.data)}")
x = series.data[0].grid1D()
xs = benchmark.grid(x.size)
benchmark_color = "#ffb3b3"

_, _, ni_poly = make_ref_polygon("ni", ref_dir)
_, _, E_poly = make_ref_polygon("E", ref_dir)
E_poly[:, 1] /= 1000
Te_low, Te_hi, Te_poly = make_ref_polygon("Te", ref_dir)
lw = 2

fig, axes = benchmark_plot(1, 3, benchmark, xscale = 0.63, yscale = 1.1, dpi = fig_dpi)
axes[0].add_patch(Polygon(E_poly, color = benchmark_color, label = "Benchmark", zorder = 2))
axes[0].plot(xs, Ex_avg, color = "black", linewidth = lw)
axes[0].set_ylabel("Electric field [kV/m]")
#axes[0].set_ylim(-5, 60)

axes[1].add_patch(Polygon(ni_poly, color = benchmark_color, label = "Benchmark", zorder = 2))
axes[1].plot(xs, ni_avg, color = "black", linewidth = lw)
axes[1].set_ylabel("Number density [m$^{-3}$]")

axes[2].add_patch(Polygon(Te_poly, color = benchmark_color, label = "Benchmark", zorder = 2))
axes[2].plot(xs, Te_avg, color = "black", label = "This work", linewidth = lw)
axes[2].set_ylabel("Electron temperature [eV]")

# Save individual subplots
fig.tight_layout()
save_subfig(fig, axes[0], output_dir, "E")
save_subfig(fig, axes[1], output_dir, "ni")
save_subfig(fig, axes[2], output_dir, "Te")

# Add legend and save main plot
lines_labels = axes[2].get_legend_handles_labels()
lines = [l[0] for l in  zip(*lines_labels)]
labels = [l[1] for l in  zip(*lines_labels)]
fig.legend(reversed(lines), reversed(labels), loc = "upper center", ncols = 5)

# Add plot numbers
for (i, ax) in enumerate(axes):
    figsize = fig.get_size_inches()
    center = ax_center(fig, ax)
    text_x = center[0] / figsize[0]
    text_y = 0.02
    label = f"({chr(ord('a')+i)})"
    fig.text(text_x, text_y, label, size = 20)

# Adjust margins to avoid overlap
fig.subplots_adjust(top = 0.85)
fig.subplots_adjust(bottom = 0.2)

# Save combined figure
fig.savefig(os.path.join(output_dir, "output.png"))
fig.savefig(os.path.join(output_dir, "output.eps"))

# Save data to file
data_fields = (xs, ni_avg, Ex_avg, Te_avg, Qe_avg[0, :])
header="x[cm],n_i[m^-3],E_x[kV/m],T_e[eV],q_x[eVm^-2s^-1]"
np.savetxt(os.path.join(output_dir, "extracted.csv"), data_fields, header = header, delimiter=',')

