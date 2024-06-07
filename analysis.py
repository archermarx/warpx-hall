import yt
yt.funcs.mylog.setLevel(50)

import numpy as np
from numpy import inf

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# some physical constants
c = 299_792_458
q_e = 1.60217663e-19
m_e = 9.1093837e-31

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
    low_points = np.genfromtxt(ref_dir + field + '_lower_raw.csv', delimiter = ',')
    hi_points = np.genfromtxt(ref_dir + field + '_upper_raw.csv', delimiter = ',')
    return low_points, hi_points, np.vstack([low_points, np.flip(hi_points, axis = 0), low_points[0, :]])

def benchmark_plot(yplots, xplots, benchmark = LandmarkBenchmark(), xscale = 1.0, yscale = 1.0):
    base_fig_width = 6.4 * xscale
    base_fig_height = 4.8 * yscale
    fig, axes = plt.subplots(yplots, xplots, figsize = (xplots * base_fig_width, yplots * base_fig_height))
    num_axes = yplots * xplots
    if (num_axes == 1):
        axes = [axes]
    for ax in axes:
        ax.set_xlim(0, benchmark.x_max)
        ax.set_xlabel(f"x [{benchmark.unit}]")
        ax.axvspan(benchmark.x_inj_0, benchmark.x_inj_1, color = 'grey', alpha = 0.2, label = "Injection region")
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

    def extent(self):
        dimensionality = self.dataset.dimensionality
        return [
            self.dataset.domain_left_edge[0], self.dataset.domain_right_edge[0],
            self.dataset.domain_left_edge[dimensionality-1], self.dataset.domain_right_edge[dimensionality-1]
        ]
    

import os

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

    def load(self, field, dimension = 1, axis = 1, jobs = 1):
        closure = load_field(field, dimension, axis)
        values = [closure(ds) for ds in self.data]
        return np.array(values)
    
    def load_averaged(self, field, dimension = 1, axis = 1):
        return np.mean(self.load(field, dimension, axis), 0)
    

case = 'diags'
os.makedirs("output", exist_ok=True)

ref_dir = "../warpx-hall/reference/"
_, _, ni_poly = make_ref_polygon("ni", ref_dir)
_, _, E_poly = make_ref_polygon("E", ref_dir)
E_poly[:, 1] /= 1000
Te_low, Te_hi, Te_poly = make_ref_polygon("Te", ref_dir)

series = DataSeries(case, start_time = 1.6e-5)
print("Data series loaded")
n_ions = series.load("rho_ions") / q_e
print("Density loaded")
E_x = series.load("Ex") / 1000
print("Electric field loaded")
Te = np.array([ds.compute_temp("electrons") for ds in series.data])
print("Temperature loaded")

ni_avg = np.mean(n_ions, 0)
Ex_avg = np.mean(E_x, 0)
Te_avg = np.mean(Te, 0)

benchmark = LandmarkBenchmark().scaled("cm")
x = series.data[0].grid1D()
xs = benchmark.grid(x.size)
benchmark_color = "#ffb3b3"

fig, axes = benchmark_plot(1, 3, benchmark, xscale = 0.8)
axes[0].add_patch(Polygon(E_poly, color = benchmark_color, label = "Benchmark", zorder = 2))
axes[0].plot(xs, Ex_avg, color = "black")
axes[0].set_title("Electric field")
axes[0].set_ylabel("Electric field [kV/m]")
axes[0].set_ylim(-5, 60)

axes[1].add_patch(Polygon(ni_poly, color = benchmark_color, label = "Benchmark", zorder = 2))
axes[1].plot(xs, ni_avg, color = "black")
axes[1].set_title("Ion number density")
axes[1].set_ylabel("Number density [m$^{-3}$]")

axes[2].add_patch(Polygon(Te_poly, color = benchmark_color, label = "Benchmark", zorder = 2))
axes[2].plot(xs, Te_avg, color = "black")
axes[2].set_title("Electron temperature")
axes[2].set_ylabel("Electron temperature [eV]")
axes[2].legend()

data_fields = (xs, ni_avg, Ex_avg, Te_avg)
header="x[cm],n_i[m^-3],E_x[kV/m],T_e[eV]"
np.savetxt("data_" + case + ".csv", data_fields, header = header, delimiter=',')

fig.savefig("output/benchmark_results.png")

ne_avg  = np.mean(series.load("rho_electrons") / q_e, 0)
uix_avg = np.mean(series.load("ux_ions") * c, 0)
uex_avg = np.mean(series.load("ux_electrons") * c, 0)
uez_avg = np.mean(series.load("uz_electrons") * c, 0)

j_i = q_e * ni_avg * uix_avg
j_e = -q_e * ne_avg * uex_avg

fig, ax = benchmark_plot(1,3, benchmark, xscale = 0.8)

ax[0].plot(xs, uix_avg / 1000, label = "Ions")
#ax[0].plot(xs, uex_avg / 1000, label = "Electrons")
ax[0].set_ylim(-5, 20)
ax[0].set_ylabel("Velocity [km/s]")
ax[0].set_title("Axial velocity")

ax[1].plot(xs, j_i / 1000, label = "Ion current")
ax[1].plot(xs, j_e / 1000, label = "Electron current")
ax[1].plot(xs, (j_e + j_i) / 1000, label = "Total current", color = 'red')
ax[1].axhline([0.4], label = "Expected ion current", linestyle = '--', color = 'black')
ax[1].set_ylabel("Current density [kA/m$^2$]")
ax[1].set_title("Axial current density")
ax[1].set_ylim(-1.5, 0.6)
ax[1].legend()

inv_hall = -uex_avg / uez_avg
inv_hall_offset = -uex_avg / (uez_avg + 8e4)

ax[2].set_ylim(1e-4, 1e2)
ax[2].axhline([0.0625], linestyle = '-', color = 'black', label = "Bohm")
ax[2].semilogy(xs, inv_hall, label = "PIC")
ax[2].semilogy(xs, inv_hall_offset, label = "PIC (offset)")
ax[2].set_title("Inverse Hall parameter")
ax[2].legend()

fig.tight_layout()
fig.savefig("output/velocities.png")

particlenumber_new = np.genfromtxt(case + "/reducedfiles/hall.txt")

seconds_to_us = 1e6

time_new = particlenumber_new[:, 1] * seconds_to_us
N_new = particlenumber_new[:, -3]

fig, ax = plt.subplots(1,1)

new_color = "#0000ee"

ax.plot(time_new, N_new, color = new_color, label = "New")
ax.set_xlim(np.min(time_new), np.max(time_new))
ax.set_xlabel("Time [$\mu$s]")
ax.set_ylabel("Physical particles in domain")
ax.set_title("Particle count")
ax.legend()
fig.savefig("output/particle_count.png")
