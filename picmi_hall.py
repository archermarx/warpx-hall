####################################################################
#                            IMPORTS                               #
####################################################################

# c.f. https://sppl.stanford.edu/wp-content/uploads/2020/09/MicroHall.pdf

import numpy as np
from math import sqrt, ceil, floor
import time

# Read command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--case', type=int)
parser.add_argument('-n', '--numgpus', type=int)
parser.add_argument('--resample', action = argparse.BooleanOptionalAction)
parser.set_defaults(resample=False)
parser.add_argument('--resample_min', type=int)
parser.add_argument('--resample_max', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--sort_interval', type=int)
parser.add_argument('--mlmg_precision', type=float)
parser.add_argument('--collision_interval', type=int)
parser.add_argument('--diag_interval', type=int)
parser.add_argument('--cathode', action = argparse.BooleanOptionalAction)
parser.set_defaults(cathode=True)
parser.add_argument('--shape_factor', type=int)

# default arguments
case = 1
numgpus = 1
resample = False
resample_min = 75
resample_max = 300
sort_interval = 500
mlmg_precision = 1e-5
collision_interval = 0
diag_interval = 5000            # Interval between diagnostic outputs (iters)
cathode=True                # Whether cathode particle injection BC is applied
shape_factor = 1

seed = np.random.randint(1_000_000);

args, leftovers = parser.parse_known_args()
if (args.case is not None):
    case = args.case
if (args.numgpus is not None):
    numgpus = args.numgpus
if (args.resample is not None):
    print("args.resample = ", args.resample)
    resample = args.resample
if (args.resample_min is not None):
    resample_min = args.resample_min
if (args.resample_max is not None):
    resample_max = args.resample_max
if (args.seed is not None):
    seed = args.seed
if (args.sort_interval is not None):
    sort_interval = args.sort_interval
if (args.mlmg_precision is not None):
    mlmg_precision = args.mlmg_precision
if (args.collision_interval is not None):
    collision_interval = args.collision_interval
if (args.diag_interval is not None):
    diag_interval = args.diag_interval
if (args.cathode is not None):
    cathode = args.cathode
if (args.shape_factor is not None):
    shape_factor = args.shape_factor

# Print parsed args
print(f'Case: {case}')
print(f'Num GPUs: {numgpus}')
print(f'Seed: {seed}')
print(f'Resample: {resample}')
if (resample):
    print(f'Resample min particles: {resample_min}')
    print(f'Resample max particles: {resample_max}')
print(f'Sort interval: {sort_interval}')
print(f'MLMG precision: {mlmg_precision}')
print(f'Collision interval: {collision_interval}')
print(f'Diag interval: {diag_interval}')
print(f'Cathode enabled: {cathode}')
print(f'Particle shape factor: {shape_factor}')

# Cases
Np_base = 75
Nx_base = 512
Nz_base = 256

if (case == 1):
    particle_factor = 1
    grid_factor = 1

elif (case == 2):
    particle_factor = 2
    grid_factor = 1

elif (case == 3):
    particle_factor = 4
    grid_factor = 1

else:
    raise Exception(f"Invalid case selected: {case}.")

# set workload parameters
Np = Np_base * particle_factor
Nx = Nx_base * grid_factor
Nz = Nz_base * grid_factor

# Set particle shape factor
if shape_factor == 1:
    particle_shape = 'linear'
elif shape_factor == 2:
    particle_shape = 'quadratic'
elif shape_factor == 3:
    particle_shape = 'cubic'
else:
    raise Exception(f"Invalid particle shape specified: {shape_factor}.")

exit()

print(f"Particles per cell, initial: {Np}")
print(f"Axial cells: {Nx}")
print(f"Azimuthal cells: {Nz}")

# Set parallization parameters
# NOTE: we currently split the domain in the azimuthal direction only,
# as our electrostatic potential callback will not work properly
# if we split the domain axially instead
numprocs = [1, numgpus]

# Import WarpX packages
from pywarpx import callbacks, fields, libwarpx, particle_containers, picmi
from periodictable import elements

# Physical constants
m_p = picmi.constants.m_p
m_e = picmi.constants.m_e
q_e = picmi.constants.q_e
ep0 = picmi.constants.ep0

####################################################################
#                     CONFIGURABLE OPTIONS                         #
####################################################################
diag_name = "hall"              # Case name
verbose = True                  # Whether to use verbose output
num_grids = 1                   # Number of subgrids to decompose domain into
dt = 5e-12                      # Timestep size (seconds)
max_time = 20e-6                # Max time (s)

# Geometric factors
# NOTE: in WarpX 2D simulations, the coordinate axes are x and z.
# We treat 'x' as the axial dimension and 'z' as the azimuthal dimension.
Lx = 2.5e-2                     # Axial domain length (m)
Lz = 1.28e-2                    # Azimuthal domain length (m)
dx = Lx / Nx                    # Axial grid spacing (m)
dz = Lz / Nz                    # Azimuthal grid spacing (m)
x1_iz = 0.1 * Lx                # Axial location of beginning of ionization region
x2_iz = 0.4 * Lx                # Axial location of end of ionization region
xm_iz = 0.5 * (x1_iz + x2_iz)   # Axial location of middle of ionization region
dx_iz = x2_iz - x1_iz           # Axial extent of ionization region
x_emission = 24/25 * Lx         # axial location of emission/cathode plane

# physics properties
U0 = 200.0                      # Discharge voltage (V)
B_max = 1e-2                    # Peak magnetic field strength (T)
B0 = 0.6 * B_max                # Magnetic field at anode
B_min = 0.1 * B_max             # Magnetic field at Lx
sigma_B_1 = 0.25 * Lx           # Width of magnetic field profile
sigma_B_2 = sigma_B_1
x_Bmax = 0.3 * Lx               # Location of peak magnetic field strength
species = "Xe"                  # Ion species
T_e = 10.0                      # Electron temperature (eV)
T_i = 0.5                       # Ion temperature (eV)
n0 = 5e16                       # Initial plasma density (m^-3)
S0 = 5.23e23                    # maxmimum ionization rate
weight = n0 * Lx * Lz / (Np * Nx * Nz)                      # macroparticle weight:
N_inject = 2 * dt * Lz * S0 * dx_iz / (weight * np.pi)      # macroparticles to inject each timestep

# Coefficients for prescribed magnetic field
exp_factor_1 = np.exp(-0.5 * (x_Bmax/sigma_B_1)**2)
exp_factor_2 = np.exp(-0.5 * ((Lx - x_Bmax)/sigma_B_2)**2)
a1 = (B_max -    B0) / (1 - exp_factor_1)
a2 = (B_max - B_min) / (1 - exp_factor_2)
b1 = (B0    - B_max * exp_factor_1) / (1 - exp_factor_1)
b2 = (B_min - B_max * exp_factor_2) / (1 - exp_factor_2)

# Magnetic field function string
B_func = f"if(x<{x_Bmax}, {a1}*exp(-0.5*( (x-{x_Bmax})/{sigma_B_1} )**2) + {b1}, {a2}*exp(-0.5*( (x-{x_Bmax})/{sigma_B_2} )**2) + {b2})"

m_i = elements.symbol(species).mass * m_p       # Ion mass
ve_std = sqrt(q_e * T_e / m_e)                  # Electron thermal speed
vi_std = sqrt(q_e * T_i / m_i)                  # Ion thermal speed
max_steps = ceil(max_time / dt)                 # Maximum simulation steps

####################################################################
#                       SIMULATION SETUP                           #
####################################################################

# Grid
grid = picmi.Cartesian2DGrid(
    number_of_cells = [Nx, Nz],
    lower_bound = [0, 0],
    upper_bound = [Lx, Lz],
    lower_boundary_conditions = ['dirichlet', 'periodic'],
    upper_boundary_conditions = ['dirichlet', 'periodic'],
    lower_boundary_conditions_particles = ['absorbing', 'periodic'],
    upper_boundary_conditions_particles = ['absorbing', 'periodic'],
    warpx_potential_lo_x = U0,
    warpx_potential_hi_x = 0.0,
)

# Applied fields
external_field = picmi.AnalyticAppliedField(
    Bx_expression = "0.0",
    By_expression = B_func,
    Bz_expression = "0.0"
)

# Initial particle setup
particle_layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell = Np, grid = grid)
dist_e = picmi.UniformDistribution(n0, rms_velocity = [ve_std, ve_std, ve_std], lower_bound = [0.0, 0.0, 0.0], upper_bound = [Lx, Lz, Lz])
dist_i = picmi.UniformDistribution(n0, rms_velocity = [vi_std, vi_std, vi_std], lower_bound = [0.0, 0.0, 0.0], upper_bound = [Lx, Lz, Lz])

# Species definitions
# We save ions and electrons that hit the left boundary so we can
# inject the correct cathode current
electrons = picmi.Species(
    particle_type = 'electron', name = 'electrons',
    initial_distribution = dist_e,
    warpx_save_particles_at_xlo=1,
    # resampling options
    warpx_do_resampling = resample,
    warpx_resampling_trigger_max_avg_ppc = resample_max,
    warpx_resampling_min_ppc = resample_min,
)

ions = picmi.Species(
    particle_type = species, name = 'ions', mass = m_i, charge = 'q_e',
    initial_distribution = dist_i,
    warpx_save_particles_at_xlo=1,
    # resampling options
    warpx_do_resampling = resample,
    warpx_resampling_trigger_max_avg_ppc = resample_max,
    warpx_resampling_min_ppc = resample_min,
)

# Field solver
solver = picmi.ElectrostaticSolver(
    grid=grid,
    required_precision=mlmg_precision,
    warpx_self_fields_verbosity=1,
)

collision_ee = picmi.CoulombCollisions("ee", species = [electrons, electrons], ndt = collision_interval)
collision_ei = picmi.CoulombCollisions("ei", species = [electrons, ions], ndt = collision_interval)
collision_ii = picmi.CoulombCollisions("ii", species = [ions, ions], ndt = collision_interval)

collisions = []
if (collision_interval > 0):
    collisions = [collision_ee, collision_ei, collision_ii]

# Initialize simulation
sim = picmi.Simulation(
    solver = solver,
    time_step_size = dt,
    max_time = max_time,
    verbose = verbose,
    warpx_random_seed = seed,
    warpx_sort_intervals = sort_interval,
    warpx_numprocs = numprocs,
    warpx_use_filter = True,
    warpx_amrex_use_gpu_aware_mpi = True,
    warpx_collisions = collisions,
    particle_shape = particle_shape
)

sim.add_species(ions, layout = particle_layout)
sim.add_species(electrons, layout = particle_layout)
sim.add_applied_field(external_field)
solver.sim = sim

# Set up infrequent checkpoints
checkpoint = picmi.Checkpoint(
    period = int(max_steps / 10),
    warpx_file_prefix = "checkpoint",
    write_dir = "checkpoint",
    warpx_file_min_digits = 10
)
sim.add_diagnostic(checkpoint)

# Particle saving options.
# Note that this is currently disabled (see commented-out line below) due to the amount of space required
particle_diag = picmi.ParticleDiagnostic(
    name = diag_name + '_particles',
    period = diag_interval * 10,
    data_list = ['position', 'momentum', 'weighting'],
    write_dir = 'diags',
    warpx_format="plotfile",
)
# sim.add_diagnostic(particle_diag)

# Grid diagnostics
# In addition to normal grid quantities, we also save gridded moments of the electron VDF for later analysis
# Note that
field_diag = picmi.FieldDiagnostic(
    name = diag_name,
    grid = grid,
    period = diag_interval,
    data_list = ['rho', 'rho_ions', 'rho_electrons', 'E', 'B', 'J', 'phi'],
    write_dir = 'diags',
    warpx_format = "plotfile",
    warpx_particle_fields_species = ['electrons', 'ions'],
    warpx_particle_fields_to_plot = [
        # ===================================================================
        # MEAN VELOCITY
        #
        # \vec{u}_e = \int \vec{v} f(\vec{v}) d^3 v
        #
        # ===================================================================
        # Mean axial velocity (units: v/c)
        picmi.ParticleFieldDiagnostic(
            name = 'ux',
            func = 'ux',
        ),
        # Mean radial velocity (units: v/c)
        picmi.ParticleFieldDiagnostic(
            name = 'uy',
            func = 'uy',
        ),
        # Mean azimuthal velocity (units: v/c)
        picmi.ParticleFieldDiagnostic(
            name = 'uz',
            func = 'uz',
        ),
        # ===================================================================
        # STRESS TENSOR
        #
        # P_{i,j}= \int v_i v_j f(\vec{v}) d^3 v
        #
        # The pressure tensor, p_{i,j}, can be retrieved as
        #
        # p = n P - m n u u.
        #
        # Kinetic temperature (in J) can be computed as
        #
        # T = m_e c^2 \frac{1}{3} \{
        #       (P_{x,x} - u_{x}^2)
        #     + (P_{y,y} - u_{y}^2)
        #     + (P_{z,z} - u_{z}^2)
        # \}
        #
        # ===================================================================
        picmi.ParticleFieldDiagnostic(
            name = 'P_xx',
            func = 'ux * ux',
        ),
        picmi.ParticleFieldDiagnostic(
            name = 'P_xy',
            func = 'ux * uy',
        ),
        picmi.ParticleFieldDiagnostic(
            name = 'P_xz',
            func = 'ux * uz',
        ),
        picmi.ParticleFieldDiagnostic(
            name = 'P_yy',
            func = 'uy * uy',
        ),
        picmi.ParticleFieldDiagnostic(
            name = 'P_yz',
            func = 'uy * uz',
        ),
        picmi.ParticleFieldDiagnostic(
            name = 'P_zz',
            func = 'uz * uz',
        ),
        # ===================================================================
        # ENERGY FLUX VECTOR (units v^3 / c^3)
        #
        # Q_i = \int v_i^2 v_j f(\vec{v}) d^3v
        #
        # Heat flux vector can be retrieved as
        #
        # q_i = Q_i - \sum_j p_{ij} u_j - 3/2 T u_i - 1/2 m |u^2| u_i
        # ===================================================================
        # Axial energy flux (units: v^3/c^3)
        picmi.ParticleFieldDiagnostic(
            name = 'Q_x',
            func = 'ux * (ux*ux + uy*uy + uz*uz)',
        ),
        # Radial heat flux (units: v^3/c^3)
        picmi.ParticleFieldDiagnostic(
            name = 'Q_y',
            func = 'uy * (ux*ux + uy*uy + uz*uz)',
        ),
        # Azimuthal heat flux (units: v^3/c^3)
        picmi.ParticleFieldDiagnostic(
            name = 'Q_z',
            func = 'uz * (ux*ux + uy*uy + uz*uz)',
        ),
    ]
)

sim.add_diagnostic(field_diag)

# Particle number diagnostic
particle_number = picmi.ReducedDiagnostic(
    name = diag_name,
    diag_type = 'ParticleNumber',
    period = diag_interval,
)

sim.add_diagnostic(particle_number)

# Initialize simulation
sim.initialize_inputs()

####################################################################
#                           CALLBACKS                              #
####################################################################
sim.initialize_warpx()

elec_wrapper = particle_containers.ParticleContainerWrapper('electrons')
ion_wrapper = particle_containers.ParticleContainerWrapper('ions')

from pywarpx.callbacks import callfrombeforeInitEsolve, callfromparticleinjection

# Inject electron-ion pairs at each timestep
def inject_particles():
    # Use a random number to decide whether to inject fractional part of N_inject
    particles_to_inject = floor(N_inject)
    remainder = N_inject - particles_to_inject
    if (np.random.rand() < remainder):
        particles_to_inject += 1

    # Sample random numbers for position
    r1 = np.random.uniform(0, 1, size = particles_to_inject)
    r2 = np.random.uniform(0, 1, size = particles_to_inject)

    xs = xm_iz + np.arcsin(2 * r1 - 1) * (dx_iz / np.pi)    # x-position determined by prescribed profile
    ys = np.zeros(particles_to_inject)                      # zero y-position
    zs = r2 * Lz                                            # Uniform in azimuthal position
    ws = np.full(particles_to_inject, weight)               # equal weights

    # Sample velocities from maxwellians
    vel_electron = ve_std * np.random.randn(particles_to_inject, 3)
    vel_ion      = vi_std * np.random.randn(particles_to_inject, 3)

    # Add particles to simulation
    elec_wrapper.add_particles(
        x = xs, y = ys, z = zs,
        ux = vel_electron[:, 0], uy = vel_electron[:, 1], uz = vel_electron[:, 2],
        w = ws
    )

    ion_wrapper.add_particles(
        x = xs, y = ys, z = zs,
        ux = vel_ion[:, 0], uy = vel_ion[:, 1], uz = vel_ion[:, 2],
        w = ws
    )

# Install ionization callback
callbacks.installparticleinjection(inject_particles)

import cupy as cp

particle_buffer = particle_containers.ParticleBoundaryBufferWrapper()

def inject_cathode():
    # Read anode boundary buffer ('x_lo') to see how many particles of each type were scraped
    Ne_anode = particle_buffer.get_particle_boundary_buffer_size("electrons", "x_lo")
    Ni_anode = particle_buffer.get_particle_boundary_buffer_size("ions", "x_lo")

    # Number of particles to inject at cathode
    N_cath = Ne_anode - Ni_anode

    # clear buffer
    particle_buffer.clear_buffer()

    # inject new electrons for current continuity
    if (N_cath > 0):
        x = np.full(N_cath, x_emission)         # All particles injected at emission plane
        y = np.zeros(N_cath)                    # Zero radial position
        z = np.random.uniform(0, Lz, N_cath)    # Uniform distribution in azimuthal dimension
        v = ve_std * np.random.randn(N_cath, 3) # 3D maxwellian velocity distribution
        w = np.full(N_cath, weight)
        elec_wrapper.add_particles(
            x = x, y = y, z = z,
            ux = v[:, 0], uy = v[:, 1], uz = v[:, 2],
            w = w
        )

if cathode:
    callbacks.installparticleinjection(inject_cathode)

x_nodes = dx * cp.arange(Nx+1)

def adjust_potential():
    # get field wrappers at level 0 (no AMR)
    phi_wrapper = fields.PhiFPWrapper()
    Ex_wrapper = fields.ExFPWrapper()

    phi_mean = 0.0
    ngv = phi_wrapper.mf.n_grow_vect
    mesh = phi_wrapper.mesh('x')

    # compute mean potential at emission plane
    ind1 = floor(x_emission / Lx * Nx) + ngv[0]
    ind2 = ind1 + 1
    a2 = (x_emission/Lx * Nx - ind1)
    a1 = 1 - a2
    for mfi in phi_wrapper:
        phi_arr = phi_wrapper.mf.array(mfi).to_cupy()
        phi_mean += cp.mean(a1 * phi_arr[ind1, :] + a2 * phi_arr[ind2, :])

    # update potential and electric field
    for mfi in phi_wrapper:
        phi_arr = phi_wrapper.mf.array(mfi).to_cupy()
        Ex_arr = Ex_wrapper.mf.array(mfi).to_cupy()
        phi_arr[1:-1, :] -= x_nodes[:, None, None, None]/x_emission * phi_mean
        Ex_arr +=  phi_mean / x_emission

callbacks.installafterEsolve(adjust_potential)

####################################################################
#                        RUN SIMULATION                            #
####################################################################
sim.step(max_steps)
