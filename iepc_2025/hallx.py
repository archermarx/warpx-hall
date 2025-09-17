"""
Library for setting up benchmark-like axial-azimuthal Hall thruster simulations in WarpX.
"""

from pywarpx import picmi, fields, callbacks, particle_containers as pc
from periodictable import elements
import numpy as np
import cupy as cp
from math import sqrt, floor, ceil
from pathlib import Path

# Physical constants
m_p = picmi.constants.m_p
m_e = picmi.constants.m_e
q_e = picmi.constants.q_e
ep0 = picmi.constants.ep0
k_B = 1.380649e-23

seed = 123_456

IONIZATION_ENERGIES = dict(
    Xe=12.1298431,
    Ar=15.7596112,
)


class BenchmarkSim:
    def __init__(
        self,
        L_x=2.5e-2,
        L_z=1.28e-2,
        B_max=1e-2,
        B_0=0.6,
        B_min=0.1,
        B_loc=0.3,
        B_width=0.25,
        propellant="Xe",
        grid_scale=1,
        particle_scale=1,
        discharge_voltage=200.0,
        max_time=20e-6,
        dt=5e-12,
        effective_potential_solver=False,
        effective_potential_factor=4,
        initial_density=5e16,
        initial_Te=10.0,
        initial_Ti=0.5,
        cathode_Te=10.0,
        particle_sort_interval=500,
        verbosity=1,
        diag_interval=5000,
        output_dir="output",
        include_neutrals=False,
        neutral_temp_K=1000.0,
        neutral_density=1e18,
        dsmc_interval=10,
        collisions: set | None = None,
        resample_max_particles: int = -1,
        resample_interval: int = 1009,
        anode_recombination: bool = False,
    ):
        self.include_neutrals = include_neutrals
        self.anode_recombination = anode_recombination
        self.dt = dt

        if collisions is None:
            self.collision_types = set()
        else:
            self.collision_types = collisions

        # Grid and geometry
        N_x_base = 512
        N_z_base = 256
        L_x_base = 25e-3
        L_z_base = 12.8e-3

        self.L_x = L_x
        self.L_z = L_z
        self.N_x = floor(N_x_base * L_x / (grid_scale * L_x_base))
        self.N_z = floor(N_z_base * L_z / (grid_scale * L_z_base))

        self.mlmg_precision = 1e-3
        self.dx = self.L_x / self.N_x
        self.dz = self.L_z / self.N_z

        particles_per_cell_base = 75
        self.particles_per_cell_init = particles_per_cell_base * particle_scale

        # Magnetic field
        self.B_max = B_max
        self.B_0 = B_0 * B_max
        self.B_min = B_min * B_max
        self.B_loc = B_loc * self.L_x
        self.B_width = B_width * self.L_x

        self.discharge_voltage = discharge_voltage
        self.propellant = propellant
        self.m_i = elements.symbol(self.propellant).mass
        print(f"{propellant=}, {self.m_i=}")
        self.m_i = self.m_i * m_p

        # "Ionization" source region
        x1_iz = 0.1 * self.L_x  # Axial location of beginning of source region
        x2_iz = 0.4 * self.L_x  # Axial location of end of source region
        self.dx_iz = x2_iz - x1_iz  # Axial extent of source region
        self.xm_iz = 0.5 * (x1_iz + x2_iz)  # Center of source region

        ionization_rate = 5.23e23
        self.x_cathode = 24 / 25 * self.L_x

        self.particles_per_cell_neutral = 5 * self.particles_per_cell_init
        self.weight = self.compute_weight(initial_density, self.particles_per_cell_init)
        self.neutral_weight = self.compute_weight(
            neutral_density, self.particles_per_cell_neutral
        )

        self.inject_per_step = (
            2 * dt * self.L_z * ionization_rate * self.dx_iz / (self.weight * np.pi)
        )

        grid = picmi.Cartesian2DGrid(
            number_of_cells=[self.N_x, self.N_z],
            lower_bound=[0.0, 0.0],
            upper_bound=[self.L_x, self.L_z],
            lower_boundary_conditions=["dirichlet", "periodic"],
            upper_boundary_conditions=["dirichlet", "periodic"],
            lower_boundary_conditions_particles=["absorbing", "periodic"],
            upper_boundary_conditions_particles=["absorbing", "periodic"],
            warpx_potential_lo_x=self.discharge_voltage,
            warpx_potential_hi_x=0.0,
        )

        field = self.applied_field()

        self.vn_std = sqrt(k_B * neutral_temp_K / self.m_i)
        # Neutrals enter axially at their speed of sound, assuming a choked reservoir
        self.vn_directed = sqrt(5 / 3 * k_B * neutral_temp_K / self.m_i)
        self.ve_std = sqrt(q_e * initial_Te / m_e)
        self.ve_cath = sqrt(q_e * cathode_Te / m_e)
        self.vi_std = sqrt(q_e * initial_Ti / self.m_i)

        solver = picmi.ElectrostaticSolver(
            grid=grid,
            required_precision=self.mlmg_precision,
            warpx_self_fields_verbosity=1,
            warpx_effective_potential=effective_potential_solver,
            warpx_effective_potential_factor=effective_potential_factor,
        )

        particle_layout = picmi.PseudoRandomLayout(
            n_macroparticles_per_cell=self.particles_per_cell_init, grid=grid
        )

        neutral_layout = picmi.PseudoRandomLayout(
            n_macroparticles_per_cell=self.particles_per_cell_neutral, grid=grid
        )

        neutral_flux = neutral_density * self.vn_directed

        nppc_neutral_flux = (neutral_flux * self.dt * self.L_z) / (
            self.N_z * self.neutral_weight
        )

        neutral_flux_layout = picmi.PseudoRandomLayout(
            n_macroparticles_per_cell=nppc_neutral_flux, grid=grid
        )

        print(f"{nppc_neutral_flux=}, {self.neutral_weight=}, {self.vn_directed=}")

        electron_distribution = picmi.UniformDistribution(
            initial_density,
            rms_velocity=[self.ve_std] * 3,
            lower_bound=[0.0] * 3,
            upper_bound=[self.L_x, self.L_z, self.L_z],
        )

        ion_distribution = picmi.UniformDistribution(
            initial_density,
            rms_velocity=[self.vi_std] * 3,
            lower_bound=[0.0] * 3,
            upper_bound=[self.L_x, self.L_z, self.L_z],
        )

        neutral_init_distribution = picmi.UniformDistribution(
            neutral_density,
            rms_velocity=[self.vn_std] * 3,
            lower_bound=[0.0] * 3,
            upper_bound=[self.L_x, self.L_z, self.L_z],
            directed_velocity=[self.vn_directed, 0.0, 0.0],
        )

        neutral_flux_distribution = picmi.UniformFluxDistribution(
            flux=neutral_flux,
            gaussian_flux_momentum_distribution=True,
            flux_direction=1,
            surface_flux_position=0,
            flux_normal_axis="x",
            rms_velocity=[self.vn_std, self.vn_std, self.vn_std],
            directed_velocity=[self.vn_directed, 0.0, 0.0],
        )

        if resample_max_particles > 0:
            resampling_options = dict(
                warpx_do_resampling=True,
                warpx_resampling_algorithm="leveling_thinning",
                warpx_resampling_trigger_intervals=resample_interval,
                warpx_resampling_trigger_max_avg_ppc=resample_max_particles,
            )
        else:
            resampling_options = {}

        self.electrons = picmi.Species(
            particle_type="electron",
            name="electrons",
            initial_distribution=electron_distribution,
            warpx_save_particles_at_xlo=1,
            **resampling_options,
        )

        self.ions = picmi.Species(
            particle_type=self.propellant,
            name="ions",
            mass=self.m_i,
            charge="q_e",
            initial_distribution=ion_distribution,
            warpx_save_particles_at_xlo=1,
            **resampling_options,
        )

        self.neutrals = picmi.Species(
            particle_type=self.propellant,
            name="neutrals",
            mass=self.m_i,
            charge=0,
            initial_distribution=[neutral_init_distribution, neutral_flux_distribution],
            warpx_save_particles_at_xlo=1,
            **resampling_options,
        )

        CROSS_SECTION_DIR = Path(
            f"/home/marksta/src/warpx-data/MCC_cross_sections/{self.propellant}"
        )

        collision_list = []

        if "iz" in self.collision_types:
            iz_collisions = picmi.DSMCCollisions(
                name="dsmc_iz_collisions",
                species=[self.electrons, self.neutrals],
                product_species=[self.electrons, self.ions],
                ndt=dsmc_interval,
                scattering_processes={
                    "ionization": {
                        "cross_section": CROSS_SECTION_DIR / "ionization.dat",
                        "energy": IONIZATION_ENERGIES[self.propellant],
                        "target_species": self.neutrals,
                    },
                },
            )
            collision_list.append(iz_collisions)

        if "en" in self.collision_types:
            en_collisions = picmi.DSMCCollisions(
                name="dsmc_en_collisions",
                species=[self.electrons, self.neutrals],
                ndt=dsmc_interval,
                scattering_processes={
                    "elastic": {
                        "cross_section": CROSS_SECTION_DIR / "electron_scattering.dat"
                    }
                },
            )
            collision_list.append(en_collisions)

        if "ei" in self.collision_types:
            ei_collisions = picmi.CoulombCollisions(
                name="ei_coulomb",
                species=[self.electrons, self.ions],
                ndt=dsmc_interval,
            )
            collision_list.append(ei_collisions)

        if "ee" in self.collision_types:
            ee_collisions = picmi.CoulombCollisions(
                name="ee_columb",
                species=[self.electrons, self.electrons],
                ndt=dsmc_interval,
            )
            collision_list.append(ee_collisions)

        if "ii" in self.collision_types:
            ii_collisions = picmi.CoulombCollisions(
                name="ii_coulomb",
                species=[self.ions, self.ions],
                ndt=dsmc_interval,
            )
            collision_list.append(ii_collisions)

        # initialize simulation
        self.sim = picmi.Simulation(
            solver=solver,
            time_step_size=dt,
            max_time=max_time,
            verbose=verbosity,
            warpx_random_seed=seed,
            warpx_sort_intervals=particle_sort_interval,
            warpx_numprocs=[1, 1],
            warpx_use_filter=True,
            warpx_amrex_use_gpu_aware_mpi=True,
            particle_shape="linear",
            warpx_field_gathering_algo="energy-conserving",
            warpx_collisions=collision_list,
        )

        self.sim.add_species(self.ions, layout=particle_layout)
        self.sim.add_species(self.electrons, layout=particle_layout)

        if include_neutrals:
            self.sim.add_species(
                self.neutrals,
                layout=[neutral_layout, neutral_flux_layout],
            )

        self.sim.add_applied_field(field)
        solver.sim = self.sim

        self.electron_wrapper = pc.ParticleContainerWrapper("electrons")
        self.ion_wrapper = pc.ParticleContainerWrapper("ions")

        if self.include_neutrals:
            self.neutral_wrapper = pc.ParticleContainerWrapper("neutrals")
        else:
            self.neutral_wrapper = None

        self.particle_buffer = pc.ParticleBoundaryBufferWrapper()

        self.max_steps = ceil(max_time / dt)
        self.diag_interval = diag_interval
        self.output_dir = Path(output_dir)
        self.field_diag_dir = self.output_dir / "fields"
        self.particle_diag_dir = self.output_dir / "particles"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.reduced_dir = self.output_dir / "reduced"

        # Simulation checkpointing
        checkpoint = picmi.Checkpoint(
            period=int(self.max_steps // 10),
            warpx_file_prefix="checkpoint",
            write_dir=self.checkpoint_dir,
            warpx_file_min_digits=10,
        )
        self.sim.add_diagnostic(checkpoint)

        # Particle saving
        particle_diag = picmi.ParticleDiagnostic(
            name="particles_",
            period=self.diag_interval * 10,
            data_list=["position", "momentum", "weighting"],
            write_dir=self.particle_diag_dir,
            warpx_format="plotfile",
        )
        self.sim.add_diagnostic(particle_diag)

        diag_species = ["electrons", "ions"]
        if self.include_neutrals:
            diag_species.append("neutrals")

        # Grid diagnostics
        # In addition to normal grid quantities we also save gridded moments of the electron VDF for later analysis
        field_diag = picmi.FieldDiagnostic(
            name="fields_",
            grid=grid,
            period=diag_interval,
            data_list=["rho", "E", "B", "J", "phi"]
            + ["rho_" + s for s in diag_species],
            write_dir=self.field_diag_dir,
            warpx_format="plotfile",
            warpx_particle_fields_species=diag_species,
            warpx_particle_fields_to_plot=[
                # ===================================================================
                # NUMBER DENSITY
                # units: m^{-3}
                # ====================================================================
                picmi.ParticleFieldDiagnostic(
                    name="n", func=f"{1 / (self.dx * self.dz)}", do_average=0
                ),
                # ===================================================================
                # MEAN VELOCITY
                #
                # \vec{u}_e = \int \vec{v} f(\vec{v}) d^3 v
                # units: v/c
                # ===================================================================
                picmi.ParticleFieldDiagnostic(name="ux", func="ux"),
                picmi.ParticleFieldDiagnostic(name="uy", func="uy"),
                picmi.ParticleFieldDiagnostic(name="uz", func="uz"),
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
                # ===================================================================
                picmi.ParticleFieldDiagnostic(name="P_xx", func="ux * ux"),
                picmi.ParticleFieldDiagnostic(name="P_xy", func="ux * uy"),
                picmi.ParticleFieldDiagnostic(name="P_xz", func="ux * uz"),
                picmi.ParticleFieldDiagnostic(name="P_yy", func="uy * uy"),
                picmi.ParticleFieldDiagnostic(name="P_yz", func="uy * uz"),
                picmi.ParticleFieldDiagnostic(name="P_zz", func="uz * uz"),
                # ===================================================================
                # ENERGY FLUX VECTOR (units v^3 / c^3)
                #
                # Q_i = \int v_i^2 v_j f(\vec{v}) d^3v
                #
                # Heat flux vector can be retrieved as
                #
                # q_i = Q_i - \sum_j p_{ij} u_j - 3/2 T u_i - 1/2 m |u^2| u_i
                #
                # units: v^3/c^3
                # ===================================================================
                picmi.ParticleFieldDiagnostic(
                    name="Q_x", func="ux * (ux*ux + uy*uy + uz*uz)"
                ),
                picmi.ParticleFieldDiagnostic(
                    name="Q_y", func="uy * (ux*ux + uy*uy + uz*uz)"
                ),
                picmi.ParticleFieldDiagnostic(
                    name="Q_z", func="uz * (ux*ux + uy*uy + uz*uz)"
                ),
            ],
        )

        self.sim.add_diagnostic(field_diag)

        # Selected reduced diagnostics
        for diag_type in ["ParticleNumber", "ParticleEnergy", "FieldEnergy"]:
            diag = picmi.ReducedDiagnostic(
                name=diag_type,
                diag_type=diag_type,
                period=diag_interval,
                path=str(self.reduced_dir) + "/",
            )

            self.sim.add_diagnostic(diag)

    def run(self):
        self.sim.initialize_inputs()
        self.sim.initialize_warpx()

        if "iz" not in self.collision_types:
            # Disable fixed particle source term if we're using self-consistent ionization
            callbacks.installparticleinjection(self.callback_particle_source)

        callbacks.installparticleinjection(self.callback_boundary_particle_injection)
        callbacks.installafterEsolve(self.callback_cathode_potential)

        print("Running")
        self.sim.step(self.max_steps)

    def compute_weight(self, density, ppc):
        return density * self.L_x * self.L_z / (ppc * self.N_x * self.N_z)

    def applied_field(self):
        # Coefficients for prescribed magnetic field
        exp_factor_1 = np.exp(-0.5 * (self.B_loc / self.B_width) ** 2)
        exp_factor_2 = np.exp(-0.5 * ((self.L_x - self.B_loc) / self.B_width) ** 2)
        a1 = (self.B_max - self.B_0) / (1 - exp_factor_1)
        a2 = (self.B_max - self.B_min) / (1 - exp_factor_2)
        b1 = (self.B_0 - self.B_max * exp_factor_1) / (1 - exp_factor_1)
        b2 = (self.B_min - self.B_max * exp_factor_2) / (1 - exp_factor_2)

        B_func_str = f"if(x<{self.B_loc}, {a1}*exp(-0.5*( (x-{self.B_loc})/{self.B_width} )**2) + {b1}, {a2}*exp(-0.5*( (x-{self.B_loc})/{self.B_width} )**2) + {b2})"

        return picmi.AnalyticAppliedField(
            Bx_expression="0.0", By_expression=B_func_str, Bz_expression="0.0"
        )

    def callback_particle_source(self):
        # Use RNG to decide whether to inject fractional part of N_inject
        particles_to_inject = floor(self.inject_per_step)
        remainder = self.inject_per_step - particles_to_inject
        if np.random.rand() < remainder:
            particles_to_inject += 1

        # Sample random numbers for position
        # Electrons and ions are injected in identical locations
        r1 = np.random.uniform(0, 1, size=particles_to_inject)
        r2 = np.random.uniform(0, 1, size=particles_to_inject)

        # Axial, radial, and azimuthal positions
        x = self.xm_iz + np.arcsin(2 * r1 - 1) * (self.dx_iz / np.pi)
        y = np.zeros(particles_to_inject)
        z = r2 * self.L_z

        # Particle weights (equal)
        w = np.full(particles_to_inject, self.weight)

        # Sample velocities from appropriate maxwellians
        vel_e = self.ve_std * np.random.randn(particles_to_inject, 3)
        vel_i = self.vi_std * np.random.randn(particles_to_inject, 3)

        # Add particles to simulation
        self.electron_wrapper.add_particles(
            x=x, y=y, z=z, ux=vel_e[:, 0], uy=vel_e[:, 1], uz=vel_e[:, 2], w=w
        )

        self.ion_wrapper.add_particles(
            x=x, y=y, z=z, ux=vel_i[:, 0], uy=vel_i[:, 1], uz=vel_i[:, 2], w=w
        )

    def callback_boundary_particle_injection(self):
        # Read anode boundary buffer ('x_lo') to see how many particles of each type were scraped
        Ne_anode = self.particle_buffer.get_particle_boundary_buffer_size(
            "electrons", "x_lo"
        )
        if Ne_anode > 0:
            w_e_anode = self.particle_buffer.get_particle_boundary_buffer(
                "electrons", "x_lo", "w", 0
            )
            total_weight_e = sum(cp.sum(x) for x in w_e_anode).get()
        else:
            total_weight_e = 0.0

        Ni_anode = self.particle_buffer.get_particle_boundary_buffer_size(
            "ions", "x_lo"
        )
        if Ni_anode > 0:
            w_i_anode = self.particle_buffer.get_particle_boundary_buffer(
                "ions", "x_lo", "w", 0
            )
            total_weight_i = sum(cp.sum(x) for x in w_i_anode).get()
        else:
            total_weight_i = 0.0

        # Clear buffer for next step
        self.particle_buffer.clear_buffer()

        # Number of particles to inject at cathode
        w_cath = total_weight_e - total_weight_i

        # Inject new electrons to establish current continuity
        if w_cath > 0:
            N_cath = ceil(w_cath / self.weight)
            weight_per_particle = w_cath / N_cath

            # Particles injected at emission plane with uniform azimuthal distribution
            x = np.full(N_cath, self.x_cathode)
            y = np.zeros(N_cath)
            z = np.random.uniform(0, self.L_z, N_cath)

            # Cathode electrons injected with 3D vdf
            v = self.ve_cath * np.random.randn(N_cath, 3)
            w = np.full(N_cath, weight_per_particle)
            self.electron_wrapper.add_particles(
                x=x, y=y, z=z, ux=v[:, 0], uy=v[:, 1], uz=v[:, 2], w=w
            )

        # Neutral recombination at the anode
        if self.anode_recombination and total_weight_i > 0:
            assert self.include_neutrals and self.neutral_wrapper is not None

            num_anode_neutrals = ceil(total_weight_i / self.neutral_weight)
            weight_per_neutral = total_weight_i / num_anode_neutrals

            # Inject neutrals at anode with uniform azimuthal distribution
            x = np.full(num_anode_neutrals, self.dx / 10)
            y = np.zeros(num_anode_neutrals)
            z = np.random.uniform(0, self.L_z, num_anode_neutrals)

            # One-sided maxwellian VDF for recombined neutrals
            v = self.vn_std * np.random.randn(num_anode_neutrals, 3)
            w = np.full(num_anode_neutrals, weight_per_neutral)
            self.neutral_wrapper.add_particles(
                x=x, y=y, z=z, ux=np.abs(v[:, 0]), uy=v[:, 1], uz=v[:, 2], w=w
            )

        return

    def callback_cathode_potential(self):
        # Get the field wrappers at level 0 (no AMR)
        phi_wrapper = fields.PhiFPWrapper()
        Ex_wrapper = fields.ExFPWrapper()

        # Number of ghost cells
        ngv = phi_wrapper.mf.n_grow_vect

        # Interpolation weights and indices for the emission plane
        cathode_ind_fractional = self.x_cathode / self.L_x * self.N_x + ngv[0]
        cathode_ind = floor(cathode_ind_fractional)
        a2 = cathode_ind_fractional - cathode_ind
        a1 = 1 - a2

        # Compute the mean potential at the cathode plane
        phi_cathode = 0.0
        for mfi in phi_wrapper:
            phi_arr = phi_wrapper.mf.array(mfi).to_cupy()
            phi_cathode += cp.mean(
                a1 * phi_arr[cathode_ind, :] + a2 * phi_arr[cathode_ind + 1, :]
            )

        # Update the potential and electric field
        x_nodes = self.dx * cp.arange(self.N_x + 1)
        for mfi in phi_wrapper:
            # Potential: phi = U - U_cathode * (x / x_c)
            phi_arr = phi_wrapper.mf.array(mfi).to_cupy()
            phi_arr[1:-1, :] -= (
                x_nodes[:, None, None, None] / self.x_cathode * phi_cathode
            )

            # Electric field: Ex = dx(U) + U_cathode / x_c
            Ex_arr = Ex_wrapper.mf.array(mfi).to_cupy()
            Ex_arr += phi_cathode / self.x_cathode
