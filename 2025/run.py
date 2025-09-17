"""
Script for running Hall thruster simulations in WarpX.
"""

from ctypes import ArgumentError
from hallx import BenchmarkSim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--case", type=str, choices=["1x", "2x", "4x", "8x"], default="1x")
parser.add_argument("--name", type=str, default="")
parser.add_argument("--propellant", type=str, choices=["Xe", "Kr", "Ar"])
parser.add_argument("--neutrals", type=bool, default=False)
parser.add_argument("--te", type=float, default=10.0)
parser.add_argument("--max-time", type=float, default=20e-6)
parser.add_argument("--density", type=float, default=5e17)
parser.add_argument("--voltage", type=float, default=200.0)
parser.add_argument("--bmax", type=float, default=100.0)
parser.add_argument("--neutral-density", type=float, default=1e19)
parser.add_argument("--particle-scale", type=float, default=1.0)
parser.add_argument("--neutral-temp", type=float, default=1000.0)
parser.add_argument(
    "--collisions",
    help="Collision types",
    type=lambda s: [str(item) for item in s.split(",")],
)
parser.add_argument("--collision-interval", type=int, default=1009)
parser.add_argument("--resample-interval", type=int, default=997)
parser.add_argument("--resample-max-particles", type=int, default=-1)
parser.add_argument("--Lx", type=float, default=2.5, help="Axial domain length in mm")
parser.add_argument(
    "--Lz", type=float, default=1.25, help="Azimuthal domain length in mm"
)
parser.add_argument("--anode-recombination", type=bool, default=False)


def main(args):
    dt_base = 5e-12

    print(f"{args=}")

    if args.case == "1x":
        grid_scale = 1
        effective_potential_solver = False
        effective_potential_factor = 4
    elif args.case == "2x":
        grid_scale = 2
        effective_potential_solver = True
        effective_potential_factor = 8
    elif args.case == "4x":
        grid_scale = 4
        effective_potential_solver = True
        effective_potential_factor = 16
    else:
        grid_scale = 8
        effective_potential_solver = True
        effective_potential_factor = 8

    dt = dt_base * grid_scale

    if args.collisions is not None:
        collisions = set(args.collisions)
    else:
        collisions = set()

    if (
        "iz" in collisions or "en" in collisions or "ex" in collisions
    ) and not args.neutrals:
        raise ArgumentError(
            f"Cannot use excitation, ionization, or e-n collisions without neutrals! Got these collisions: {collisions}."
        )

    print(f"{collisions=}")

    sim = BenchmarkSim(
        grid_scale=grid_scale,
        effective_potential_solver=effective_potential_solver,
        effective_potential_factor=effective_potential_factor,
        propellant=args.propellant,
        dt=dt,
        output_dir=f"output{'_' if args.name else ''}{args.name}",
        include_neutrals=args.neutrals,
        initial_Te=args.te,
        max_time=args.max_time,
        initial_density=args.density,
        neutral_density=args.neutral_density,
        particle_scale=args.particle_scale,
        discharge_voltage=args.voltage,
        B_max=args.bmax / 10_000,
        neutral_temp_K=args.neutral_temp,
        collisions=collisions,
        dsmc_interval=args.collision_interval,
        resample_interval=args.resample_interval,
        resample_max_particles=args.resample_max_particles,
        L_x=args.Lx,
        L_z=args.Lz,
        anode_recombination=args.anode_recombination,
    )
    sim.run()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
