import taichi as ti
import neighbor_search as ns
import particle as pa


# Equations of State
@ ti.data_oriented
class EOS:
    def __init__(self, particle_system: pa.ParticleSystem, neighbor_searcher: ns.NeighborSearcher):
        self.particle_system = None  # particle system
        self.neighbor_searcher = None  # neighbor searcher

        # pressure parameter
        self.initial_density = neighbor_searcher.initial_density
        self.eosScale = 2E5
        self.eosExponent = 7.0
        self.negativePressureScale = 0.0

        # viscosity parameter
        self.viscosity = 0.1

    def random_initialize(self, max_particles: int):
        self.particle_system = pa.ParticleSystem(max_particles)
        self.particle_system.random_initialize()

        # update neighbor searcher
        self.neighbor_searcher = ns.NeighborSearcher(self.particle_system)
        self.radius = self.neighbor_searcher.radius

    # * update EOS state and update neighbor searcher
    def update_EOS(self, particle_system: pa.ParticleSystem):
        self.particle_system = particle_system
        self.neighbor_searcher = ns.NeighborSearcher(self.particle_system)
        self.initial_density = self.neighbor_searcher.initial_density
        self.neighbor_searcher.SPH_initialization()

    # * EOS computation: pressure

    @ ti.func
    def computePressure(self, density: ti.template()):
        return self.eosScale * (density / self.initial_density - 1) ** self.eosExponent / self.eosExponent

    @ ti.kernel
    def computePressureFromEos(self, density: ti.template(), position: ti.template(), pressure: ti.template(), max_particles: ti.i32):

        for i in range(max_particles):
            # consider the surface effect
            if density[i] < self.initial_density:
                pressure[i] = self.computePressure(
                    density[i]) * self.negativePressureScale
            else:
                pressure[i] = self.computePressure(density[i])

    @ ti.kernel
    def computePressureFromEos_gradient(self, mass: ti.template(), density: ti.template(), gradient: ti.template(), output: ti.template(), max_particles: ti.i32):
        for i in range(max_particles):
            output[i] -= mass[i] * gradient[i] / \
                density[i]

            # ? debug
            # if i == 0:
            # print("Poutput:", output[i])

    def computePressureFromEos_force(self, forces: ti.template(), max_particles: ti.i32):
        temp_pressure = ti.field(float, self.particle_system.max_particles)
        temp_forces = ti.Vector.field(
            3, float, self.particle_system.max_particles)
        # calculate pressure
        self.computePressureFromEos(self.neighbor_searcher.density,
                                    self.particle_system.position,
                                    temp_pressure,
                                    max_particles)
        # interpolate pressure gradient
        self.neighbor_searcher.SPH_interpolation(self.neighbor_searcher.position,
                                                 temp_pressure,
                                                 temp_forces,
                                                 max_particles,
                                                 types="sync",
                                                 mode="self")
        # calculate pressure force
        self.computePressureFromEos_gradient(self.particle_system.mass,
                                             self.neighbor_searcher.density,
                                             temp_forces,
                                             forces,
                                             max_particles)

    # * EOS computation: viscosity

    @ ti.kernel
    def extract_velocity_scalar(self, velocity: ti.template(), output_x: ti.template(), output_y: ti.template(), output_z: ti.template(), max_particles: ti.i32):
        for i in range(max_particles):
            output_x[i] = velocity[i][0]
            output_y[i] = velocity[i][1]
            output_z[i] = velocity[i][2]

    @ ti.kernel
    def merge_force_scalar(self, mass: ti.template(), force_x: ti.template(), force_y: ti.template(), force_z: ti.template(), output: ti.template(), max_particles: ti.i32):
        for i in range(max_particles):
            output[i] += ti.Vector(
                [force_x[i], force_y[i], force_z[i]]) * self.viscosity * mass[i]

            # ? debug
            # if i == 0:
            # print("Loutput:", output[i])

    def computeViscosityFromEOS_force(self, forces: ti.template(), max_particles: ti.i32):
        temp_vx = ti.field(float, self.particle_system.max_particles)
        temp_vy = ti.field(float, self.particle_system.max_particles)
        temp_vz = ti.field(float, self.particle_system.max_particles)

        forces_x = ti.field(float, self.particle_system.max_particles)
        forces_y = ti.field(float, self.particle_system.max_particles)
        forces_z = ti.field(float, self.particle_system.max_particles)

        # calculate velocity
        self.extract_velocity_scalar(self.particle_system.velocity,
                                     temp_vx,
                                     temp_vy,
                                     temp_vz,
                                     max_particles)

        # interpolate velocity laplacian
        self.neighbor_searcher.SPH_interpolation(self.neighbor_searcher.position,
                                                 temp_vx,
                                                 forces_x,
                                                 max_particles,
                                                 types="laplacian",
                                                 mode="self")
        self.neighbor_searcher.SPH_interpolation(self.neighbor_searcher.position,
                                                 temp_vy,
                                                 forces_y,
                                                 max_particles,
                                                 types="laplacian",
                                                 mode="self")
        self.neighbor_searcher.SPH_interpolation(self.neighbor_searcher.position,
                                                 temp_vz,
                                                 forces_z,
                                                 max_particles,
                                                 types="laplacian",
                                                 mode="self")

        # merge force
        self.merge_force_scalar(self.particle_system.mass,
                                forces_x,
                                forces_y,
                                forces_z,
                                forces,
                                max_particles)

    @ ti.func
    def computeViscosity(self, velocity: ti.template(), neighbor_velocity: ti.template(), neighbor_mass: ti.template(), distance: ti.template()):
        return 0.0

    # * visualization
    def visualization_scalar(self, scalar_name: str, output: ti.template(), moreScalar: ti.template() = None):
        max_particles = self.particle_system.max_particles

        # ! pressure visualization
        if scalar_name == "pressure":
            self.SPH_initialization()
            self.computePressureFromEos(self.neighbor_searcher.density,
                                        self.particle_system.position,
                                        output,
                                        max_particles)

        # ! gradient visualization
        if scalar_name == "pressure gradient":
            self.SPH_initialization()
            temp_pressure = ti.field(float, self.particle_system.max_particles)
            self.computePressureFromEos(self.neighbor_searcher.density,
                                        self.particle_system.position,
                                        temp_pressure,
                                        max_particles)
            self.neighbor_searcher.SPH_interpolation(self.neighbor_searcher.position,
                                                     temp_pressure,
                                                     output,
                                                     max_particles,
                                                     types="sync")
