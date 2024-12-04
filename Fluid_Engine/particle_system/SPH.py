import taichi as ti
import neighbor_search as ns
import particle as pa


# Equations of State
@ ti.data_oriented
class SPH_Solver:
    def __init__(self, max_particles: int):
        # * particle system
        self.particle_system = pa.ParticleSystem(max_particles)

        # * forces
        self.accelerations = ti.Vector.field(3, float, max_particles)
        # gravity
        self.gravity = [0.0, -9.8, 0.0]

        # pressure parameter
        self.pressure = ti.field(float, max_particles)
        self.gradient = ti.Vector.field(3, float, max_particles)
        self.initial_density = 80000
        self.eosScale = 2E4
        self.eosExponent = 6.0
        self.negativePressureScale = 0.0

        # viscosity parameter
        self.laplacian = ti.Vector.field(3, float, max_particles)
        self.viscosity = 0.005

        # neighbor searcher
        self.max_neighbors = 100
        self.neighbor_radius = 0.04
        self.neighbors = ti.field(int, (max_particles, self.max_neighbors))
        self.neighbors_num = ti.field(int, max_particles)
        self.neighbor_searcher = ns.NeighborSearcher(radius=self.neighbor_radius,
                                                     max_neighbors=self.max_neighbors,
                                                     initial_density=self.initial_density,
                                                     hash_ratio=0.05)

    # * initialization
    def random_initialize(self):
        self.particle_system.random_initialize()

    def uniform_initialize(self, cube_len_num: int):
        self.particle_system.uniform_initialize(cube_len_num)

    # * forces computation: pressure forces

    @ ti.kernel
    def compute_pressure(self, density: ti.template(), pressure: ti.template(), max_particles: int):
        for i in range(max_particles):
            pressure[i] = self.eosScale * \
                (density[i] / self.initial_density -
                 1) ** self.eosExponent / self.eosExponent

            if density[i] < self.initial_density:
                pressure[i] *= self.negativePressureScale

    def compute_pressure_force(self):
        # compute pressure
        self.compute_pressure(
            self.particle_system.density, self.pressure, self.particle_system.max_particles)

        # compute gradient
        self.neighbor_searcher.interpolation_property_sync(
            self.particle_system.position, self.particle_system.mass, self.neighbors, self.neighbors_num, self.particle_system.density, self.pressure, self.gradient, self.particle_system.max_particles)

    # * forces computation: non-pressure forces

    def compute_viscosity_force(self):
        # compute laplacian
        self.neighbor_searcher.interpolation_property_laplacian(
            self.particle_system.position, self.particle_system.mass, self.neighbors, self.neighbors_num, self.particle_system.density, self.particle_system.velocity, self.laplacian, self.particle_system.max_particles)

    # * forces computation: All accelerations
    @ ti.kernel
    def compute_all_forces(self):
        for i in range(self.particle_system.max_particles):
            self.accelerations[i] = self.gravity
            self.accelerations[i] -= self.gradient[i] / \
                self.particle_system.density[i]
            self.accelerations[i] += self.viscosity * self.laplacian[i]

    def compute_accelerations(self):
        # get neighbors
        self.neighbor_searcher.find_all_neighbors(
            self.particle_system.position, self.neighbors, self.neighbors_num, self.particle_system.max_particles)

        # compute density
        self.neighbor_searcher.interpolation_density(
            self.particle_system.position, self.particle_system.mass, self.neighbors, self.neighbors_num, self.particle_system.density, self.particle_system.max_particles)

        # compute pressure force
        self.compute_pressure_force()

        # compute viscosity force
        self.compute_viscosity_force()

        # compute all forces
        self.compute_all_forces()

        # ? debug
        # print("accelerations:", self.accelerations.to_numpy())

    def update(self, dt: float):
        self.particle_system.update(self.accelerations, dt)


if __name__ == "__main__":
    # * set device
    ti.init(arch=ti.gpu)

    # * object settings
    max_particles = 1000
    sph_solver = SPH_Solver(max_particles)
    sph_solver.uniform_initialize(10)

    # * compute force
    for t in range(1000):
        sph_solver.compute_accelerations()
