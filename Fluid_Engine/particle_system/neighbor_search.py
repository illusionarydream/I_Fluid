import taichi as ti
import particle as pa
import numpy as np


@ti.data_oriented
class NeighborSearcher:
    def __init__(self,
                 radius: float = 0.05,
                 grid_resolution: int = 20,
                 hash_ratio: float = 0.05,
                 max_neighbors: int = 100,
                 bin_size: int = 4096,
                 initial_density: float = 5000):

        # * particle system attributes
        self.radius = radius
        self.ball_radius = 1E-4 * self.radius

        # * neighbor searcher attributes
        self.grid_resolution = grid_resolution
        self.hash_table_size = int(hash_ratio * self.grid_resolution**3)

        self.hash_table = ti.field(int)
        self.bin_size = bin_size
        self.S_node = ti.root.dense(
            ti.i, self.hash_table_size).dynamic(ti.j, bin_size, chunk_size=256)
        self.S_node.place(self.hash_table)
        self.max_neighbors = max_neighbors

        # * SPH interpolation attributes
        self.kernel_radius = self.radius
        self.initial_density = initial_density
        self.Muller_kernel_factor = 15 / (3.14159 * self.kernel_radius**3)
        self.Poly6_kernel_factor = 315 / (64 * 3.14159 * self.kernel_radius**9)

    # * Find Neighbors

    @ ti.func
    def hash_mapping(self, grid_idx: ti.template()) -> int:
        hash_value = (grid_idx[0] * 73856093) ^ (grid_idx[1]
                                                 * 19349663) ^ (grid_idx[2] * 83492791)
        return hash_value % self.hash_table_size

    @ ti.func
    def get_grid_idx(self, position: ti.template()) -> ti.template():
        grid_idx = (position * self.grid_resolution).cast(int)
        return grid_idx

    @ ti.kernel
    def build_hash_table(self, position: ti.template(), max_particles: int):
        for i in range(max_particles):
            grid_idx = self.get_grid_idx(position[i])
            if (grid_idx < 0).any() or (grid_idx >= self.grid_resolution).any():
                continue
            hash_idx = self.hash_mapping(grid_idx)
            self.hash_table[hash_idx].append(i)
            if self.hash_table[hash_idx].length() >= self.bin_size:
                print("hash_table is full!")

    @ ti.kernel
    def clear_hash_table(self):
        for i in range(self.hash_table_size):
            ti.deactivate(self.S_node, [i])

    @ ti.kernel
    def find_all_neighbors_Kernel(self, position: ti.template(), neighbors: ti.template(), neighbors_num: ti.template(), max_particles: int):
        for idx in range(max_particles):

            grid_idx = self.get_grid_idx(position[idx])
            hash_idx = self.hash_mapping(grid_idx)
            neighbor_num = 0

            # for each grid cell
            for i in range(27):

                # get neighbor grid cell
                offset = ti.Vector([i // 9 - 1, (i // 3) % 3 - 1, i % 3 - 1])
                neighbor_grid_idx = grid_idx + offset
                if (neighbor_grid_idx < 0).any() or (neighbor_grid_idx >= self.grid_resolution).any():
                    continue

                # get neighbor particles
                neighbor_hash_idx = self.hash_mapping(neighbor_grid_idx)

                for j in range(self.hash_table[neighbor_hash_idx].length()):

                    # if the neighbor number is full
                    if neighbor_num >= self.max_neighbors:
                        break

                    particle_idx = self.hash_table[neighbor_hash_idx, j]

                    # check the distance
                    distance = (position[particle_idx] - position[idx]).norm()

                    if distance < self.radius and distance > self.ball_radius:
                        # remove those overlapping particles
                        already_added = False
                        for k in range(neighbor_num):
                            if neighbors[idx, k] == particle_idx:
                                already_added = True
                                break
                        if not already_added:
                            neighbors[idx, neighbor_num] = particle_idx
                            neighbor_num += 1

            neighbors_num[idx] = neighbor_num

    def find_all_neighbors(self, position: ti.template(), neighbors: ti.template(), neighbors_num: ti.template(), max_particles: int):
        # build hash table first
        self.clear_hash_table()
        self.build_hash_table(position, max_particles)

        # find all neighbors
        self.find_all_neighbors_Kernel(
            position, neighbors, neighbors_num, max_particles)

    # # * SPH Interpolation

    @ ti.func
    def SPH_kernel(self, r: float) -> float:
        # use Muller kernel
        h = self.radius
        q = r / h
        return self.Muller_kernel_factor * (1 - q)**3

    @ ti.func
    def SPH_kernel_gradient(self, r: float) -> float:
        # use Muller kernel
        h = self.radius
        q = r / h
        return -3 * self.Muller_kernel_factor * (1 - q)**2 / h

    @ ti.func
    def SPH_kernel_laplacian(self, r: float) -> float:
        # use Muller
        h = self.radius
        q = r / h
        return 6 * self.Muller_kernel_factor * (1 - q) / h**2

    # # * Basic Numerical
    @ ti.kernel
    def interpolation_density(self, position: ti.template(), mass: ti.template(),  neighbors: ti.template(), neighbors_num: ti.template(), density: ti.template(), max_particles: int):
        for i in range(max_particles):
            density_sum = mass[i] * self.SPH_kernel(0.0)
            for j in range(neighbors_num[i]):
                neighbor_idx = neighbors[i, j]
                r = (position[i] - position[neighbor_idx]).norm()
                if r < self.radius and r > self.ball_radius:
                    density_sum += mass[neighbor_idx] * self.SPH_kernel(r)

            density[i] = density_sum

    # * Gradient
    @ ti.kernel
    def interpolation_property_sync(self, position: ti.template(), mass: ti.template(), neighbors: ti.template(), neighbors_num: ti.template(),  density: ti.template(), property: ti.template(), gradient: ti.template(), max_particles: int):
        for i in range(max_particles):

            property_sum = ti.Vector([0.0, 0.0, 0.0])

            for j in range(neighbors_num[i]):
                neighbor_idx = neighbors[i, j]

                R = position[neighbor_idx] - position[i]
                r = R.norm()
                R /= r

                if r < self.radius:
                    old_val = property[neighbor_idx] / \
                        (density[neighbor_idx] ** 2)
                    new_val = property[i] / \
                        (density[i] ** 2)
                    property_sum += ((old_val + new_val) * mass[neighbor_idx] * self.SPH_kernel_gradient(
                        r) * (-R) * density[i])

            gradient[i] = property_sum

    # * Laplacian
    @ ti.kernel
    def interpolation_property_laplacian(self, position: ti.template(), mass: ti.template(), neighbors: ti.template(), neighbors_num: ti.template(),  density: ti.template(), property: ti.template(), laplacian: ti.template(), max_particles: int):
        for i in range(max_particles):

            property_sum = ti.Vector([0.0, 0.0, 0.0])

            for j in range(neighbors_num[i]):
                neighbor_idx = neighbors[i, j]

                R = position[i] - position[neighbor_idx]
                r = R.norm()

                if r < self.radius:
                    # avoid division by zero
                    property_sum += ((property[neighbor_idx] - property[i]) * mass[neighbor_idx] *
                                     self.SPH_kernel_laplacian(r) / density[neighbor_idx])

            laplacian[i] = property_sum


if __name__ == "__main__":
    # * set device
    ti.init(arch=ti.gpu)

    # * object settings
    max_particles = 64000
    particle_system = pa.ParticleSystem(max_particles)
    particle_system.uniform_initialize(40)

    # * build neighbor searcher
    neighbor_searcher = NeighborSearcher()

    # * neighbor search
    neighbors = ti.field(int, shape=(
        max_particles, neighbor_searcher.max_neighbors))
    neighbors_num = ti.field(int, shape=(max_particles))

    neighbor_searcher.find_all_neighbors(
        particle_system.position, neighbors, neighbors_num, max_particles)

    print("neighbors_num:", neighbors_num.to_numpy())

    neighbor_searcher.interpolation_density(
        particle_system.position, particle_system.mass, neighbors, neighbors_num, particle_system.density, max_particles)

    # print("density:", particle_system.density.to_numpy())

    # neighbor_searcher.interpolation_property_laplacian(
    # particle_system.position, particle_system.mass, neighbors, neighbors_num, particle_system.density, particle_system.velocity, particle_system.color, max_particles)

    # print("color:", particle_system.density.to_numpy())
