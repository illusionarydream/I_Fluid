import taichi as ti
import particle as pa
import numpy as np


@ti.data_oriented
class NeighborSearcher:
    def __init__(self,
                 particle_system: pa.ParticleSystem,
                 radius_ratio: float = 0.2,
                 grid_resolution: int = 40,
                 hash_ratio: float = 0.01,
                 max_neighbors: int = 200):

        # * particle system attributes
        self.max_particles = particle_system.max_particles
        self.radius = (max_neighbors * radius_ratio /
                       self.max_particles)**(1/3)
        self.position = particle_system.position
        self.mass = particle_system.mass

        # * neighbor searcher attributes
        self.hash_table_size = int(hash_ratio * grid_resolution**3)
        self.hash_table = ti.field(ti.i32)
        self.S_node = ti.root.dense(
            ti.i, self.hash_table_size).dynamic(ti.j, 1024, chunk_size=32)
        self.S_node.place(self.hash_table)

        self.max_neighbors = max_neighbors
        self.grid_resolution = grid_resolution

        # * SPH interpolation attributes
        self.kernel_radius = self.radius
        self.Muller_kernel_factor = 15 / (3.14159 * self.kernel_radius**3)
        self.density = ti.field(ti.f32, shape=self.max_particles)

    # * Grid Hashing
    @ ti.func
    def hash_mapping(self, grid_idx: ti.template()) -> ti.i32:
        hash_value = (grid_idx[0] * 73856093) ^ (grid_idx[1]
                                                 * 19349663) ^ (grid_idx[2] * 83492791)
        return hash_value % self.hash_table_size

    @ ti.func
    def get_grid_idx(self, position: ti.template()) -> ti.template():
        grid_idx = (position / self.radius).cast(ti.i32)
        return grid_idx

    @ ti.kernel
    def build(self):
        for i in range(self.max_particles):
            grid_idx = self.get_grid_idx(self.position[i])
            hash_idx = self.hash_mapping(grid_idx)
            self.hash_table[hash_idx].append(i)

    # * Find Neighbors
    @ ti.func
    def find_neighbors(self, position: ti.template(), neighbors: ti.template(), idx: ti.i32) -> ti.i32:
        grid_idx = self.get_grid_idx(position)
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
                distance = (self.position[particle_idx] - position).norm()

                if distance < self.radius and distance > 1e-5:
                    # remove those overlapping particles
                    already_added = False
                    for k in range(neighbor_num):
                        if neighbors[idx, k] == particle_idx:
                            already_added = True
                            break
                    if not already_added:
                        neighbors[idx, neighbor_num] = particle_idx
                        neighbor_num += 1

        return neighbor_num

    @ ti.kernel
    def find_all_neighbors(self, position: ti.template(), neighbors: ti.template(), neighbors_num: ti.template(), max_particles: ti.i32):
        for i in range(max_particles):
            neighbor_num = self.find_neighbors(position[i], neighbors, i)
            neighbors_num[i] = neighbor_num

    # * SPH Interpolation
    @ ti.func
    def SPH_kernel(self, r: ti.f32) -> ti.f32:
        # use Muller kernel
        h = self.radius
        q = r / h
        return self.Muller_kernel_factor * (1 - q)**3

    @ ti.kernel
    def interpolation_density(self, neighbors: ti.template(), neighbors_num: ti.template()):
        for i in range(self.max_particles):
            density = 0.0
            for j in range(neighbors_num[i]):
                neighbor_idx = neighbors[i, j]
                r = (self.position[i] - self.position[neighbor_idx]).norm()
                if r < self.radius:
                    density += self.mass[neighbor_idx] * self.SPH_kernel(r)

            self.density[i] = density

    @ ti.kernel
    def interpolation_property(self, neighbors: ti.template(), neighbors_num: ti.template(), old_property: ti.template(), position: ti.template(), output: ti.template(), new_particle_num: ti.i32):

        # new position, old property, output property
        for i in range(new_particle_num):
            property_sum = ti.Vector([0.0, 0.0, 0.0])
            for j in range(neighbors_num[i]):
                neighbor_idx = neighbors[i, j]
                r = (position[i] - self.position[neighbor_idx]).norm()
                if r < self.radius:
                    # avoid division by zero
                    density_val = max(self.density[neighbor_idx], 1e-8)
                    property_sum += (old_property[neighbor_idx] *
                                     self.SPH_kernel(r) / density_val)

                    # ? debug
                    # if i == 0:
                    #     print("old_property[", neighbor_idx, "]:",
                    #           old_property[neighbor_idx])
                    #     print("self.SPH_kernel(", r, "):",
                    #           self.SPH_kernel(r))
                    #     print("density_val:", density_val)
                    #     print("property_sum:", property_sum)

            output[i] = property_sum

    def SPH_interpolation(self, position: ti.template(), old_property: ti.template(), output: ti.template(), new_particle_num: int):
        # build hash table
        self.build()

        # find all neighbors for each particle
        neighbors = ti.field(ti.i32, shape=(
            self.max_particles, self.max_neighbors))
        neighbors_num = ti.field(ti.i32, shape=(self.max_particles))
        self.find_all_neighbors(self.position, neighbors,
                                neighbors_num, self.max_particles)

        # SPH interpolation: old density field
        self.interpolation_density(neighbors, neighbors_num)

        # find new particle neighbors
        neighbors = ti.field(ti.i32, shape=(
            new_particle_num, self.max_neighbors))
        neighbors_num = ti.field(ti.i32, shape=(new_particle_num))
        self.find_all_neighbors(position, neighbors,
                                neighbors_num, new_particle_num)

        # ? debug
        # print("neighbors_num:", neighbors_num.to_numpy())

        # SPH interpolation
        self.interpolation_property(
            neighbors, neighbors_num, old_property, position, output, new_particle_num)

    @ ti.kernel
    def print_hash_table(self):
        for i in range(self.hash_table_size):
            dynamic_length = self.hash_table[i].length()
            print("hash_table[", i, "]:", dynamic_length)

    def print_info(self):
        print("position:", self.position.to_numpy())
        self.print_hash_table()


if __name__ == "__main__":
    # * set device
    ti.init(arch=ti.gpu)

    # * object settings
    max_particles = 10000
    particle_system = pa.ParticleSystem(max_particles)
    neighbor_searcher = NeighborSearcher(particle_system)

    # * initialize
    particle_system.random_initialize()

    # * find neighbors
    new_particle_system = pa.ParticleSystem(100)
    new_particle_system.random_initialize()
    neighbor_searcher.SPH_interpolation(new_particle_system.position,
                                        particle_system.color,
                                        new_particle_system.color,
                                        new_particle_system.max_particles)

    # * print info
    print("new_particle_system.color:", new_particle_system.color.to_numpy())
