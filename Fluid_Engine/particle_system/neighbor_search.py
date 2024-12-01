import taichi as ti
import particle as pa


@ti.data_oriented
class NeighborSearcher:
    def __init__(self,
                 particle_system: pa.ParticleSystem,
                 radius: float = 0.1,
                 grid_resolution: int = 20,
                 hash_table_size: int = 200,
                 max_neighbors: int = 100):

        self.max_particles = particle_system.max_particles
        self.radius = radius
        self.position = particle_system.position

        self.hash_table_size = hash_table_size
        self.hash_table = ti.field(ti.i32)
        self.S_node = ti.root.dense(
            ti.i, self.hash_table_size).dynamic(ti.j, 1024, chunk_size=32)
        self.S_node.place(self.hash_table)

        self.max_neighbors = max_neighbors
        self.grid_resolution = grid_resolution

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

    @ ti.kernel
    def find_neighbors(self, position: ti.template(), neighbors: ti.template()) -> ti.i32:
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
                particle_idx = self.hash_table[neighbor_hash_idx, j]
                if (self.position[particle_idx] - position).norm() < self.radius:
                    # remove those overlapping particles
                    already_added = False
                    for k in range(neighbor_num):
                        if neighbors[k] == particle_idx:
                            already_added = True
                            break
                    if not already_added:
                        neighbors[neighbor_num] = particle_idx
                        neighbor_num += 1

        return neighbor_num

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
    max_particles = 1000
    particle_system = pa.ParticleSystem(max_particles)
    neighbor_searcher = NeighborSearcher(particle_system)

    # * initialize
    particle_system.random_initialize()
    neighbor_searcher.build()
    neighbor_searcher.print_info()

    # * find neighbors
    neighbors = ti.field(ti.i32, 100)
    neighbor_num = neighbor_searcher.find_neighbors(
        particle_system.position[100],
        neighbors)

    print("neighbor_num:", neighbor_num)
    for i in range(neighbor_num):
        particle_idx = neighbors[i]
        print("particle_idx:", particle_idx)
        print("particle_idx:", particle_system.position[particle_idx])
