import taichi as ti
import numpy as np


@ti.data_oriented
class ParticleSystem:
    def __init__(self,
                 max_particles=1000,
                 grid_resolution=20):
        self.max_particles = max_particles
        self.grid_resolution = grid_resolution
        self.dd = 1 / grid_resolution

        # initialize particles
        self.position = ti.Vector.field(
            3, dtype=ti.f32, shape=self.max_particles)
        self.color = ti.Vector.field(
            3, dtype=ti.f32, shape=self.max_particles)
        self.velocity = ti.Vector.field(
            3, dtype=ti.f32, shape=self.max_particles)
        self.weights = ti.field(
            dtype=ti.f32, shape=(self.max_particles, 8))

        # initialize grid
        self.grid_vel = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.grid_resolution + 1, self.grid_resolution + 1, self.grid_resolution + 1))
        self.grid_weight = ti.field(
            dtype=ti.f32, shape=(self.grid_resolution + 1, self.grid_resolution + 1, self.grid_resolution + 1))
        self.delta_grid_vel = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.grid_resolution + 1, self.grid_resolution + 1, self.grid_resolution + 1))

        # pressure variables initialization
        res = (self.grid_resolution + 1) ** 3
        self.pressure_X = ti.field(dtype=ti.f32, shape=(
            self.grid_resolution + 1, self.grid_resolution + 1, self.grid_resolution + 1))
        self.pressure_matrix = ti.linalg.SparseMatrixBuilder(
            num_rows=res, num_cols=res, max_num_triplets=res * 7)
        self.pressure_b = ti.ndarray(shape=res, dtype=ti.f32)

        # pressure solver
        self.solver = ti.linalg.SparseSolver(solver_type="LLT")
        self.build_pressure_matrix(self.pressure_matrix)
        K = self.pressure_matrix.build()
        self.solver.analyze_pattern(K)
        self.solver.factorize(K)

        # physics settings
        self.gravity = ti.Vector([0, -9.8, 0])
        self.beta_viscosity = 1E-3
        self.beta_pressure = 3E-2
        self.beta_collision = 1.5

        # boundary settings
        self.velocity_bound = ti.Vector([0.0, 0.0, 0.0])
        self.pressure_bound = 0.0
        self.damping = 1.0

    # * Grid operation
    # get grid index
    @ ti.func
    def get_grid_index(self, pos: ti.template()):
        index = ti.cast(pos * self.grid_resolution, ti.i32)

        # clamp the index
        index = ti.max(0, ti.min(index, self.grid_resolution - 1))
        return index

    @ ti.func
    def get_grid_velocity(self, i, j, k):
        vel = self.velocity_bound
        if 0 <= i <= self.grid_resolution and 0 <= j <= self.grid_resolution and 0 <= k <= self.grid_resolution:
            vel = self.grid_vel[i, j, k]

        return vel

    @ ti.func
    def get_grid_pressure(self, i, j, k):
        # pressure = self.pressure_bound
        # if 0 <= i <= self.grid_resolution and 0 <= j <= self.grid_resolution and 0 <= k <= self.grid_resolution:
        # pressure = self.pressure_X[i, j, k]
        pressure = 27.0
        if 0 <= i <= self.grid_resolution and 0 <= j <= self.grid_resolution and 0 <= k <= self.grid_resolution:
            if self.grid_weight[i, j, k] > pressure:
                pressure = self.grid_weight[i, j, k]

        return pressure

    @ ti.func
    def set_boundary_velocity(self, i, j, k):
        # one-directional constraints
        # if i == 0 or i == self.grid_resolution:
        #     self.delta_grid_vel[i, j, k][0] = self.velocity_bound[0]
        # if j == 0 or j == self.grid_resolution:
        #     self.delta_grid_vel[i, j, k][1] = self.velocity_bound[1]
        # if k == 0 or k == self.grid_resolution:
        #     self.delta_grid_vel[i, j, k][2] = self.velocity_bound[2]

        # multi-directional constraints
        # if i == 0 or i == self.grid_resolution or j == 0 or j == self.grid_resolution or k == 0 or k == self.grid_resolution:
        #     self.delta_grid_vel[i, j, k] = self.velocity_bound

        pass

        # update gravity
    @ ti.kernel
    def update_gravity(self, dt: ti.f32):
        for i, j, k in self.grid_vel:
            self.delta_grid_vel[i, j, k] += self.gravity * dt

            # set the boundary
            self.set_boundary_velocity(i, j, k)

    # update viscosity
    # ! use explicit euler

    @ ti.kernel
    def update_viscosity(self, dt: ti.f32):
        for i, j, k in self.grid_vel:

            # loop over the neighbors
            for t in ti.static(range(6)):
                neighbor = ti.Vector([i, j, k]) + ti.Vector(
                    [t == 0, t == 1, t == 2]) - ti.Vector([t == 3, t == 4, t == 5])

                # clamp the index
                neighbor_vel = self.get_grid_velocity(
                    neighbor[0], neighbor[1], neighbor[2])
                # update the velocity
                self.delta_grid_vel[i, j, k] += self.beta_viscosity * (
                    neighbor_vel - self.grid_vel[i, j, k]) * dt / self.dd**2

            # set the boundary
            self.set_boundary_velocity(i, j, k)

    # update pressure

    @ti.func
    def get_idx(self, i, j, k):
        return i * (self.grid_resolution + 1) ** 2 + j * (self.grid_resolution + 1) + k

    @ ti.kernel
    def build_pressure_matrix(self, pressure_matrix: ti.types.sparse_matrix_builder()):
        for i, j, k in self.grid_vel:
            idx = self.get_idx(i, j, k)
            pressure_matrix[idx, idx] += 6

            # i+1, j, k
            if i < self.grid_resolution:
                pressure_matrix[idx, self.get_idx(i + 1, j, k)] -= 1
            # i-1, j, k
            if i > 0:
                pressure_matrix[idx, self.get_idx(i - 1, j, k)] -= 1
            # i, j+1, k
            if j < self.grid_resolution:
                pressure_matrix[idx, self.get_idx(i, j + 1, k)] -= 1
            # i, j-1, k
            if j > 0:
                pressure_matrix[idx, self.get_idx(i, j - 1, k)] -= 1
            # i, j, k+1
            if k < self.grid_resolution:
                pressure_matrix[idx, self.get_idx(i, j, k + 1)] -= 1
            # i, j, k-1
            if k > 0:
                pressure_matrix[idx, self.get_idx(i, j, k - 1)] -= 1

    @ ti.kernel
    def build_pressure_b(self, pressure_b: ti.types.ndarray(), dt: ti.f32):
        coeff = 0.5 * self.dd / dt

        for i, j, k in self.grid_vel:
            idx = self.get_idx(i, j, k)
            # delta_vel = self.get_grid_velocity(i - 1, j, k) - self.get_grid_velocity(i + 1, j, k) +\
            #     self.get_grid_velocity(i, j - 1, k) - self.get_grid_velocity(i, j + 1, k) + \
            #     self.get_grid_velocity(i, j, k - 1) - \
            #     self.get_grid_velocity(i, j, k + 1)
            # pressure_b[idx] += coeff * \
            #     (delta_vel[0] + delta_vel[1] + delta_vel[2])

            pressure_b[idx] += (self.get_grid_velocity(i - 1, j, k)[0] - self.get_grid_velocity(i, j, k)[0] +
                                self.get_grid_velocity(i, j - 1, k)[1] - self.get_grid_velocity(i, j, k)[1] +
                                self.get_grid_velocity(i, j, k - 1)[2] - self.get_grid_velocity(i, j, k)[2]) * coeff

            cnt = 0
            if i == 0:
                cnt += 1
            if i == self.grid_resolution:
                cnt += 1
            if j == 0:
                cnt += 1
            if j == self.grid_resolution:
                cnt += 1
            if k == 0:
                cnt += 1
            if k == self.grid_resolution:
                cnt += 1

            pressure_b[idx] += cnt * self.pressure_bound

    @ ti.kernel
    def update_pressure(self, dt: ti.f32):
        coeff = dt / self.dd

        for i, j, k in self.grid_vel:

            pressure_grad = ti.Vector([0.0, 0.0, 0.0])
            pressure_grad[0] = self.get_grid_pressure(
                i - 1, j, k) - self.get_grid_pressure(i + 1, j, k)
            pressure_grad[1] = self.get_grid_pressure(
                i, j - 1, k) - self.get_grid_pressure(i, j + 1, k)
            pressure_grad[2] = self.get_grid_pressure(
                i, j, k - 1) - self.get_grid_pressure(i, j, k + 1)

            self.delta_grid_vel[i, j, k] += pressure_grad * \
                self.beta_pressure * coeff

            # set the boundary
            self.set_boundary_velocity(i, j, k)

    # * P2C and C2P transfer
    # linear interpolation

    @ ti.func
    def linear_interpolation(self, particle_index: ti.i32):
        index = self.get_grid_index(self.position[particle_index])

        # get x, y, z offset
        offset = self.position[particle_index] * self.grid_resolution - index

        # get the weight
        for local_idx in ti.static(range(8)):

            i = local_idx % 2
            j = local_idx // 2 % 2
            k = local_idx // 4

            dx = i * offset[0] + (1 - i) * (1 - offset[0])
            dy = j * offset[1] + (1 - j) * (1 - offset[1])
            dz = k * offset[2] + (1 - k) * (1 - offset[2])

            self.weights[particle_index, local_idx] = dx * dy * dz

    # Particle to cell

    @ ti.kernel
    def P2C(self):
        # clear the grid_weight, delta_grid_vel and grid_vel
        for i, j, k in self.grid_vel:
            self.delta_grid_vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_weight[i, j, k] = 0.0

        # accumulate the velocity
        for i in range(self.max_particles):
            # linear interpolation
            self.linear_interpolation(i)

            # get the grid index
            index = self.get_grid_index(self.position[i])

            # add the velocity to the grid
            for j in ti.static(range(8)):
                local_index = ti.Vector([j % 2, j // 2 % 2, j // 4]) + index

                # clamp the index
                if 0 <= local_index[0] <= self.grid_resolution and 0 <= local_index[1] <= self.grid_resolution and 0 <= local_index[2] <= self.grid_resolution:
                    self.grid_vel[local_index] += self.velocity[i] * \
                        self.weights[i, j]
                    self.grid_weight[local_index] += self.weights[i, j]

        # normalize the velocity
        for i, j, k in self.grid_vel:
            if self.grid_weight[i, j, k] > 0.001:
                self.grid_vel[i, j, k] /= self.grid_weight[i, j, k]

    # Cell to particle
    # ! reuse the result of P2C

    @ ti.kernel
    def C2P(self):
        for i in range(self.max_particles):
            # get the grid index
            index = self.get_grid_index(self.position[i])

            # add the velocity to the grid
            for j in ti.static(range(8)):
                local_index = ti.Vector([j % 2, j // 2 % 2, j // 4]) + index

                # clamp the index
                if 0 <= local_index[0] <= self.grid_resolution and 0 <= local_index[1] <= self.grid_resolution and 0 <= local_index[2] <= self.grid_resolution:
                    self.velocity[i] = self.delta_grid_vel[local_index] * \
                        self.weights[i, j] + self.velocity[i] * self.damping

    # * Particle operation

    @ ti.func
    def SDF_plane(self, point: ti.template(), plane_point: ti.template(), plane_normal: ti.template()):
        return (point - plane_point).dot(plane_normal)

    @ ti.func
    def collision_handling(self, idx: int, plane_point: ti.template(), plane_normal: ti.template()):
        # use impulse
        sdf = self.SDF_plane(self.position[idx], plane_point, plane_normal)
        if sdf < 0:
            self.position[idx] -= sdf * plane_normal
            self.velocity[idx] = self.velocity[idx] - \
                self.velocity[idx].dot(plane_normal) * \
                plane_normal * self.beta_collision

    @ ti.kernel
    def update_position(self, dt: ti.f32):
        for i in range(self.max_particles):
            self.position[i] += self.velocity[i] * dt

            # handle the boundary
            self.collision_handling(i, ti.Vector(
                [0.5, 0.0, 0.5]), ti.Vector([0.0, 1.0, 0.0]))
            self.collision_handling(i, ti.Vector(
                [0.5, 1.0, 0.5]), ti.Vector([0.0, -1.0, 0.0]))
            self.collision_handling(i, ti.Vector(
                [0.0, 0.5, 0.5]), ti.Vector([1.0, 0.0, 0.0]))
            self.collision_handling(i, ti.Vector(
                [0.5, 0.5, 0.0]), ti.Vector([0.0, 0.0, 1.0]))
            self.collision_handling(i, ti.Vector(
                [0.5, 0.5, 1.0]), ti.Vector([0.0, 0.0, -1.0]))
            self.collision_handling(i, ti.Vector(
                [1.0, 0.5, 0.5]), ti.Vector([-1.0, 0.0, 0.0]))

    def update(self, dt):

        # grid based operation: update non-advection forces
        # transfer the velocity to the grid
        self.P2C()
        # update gravity
        self.update_gravity(dt)
        # update viscosity
        self.update_viscosity(dt)
        # update pressure
        self.pressure_b.fill(0)
        self.build_pressure_b(self.pressure_b, dt)
        self.pressure_X.from_numpy(self.solver.solve(self.pressure_b).to_numpy().reshape(
            (self.grid_resolution + 1, self.grid_resolution + 1, self.grid_resolution + 1)))
        self.update_pressure(dt)
        # transfer the velocity back to the particles
        self.C2P()

        # update particle position
        self.update_position(dt)

    # * initialize particles

    @ ti.kernel
    def _uniform_initialize(self,
                            max_corners: ti.template(),
                            min_corners: ti.template(),
                            cube_len: ti.f32):
        dd = (max_corners - min_corners)
        for i in range(self.max_particles):
            self.position[i] = min_corners + \
                ti.Vector([i % cube_len, i // cube_len % cube_len,
                           i // cube_len // cube_len]) * dd / cube_len

            self.color[i] = ti.Vector([0.2, 0.2, 0.8])

    def uniform_initialize(self,
                           max_corners=ti.Vector([1, 1, 1]),
                           min_corners=ti.Vector([0, 0, 0])):
        # judge wether the max_particles is n^3
        assert int(self.max_particles**(1/3) + 0.0001
                   )**3 == self.max_particles, "max_particles should be n^3"

        # initialize particles
        cube_len = self.max_particles**(1/3)
        self._uniform_initialize(max_corners, min_corners, cube_len)
