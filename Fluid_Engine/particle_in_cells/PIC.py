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
        self.weights = ti.Vector.field(
            8, dtype=ti.f32, shape=self.max_particles)

        # initialize grid
        self.grid_vel = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.grid_resolution, self.grid_resolution, self.grid_resolution))
        self.delta_grid_vel = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.grid_resolution, self.grid_resolution, self.grid_resolution))

        # physics settings
        self.gravity = ti.Vector([0, -0.8, 0])

    # * Grid operation
    # get grid index
    @ti.func
    def get_grid_index(self, pos: ti.template()):
        index = ti.cast(pos * self.grid_resolution, ti.i32)

        # clamp the index
        index = ti.max(0, ti.min(index, self.grid_resolution - 1))
        return index

    # update gravity

    @ti.kernel
    def update_gravity(self, dt: ti.f32):
        for i, j, k in self.grid_vel:
            self.delta_grid_vel[i, j, k] = self.gravity * dt

    # * P2C and C2P transfer
    # linear interpolation
    @ti.func
    def linear_interpolation(self,
                             pos: ti.template(),
                             weights: ti.template()):
        index = self.get_grid_index(pos)

        # get x, y, z offset
        offset = pos * self.grid_resolution - index

        # get the weight
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                for k in ti.static(range(2)):
                    global_idx = i + j * 2 + k * 4
                    weights[global_idx] = (1 - i) * (1 - j) * (1 - k) * offset[0] + \
                        i * (1 - j) * (1 - k) * (1 - offset[0]) + \
                        (1 - i) * j * (1 - k) * offset[1] + \
                        i * j * (1 - k) * (1 - offset[1]) + \
                        (1 - i) * (1 - j) * k * offset[2] + \
                        i * (1 - j) * k * (1 - offset[2]) + \
                        (1 - i) * j * k * offset[0] + \
                        i * j * k * (1 - offset[0])

    # Particle to cell
    @ti.kernel
    def P2C(self):
        for i in range(self.max_particles):
            # linear interpolation
            self.linear_interpolation(self.position[i], self.weights[i])

            # get the grid index
            index = self.get_grid_index(self.position[i])

            # add the velocity to the grid
            for j in ti.static(range(8)):
                self.grid_vel[index + ti.Vector([j % 2, j // 2 % 2, j // 4])
                              ] += self.velocity[i] * self.weights[i][j]

    # Cell to particle
    # ! reuse the result of P2C

    @ti.kernel
    def C2P(self):
        for i in range(self.max_particles):
            # get the grid index
            index = self.get_grid_index(self.position[i])

            # add the velocity to the grid
            for j in ti.static(range(8)):
                self.velocity[i] += self.delta_grid_vel[index +
                                                        ti.Vector([j % 2, j // 2 % 2, j // 4])] * self.weights[i][j]

    # * Particle operation

    @ti.kernel
    def update_position(self, dt: ti.f32):
        for i in range(self.max_particles):
            self.position[i] += self.velocity[i] * dt

    def update(self, dt):

        # grid based operation: update non-advection forces
        self.P2C()
        self.update_gravity(dt)
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
