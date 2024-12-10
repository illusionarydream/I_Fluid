import taichi as ti


@ ti.data_oriented
class Grid2D:
    def __init__(self, res_x: int = 60, res_y: int = 40):
        self.res_x = res_x
        self.res_y = res_y
        self.dx = 100.0/res_x
        self.dy = 100.0/res_y

        # velocity field: staggered grid
        self.velocity_x = ti.field(dtype=ti.f32, shape=(res_x + 1, res_y))
        self.velocity_y = ti.field(dtype=ti.f32, shape=(res_x, res_y + 1))
        self.vel_grad_x = ti.field(dtype=ti.f32, shape=(res_x + 1, res_y))
        self.vel_grad_y = ti.field(dtype=ti.f32, shape=(res_x, res_y + 1))
        self.vel_lap_x = ti.field(dtype=ti.f32, shape=(res_x + 1, res_y))
        self.vel_lap_y = ti.field(dtype=ti.f32, shape=(res_x, res_y + 1))
        self.vel_centroid = ti.Vector.field(
            2, dtype=ti.f32, shape=(res_x, res_y))

        # boundary condition
        self.boundary_type = "Dirichlet"

        # pressure field
        self.pressure = ti.field(dtype=ti.f32, shape=(res_x, res_y))

        # pressure solver
        self.pressure_matrix = ti.linalg.SparseMatrixBuilder(
            res_x * res_y, res_x * res_y, max_num_triplets=7*res_x*res_y)
        self.pressure_X = ti.ndarray(ti.i32, shape=(res_x * res_y))
        self.pressure_b = ti.ndarray(ti.f32, shape=(res_x * res_y))

        # precompute the pressure matrix
        self.solver = ti.linalg.SparseSolver(solver_type="LLT")
        self.build_pressure_matrix(self.pressure_matrix)
        K = self.pressure_matrix.build()
        self.solver.compute(K)

        # physical constants
        self.gravity = 9.8
        self.standard_pressure = 0.0
        self.miu = 0.01  # dynamic viscosity/diffusion coefficient
        self.beta = 10000000  # buoyancy coefficient
        self.base_temperature = 0.0

        # smoke temperature field and density field
        self.cenroid_lap = ti.field(dtype=ti.f32, shape=(res_x, res_y))
        self.cenroid_interpolate = ti.field(dtype=ti.f32, shape=(res_x, res_y))
        self.temperature = ti.field(dtype=ti.f32, shape=(res_x, res_y))
        self.density = ti.field(dtype=ti.f32, shape=(res_x, res_y))

    # * gradient operation
    @ ti.kernel
    def gradient(self, field: ti.template(), field_bound: float, grad_x: ti.template(), grad_y: ti.template()):
        # grid cenroid data -> grid edge gradient data
        for i, j in grad_x:
            if i > 0 and i < self.res_x:
                grad_x[i, j] = (field[i, j] - field[i - 1, j]) / self.dx
            elif i == 0:
                grad_x[i, j] = (field[i, j] - field_bound) / self.dx
            else:
                grad_x[i, j] = (field_bound - field[i - 1, j]) / self.dx

        for i, j in grad_y:
            if j > 0 and j < self.res_y:
                grad_y[i, j] = (field[i, j] - field[i, j - 1]) / self.dy
            elif j == 0:
                grad_y[i, j] = (field[i, j] - field_bound) / self.dy
            else:
                grad_y[i, j] = (field_bound - field[i, j - 1]) / self.dy

    # * laplacian operation
    @ ti.kernel
    def laplacian(self, field_x: ti.template(), field_y: ti.template(), field_bound: float, lap_x: ti.template(), lap_y: ti.template()):
        for i, j in lap_x:
            # i: (0, res_x), j: (0, res_y-1)
            temp_lap = 0.0
            temp_lap += field_x[i + 1, j] if i < self.res_x else field_bound
            temp_lap += field_x[i - 1, j] if i > 0 else field_bound
            temp_lap += field_x[i, j +
                                1] if j < self.res_y - 1 else field_bound
            temp_lap += field_x[i, j - 1] if j > 0 else field_bound
            temp_lap -= 4 * field_x[i, j]

            lap_x[i, j] = temp_lap / (self.dx**2)

        for i, j in lap_y:
            # i: (0, res_x-1), j: (0, res_y)
            temp_lap = 0.0
            temp_lap += field_y[i + 1,
                                j] if i < self.res_x - 1 else field_bound
            temp_lap += field_y[i - 1, j] if i > 0 else field_bound
            temp_lap += field_y[i, j +
                                1] if j < self.res_y else field_bound
            temp_lap += field_y[i, j - 1] if j > 0 else field_bound
            temp_lap -= 4 * field_y[i, j]

            lap_y[i, j] = temp_lap / (self.dy**2)

    @ ti.kernel
    def laplacian_centroid(self, field: ti.template(), field_bound: float, lap: ti.template()):
        for i, j in lap:
            temp_lap_x = 0.0
            temp_lap_y = 0.0

            temp_lap_x += field[i + 1, j] if i < self.res_x else field_bound
            temp_lap_x += field[i - 1, j] if i > 0 else field_bound
            temp_lap_x -= 2 * field[i, j]

            temp_lap_y += field[i, j + 1] if j < self.res_y else field_bound
            temp_lap_y += field[i, j - 1] if j > 0 else field_bound
            temp_lap_y -= 2 * field[i, j]

            lap[i, j] = temp_lap_x / (self.dx**2) + temp_lap_y / (self.dy**2)

    # * interpolate operation

    @ti.func
    def cubic_kernel(self, t: ti.template()):
        result = 0.0
        if t <= 1.0:
            result = 1.0 - 2.0 * t**2 + t**3
        elif t <= 2.0:
            result = 4.0 - 8.0 * t + 5.0 * t**2 - t**3
        return result

    @ ti.func
    def linear_kernel(self, t: ti.template()):
        result = 0.0
        if t <= 1.0:
            result = 1.0 - t
        return result

    @ ti.func
    def interpolate_centroid(self, field: ti.template(), field_bound: float, x: float, y: float):
        # idx computation
        ii = x/self.dx - 0.5
        jj = y/self.dy - 0.5
        i = int(ti.floor(ii))
        j = int(ti.floor(jj))
        fx = ii - i
        fy = jj - j

        v1 = field_bound
        if i >= 0 and j >= 0:
            v1 = field[i, j]

        v2 = field_bound
        if i + 1 < self.res_x and j >= 0:
            v2 = field[i + 1, j]

        v3 = field_bound
        if i >= 0 and j + 1 < self.res_y:
            v3 = field[i, j + 1]

        v4 = field_bound
        if i + 1 < self.res_x and j + 1 < self.res_y:
            v4 = field[i + 1, j + 1]

        # wx0 = self.cubic_kernel(fx)
        # wx1 = self.cubic_kernel(1.0 - fx)
        # wy0 = self.cubic_kernel(fy)
        # wy1 = self.cubic_kernel(1.0 - fy)

        wx0 = self.linear_kernel(fx)
        wx1 = self.linear_kernel(1.0 - fx)
        wy0 = self.linear_kernel(fy)
        wy1 = self.linear_kernel(1.0 - fy)

        w0 = wx0 * wy0
        w1 = wx1 * wy0
        w2 = wx0 * wy1
        w3 = wx1 * wy1

        res = (v1 * w0 + v2 * w1 + v3 * w2 + v4 * w3)/(w0 + w1 + w2 + w3)

        return res

    @ ti.kernel
    def interpolate_centroid_velocity(self, vel_x: ti.template(), vel_y: ti.template(), vel_bound: ti.template(), vel_centroid: ti.template()):
        for i, j in vel_centroid:
            temp_vel = ti.Vector([0.0, 0.0])
            temp_vel += ti.Vector([vel_x[i, j], vel_y[i, j]])
            temp_vel += ti.Vector([vel_x[i + 1, j], vel_y[i, j+1]])

            vel_centroid[i, j] = temp_vel * 0.5

    # * Update velocity field

    # update gravity
    @ ti.kernel
    def add_gravity(self, dt: float):
        for i, j in self.velocity_y:
            self.velocity_y[i, j] -= self.gravity * dt

    # update buoyancy
    @ ti.kernel
    def add_buoyancy(self, dt: float):
        for i, j in self.velocity_y:
            self.velocity_y[i, j] += self.beta * \
                (self.temperature[i, j] - self.base_temperature) * dt

    # update advection
    @ ti.kernel
    def add_advection(self, dt: float, bound_vel: ti.template()):
        # use semi-Lagrangian method
        times = 2
        inv_times = 1 / times

        for i, j in self.velocity_x:
            x = i * self.dx
            y = (j + 0.5) * self.dy

            # boundary condition
            if i == 0 or i == self.res_x:
                self.velocity_x[i, j] = bound_vel.x
                continue

            # backtrace
            temp_vec = self.interpolate_centroid(
                self.vel_centroid, bound_vel, x, y)
            for t in range(times):
                x -= temp_vec.x * dt * inv_times
                y -= temp_vec.y * dt * inv_times
                temp_vec = self.interpolate_centroid(
                    self.vel_centroid, bound_vel, x, y)

            self.velocity_x[i, j] = temp_vec.x

        for i, j in self.velocity_y:
            x = (i + 0.5) * self.dx
            y = j * self.dy

            # boundary condition
            if j == 0 or j == self.res_y:
                self.velocity_y[i, j] = bound_vel.y
                continue

            # backtrace
            temp_vec = self.interpolate_centroid(
                self.vel_centroid, bound_vel, x, y)
            for t in range(times):
                x -= temp_vec.x * dt * inv_times
                y -= temp_vec.y * dt * inv_times
                temp_vec = self.interpolate_centroid(
                    self.vel_centroid, bound_vel, x, y)

            self.velocity_y[i, j] = temp_vec.y

    # update diffusion

    @ ti.kernel
    def add_diffusion(self, dt: float):
        for i, j in self.velocity_x:
            # boundary condition
            if i == 0 or i == self.res_x:
                self.velocity_x[i, j] = 0.0
                continue

            self.velocity_x[i, j] += self.vel_lap_x[i, j] * self.miu * dt

        for i, j in self.velocity_y:
            # boundary condition
            if j == 0 or j == self.res_y:
                self.velocity_y[i, j] = 0.0
                continue

            self.velocity_y[i, j] += self.vel_lap_y[i, j] * self.miu * dt

    # update pressure gradient

    @ ti.kernel
    def add_pressure_gradient(self, dt: float):
        for i, j in self.velocity_x:
            self.velocity_x[i, j] -= self.vel_grad_x[i, j] * dt

        for i, j in self.velocity_y:
            self.velocity_y[i, j] -= self.vel_grad_y[i, j] * dt

    @ ti.func
    def get_idx(self, i, j):
        return i * self.res_y + j

    @ ti.kernel
    def build_pressure_matrix(self, pressure_matrix: ti.types.sparse_matrix_builder()):
        coeff_x = 1 / self.dx
        coeff_y = 1 / self.dy

        for i, j in self.pressure:
            # i, j
            idx = self.get_idx(i, j)
            pressure_matrix[idx, idx] += 2 * coeff_x + 2 * coeff_y
            # i+1, j
            if i + 1 < self.res_x:
                pressure_matrix[idx, self.get_idx(i + 1, j)] -= coeff_x
            # i-1, j
            if i > 0:
                pressure_matrix[idx, self.get_idx(i - 1, j)] -= coeff_x
            # i, j+1
            if j + 1 < self.res_y:
                pressure_matrix[idx, self.get_idx(i, j + 1)] -= coeff_y
            # i, j-1
            if j > 0:
                pressure_matrix[idx, self.get_idx(i, j - 1)] -= coeff_y

    @ ti.kernel
    def build_pressure_b(self, pressure_b: ti.types.ndarray(), bound_pressure: float, dt: float, vel_x: ti.template(), vel_y: ti.template()):
        inv_dt = 1 / dt
        coeff_x = 1 / self.dx
        coeff_y = 1 / self.dy

        for i, j in self.pressure:
            # i, j
            idx = self.get_idx(i, j)

            pressure_b[idx] += (vel_x[i, j] - vel_x[i + 1, j] +
                                vel_y[i, j] - vel_y[i, j + 1]) * inv_dt

            # i+1, j
            if i == self.res_x - 1:
                pressure_b[idx] += bound_pressure * coeff_x
            # i-1, j
            if i == 0:
                pressure_b[idx] += bound_pressure * coeff_x
            # i, j+1
            if j == self.res_y - 1:
                pressure_b[idx] += bound_pressure * coeff_y
            # i, j-1
            if j == 0:
                pressure_b[idx] += bound_pressure * coeff_y

    def update_velocity(self, dt: float):

        # * add gravity
        self.add_gravity(dt)

        # * add buoyancy
        self.add_buoyancy(dt)

        # * add advection
        bound_vel = ti.Vector([0.0, 0.0])
        self.interpolate_centroid_velocity(
            self.velocity_x, self.velocity_y, bound_vel, self.vel_centroid)

        self.add_advection(dt, bound_vel)

        # * add diffusion
        self.laplacian(self.velocity_x, self.velocity_y,
                       bound_vel.x, self.vel_lap_x, self.vel_lap_y)

        self.add_diffusion(dt)

        # * add pressure projection
        # initial the pressure field
        self.pressure_b.fill(0)

        # fill pressure matrix
        self.build_pressure_b(
            self.pressure_b, self.standard_pressure, dt, self.velocity_x, self.velocity_y)

        # solve process
        self.pressure.from_numpy(self.solver.solve(self.pressure_b).to_numpy().reshape(
            (self.res_x, self.res_y)))

        # compute pressure gradient
        self.gradient(self.pressure, self.standard_pressure,
                      self.vel_grad_x, self.vel_grad_y)

        self.add_pressure_gradient(dt)

    # * Update common field
    @ ti.kernel
    def update_adevection(self, field: ti.template(), temp_field: ti.template(), dt: float):
        # Apply semi-Lagrangian advection
        times = 2
        inv_times = 1 / times
        vel_bound = ti.Vector([0.0, 0.0])

        for i, j in field:

            # the src
            if i >= 195 and i < 205 and j > 0 and j < 20:
                temp_field[i, j] = 10.0
                continue

            x = (i + 0.5) * self.dx
            y = (j + 0.5) * self.dy

            temp_vel = self.interpolate_centroid(
                self.vel_centroid, vel_bound, x, y)

            # backtrace
            for t in range(times):
                x -= temp_vel.x * dt * inv_times
                y -= temp_vel.y * dt * inv_times
                temp_vel = self.interpolate_centroid(
                    self.vel_centroid, vel_bound, x, y)

            temp_field[i, j] = self.interpolate_centroid(
                field, 0.0, x, y)

    @ ti.kernel
    def update_diffusion(self, field: ti.template(), lap_field: ti.template(), dt: float):
        for i, j in field:
            # the src
            if i >= 195 and i < 205 and j > 0 and j < 20:
                field[i, j] = 10.0
                continue

            field[i, j] += lap_field[i, j] * self.miu * dt

    def update_temperature(self, dt: float):
        # adevection
        self.update_adevection(self.temperature, self.cenroid_interpolate, dt)
        self.temperature = self.cenroid_interpolate

        # diffusion
        self.laplacian_centroid(self.temperature, 0.0, self.cenroid_lap)
        self.update_diffusion(self.temperature, self.cenroid_lap, dt)

    @ ti.kernel
    def SetField(self, field: ti.template()):
        for i, j in field:
            if i >= 195 and i < 205 and j > 0 and j < 20:
                field[i, j] = 10.0


# * tools
@ ti.kernel
def Field_to_Pixel(field: ti.template(), pixel: ti.template()):
    for i, j in pixel:
        pixel[i, j] = field[i, j] * 0.08


if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    grid = Grid2D(400, 400)
    grid.SetField(grid.temperature)

    # Initialize GUI
    gui_shape = (grid.res_x, grid.res_y)
    gui = ti.GUI("2D Grid", res=gui_shape)

    # Create a pixel field to display the pressure field
    pixel_field = ti.Vector.field(3, dtype=ti.f32, shape=gui_shape)

    while gui.running:
        # for i in range(5):

        # Update the velocity field
        grid.update_velocity(0.00002)
        grid.update_temperature(0.00002)

        # Convert the pressure field to pixel values
        Field_to_Pixel(grid.temperature, pixel_field)

        # Display the pixel field
        gui.set_image(pixel_field)
        gui.show()
