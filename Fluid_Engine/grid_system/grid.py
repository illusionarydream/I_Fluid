import taichi as ti


@ ti.data_oriented
class Grid2D:
    def __init__(self, res_x: int = 60, res_y: int = 40):
        self.res_x = res_x
        self.res_y = res_y
        self.dx = 1.0/res_x
        self.dy = 1.0/res_y

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

        # pressure solver
        self.pressure_matrix = ti.linalg.SparseMatrixBuilder(
            res_x * res_y, res_x * res_y, max_num_triplets=7*res_x*res_y)
        self.pressure_X = ti.ndarray(ti.i32, shape=(res_x * res_y))
        self.pressure_b = ti.ndarray(ti.f32, shape=(res_x * res_y))
        self.solver = ti.linalg.SparseSolver(solver_type="LLT")
        # pressure field
        self.pressure = ti.field(dtype=ti.f32, shape=(res_x, res_y))

        # physical constants
        self.gravity = 9.8
        self.miu = 0.001  # dynamic viscosity/diffusion coefficient

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

    # * interpolate operation
    @ ti.func
    def interpolate_centroid(self, field: ti.template(), field_bound: float, x: float, y: float):
        i = int(x/self.dx - 0.5)
        j = int(y/self.dy - 0.5)
        fx = (x/self.dx - 0.5) - i
        fy = (y/self.dy - 0.5) - j

        v1 = field[i, j] if i >= 0 and j >= 0 else field_bound
        v2 = field[i + 1, j] if i + 1 < self.res_x and j >= 0 else field_bound
        v3 = field[i, j + 1] if i >= 0 and j + 1 < self.res_y else field_bound
        v4 = field[i + 1, j + 1] if i + 1 < self.res_x and j + \
            1 < self.res_y else field_bound

        res = (1 - fx) * (1 - fy) * v1 + fx * (1 - fy) * \
            v2 + (1 - fx) * fy * v3 + fx * fy * v4

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
            self.velocity_y[i, j] += self.gravity * dt

    # update advection
    @ ti.kernel
    def add_advection(self, dt: float, bound_vel: ti.template()):
        # use semi-Lagrangian method

        for i, j in self.velocity_x:
            x = i * self.dx
            y = j * self.dy
            x -= self.velocity_x[i, j] * dt

            self.velocity_x[i, j] = self.interpolate_centroid(
                self.velocity_x, bound_vel.x, x, y)

        for i, j in self.velocity_y:
            x = i * self.dx
            y = j * self.dy
            y -= self.velocity_y[i, j] * dt

            self.velocity_y[i, j] = self.interpolate_centroid(
                self.velocity_y, bound_vel.y, x, y)

    # update diffusion
    @ ti.kernel
    def add_diffusion(self, dt: float):
        for i, j in self.velocity_x:
            self.velocity_x[i, j] += self.vel_lap_x[i, j] * self.miu * dt

        for i, j in self.velocity_y:
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
    def set_pressure_matrix(self, pressure_matrix: ti.types.sparse_matrix_builder(), pressure_b: ti.types.ndarray(),  bound_pressure: float, dt: float, vel_x: ti.template(), vel_y: ti.template()):
        coeff_x = dt / self.dx
        coeff_y = dt / self.dy

        for i, j in self.pressure:
            # i, j
            idx = self.get_idx(i, j)
            pressure_matrix[idx, idx] += 2 * coeff_x + 2 * coeff_y

            pressure_b[idx] += (vel_x[i + 1, j] - vel_x[i, j]) + \
                (vel_y[i, j + 1] - vel_y[i, j])

            # i+1, j
            if i + 1 < self.res_x:
                pressure_matrix[idx, self.get_idx(i + 1, j)] -= coeff_x
            else:
                pressure_b[idx] += coeff_x * bound_pressure

            # i-1, j
            if i > 0:
                pressure_matrix[idx, self.get_idx(i - 1, j)] -= coeff_x
            else:
                pressure_b[idx] += coeff_x * bound_pressure

            # i, j+1
            if j + 1 < self.res_y:
                pressure_matrix[idx, self.get_idx(i, j + 1)] -= coeff_y
            else:
                pressure_b[idx] += coeff_y * bound_pressure

            # i, j-1
            if j > 0:
                pressure_matrix[idx, self.get_idx(i, j - 1)] -= coeff_y
            else:
                pressure_b[idx] += coeff_y * bound_pressure

    def update_velocity(self, dt: float):

        # * add gravity
        # self.add_gravity(dt)

        # * add advection
        # bound_vel = ti.Vector([0.0, 0.0])
        # self.interpolate_centroid_velocity(
        # self.velocity_x, self.velocity_y, bound_vel, self.vel_centroid)

        # self.add_advection(dt, bound_vel)

        # * add diffusion
        # self.laplacian(self.velocity_x, self.velocity_y,
        #    bound_vel.x, self.vel_lap_x, self.vel_lap_y)

        # print(self.vel_lap_x[0, 0], self.vel_lap_y[0, 0])

        # self.add_diffusion(dt)

        # * add pressure projection
        # initial the pressure field
        self.pressure_b.fill(0)

        # fill pressure matrix
        self.set_pressure_matrix(
            self.pressure_matrix, self.pressure_b, 10.0, dt, self.velocity_x, self.velocity_y)

        # solve process
        K = self.pressure_matrix.build()
        # self.solver.analyze_pattern(K)
        # self.solver.factorize(K)
        self.solver.compute(K)
        self.pressure.from_numpy(self.solver.solve(self.pressure_b).to_numpy().reshape(
            (self.res_x, self.res_y)))

        self.gradient(self.pressure, 0.0, self.vel_grad_x, self.vel_grad_y)
        self.add_pressure_gradient(dt)

    @ ti.kernel
    def SetField(self, field: ti.template()):
        for i, j in field:
            field[i, j] = j * self.dx


# * tools
@ ti.kernel
def Field_to_Pixel(field: ti.template(), pixel: ti.template()):
    for i, j in pixel:
        pixel[i, j] = [field[i, j].x, field[i, j].y, 0.0]


if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    grid = Grid2D(400, 300)

    # Initialize GUI
    gui_shape = (grid.res_x, grid.res_y)
    gui = ti.GUI("2D Grid", res=gui_shape)

    # Create a pixel field to display the pressure field
    pixel_field = ti.Vector.field(3, dtype=ti.f32, shape=gui_shape)

    while gui.running:

        # Update the velocity field
        grid.update_velocity(0.0001)

        # Convert the pressure field to pixel values
        Field_to_Pixel(grid.vel_centroid, pixel_field)

        # Display the pixel field
        gui.set_image(pixel_field)
        gui.show()
