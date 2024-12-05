import taichi as ti


@ ti.data_oriented
class Grid:
    def __init__(self, resolution: int = 20):

        # basic grid information
        self.resolution = resolution
        self.cell_size = resolution ** 3
        self.mesh_size = (resolution + 1) * resolution ** 2
        self.dd = 1 / resolution

        # cell data
        self.density = ti.field(dtype=ti.f32, shape=(
            resolution, resolution, resolution))
        self.pressure = ti.field(dtype=ti.f32, shape=(
            resolution, resolution, resolution))

        # mesh data
        self.U_velocity = ti.Vector.field(3, dtype=ti.f32, shape=(
            resolution + 1, resolution, resolution))
        self.V_velocity = ti.Vector.field(3, dtype=ti.f32, shape=(
            resolution, resolution + 1, resolution))
        self.W_velocity = ti.Vector.field(3, dtype=ti.f32, shape=(
            resolution, resolution, resolution + 1))

        self.U_gradient = ti.Vector.field(3, dtype=ti.f32, shape=(
            resolution, resolution, resolution))
        self.V_gradient = ti.Vector.field(3, dtype=ti.f32, shape=(
            resolution, resolution, resolution))
        self.W_gradient = ti.Vector.field(3, dtype=ti.f32, shape=(
            resolution, resolution, resolution))

        self.U_laplacian = ti.Vector.field(3, dtype=ti.f32, shape=(
            resolution, resolution, resolution))
        self.V_laplacian = ti.Vector.field(3, dtype=ti.f32, shape=(
            resolution, resolution, resolution))
        self.W_laplacian = ti.Vector.field(3, dtype=ti.f32, shape=(
            resolution, resolution, resolution))

    # * helper functions
    @ ti.func
    def get_ijk(self, idx: int) -> ti.Vector:
        i = idx // (self.resolution * self.resolution)
        j = (idx // self.resolution) % self.resolution
        k = idx % self.resolution
        return ti.Vector([i, j, k])

    @ ti.kernel
    def get_cell_position(self, position: ti.template()):

        for idx in range(self.cell_size):

            ijk = self.get_ijk(idx)

            position[idx] = ijk * self.dd

    @ ti.kernel
    def grid_to_particle(self, position: ti.template(), color: ti.template()):

        for idx in range(self.cell_size):

            ijk = self.get_ijk(idx)

            position[idx] = ijk * self.dd + 0.5 * self.dd
            color[idx] = ijk * self.dd + 0.5 * self.dd

    # * gradient functions
    @ ti.kernel
    def get_property_gradient(self, property: ti.template(), gradient_x: ti.template(), gradient_y: ti.template(), gradient_z: ti.template(), bound_property: ti.template()):
        # ! cell data to mesh gradient
        for idx in range(self.mesh_size):

            i, j, k = self.get_ijk(idx)

            # x gradient
            if i == self.resolution:
                gradient_x[i, j, k] = (
                    bound_property - property[i - 1, j, k]) / self.dd
            elif i == 0:
                gradient_x[i, j, k] = (
                    property[i, j, k] - bound_property) / self.dd
            else:
                gradient_x[i, j, k] = (
                    property[i, j, k] - property[i - 1, j, k]) / self.dd

            # y gradient
            if i == self.resolution:
                gradient_y[k, i, j] = (
                    bound_property - property[k, i - 1, j]) / self.dd
            elif i == 0:
                gradient_y[k, i, j] = (
                    property[k, i, j] - bound_property) / self.dd
            else:
                gradient_y[k, i, j] = (
                    property[k, i, j] - property[k, i - 1, j]) / self.dd

            # z gradient
            if i == self.resolution:
                gradient_z[j, k, i] = (
                    bound_property - property[j, k, i - 1]) / self.dd
            elif i == 0:
                gradient_z[j, k, i] = (
                    property[j, k, i] - bound_property) / self.dd
            else:
                gradient_z[j, k, i] = (
                    property[j, k, i] - property[j, k, i - 1]) / self.dd

    # * laplacian functions
    @ ti.kernel
    def get_property_laplacian(self, property_x: ti.template(), property_y: ti.template(), property_z: ti.template(),
                               laplacian_x: ti.template(), laplacian_y: ti.template(), laplacian_z: ti.template()):
        # ! mesh data to cell laplacian
        for idx in range(self.mesh_size):

            i, j, k = self.get_ijk(idx)

            # ! apply dirichlet boundary condition
            xn00, xp00, x0n0, x0p0, x00n, x00p = 0, 0, 0, 0, 0, 0
            yn00, yp00, y0n0, y0p0, y00n, y00p = 0, 0, 0, 0, 0, 0
            zn00, zp00, z0n0, z0p0, z00n, z00p = 0, 0, 0, 0, 0, 0

            if i > 0:
                xn00 = property_x[i - 1, j, k]
                y0n0 = property_y[k, i - 1, j]
                z00n = property_z[j, k, i - 1]
            if i < self.resolution:
                xp00 = property_x[i + 1, j, k]
                y0p0 = property_y[k, i + 1, j]
                z00p = property_z[j, k, i + 1]
            if j > 0:
                x0n0 = property_x[i, j - 1, k]
                y00n = property_y[k, i, j - 1]
                zn00 = property_z[j - 1, k, i]
            if j < self.resolution:
                x0p0 = property_x[i, j + 1, k]
                y00p = property_y[k, i, j + 1]
                zp00 = property_z[j + 1, k, i]
            if k > 0:
                x00n = property_x[i, j, k - 1]
                yn00 = property_y[k - 1, i, j]
                z0n0 = property_z[j, k - 1, i]
            if k < self.resolution:
                x00p = property_x[i, j, k + 1]
                yp00 = property_y[k + 1, i, j]
                z0p0 = property_z[j, k + 1, i]

            # x laplacian
            laplacian_x[i, j, k] = (xp00 + xn00 + y0p0 + y0n0 + z00p + z00n -
                                    6 * property_x[i, j, k]) / (self.dd ** 2)
            # y laplacian
            laplacian_y[k, i, j] = (x0p0 + x0n0 + yp00 + yn00 + zp00 + zn00 -
                                    6 * property_y[k, i, j]) / (self.dd ** 2)
            # z laplacian
            laplacian_z[j, k, i] = (x00p + x00n + y00p + y00n + z0p0 + z0n0 -
                                    6 * property_z[j, k, i]) / (self.dd ** 2)
    # * update

    def update(self):
        pass


if __name__ == '__main__':

    ti.init(arch=ti.gpu)

    grid = Grid(5)

    position = ti.Vector.field(3, dtype=ti.f32, shape=(5**3))
    grid.get_cell_position(position=position)
