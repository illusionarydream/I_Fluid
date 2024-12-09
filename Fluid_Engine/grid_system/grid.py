import taichi as ti


@ ti.data_oriented
class Grid2D:
    def __init__(self, res_x: int = 60, res_y: int = 40):
        self.res_x = res_x
        self.res_y = res_y
        self.dx = 1/res_x
        self.dy = 1/res_y

        # velocity field: staggered grid
        self.velocity_x = ti.field(dtype=ti.f32, shape=(res_x + 1, res_y))
        self.velocity_y = ti.field(dtype=ti.f32, shape=(res_x, res_y + 1))

        # boundary condition
        self.boundary_type = "Dirichlet"

        # pressure field
        self.pressure = ti.field(dtype=ti.f32, shape=(res_x, res_y))

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

    @ ti.kernel
    def SetField(self, field: ti.template()):
        for i, j in field:
            field[i, j] = i * self.dx


# * tools
@ ti.kernel
def Field_to_Pixel(field: ti.template(), pixel: ti.template()):
    for i, j in pixel:
        pixel[i, j] = field[i, j]


if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    grid = Grid2D(800, 600)
    grid.SetField(grid.pressure)

    # Initialize GUI
    gui_shape = (grid.res_x, grid.res_y)
    gui = ti.GUI("2D Grid", res=gui_shape)

    # Create a pixel field to display the pressure field
    pixel_field = ti.field(dtype=ti.f32, shape=gui_shape)

    while gui.running:
        # Update the gradient
        grid.gradient(grid.pressure, 0, grid.velocity_x, grid.velocity_y)

        # Convert the pressure field to pixel values
        Field_to_Pixel(grid.velocity_y, pixel_field)

        # Display the pixel field
        gui.set_image(pixel_field)
        gui.show()
