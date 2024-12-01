import taichi as ti

# * set device
ti.init(arch=ti.gpu)

# * object settings
vec3 = ti.types.vector(3, ti.f32)


@ti.dataclass
class Wave1D:
    waveX: ti.f32
    waveHeight: ti.f32
    waveLength: ti.f32
    waveVelocity: ti.f32

    @ti.func
    def update(self, dt: ti.f32):
        self.waveX += self.waveVelocity * dt

        # solve the boundary condition
        if self.waveX > 1.0:
            self.waveX = 2.0 - self.waveX
            self.waveVelocity *= -1.0
        elif self.waveX < 0.0:
            self.waveX = -self.waveX
            self.waveVelocity *= -1.0

    @ti.func
    def getHeight(self, x: ti.f32) -> ti.f32:
        return self.waveHeight * ti.exp(-(x - self.waveX) ** 2 / self.waveLength)


# * kernel functions
@ti.kernel
def updateWaves(waves: ti.template(), dt: ti.f32):
    for i in range(waves.shape[0]):
        waves[i].update(dt)


@ti.kernel
def drawWave(waves: ti.template(),
             img: ti.template(),
             width: int,
             height: int,
             dx: ti.f32):

    for i, j in img:
        img[i, j] = vec3(0.0, 0.0, 0.0)

    for i in range(width):
        h = 0.0
        for j in ti.static(range(waves.shape[0])):
            h += waves[j].getHeight(i * dx) * height

        h = int(h)
        if 0 <= h < height:
            img[i, int(h)] = vec3(1.0, 1.0, 1.0)


# * main
if __name__ == "__main__":
    # * basic settings
    FPS = 12
    dt = 1.0 / FPS
    width = 800
    height = 800
    dx = 1.0 / width

    # * basic canvas settings
    # set window
    gui = ti.GUI('Wave1D', res=(width, height))

    # set wave
    waves = Wave1D.field(shape=(2,))
    waves[0] = Wave1D(0.0, 0.5, 0.02, 0.1)
    waves[1] = Wave1D(1.0, 0.2, 0.01, -0.05)

    img = ti.Vector.field(3, ti.f32, (width, height))

    # * Loop
    while gui.running:
        # update wave
        updateWaves(waves, dt)

        # draw wave
        drawWave(waves, img, width, height, dx)

        # show image
        gui.set_image(img)

        # update window
        gui.show()
