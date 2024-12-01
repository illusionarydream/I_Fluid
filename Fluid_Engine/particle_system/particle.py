import taichi as ti


vec3 = ti.types.vector(3, float)


@ti.data_oriented
class ParticleSystem:
    def __init__(self, max_particles: int):
        self.max_particles = max_particles
        self.position = ti.Vector.field(3, float, max_particles)
        self.velocity = ti.Vector.field(3, float, max_particles)
        self.color = ti.Vector.field(3, float, max_particles)
        self.mass = ti.field(float, max_particles)

    @ti.kernel
    def random_initialize(self):
        for i in range(self.max_particles):
            self.position[i] = [ti.random(), ti.random(), ti.random()]
            self.velocity[i] = [0.0, 0.0, 0.0]
            self.color[i] = [0.5, 0.5, 0.5]
            self.mass[i] = 1.0

    def print_info(self):
        print("position:", self.position.to_numpy())
        print("velocity:", self.velocity.to_numpy())
        print("color:", self.color.to_numpy())
        print("mass:", self.mass.to_numpy())


if __name__ == "__main__":
    # * set device
    ti.init(arch=ti.gpu)

    # * object settings
    max_particles = 10
    particle_system = ParticleSystem(max_particles)

    # * initialize
    particle_system.random_initialize()
    particle_system.print_info()
