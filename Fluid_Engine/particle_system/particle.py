import taichi as ti


vec3 = ti.types.vector(3, float)


@ti.data_oriented
class ParticleSystem:
    def __init__(self, max_particles: int):
        self.max_particles = max_particles
        self.position = ti.Vector.field(3, float, max_particles)
        self.velocity = ti.Vector.field(3, float, max_particles)
        self.color = ti.Vector.field(3, float, max_particles)
        self.density = ti.field(float, max_particles)
        self.mass = ti.field(float, max_particles)

        # physics
        self.gravity = [0.0, -9.8, 0.0]
        self.damping = 0.99999

    # * initialization
    @ti.kernel
    def random_initialize(self):
        for i in range(self.max_particles):
            self.position[i] = [ti.random() * 0.5, ti.random() *
                                0.5 + 0.5, ti.random() * 0.5]
            self.velocity[i] = [0.0, 0.0, 0.0]
            self.color[i] = [0.3, 0.3, 0.8]
            self.mass[i] = 1.0

    @ti.kernel
    def uniform_initialize(self, cube_len_num: int):
        for i in range(self.max_particles):
            x = i % cube_len_num
            y = (i // cube_len_num) % cube_len_num
            z = i // cube_len_num // cube_len_num

            self.position[i] = [x / cube_len_num * 0.5,
                                y / cube_len_num * 0.5 + 0.5,
                                z / cube_len_num * 0.5]
            self.velocity[i] = [0.0, 0.0, 0.0]
            self.color[i] = [0.3, 0.3, 0.8]
            self.mass[i] = 1.0

    def print_info(self):
        print("position:", self.position.to_numpy())
        print("velocity:", self.velocity.to_numpy())
        print("color:", self.color.to_numpy())
        print("mass:", self.mass.to_numpy())

    # * collision handling
    @ ti.func
    def SDF_plane(self, point: vec3, plane_point: vec3, plane_normal: vec3):
        return (point - plane_point).dot(plane_normal)

    @ ti.func
    def collision_handling(self, idx: int, plane_point: vec3, plane_normal: vec3):
        # use impulse
        sdf = self.SDF_plane(self.position[idx], plane_point, plane_normal)
        if sdf < 0:
            self.position[idx] -= sdf * plane_normal
            self.velocity[idx] = self.velocity[idx] - \
                self.velocity[idx].dot(plane_normal) * plane_normal * 2.0

    # * update
    @ ti.kernel
    def update(self, accelerations: ti.template(), dt: float):
        for i in range(self.max_particles):
            self.velocity[i] = accelerations[i] * \
                dt + self.velocity[i] * self.damping
            if self.velocity[i].norm() > 10.0:
                self.velocity[i] = self.velocity[i] / \
                    self.velocity[i].norm() * 10.0

            self.position[i] += self.velocity[i] * dt

            # ? debug
            # print("velocity:", self.velocity[i])

            # collision handling
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


if __name__ == "__main__":
    # * set device
    ti.init(arch=ti.gpu)

    # * object settings
    max_particles = 10
    particle_system = ParticleSystem(max_particles)

    # * initialize
    particle_system.random_initialize()
    particle_system.print_info()
