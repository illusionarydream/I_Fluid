import taichi as ti
import neighbor_search as ns
import particle as pa


# * set device
ti.init(arch=ti.gpu)


# * object settings
vec3 = ti.types.vector(3, float)

# particle system
max_particles = 100000
particle_system = pa.ParticleSystem(max_particles)
particle_system.random_initialize()

# build neighbor searcher
neighbor_searcher = ns.NeighborSearcher(particle_system)

# new particle system
new_particle_num = 100
new_particle_system = pa.ParticleSystem(new_particle_num)
new_particle_system.random_initialize()

# SPH interpolation


@ ti.kernel
def old_property_init(position: ti.template(), old_property: ti.template()):
    for i in range(particle_system.max_particles):
        old_property[i] = 0.5


@ ti.kernel
def new_property_init(color: ti.template(), new_property: ti.template()):
    for i in range(new_particle_system.max_particles):
        val = new_property[i]
        color[i] = [val, val, val]


old_property = ti.field(float, particle_system.max_particles)
old_property_init(particle_system.position, old_property)
new_property = ti.field(float, new_particle_system.max_particles)
neighbor_searcher.SPH_interpolation(new_particle_system.position,
                                    old_property,
                                    new_property,
                                    new_particle_system.max_particles,
                                    types="laplacian")
new_property_init(new_particle_system.color, new_property)

# print("new_particle_system.mass:", new_particle_system.mass.to_numpy())
# print("new_particle_system.color:", new_particle_system.color.to_numpy())
# print("new_particle_system.position:", new_particle_system.position.to_numpy())
# print("color - position:", new_particle_system.color.to_numpy() -
#   new_particle_system.position.to_numpy())

# * basic canvas settings
# set window
window = ti.ui.Window("Test", (800, 800))

# set canvas
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))

# get camera
camera = ti.ui.Camera()

# get scene
scene = ti.ui.Scene()


# * Loop
while window.running:

    # * set camera
    camera.fov = 0.8
    camera.position(0.5, 0.5, 3)
    camera.lookat(0.5, 0.5, 0)
    scene.set_camera(camera)

    # * set light
    scene.point_light(pos=[2, 2, 2], color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    # * set scene
    # new particle system
    scene.particles(new_particle_system.position,
                    per_vertex_color=new_particle_system.color,
                    radius=0.01)
    canvas.scene(scene)

    # * show window
    window.show()
