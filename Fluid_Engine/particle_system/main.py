import taichi as ti
import neighbor_search as ns
import particle as pa


# * set device
ti.init(arch=ti.gpu)


# * object settings
vec3 = ti.types.vector(3, float)

# particle system
max_particles = 10000
particle_system = pa.ParticleSystem(max_particles)
particle_system.random_initialize()

# build neighbor searcher
origin_idx = 0
neighbor_searcher = ns.NeighborSearcher(particle_system)
neighbor_searcher.build()
neighbors = ti.field(ti.i32, 100)
neighbor_num = neighbor_searcher.find_neighbors(
    particle_system.position[origin_idx],
    neighbors)

# set different color for each particle
for i in range(neighbor_num):
    particle_idx = neighbors[i]
    particle_system.color[particle_idx] = [1.0, 0.0, 0.0]
particle_system.color[origin_idx] = [0.0, 0.0, 1.0]

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
    scene.particles(particle_system.position,
                    radius=0.01,
                    per_vertex_color=particle_system.color)
    canvas.scene(scene)

    # * show window
    window.show()
