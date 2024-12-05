import taichi as ti
from grid import Grid

# * set device
ti.init(arch=ti.gpu)


# * object settings
res = 5
grid = Grid(res)
position = ti.Vector.field(3, dtype=ti.f32, shape=(res**3))
color = ti.Vector.field(3, dtype=ti.f32, shape=(res**3))
grid.grid_to_particle(position=position, color=color)


# * basic canvas settings
# set window
window = ti.ui.Window("Test", (800, 800))

# set canvas
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))

# get camera
camera = ti.ui.Camera()
camera.fov = 0.8
camera.position(0, 0, 3)
camera.lookat(0, 0, 0)

# get scene
scene = ti.ui.Scene()

# set light
scene.point_light(pos=[2, 2, 2], color=(1, 1, 1))
scene.ambient_light((0.5, 0.5, 0.5))

# * set motions
space_pressed = False
movement_speed = 0.002

# * Loop
while window.running:

    # set camera
    scene.set_camera(camera)

    # set particles
    scene.particles(position, per_vertex_color=color, radius=0.02)
    canvas.scene(scene)

    # * handle mouse input and keyboard input
    camera.track_user_inputs(
        window, movement_speed=movement_speed, hold_key=ti.ui.LMB)

    # break
    if window.is_pressed(ti.ui.ESCAPE):
        break

    # start
    if window.is_pressed(ti.ui.SPACE) and not space_pressed:
        space_pressed = True

    # show window
    window.show()
