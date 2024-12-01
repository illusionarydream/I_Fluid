import taichi as ti

ti.init(arch=ti.gpu)

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

    # set camera
    camera.fov = 0.8
    camera.position(0, 0, 3)
    camera.lookat(0, 0, 0)
    scene.set_camera(camera)

    # set scene
    scene.point_light(pos=[2, 2, 2], color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    # set particles
    ball_center = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
    ball_center[0] = [0, 0, 0]
    scene.particles(centers=ball_center, radius=0.5)
    canvas.scene(scene)

    window.show()
