import taichi as ti


# * set device
ti.init(arch=ti.gpu)


# * object settings
vec3 = ti.types.vector(3, float)


@ti.dataclass
class Ball:
    center: vec3
    radius: float
    color: vec3


balls_field = ti.Struct.field({
    "center": vec3,
    "radius": float,
    "color": vec3
}, shape=(1,))


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

    # set camera
    camera.fov = 0.8
    camera.position(0, 0, 3)
    camera.lookat(0, 0, 0)
    scene.set_camera(camera)

    # set scene
    scene.point_light(pos=[2, 2, 2], color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    # set particles
    ball = Ball(center=[0, 0, 0], radius=0.5, color=(1, 0, 0))
    balls_field[0] = ball
    scene.particles(balls_field.center, ball.radius, ball.color)
    canvas.scene(scene)

    window.show()
