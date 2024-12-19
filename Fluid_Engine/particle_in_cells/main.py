import taichi as ti
from PIC import ParticleSystem as PS
import numpy as np


if __name__ == "__main__":

    # * set device
    ti.init(arch=ti.cpu)

    # * object settings
    # build SPH solver
    max_particles = 1000
    grid_resolution = 20
    # System settings
    dt = 3E-4
    # build particle system
    particle_system = PS(max_particles=max_particles,
                         grid_resolution=grid_resolution)
    particle_system.uniform_initialize(max_corners=ti.Vector([0.3, 1.0, 0.3]),
                                       min_corners=ti.Vector([0, 0.7, 0]))

    # * basic canvas settings
    # set window
    window = ti.ui.Window("Test", (800, 800))

    # set canvas
    canvas = window.get_canvas()
    canvas.set_background_color((0.0, 0.0, 0.0))

    # get camera
    camera = ti.ui.Camera()
    movement_speed = 0.02

    camera.fov = 0.8
    camera.position(2, 0.5, 2)
    camera.lookat(0.5, 0.5, 0)

    # get scene
    scene = ti.ui.Scene()

    # * Loop
    while window.running:
        # * set camera

        scene.set_camera(camera)

        # * set light
        scene.point_light(pos=[2, 2, 2], color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        # * update particle system
        particle_system.update(dt)

        # * render particles
        scene.particles(particle_system.position,
                        per_vertex_color=particle_system.color,
                        radius=0.01)
        canvas.scene(scene)

        # * handle mouse input and keyboard input
        camera.track_user_inputs(
            window, movement_speed=movement_speed, hold_key=ti.ui.LMB)

        # * show window
        window.show()
