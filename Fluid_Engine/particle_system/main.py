import taichi as ti
import neighbor_search as ns
from tqdm import tqdm
import imageio
import numpy as np
import particle as pa
import SPH
import os

if __name__ == "__main__":

    # * set device
    ti.init(arch=ti.gpu, device_memory_fraction=0.8)

    # * object settings
    # build SPH solver
    max_particles = 8000
    cube_len = 20
    SPH_solver = SPH.SPH_Solver(max_particles)
    SPH_solver.uniform_initialize(cube_len)

    # System settings
    dt = 3E-4

    # * basic canvas settings
    # set window
    window = ti.ui.Window("Test", (800, 800))

    # set canvas
    canvas = window.get_canvas()
    canvas.set_background_color((0.0, 0.0, 0.0))

    # get camera
    camera = ti.ui.Camera()
    movement_speed = 0.002

    camera.fov = 0.8
    camera.position(2, 0.5, 2)
    camera.lookat(0.5, 0.5, 0)

    # get scene
    scene = ti.ui.Scene()

    falling = False

    # * Loop
    while window.running:
        # * set camera

        scene.set_camera(camera)

        # * set light
        scene.point_light(pos=[2, 2, 2], color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        # * update particle system
        if falling:
            SPH_solver.compute_accelerations()
            SPH_solver.update(dt)

        # * render particles
        scene.particles(SPH_solver.particle_system.position,
                        per_vertex_color=SPH_solver.particle_system.color,
                        radius=0.01)
        canvas.scene(scene)

        # * handle mouse input and keyboard input
        camera.track_user_inputs(
            window, movement_speed=movement_speed, hold_key=ti.ui.LMB)

        if window.is_pressed(ti.ui.ESCAPE):
            break
        if window.is_pressed(ti.ui.SPACE):
            falling = not falling

        # * show window
        window.show()
