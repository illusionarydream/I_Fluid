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
    canvas.set_background_color((1, 1, 1))

    # get camera
    camera = ti.ui.Camera()

    # get scene
    scene = ti.ui.Scene()

    # * Directory for saving frames
    if_save = 0
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    # * Frame index
    frame_index = 0

    # * Loop
    while window.running:
        # * set camera
        camera.fov = 0.8
        camera.position(2, 0.5, 2)
        camera.lookat(0.5, 0.5, 0)
        scene.set_camera(camera)

        # * set light
        scene.point_light(pos=[2, 2, 2], color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        # * update particle system
        SPH_solver.compute_accelerations()
        SPH_solver.update(dt)

        # * render particles
        scene.particles(SPH_solver.particle_system.position,
                        per_vertex_color=SPH_solver.particle_system.color,
                        radius=0.01)
        canvas.scene(scene)

        # * save current frame as image
        if if_save:
            frame_image = window.get_image_buffer_as_numpy()
            frame_image = (frame_image * 255).astype(np.uint8)  # 转换为 uint8 类型
            imageio.imwrite(
                f'{output_dir}/frame_{frame_index:04d}.png', frame_image)
            frame_index += 1

        # * show window
        window.show()

    print("Frames saved in:", output_dir)
