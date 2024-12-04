import taichi as ti
import neighbor_search as ns
from tqdm import tqdm
import imageio
import numpy as np
import particle as pa
import EOS
import os


if __name__ == "__main__":

    # * set device
    ti.init(arch=ti.cpu, device_memory_GB=4)

    # * object settings
    vec3 = ti.types.vector(3, float)

    # particle system
    max_particles = 1500
    particle_system = pa.ParticleSystem(max_particles)
    particle_system.random_initialize()

    # build neighbor searcher
    neighbor_searcher = ns.NeighborSearcher(particle_system)

    # build EOS
    eos = EOS.EOS(particle_system, neighbor_searcher)

    # System settings
    dt = 4E-3

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
    if_save = 1
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    # * Frame index
    frame_num = 1000

    # * Loop
    for frame_index in tqdm(range(frame_num)):
        # * set camera
        camera.fov = 0.8
        camera.position(2, 0.5, 2)
        camera.lookat(0.5, 0.5, 0)
        scene.set_camera(camera)

        # * set light
        scene.point_light(pos=[2, 2, 2], color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        # * update particle system
        forces = ti.Vector.field(3, float, max_particles)
        eos.update_EOS(particle_system)
        eos.computePressureFromEos_force(forces, max_particles)
        eos.computeViscosityFromEOS_force(forces, max_particles)

        particle_system.update(forces, dt)

        # * render particles
        scene.particles(particle_system.position,
                        per_vertex_color=particle_system.color,
                        radius=0.02)
        canvas.scene(scene)

        # * save current frame as image
        if if_save:
            frame_image = window.get_image_buffer_as_numpy()
            frame_image = (frame_image * 255).astype(np.uint8)  # 转换为 uint8 类型
            imageio.imwrite(
                f'{output_dir}/frame_{frame_index:04d}.png', frame_image)

        # * show window
        window.show()

    print("Frames saved in:", output_dir)
