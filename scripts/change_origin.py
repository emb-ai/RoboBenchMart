import numpy as np
import scipy.spatial
from stl import mesh
import tqdm
from multiprocessing import Pool

import trimesh

def change_origin_frame(path, T):
    mesh = trimesh.load(path)
    mesh.show(flags={'axis': True})
    mesh.apply_transform(T)
    mesh.show(flags={'axis': True})
    mesh.export(path)

def change_origin_frame_dae(path, T):
    # mesh_params = trimesh.exchange.dae.load_collada('/home/kvsoshin/Work/ManiSkill/mani_skill/assets/robots/fetch/fetch_description/meshes/gripper_link.dae')
    # mesh = trimesh.Trimesh(**mesh_params)
    mesh = trimesh.load(path).to_mesh()
    mesh.show(flags={'axis': True})
    mesh.apply_transform(T)
    mesh.show(flags={'axis': True})
    export = trimesh.exchange.dae.export_collada(mesh)

    with open(path, "w") as text_file:
        text_file.write(export.decode())
    

if __name__ == '__main__':
    gripper_T = np.array([
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    pi_2 = 3.1416 / 2
    rot_y = np.array([
        [np.cos(pi_2), 0, -np.sin(pi_2), 0],
        [0, 1, 0, 0],
        [np.sin(pi_2), 0, np.cos(pi_2), 0],
        [0, 0, 0, 1]
    ])
    change_origin_frame_dae('/home/kvsoshin/Work/ManiSkill/mani_skill/assets/robots/fetch/fetch_description/meshes/gripper_link.dae', gripper_T)
    # change_origin_frame_dae('/home/kvsoshin/Work/ManiSkill/mani_skill/assets/robots/fetch/fetch_description/meshes/r_gripper_finger_link.dae')
    # change_origin_frame_dae('/home/kvsoshin/Work/ManiSkill/mani_skill/assets/robots/fetch/fetch_description/meshes/l_gripper_finger_link.dae')
    change_origin_frame('/home/kvsoshin/Work/ManiSkill/mani_skill/assets/robots/fetch/fetch_description/meshes/gripper_link.STL', gripper_T)
    change_origin_frame('/home/kvsoshin/Work/ManiSkill/mani_skill/assets/robots/fetch/fetch_description/meshes/r_gripper_finger_link.STL', rot_y)
    change_origin_frame('/home/kvsoshin/Work/ManiSkill/mani_skill/assets/robots/fetch/fetch_description/meshes/l_gripper_finger_link.STL', rot_y)