import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
import numpy as np
np.set_printoptions(suppress=True)
import mplib
from transforms3d.euler import euler2quat, quat2euler
from mplib.sapien_utils.conversion import convert_object_name
from mplib.collision_detection.fcl import CollisionGeometry
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib.collision_detection.fcl import Convex, CollisionObject, FCLObject
from mplib.collision_detection import fcl
import sapien
import sapien.physx as physx
from sapien import Entity
from sapien.physx import (
    PhysxArticulation,
    PhysxArticulationLinkComponent,
    PhysxCollisionShapeConvexMesh
)


from typing import Literal, Optional, Sequence, Union
import sys
import trimesh
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube
from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils import common

from dsynth.envs import *

from dsynth.planning.motionplanner import (
    PandaArmMotionPlanningSolverV2, 
    PandaArmMotionPlanningSapienSolver,
    FetchStaticArmMotionPlanningSapienSolver,
    FetchQuasiStaticArmMotionPlanningSapienSolver,
    FetchMotionPlanningSapienSolver
)
from dsynth.planning.utils import (
    get_fcl_object_name, 
    compute_box_grasp_thin_side_info,
    convert_actor_convex_mesh_to_fcl,
    is_mesh_cylindrical
)

def solve_fetch_open_door_showcase(env: OpenDoorFridgeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        disable_actors_collision=False,
    )
    def get_obb_center(obb):
        T = np.array(obb.primitive.transform)
        return T[:3, 3]

    def get_base_pose():
        return env.agent.base_link.pose
    
    def get_tcp_pose():
        return env.agent.tcp.pose

    def get_tcp_matrix():
        tcp_pose = get_tcp_pose()
        return tcp_pose.to_transformation_matrix()[0].cpu().numpy()
    
    def get_tcp_center():
        return get_tcp_matrix()[:3, 3]


    FINGER_LENGTH = 0.07
    env = env.unwrapped
    
    direction_to_shelf = env.directions_to_shelf[0]
    direction_to_shelf /= np.linalg.norm(direction_to_shelf)

    perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

    cell_i, cell_j = env.init_cells[0]
    start_point = np.array([
        cell_i * CELL_SIZE + CELL_SIZE / 2,
        cell_j * CELL_SIZE + CELL_SIZE / 2,
        0.
    ])
    door_name = env.target_door_names[0]
    door_idx = env.DOOR_NAMES_2_IDX[door_name]

    if door_idx == 1:
        start_point += -1.25 * perp_direction + 0.6 * direction_to_shelf
    if door_idx == 2:
        start_point += -0.3 * perp_direction + 0.6 * direction_to_shelf
    if door_idx == 3:
        start_point += 0.3 * perp_direction + 0.6 * direction_to_shelf
    if door_idx == 4:
        start_point += 1.25 * perp_direction + 0.6 * direction_to_shelf

    target_showcase_name = env.target_actor_name[0]
    target_showcase = env.actors['fixtures']['shelves'][target_showcase_name]

    handle = target_showcase.links_map[f'door{door_idx}_handle_link']

    mesh = get_component_mesh(handle._bodies[0])
    obb: trimesh.primitives.Box = mesh.bounding_box_oriented
    grasp_center = get_obb_center(obb)
    grasp_center[2] -= 0.1

    if door_idx in [2, 4]:
        grasp_approaching = -1.9 * perp_direction + direction_to_shelf
    else:
        grasp_approaching = 1.8 * perp_direction + direction_to_shelf
    grasp_approaching /= np.linalg.norm(grasp_approaching)
    grasp_closing = np.cross(grasp_approaching, [0, 0, 1])

    pre_grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
    pre_grasp_pose = pre_grasp_pose * sapien.Pose([0, 0, -0.2])

    grasp_pose = pre_grasp_pose * sapien.Pose([0.0, -0.05, 0.17])
    
    # direction_handle = grasp_center - get_base_pose().sp.p
    direction_handle = grasp_center - start_point
    direction_handle[2] = 0.
    # res = planner.rotate_base_z(direction_handle)
    # if res == -1:
    #     return res
    res = planner.drive_base(target_pos=start_point, target_view_vec=direction_handle)
    if res == -1:
        return res
    
    res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{target_showcase_name}_door{door_idx}_link', True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{target_showcase_name}_door{door_idx}_handle_link', True
    )

    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res

    res = planner.close_gripper()
    if res == -1:
        return res
    
    
    res = planner.move_forward_delta(delta=-0.3)
    if res == -1:
        return res
    
    # res = planner.static_manipulation(get_tcp_pose().sp * sapien.Pose([0.0, 0.0, -0.1]), n_init_qpos=400, disable_lift_joint=False)
    # if res == -1:
    #     return res
    
    res = planner.open_gripper()
    if res == -1:
        return res
    
    #=======
    # res = planner.static_manipulation(get_tcp_pose().sp * sapien.Pose([0, 0, -0.2]), n_init_qpos=400, disable_lift_joint=False)
    # if res == -1:
    #     return res
    # res = planner.move_forward_delta(delta=0.15)
    # if res == -1:
    #     return res


    # grasp_center = get_obb_center(get_component_mesh(handle._bodies[0]).bounding_box_oriented)
    # grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
    # grasp_pose = grasp_pose * sapien.Pose([-0.1, 0.00, -0.02])
    # res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=False)
    # if res == -1:
    #     return res
    # res = planner.close_gripper()

    # res = planner.move_forward_delta(delta=-0.2)
    # if res == -1:
    #     return res

    # res = planner.open_gripper()
    #=======
    
    res = planner.static_manipulation(get_tcp_pose().sp * sapien.Pose([0, 0, -0.2]), n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    
    # res = planner.rotate_base_z(-perp_direction)
    # if res == -1:
    #     return res
    res = planner.rotate_z_delta(0.9)
    if res == -1:
        return res
    
    res = planner.move_forward_delta(delta=0.6)
    if res == -1:
        return res
    res = planner.rotate_z_delta(-0.8)
    if res == -1:
        return res
    
    # res = planner.rotate_base_z(direction_to_shelf)
    # if res == -1:
    #     return res
    
    pre_open_approaching = handle.pose.sp.to_transformation_matrix()[:3, 0]
    pre_open_center = get_obb_center(get_component_mesh(handle._bodies[0]).bounding_box_oriented)
    pre_open_closing = np.cross(pre_open_approaching, [0, 0, 1])
    pre_open_pose = env.agent.build_grasp_pose(pre_open_approaching, pre_open_closing, pre_open_center)
    pre_open_pose = pre_open_pose * sapien.Pose([-0.3, -0.1, 0.05])
    res = planner.static_manipulation(pre_open_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    
    res = planner.rotate_z_delta(-0.4)
    if res == -1:
        return res

    open_pose_1 = pre_open_pose * sapien.Pose([0.,0.3, 0.2])
    res = planner.static_manipulation(open_pose_1, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    
    res = planner.rotate_z_delta(-0.6)
    if res == -1:
        return res

    open_pose_2 = get_tcp_pose().sp * sapien.Pose([0.,0.0, 0.1])
    res = planner.static_manipulation(open_pose_2, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
   
    res = planner.idle_steps(t=10)

    planner.render_wait()
    return res
