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
    BAD_ENV_ERROR_CODE,
    get_fcl_object_name, 
    compute_box_grasp_thin_side_info,
    convert_actor_convex_mesh_to_fcl,
    is_mesh_cylindrical
)

def solve_fetch_open_door_fridge_cont(env: OpenDoorShowcaseContEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        disable_actors_collision=False,
        verbose=debug
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
    
    if len(planner.planner.planning_world.check_collision()) > 0:
        return BAD_ENV_ERROR_CODE
    
    env = env.unwrapped
    
    direction_to_shelf = env.directions_to_shelf[0]
    direction_to_shelf /= np.linalg.norm(direction_to_shelf)

    perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

    # -------------------------------------------------------------------------- #
    # Drive to the starting point
    # -------------------------------------------------------------------------- #
    target_fridge_pose = env.actors["fixtures"]["shelves"][env.target_actor_name[0]].pose
    target_fridge_center = target_fridge_pose.sp.p

    drive_pose = target_fridge_pose * sapien.Pose(p=[0.5, 1.2, 0])
    drive_pos = drive_pose.sp.p
    drive_pos[2] = 0
    
    res = planner.drive_base(target_pos=drive_pos, target_view_vec=direction_to_shelf)
    if res == -1:
        return res
    planner.planner.update_from_simulation()


    # -------------------------------------------------------------------------- #
    # Lift hand
    # -------------------------------------------------------------------------- #
    hand_pose = target_fridge_pose * sapien.Pose(p=[0.641, 0.3, 1.1])
    grasp_approaching = 2 * np.array([0, 0, -1]) + direction_to_shelf
    grasp_approaching /= np.linalg.norm(grasp_approaching)
    grasp_closing = get_tcp_matrix()[:3, 1]
    grasp_closing = grasp_closing - grasp_approaching * (grasp_closing @ grasp_approaching) / np.linalg.norm(grasp_approaching)
    pre_grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, hand_pose.sp.p)

    res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    

    # -------------------------------------------------------------------------- #
    # Grab handle
    # -------------------------------------------------------------------------- #
    res = planner.move_forward_delta(delta=0.15)

    # 'scene-0-[ENV#0]_active_open_fridge_1_b570ab7c_0_fridge_0_right_cover'
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{env.target_actor_name[0]}_right_cover', True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{env.target_actor_name[0]}_base_link', True
    )
    planner.close_gripper()

    grasp_pose = get_tcp_pose().sp * sapien.Pose([-0.1, -0.05, 0.1])
    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    

    # -------------------------------------------------------------------------- #
    # Open
    # -------------------------------------------------------------------------- #
    res = planner.rotate_z_delta(-1.7, rotate_recalculation_enabled=False)


    planner.render_wait()
    return res


def solve_fetch_open_door_showcase_cont(env: OpenDoorShowcaseContEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        disable_actors_collision=False,
        verbose=debug
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
    
    if len(planner.planner.planning_world.check_collision()) > 0:
        return BAD_ENV_ERROR_CODE

    env = env.unwrapped

    default_tcp_pose_wrt_to_base = get_base_pose().sp.inv() * get_tcp_pose().sp
    
    direction_to_shelf = env.directions_to_shelf[0]
    direction_to_shelf /= np.linalg.norm(direction_to_shelf)

    perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

    # cell_i, cell_j = env.init_cells[0]
    # start_point = np.array([
    #     cell_i * CELL_SIZE + CELL_SIZE / 2,
    #     cell_j * CELL_SIZE + CELL_SIZE / 2,
    #     0.
    # ])
    start_point = env.actors["fixtures"]["shelves"][env.target_actor_name[0]].pose.sp.p
    start_point = start_point - 1.55 * direction_to_shelf

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

    # -------------------------------------------------------------------------- #
    # Drive to the starting point
    # -------------------------------------------------------------------------- #

    direction_handle = grasp_center - start_point
    direction_handle[2] = 0.
    res = planner.drive_base(target_pos=start_point, target_view_vec=direction_handle)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Raise the gripper
    # -------------------------------------------------------------------------- #

    pre_grasp_center = grasp_center.copy()
    pre_grasp_center[:2] = get_base_pose().sp.p[:2] + 0.8 * direction_handle[:2] / np.linalg.norm(direction_handle)
    pre_grasp_approaching = direction_handle / np.linalg.norm(direction_handle)
    pre_grasp_closing = np.cross(pre_grasp_approaching, [0, 0, 1])
    pre_grasp_pose = env.agent.build_grasp_pose(pre_grasp_approaching, pre_grasp_closing, pre_grasp_center)
    res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=400, disable_lift_joint=True) #disable lift joint
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Drive to handle
    # -------------------------------------------------------------------------- #

    res = planner.move_forward_delta(delta=0.1)
    # if res == -1:
    #     return res
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Pick up the door
    # -------------------------------------------------------------------------- #
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{target_showcase_name}_door{door_idx}_link', True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{target_showcase_name}_door{door_idx}_handle_link', True
    )
    if door_idx in [1, 3]:
        grasp_approaching = 1.9 * perp_direction + direction_to_shelf
    else:
        grasp_approaching = -1.9 * perp_direction + direction_to_shelf
    grasp_approaching /= np.linalg.norm(grasp_approaching)
    grasp_closing = np.cross(grasp_approaching, [0, 0, 1])
    grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
    if door_idx in [1, 3]:
        grasp_pose = grasp_pose * sapien.Pose([0, 0.04, -0.04])
    else:
        grasp_pose = grasp_pose * sapien.Pose([0, -0.04, -0.04])

    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    
    planner.close_gripper()
    
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Pull back
    # -------------------------------------------------------------------------- #

    res = planner.move_forward_delta(delta=-0.1)
    if res == -1:
        return res
    
    planner.open_gripper()

    if door_idx in [1, 3]:
        grasp_pose = grasp_pose * sapien.Pose([0, 0.1, -0.1])
    else:
        grasp_pose = grasp_pose * sapien.Pose([0, -0.1, -0.1])

    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    neutral_pose = get_base_pose().sp * default_tcp_pose_wrt_to_base
    res = planner.static_manipulation(neutral_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Move to the center of the shelf
    # -------------------------------------------------------------------------- #
    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(-0.98)
    else:
        res = planner.rotate_z_delta(0.98)
    
    res = planner.move_forward_delta(0.8)

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(1.8)
    else:
        res = planner.rotate_z_delta(-1.8)

    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Take a stable grasp
    # -------------------------------------------------------------------------- #
    stable_grasp_center = get_obb_center(get_component_mesh(handle._bodies[0]).bounding_box_oriented)
    stable_grasp_center[2] -= 0.1 
    stable_grasp_approaching = handle.pose.sp.to_transformation_matrix()[:3, 1]
    stable_grasp_closing = np.cross(stable_grasp_approaching, [0, 0, 1])
    grasp_pose = env.agent.build_grasp_pose(stable_grasp_approaching, stable_grasp_closing, stable_grasp_center)
    grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.02])

    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.2])

    res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    planner.close_gripper()
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Rotate the robot to open the door
    # -------------------------------------------------------------------------- #

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(1.7, rotate_recalculation_enabled=False) # disable second rotation
    else:
        res = planner.rotate_z_delta(-1.7, rotate_recalculation_enabled=False) # disable second rotation
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Rotate and push the door from the interior
    # -------------------------------------------------------------------------- #
    neutral_pose = get_base_pose().sp * default_tcp_pose_wrt_to_base
    res = planner.static_manipulation(neutral_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(-1.)
    else:
        res = planner.rotate_z_delta(1.)
    if res == -1:
        return res
    
    push_center = get_obb_center(get_component_mesh(handle._bodies[0]).bounding_box_oriented)
    push_center[2] -= 0.1
    push_approaching = perp_direction.copy()
    if door_idx in [1, 3]:
        push_approaching /= -np.linalg.norm(push_approaching)
    else:
        push_approaching /= np.linalg.norm(push_approaching)
    push_closing = np.cross(push_approaching, [0, 0, 1])
    push_pose = env.agent.build_grasp_pose(push_approaching, push_closing, push_center)
    if door_idx in [1, 3]:
        push_pose = push_pose * sapien.Pose([0, 0.2, 0.])
    else:
        push_pose = push_pose * sapien.Pose([0, -0.2, 0.])
    res = planner.static_manipulation(push_pose, n_init_qpos=400, disable_lift_joint=True)

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(0.6)
    else:
        res = planner.rotate_z_delta(-0.6)
    
    res = planner.move_forward_delta(0.3)

    planner.planner.update_from_simulation()

    res = planner.idle_steps(t=1)

    planner.render_wait()
    return res

def solve_fetch_close_door_showcase_cont(env: CloseDoorShowcaseContEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        disable_actors_collision=False,
        verbose=debug
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
    
    if len(planner.planner.planning_world.check_collision()) > 0:
        return BAD_ENV_ERROR_CODE
    
    env = env.unwrapped
    
    direction_to_shelf = env.directions_to_shelf[0]
    direction_to_shelf /= np.linalg.norm(direction_to_shelf)

    perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

    door_name = env.target_door_names[0]
    door_idx = env.DOOR_NAMES_2_IDX[door_name]

    target_showcase_name = env.target_actor_name[0]
    target_showcase = env.actors['fixtures']['shelves'][target_showcase_name]

    handle = target_showcase.links_map[f'door{door_idx}_handle_link']

    mesh = get_component_mesh(handle._bodies[0])
    obb: trimesh.primitives.Box = mesh.bounding_box_oriented
    grasp_center = get_obb_center(obb)
    grasp_center[2] -= 0.2

    if door_idx in [1, 3]:
        grasp_center += - 0.2 * perp_direction
    if door_idx in [2, 4]:
        grasp_center += 0.2 * perp_direction
    
    start_pose_center = grasp_center - 0.8 * direction_to_shelf
    start_pose_center[2] = 0.

    # -------------------------------------------------------------------------- #
    # Drive to the starting point
    # -------------------------------------------------------------------------- #
    res = planner.drive_base(target_pos=start_pose_center, target_view_vec=direction_to_shelf)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Place the gripper near the handle
    # -------------------------------------------------------------------------- #
    planner.close_gripper()
    
    grasp_pose = env.agent.build_grasp_pose(direction_to_shelf, perp_direction, grasp_center)
    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=True) #disable lift joint
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Rotate
    # -------------------------------------------------------------------------- #
    res = planner.move_forward_delta(0.1)

    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{target_showcase_name}_door{door_idx}_link', True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{target_showcase_name}_door{door_idx}_handle_link', True
    )

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(-0.98, rotate_recalculation_enabled=False)
    else:
        res = planner.rotate_z_delta(0.98, rotate_recalculation_enabled=False)

    # res = planner.rotate_base_z(direction_to_shelf)
    # res = planner.move_forward_delta(0.1)

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(0.26, rotate_recalculation_enabled=False)
    else:
        res = planner.rotate_z_delta(-0.26, rotate_recalculation_enabled=False)

    res = planner.move_forward_delta(0.7)

    # -------------------------------------------------------------------------- #
    # Push
    # -------------------------------------------------------------------------- #

    push_center = get_tcp_center() + 0.2 * direction_to_shelf
    push_pose = env.agent.build_grasp_pose(direction_to_shelf, perp_direction, push_center)
    res = planner.static_manipulation(push_pose, n_init_qpos=400, disable_lift_joint=True) #disable lift joint
    # if res == -1:
    #     return res

    res = planner.idle_steps(t=1)

    planner.render_wait()
    return res


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

    env = env.unwrapped

    default_tcp_pose_wrt_to_base = get_base_pose().sp.inv() * get_tcp_pose().sp
    
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

    # -------------------------------------------------------------------------- #
    # Drive to the starting point
    # -------------------------------------------------------------------------- #

    direction_handle = grasp_center - start_point
    direction_handle[2] = 0.
    res = planner.drive_base(target_pos=start_point, target_view_vec=direction_handle)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Raise the gripper
    # -------------------------------------------------------------------------- #

    pre_grasp_center = grasp_center.copy()
    pre_grasp_center[:2] = get_base_pose().sp.p[:2] + 0.8 * direction_handle[:2] / np.linalg.norm(direction_handle)
    pre_grasp_approaching = direction_handle / np.linalg.norm(direction_handle)
    pre_grasp_closing = np.cross(pre_grasp_approaching, [0, 0, 1])
    pre_grasp_pose = env.agent.build_grasp_pose(pre_grasp_approaching, pre_grasp_closing, pre_grasp_center)
    res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=400, disable_lift_joint=True) #disable lift joint
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Drive to handle
    # -------------------------------------------------------------------------- #

    res = planner.move_forward_delta(delta=0.1)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Pick up the door
    # -------------------------------------------------------------------------- #
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{target_showcase_name}_door{door_idx}_link', True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        f'scene-0-{target_showcase_name}_door{door_idx}_handle_link', True
    )
    if door_idx in [1, 3]:
        grasp_approaching = 1.9 * perp_direction + direction_to_shelf
    else:
        grasp_approaching = -1.9 * perp_direction + direction_to_shelf
    grasp_approaching /= np.linalg.norm(grasp_approaching)
    grasp_closing = np.cross(grasp_approaching, [0, 0, 1])
    grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
    if door_idx in [1, 3]:
        grasp_pose = grasp_pose * sapien.Pose([0, 0.04, -0.04])
    else:
        grasp_pose = grasp_pose * sapien.Pose([0, -0.04, -0.04])

    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    
    planner.close_gripper()
    
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Pull back
    # -------------------------------------------------------------------------- #

    res = planner.move_forward_delta(delta=-0.1)
    if res == -1:
        return res
    
    planner.open_gripper()

    if door_idx in [1, 3]:
        grasp_pose = grasp_pose * sapien.Pose([0, 0.1, -0.1])
    else:
        grasp_pose = grasp_pose * sapien.Pose([0, -0.1, -0.1])

    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    neutral_pose = get_base_pose().sp * default_tcp_pose_wrt_to_base
    res = planner.static_manipulation(neutral_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Move to the center of the shelf
    # -------------------------------------------------------------------------- #
    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(-0.98)
    else:
        res = planner.rotate_z_delta(0.98)
    
    res = planner.move_forward_delta(0.8)

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(1.7)
    else:
        res = planner.rotate_z_delta(-1.7)

    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Take a stable grasp
    # -------------------------------------------------------------------------- #
    stable_grasp_center = get_obb_center(get_component_mesh(handle._bodies[0]).bounding_box_oriented)
    stable_grasp_center[2] -= 0.1 
    stable_grasp_approaching = handle.pose.sp.to_transformation_matrix()[:3, 1]
    stable_grasp_closing = np.cross(stable_grasp_approaching, [0, 0, 1])
    grasp_pose = env.agent.build_grasp_pose(stable_grasp_approaching, stable_grasp_closing, stable_grasp_center)
    grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.02])

    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.2])

    res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    planner.close_gripper()
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Rotate the robot to open the door
    # -------------------------------------------------------------------------- #

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(1.7, rotate_recalculation_enabled=False) # disable second rotation
    else:
        res = planner.rotate_z_delta(-1.7, rotate_recalculation_enabled=False) # disable second rotation
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Rotate and push the door from the interior
    # -------------------------------------------------------------------------- #
    neutral_pose = get_base_pose().sp * default_tcp_pose_wrt_to_base
    res = planner.static_manipulation(neutral_pose, n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(-1.)
    else:
        res = planner.rotate_z_delta(1.)
    if res == -1:
        return res
    
    push_center = get_obb_center(get_component_mesh(handle._bodies[0]).bounding_box_oriented)
    push_center[2] -= 0.1
    push_approaching = perp_direction.copy()
    if door_idx in [1, 3]:
        push_approaching /= -np.linalg.norm(push_approaching)
    else:
        push_approaching /= np.linalg.norm(push_approaching)
    push_closing = np.cross(push_approaching, [0, 0, 1])
    push_pose = env.agent.build_grasp_pose(push_approaching, push_closing, push_center)
    if door_idx in [1, 3]:
        push_pose = push_pose * sapien.Pose([0, 0.2, 0.])
    else:
        push_pose = push_pose * sapien.Pose([0, -0.2, 0.])
    res = planner.static_manipulation(push_pose, n_init_qpos=400, disable_lift_joint=True)

    if door_idx in [1, 3]:
        res = planner.rotate_z_delta(0.6)
    else:
        res = planner.rotate_z_delta(-0.6)
    
    res = planner.move_forward_delta(0.3)

    planner.planner.update_from_simulation()

    res = planner.idle_steps(t=1)

    planner.render_wait()
    return res



def solve_fetch_open_door_showcase_old(env: OpenDoorFridgeEnv, seed=None, debug=False, vis=False):
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
