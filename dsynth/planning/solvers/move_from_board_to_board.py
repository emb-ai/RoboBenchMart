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

def solve_fetch_move_to_board_cont_one_prod(env: MoveFromBoardToBoardContEnv, seed=None, debug=False, vis=False):
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

    FINGER_LENGTH = 0.04
    env = env.unwrapped

    # -------------------------------------------------------------------------- #
    # Setup target product
    # -------------------------------------------------------------------------- #

    #find the closest to gripper as target product
    max_dist = np.inf
    target_product_name = ''
    for target_actor_name in env.target_products_df['actor_name']:
        prod_pos = env.actors['products'][target_actor_name].pose.sp.p
    
        if np.linalg.norm(prod_pos - get_base_pose().sp.p) < max_dist:
            max_dist = np.linalg.norm(prod_pos - get_base_pose().sp.p)
            target_product_name = target_actor_name

    target_product_actor = env.actors['products'][target_product_name]
    obb = get_actor_obb(target_product_actor)
    target_product_center = get_obb_center(obb)

    
    # -------------------------------------------------------------------------- #
    # Go to shelf
    # -------------------------------------------------------------------------- #
    actor_shelf_name = env.active_shelves[0][0]
    shelf_pose = env.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp
    origin = shelf_pose.p - 1.4 * env.directions_to_shelf[0]

    res = planner.drive_base(origin)
    if res == -1:
        return res
    view_to_target = target_product_center - get_base_pose().sp.p
    view_to_target[2] = 0.
    res = planner.rotate_base_z(view_to_target)
    if res == -1:
        return res

    planner.planner.update_from_simulation()


    # -------------------------------------------------------------------------- #
    # Lift end-effector hand
    # -------------------------------------------------------------------------- #

    lift_ee_pos = get_tcp_pose().sp.p

     # lift the hand to the level of product
    lift_ee_pos[2] = target_product_center[2]

    lift_ee_pose = sapien.Pose(p=lift_ee_pos, q=get_tcp_pose().sp.q)

    # lift 0.45 in front of the robot
    lift_ee_pose = lift_ee_pose * sapien.Pose(p=[0, 0, 0.4]) # lift 

    res = planner.static_manipulation(lift_ee_pose, n_init_qpos=50, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Move to pre-grasp pose
    # -------------------------------------------------------------------------- #

    pre_grasp_base_translation = target_product_center - get_tcp_center()
    pre_grasp_base_translation[2] = 0.
    # pre_grasp_base_direction = common.np_normalize_vector(pre_grasp_base_translation)

    # move base to position 0.15m in fornt of the target object
    base_target_pos = get_base_pose().sp.p + \
        (1 - 0.15 / np.linalg.norm(pre_grasp_base_translation)) * pre_grasp_base_translation
    
    res = planner.drive_base(base_target_pos)
    if res == -1:
        return res
    view_to_target = target_product_center - get_base_pose().sp.p
    view_to_target[2] = 0.
    res = planner.rotate_base_z(view_to_target)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Grasp target object
    # -------------------------------------------------------------------------- #

    if is_mesh_cylindrical(target_product_actor):
        # if the mesh is cylindrical (can, bottle, etc.) we can grasp it from any side
        # we assume that diameter is less than grasp width
        # grasp_approaching = target_product_center - get_tcp_center()
        grasp_approaching = env.directions_to_shelf[0].copy()
        grasp_approaching[2] = 0.
        grasp_approaching = common.np_normalize_vector(grasp_approaching)



        grasp_closing = np.cross(grasp_approaching, [0., 0., 1.])
        grasp_center = target_product_center

    else: # mesh is a rectangular box, pick from the thinnest side
        # grasp_info = compute_box_grasp_thin_side_info(
        #     obb,
        #     target_closing=get_tcp_matrix()[:3, 1],
        #     ee_direction=get_tcp_matrix()[:3, 2],
        #     depth=FINGER_LENGTH,
        #     use_thick_side=True
        # )
        grasp_info = compute_grasp_info_by_obb(obb,
                                  approaching=get_tcp_matrix()[:3, 2],
                                  target_closing=get_tcp_matrix()[:3, 1],
                                  depth=FINGER_LENGTH,)
        grasp_closing, grasp_center, grasp_approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]

    grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)

    final_pose_center = grasp_pose.p + [0, 0, env.get_interboard_height()]
    final_pose = sapien.Pose(p=final_pose_center, q=grasp_pose.q)

    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target_product_actor), True
    )
    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    
    res = planner.close_gripper()
    if res == -1:
        return res
    
    kwargs = {"name": get_fcl_object_name(target_product_actor), "art_name": 'scene-0_ds_fetch_basket_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Lift product
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0.06, 0., 0.])
    res = planner.static_manipulation(lift_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    # -------------------------------------------------------------------------- #
    # Move backward
    # -------------------------------------------------------------------------- #
    tcp_pose = get_tcp_pose().sp

    res = planner.move_forward_delta(delta=-0.4)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    res = planner.static_manipulation(tcp_pose * sapien.Pose([0, 0, -0.4]), n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()



    res = planner.static_manipulation(get_tcp_pose().sp * sapien.Pose([0, 0, -0.15]), n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Lift end-effector hand
    # -------------------------------------------------------------------------- #
    # lift_center = get_tcp_pose().sp.p
    # lift_center[2] = final_pose.p[2] + 0.1

    # lift_pose = sapien.Pose(p=lift_center, q=get_tcp_pose().sp.q) * sapien.Pose([0, 0, 0.15])

    lift_pose = final_pose * sapien.Pose([0.05, 0, -0.25])

    res = planner.static_manipulation(lift_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Move forward
    # -------------------------------------------------------------------------- #
    tcp_pose = get_tcp_pose().sp
    res = planner.move_forward_delta(delta=0.3)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    res = planner.static_manipulation(tcp_pose * sapien.Pose([0, 0, 0.3]), n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()


    # -------------------------------------------------------------------------- #
    # Place
    # -------------------------------------------------------------------------- #
    # final_pose = sapien.Pose(p=final_target_pose.sp.p + 0.1,
    #                          q=get_tcp_pose().sp.q)
    res = planner.static_manipulation(final_pose * sapien.Pose([0.05, 0., 0.]), n_init_qpos=400, disable_lift_joint=True)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
   
    
    res = planner.open_gripper()
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    res = planner.idle_steps(t=10)

    planner.render_wait()
    return res




def solve_fetch_static_from_board_to_board(env: MoveFromBoardToBoardStaticEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.07
    env = env.unwrapped
    
    target = env.actors['products'][env.target_product_name]
    # retrieves the object oriented bounding box (trimesh box object)

    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_approaching = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    ee_direction = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    tcp_center = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()

    obb = get_actor_obb(target)

    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda


    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        target_closing=target_closing,
        ee_direction=ee_direction,
        depth=FINGER_LENGTH,
    )
    closing, center, approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.04])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])

    goal_center = env.target_volume.pose.sp.p
    goal_pose = env.agent.build_grasp_pose(approaching, closing, goal_center)
    
    if is_mesh_cylindrical(target):
        approaching = center - tcp_center
        approaching[2] = 0.
        approaching = common.np_normalize_vector(approaching)
        closing = np.cross(approaching, [0., 0., 1.])

        grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

        grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.02])
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.25])
        goal_pose = env.agent.build_grasp_pose(approaching, closing, goal_center)
        
    goal_pose = goal_pose * sapien.Pose([0, 0, -0.02])
    pre_goal_pose = goal_pose * sapien.Pose([0, 0, -0.2])
    
    # WTF: idk why this collision happens
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        get_fcl_object_name(target), 'scene-0-ds_fetch_gripper_link', True
    )
    # planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
    #     get_fcl_object_name(target), True
    # )

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(reach_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    # planner.move_base_x_and_manipulation(reach_pose, n_init_qpos=100)
    res = planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(grasp_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    res = planner.close_gripper()
    planner.planner.update_from_simulation()
    
    kwargs = {"name": get_fcl_object_name(target), "art_name": 'scene-0_ds_fetch_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0.06, 0., 0.])
    res = planner.static_manipulation(lift_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Pull
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(reach_pose * sapien.Pose([0.06, 0., 0.]), n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Move to goal board
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(pre_goal_pose * sapien.Pose([0.06, 0., 0.]), n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Place
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(goal_pose * sapien.Pose([0.05, 0., 0.]), n_init_qpos=200, disable_lift_joint=False)
    planner.planner.update_from_simulation()
    res = planner.open_gripper()


    planner.render_wait()
    return res

def solve_fetch_move_from_board_to_board(env: MoveFromBoardToBoardEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.07
    env = env.unwrapped
    
    target = env.actors['products'][env.target_product_name]
    # retrieves the object oriented bounding box (trimesh box object)

    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_approaching = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    ee_direction = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    tcp_center = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()

    obb = get_actor_obb(target)

    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda


    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        target_closing=target_closing,
        ee_direction=ee_direction,
        depth=FINGER_LENGTH,
    )
    closing, center, approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.04])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])

    goal_center = env.target_volume.pose.sp.p
    goal_pose = env.agent.build_grasp_pose(approaching, closing, goal_center)
    
    if is_mesh_cylindrical(target):
        approaching = center - tcp_center
        approaching[2] = 0.
        approaching = common.np_normalize_vector(approaching)
        closing = np.cross(approaching, [0., 0., 1.])

        grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

        grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.02])
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.25])
        goal_pose = env.agent.build_grasp_pose(approaching, closing, goal_center)
        
    goal_pose = goal_pose * sapien.Pose([0, 0, -0.02])
    pre_goal_pose = goal_pose * sapien.Pose([0, 0, -0.2])
    
    # WTF: idk why this collision happens
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        get_fcl_object_name(target), 'scene-0-ds_fetch_gripper_link', True
    )
    # planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
    #     get_fcl_object_name(target), True
    # )
    # -------------------------------------------------------------------------- #
    # Drive
    # -------------------------------------------------------------------------- #
    # robot_base_coords = planner.base_env.agent.base_link.pose.sp.p.copy()
    # robot_base_coords[:, :, 0] = 0.

    # target_coords = env.agent.tcp.pose.sp.p.copy()
    # target_coords[:, :, 0] = 0.
    # env.direction_to_shelf
    drive_pos = sapien.Pose(
        p = reach_pose.p - env.direction_to_shelf * 0.4 * CELL_SIZE,
        q = env.agent.tcp.pose.sp.q
    )
    planner.drive_base(drive_pos, reach_pose)
    res = planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(reach_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    # planner.move_base_x_and_manipulation(reach_pose, n_init_qpos=100)
    res = planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(grasp_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    res = planner.close_gripper()
    planner.planner.update_from_simulation()
    
    kwargs = {"name": get_fcl_object_name(target), "art_name": 'scene-0_ds_fetch_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0.06, 0., 0.])
    res = planner.static_manipulation(lift_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Pull
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(reach_pose * sapien.Pose([0.06, 0., 0.]), n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Move to goal board
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(pre_goal_pose * sapien.Pose([0.06, 0., 0.]), n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Place
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(goal_pose * sapien.Pose([0.05, 0., 0.]), n_init_qpos=200, disable_lift_joint=False)
    planner.planner.update_from_simulation()
    res = planner.open_gripper()


    planner.render_wait()
    return res

