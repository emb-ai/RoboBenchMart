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
    is_mesh_cylindrical,
    BAD_ENV_ERROR_CODE
)

def solve_fetch_pick_to_basket_cont_one_prod(env: PickToBasketContEnv, seed=None, debug=False, vis=False):
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

    res = planner.move_forward_delta(delta=-0.4)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Drop to basket
    # -------------------------------------------------------------------------- #

    goal_center = env.calc_target_pose().sp.p
    goal_center = goal_center # add shift from base to basket

    goal_approaching = np.array([0, 0., -1.])
    goal_closing = - get_base_pose().sp.to_transformation_matrix()[:3, 1]

    goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)
    goal_pose = goal_pose * sapien.Pose(p=[-0.03, 0., -0.35])

    res = planner.static_manipulation(goal_pose, n_init_qpos=100, disable_lift_joint=False)
    if res == -1:
        return res
    
    res = planner.open_gripper()
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    res = planner.idle_steps(t=10)

    planner.render_wait()
    return res




def solve_fetch_pick_to_basket_one_prod(env: PickToBasketEnv, seed=None, debug=False, vis=False):
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
    
    # -------------------------------------------------------------------------- #
    # Go to shelf
    # -------------------------------------------------------------------------- #
    drive_pos = np.array([env.init_cells[0, 0] * CELL_SIZE + CELL_SIZE / 2, 
                          env.init_cells[0, 1] * CELL_SIZE + CELL_SIZE / 2, 0])

    res = planner.drive_base(drive_pos, target_view_vec=env.directions_to_shelf[0])
    if res == -1:
        return res
    planner.planner.update_from_simulation()

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

    # move base to position 0.1m in fornt of the target object
    base_target_pos = get_base_pose().sp.p + \
        (1 - 0.1 / np.linalg.norm(pre_grasp_base_translation)) * pre_grasp_base_translation
    
    res = planner.drive_base(base_target_pos, env.directions_to_shelf[0])
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Grasp target object
    # -------------------------------------------------------------------------- #

    if is_mesh_cylindrical(target_product_actor):
        # if the mesh is cylindrical (can, bottle, etc.) we can grasp it from any side
        # we assume that diameter is less than grasp width
        grasp_approaching = target_product_center - get_tcp_center()
        # grasp_approaching = env.directions_to_shelf[0].copy()
        grasp_approaching[2] = 0.
        grasp_approaching = common.np_normalize_vector(grasp_approaching)



        grasp_closing = np.cross(grasp_approaching, [0., 0., 1.])
        grasp_center = target_product_center

    else: # mesh is a rectangular box, pick from the thinnest side
        grasp_info = compute_box_grasp_thin_side_info(
            obb,
            target_closing=get_tcp_matrix()[:3, 1],
            ee_direction=get_tcp_matrix()[:3, 2],
            depth=FINGER_LENGTH,
        )
        grasp_closing, grasp_center, grasp_approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]

    grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)

    # WTF: idk why this collision happens
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

    res = planner.move_forward_delta(delta=-0.5)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Drop to basket
    # -------------------------------------------------------------------------- #

    goal_center = env.calc_target_pose().sp.p
    goal_center = goal_center + np.array([0.1, 0., 0.2]) # add shift from base to basket

    goal_approaching = np.array([0, 0., -1.])
    goal_closing = - get_base_pose().sp.to_transformation_matrix()[:3, 1]

    goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)

    res = planner.static_manipulation(goal_pose, n_init_qpos=100, disable_lift_joint=False)
    if res == -1:
        return res
    
    res = planner.open_gripper()
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    res = planner.idle_steps(t=10)

    planner.render_wait()
    return res


def solve_fetch_pick_to_basket_static_one_prod_OLD(env: PickToBasketEnv, seed=None, debug=False, vis=False):
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
    FINGER_LENGTH = 0.07
    env = env.unwrapped
    
    tcp_pose = env.agent.tcp.pose

    #find the closest to gripper as target product
    max_dist = np.inf
    target_product_name = ''
    for target_actor_name in env.target_products_df['actor_name']:
        prod_pos = env.actors['products'][target_actor_name].pose.sp.p
    
        if np.linalg.norm(prod_pos - tcp_pose.sp.p) < max_dist:
            max_dist = np.linalg.norm(prod_pos - tcp_pose.sp.p)
            target_product_name = target_actor_name

    target = env.actors['products'][target_product_name]




    target_closing = tcp_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_approaching = tcp_pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    ee_direction = tcp_pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    tcp_center = tcp_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()
    tcp_closing = tcp_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    obb = get_actor_obb(target)


    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        target_closing=target_closing,
        ee_direction=ee_direction,
        depth=FINGER_LENGTH,
    )
    closing, center, approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    
    if is_mesh_cylindrical(target):
        # if the mesh is cylindrical we can grasp it from any side
        approaching = center - tcp_center
        approaching[2] = 0.
        approaching = common.np_normalize_vector(approaching)
        closing = np.cross(approaching, [0., 0., 1.])

    #compute grasp pose
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    #pre-grasp pose - move mobile base with static hand to this pose
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    
    approaching = center - tcp_center
    approaching[2] = 0.
    reach_p = grasp_pose.p - approaching * (1 - 0.4 / np.linalg.norm(approaching))
    reach_pose = sapien.Pose(p=reach_p, q=grasp_pose.q)
    
    # WTF: idk why this collision happens
    # planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
    #     get_fcl_object_name(target), True
    # )
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target), True
    )

    # planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
    #     get_fcl_object_name(target), 'scene-0-ds_fetch_basket_gripper_link', True
    # )

    # planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
    #     get_fcl_object_name(target), get_fcl_object_name(env.scene.actors['floor_room_0']), True
    # )
    # planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
    #     get_fcl_object_name(target), True
    # )
    



    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(reach_pose, n_init_qpos=100, disable_lift_joint=False)
    if res == -1:
        return res
    # planner.move_base_x_and_manipulation(reach_pose, n_init_qpos=100)
    # res = planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Drive Forward
    # -------------------------------------------------------------------------- #
    tcp_pose = env.agent.tcp.pose
    tcp_center = tcp_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()
    move_vec = grasp_pose.p - tcp_center
    move_vec[2] = 0
    drive_pos = env.agent.base_link.pose.sp.p + move_vec * (1 - 0.1 / np.linalg.norm(move_vec))
    drive_pos[2] = 0
    res = planner.drive_base(drive_pos)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    res = planner.close_gripper()
    planner.planner.update_from_simulation()
    
    kwargs = {"name": get_fcl_object_name(target), "art_name": 'scene-0_ds_fetch_basket_1', "link_id": planner.planner.move_group_link_id}
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
    # Drive backward
    # -------------------------------------------------------------------------- #

    res = planner.move_forward_delta(delta=-0.5)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_center = env.calc_target_pose().sp.p
    goal_center = goal_center + np.array([0.1, 0., 0.5])

    goal_approaching = np.array([0, 0., -1.])
    goal_approaching /= np.linalg.norm(goal_approaching)


    goal_closing = np.array([1., -1., 0.])
    # goal_closing = tcp_closing - goal_approaching * (goal_approaching @ tcp_closing)
    goal_closing /= np.linalg.norm(goal_closing)
    goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)
    goal_pose = goal_pose* sapien.Pose([0., 0., 0.2])

    res = planner.static_manipulation(goal_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Place
    # -------------------------------------------------------------------------- #

    res = planner.open_gripper()

    res = planner.idle_steps(t=40)


    planner.render_wait()
    return res



def solve_fetch_pick_to_basket_one_prod_old(env: PickToBasketSpriteEnv, seed=None, debug=False, vis=False):
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
    
    # -------------------------------------------------------------------------- #
    # Drive
    # -------------------------------------------------------------------------- #
    
    drive_pos = env.target_drive_position

    res = planner.drive_base(drive_pos, env.direction_to_shelf)
    planner.planner.update_from_simulation()

    
    tcp_pose = env.agent.tcp.pose

    #find the closest to gripper as target product
    max_dist = np.inf
    target_product_name = ''
    for target_actor_name in env.target_product_names:
        prod_pos = env.actors['products'][target_actor_name].pose.sp.p
        if np.linalg.norm(prod_pos - tcp_pose.sp.p) < max_dist:
            max_dist = np.linalg.norm(prod_pos - tcp_pose.sp.p)
            target_product_name = target_actor_name

    target = env.actors['products'][target_product_name]
    # retrieves the object oriented bounding box (trimesh box object)

    target_closing = tcp_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_approaching = tcp_pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    ee_direction = tcp_pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    tcp_center = tcp_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()
    tcp_closing = tcp_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    obb = get_actor_obb(target)

    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = tcp_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
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
    goal_center = goal_center + np.array([0.1, 0., 0.45])

    goal_approaching = np.array([0, 0., -1.])
    goal_approaching /= np.linalg.norm(goal_approaching)


    goal_closing = np.array([1., -1., 0.])
    # goal_closing = tcp_closing - goal_approaching * (goal_approaching @ tcp_closing)
    goal_closing /= np.linalg.norm(goal_closing)
    goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)
    goal_pose = goal_pose* sapien.Pose([0., 0., 0.2])
    
    if is_mesh_cylindrical(target):
        approaching = center - tcp_center
        approaching[2] = 0.
        approaching = common.np_normalize_vector(approaching)
        closing = np.cross(approaching, [0., 0., 1.])

        grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

        grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.02])
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.25])
        
    pre_goal_pose = goal_pose * sapien.Pose([0, 0, -0.2])
    
    # WTF: idk why this collision happens
    # planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
    #     get_fcl_object_name(target), True
    # )
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target), True
    )

    # planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
    #     get_fcl_object_name(target), 'scene-0-ds_fetch_basket_gripper_link', True
    # )

    # planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
    #     get_fcl_object_name(target), get_fcl_object_name(env.scene.actors['floor_room_0']), True
    # )
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
    
    kwargs = {"name": get_fcl_object_name(target), "art_name": 'scene-0_ds_fetch_basket_1', "link_id": planner.planner.move_group_link_id}
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
    # Move to goal pose
    # -------------------------------------------------------------------------- #

    res = planner.static_manipulation(goal_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Place
    # -------------------------------------------------------------------------- #

    res = planner.open_gripper()

    res = planner.idle_steps(t=40)


    planner.render_wait()
    return res



