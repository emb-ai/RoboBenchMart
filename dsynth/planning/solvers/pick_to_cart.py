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
    get_fcl_object_name, 
    compute_box_grasp_thin_side_info,
    convert_actor_convex_mesh_to_fcl,
    is_mesh_cylindrical
)

def solve_fetch_pick_to_basket_static(env: PickToCartStaticEnv, seed=None, debug=False, vis=False):
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
    tcp_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

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

    
    
    # goal_center = env.target_volume.pose.sp.p
    # goal_center = goal_center + np.array([0.2, 0., 0.15])

    # goal_approaching = np.array([-5., 0., -1.])
    # goal_approaching /= np.linalg.norm(goal_approaching)

    # goal_closing = tcp_closing - goal_approaching * (goal_approaching @ tcp_closing)
    # goal_closing /= np.linalg.norm(goal_closing)
    # goal_closing = -goal_closing
    # goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)

    goal_center = env.target_volume.pose.sp.p
    goal_center = goal_center + np.array([0.2, 0., 0.15])

    goal_approaching = np.array([0, 0., -1.])
    goal_approaching /= np.linalg.norm(goal_approaching)


    goal_closing = np.array([1., 0., 0.])
    # goal_closing = tcp_closing - goal_approaching * (goal_approaching @ tcp_closing)
    goal_closing /= np.linalg.norm(goal_closing)
    goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)
    
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
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        get_fcl_object_name(target), 'scene-0-ds_fetch_basket_gripper_link', True
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

    res = planner.static_manipulation(goal_pose * sapien.Pose([0.05, 0., 0.]), n_init_qpos=200, disable_lift_joint=False)
    planner.planner.update_from_simulation()
    res = planner.open_gripper()


    planner.render_wait()
    return res


