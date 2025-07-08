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

    target_showcase_name = env.target_actor_name[0]
    target_showcase = env.actors['fixtures']['shelves'][target_showcase_name]

    handle = target_showcase.links_map['door2_handle_link']

    grasp_center = handle.pose.sp.p
    grasp_approaching = np.array([-1., -1., 0.])
    grasp_approaching /= np.linalg.norm(grasp_approaching)
    grasp_closing = np.array([-1., 1., 0.])
    grasp_closing /= np.linalg.norm(grasp_closing)

    mesh = get_component_mesh(handle._bodies[0])
    obb: trimesh.primitives.Box = mesh.bounding_box_oriented
    grasp_center = get_obb_center(obb)

    pre_grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
    pre_grasp_pose = pre_grasp_pose * sapien.Pose([0, 0, -0.1])
    res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    
    # planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
    #     'scene-0-[ENV#0]_SHELF_1_zone2.freezer_large1_door2_link', True
    # )
    # planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
    #     'scene-0-[ENV#0]_SHELF_1_zone2.freezer_large1_door2_handle_link', True
    # )
    # planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
    #     'scene-0-ds_fetch_basket_l_gripper_finger_link', True
    # )
    # planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
    #     'scene-0-ds_fetch_basket_r_gripper_finger_link', True
    # )

    grasp_pose = pre_grasp_pose * sapien.Pose([0, 0, 0.05])
    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    res = planner.close_gripper()
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    
    # open_pose = sapien.Pose(p=pre_grasp_pose.p + [0, 0.08, -0.1], q=pre_grasp_pose.q)
    # res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=400, disable_lift_joint=False)
    # if res == -1:
    #     return res
    
    res = planner.move_forward_delta(delta=-0.2)
    if res == -1:
        return res
    planner.planner.update_from_simulation()
    

    closing = np.cross(direction_to_shelf, [0, 0, 1.])
    neutral_pose_center = get_base_pose().sp.p + direction_to_shelf * 0.5 + np.array([0., 0., 0.7])
    neutral_pose = env.agent.build_grasp_pose(direction_to_shelf, closing, neutral_pose_center)
    res = planner.static_manipulation(neutral_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    res = planner.move_forward_delta(delta=0.2)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        'scene-0-[ENV#0]_SHELF_1_zone2.freezer_large1_door2_link', False
    )

    mesh = get_component_mesh(handle._bodies[0])
    obb: trimesh.primitives.Box = mesh.bounding_box_oriented
    grasp_center = get_obb_center(obb)

    grasp_approaching = np.array([-1., 1., 0.]) 
    grasp_approaching /= np.linalg.norm(grasp_approaching)
    grasp_closing = np.array([1., 1., 0.])
    grasp_closing /= np.linalg.norm(grasp_closing)
    open_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)

    open_pose = open_pose * sapien.Pose([ 0.0, -0.05, -0.05,])

    res = planner.static_manipulation(open_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        'scene-0-[ENV#0]_SHELF_1_zone2.freezer_large1_door2_link', True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        'scene-0-[ENV#0]_SHELF_1_zone2.freezer_large1_door2_handle_link', True
    )
    planner.planner.update_from_simulation()


    res = planner.static_manipulation(open_pose * sapien.Pose([0, 0.1, 0.05]), n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        return res

    # res = planner.rotate_z_delta(delta=-0.5)
    # if res == -1:
    #     return res
    # -------------------------------------------------------------------------- #
    # Go to shelf
    # -------------------------------------------------------------------------- #
    

    res = planner.idle_steps(t=10)

    planner.render_wait()
    return res
