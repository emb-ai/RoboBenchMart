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
    BAD_ENV_ERROR_CODE,
    compute_box_grasp_thin_side_info,
    convert_actor_convex_mesh_to_fcl,
    is_mesh_cylindrical
)

def solve_fetch_pick_from_floor_cont(env: PickFromFloorContEnv, seed=None, debug=False, vis=False):
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

    target_pose = env.target_pose[0]
    target_center = target_pose.sp.p
    target_center[2] += 0.15

    # -------------------------------------------------------------------------- #
    # Setup target product
    # -------------------------------------------------------------------------- #

    target_product_actor = env.actors['products'][env.fallen_items[0]]
    obb = get_actor_obb(target_product_actor)
    target_product_center = get_obb_center(obb)
    
    # -------------------------------------------------------------------------- #
    # Go to the fallen item
    # -------------------------------------------------------------------------- #
    drive_vec = target_product_center - get_base_pose().sp.p
    drive_vec[2] = 0 
    
    if np.linalg.norm(drive_vec) > 0.5:
        drive_pos = get_base_pose().sp.p + drive_vec * (1 - 0.5 / np.linalg.norm(drive_vec)) 

        res = planner.drive_base(drive_pos)
        # if res == -1:
        #     return res
    else:
        res = planner.rotate_base_z(drive_vec)
        # if res == -1:
        #     return res
    
    res = planner.rotate_z_delta(0.1)
    # if res == -1:
    #     return res

    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Pick item
    # -------------------------------------------------------------------------- #
    obb = get_actor_obb(target_product_actor)
    target_product_center = get_obb_center(obb)

    grasp_info = compute_grasp_info_by_obb(obb,
                                  approaching=[0, 0, -1],
                                  target_closing=get_tcp_matrix()[:3, 1],
                                  depth=FINGER_LENGTH,)
    
    grasp_closing, grasp_center, grasp_approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.3])
    
    res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=100, disable_lift_joint=True)
    if res == -1:
        return res
    
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target_product_actor), True
    )

    res = planner.static_manipulation(grasp_pose, n_init_qpos=100, disable_lift_joint=False)
    if res == -1:
        return res

    planner.planner.update_from_simulation()

    res = planner.close_gripper()
    # if res == -1:
    #     return res
    
    kwargs = {"name": get_fcl_object_name(target_product_actor), "art_name": 'scene-0_ds_fetch_basket_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=100, disable_lift_joint=False)
    if res == -1:
        return res
    
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Go to start pose
    # -------------------------------------------------------------------------- #
    actor_shelf_name = env.active_shelves[0][0]
    shelf_pose = env.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp
    origin = shelf_pose.p - 1.4 * env.directions_to_shelf[0]

    res = planner.drive_base(origin)
    # if res == -1:
    #     return res
    
    view_to_target = target_center - get_base_pose().sp.p
    view_to_target[2] = 0.

    res = planner.rotate_base_z(view_to_target)
    # if res == -1:
    #     return res
    
    # -------------------------------------------------------------------------- #
    # Lift hand
    # -------------------------------------------------------------------------- #
    lift_ee_center = get_tcp_pose().sp.p
    lift_ee_center[2] = target_center[2]
    lift_approaching = view_to_target / np.linalg.norm(view_to_target)
    lift_closing = np.cross(lift_approaching, [0, 0, 1])
    lift_pose = env.agent.build_grasp_pose(lift_approaching, lift_closing, lift_ee_center)

    lift_pose = lift_pose * sapien.Pose([0, 0, 0.3])

    res = planner.static_manipulation(lift_pose, n_init_qpos=100, disable_lift_joint=False)
    if res == -1:
        return res
    
    # -------------------------------------------------------------------------- #
    # Move to shelf
    # -------------------------------------------------------------------------- #
    tcp_pose = get_tcp_pose().sp
    drive_vec = target_center - get_tcp_center()
    drive_vec[2] = 0
    drive_delta = drive_vec * (1 - 0.2 / np.linalg.norm(drive_vec))
    drive_pos = get_base_pose().sp.p + drive_delta 
    
    res = planner.drive_base(drive_pos)
    # if res == -1:
    #     return res
    
    res = planner.static_manipulation(tcp_pose * sapien.Pose([0, 0, np.linalg.norm(drive_delta)]), n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Place
    # -------------------------------------------------------------------------- #
    target_approaching = env.directions_to_shelf[0]
    target_closing = np.cross(target_approaching, [0, 0, 1])

    target_pose = env.agent.build_grasp_pose(target_approaching, target_closing, target_center)

    res = planner.static_manipulation(target_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return res
    planner.planner.update_from_simulation()

    res = planner.open_gripper()
    res = planner.idle_steps(t=10)

    planner.render_wait()
    return res


