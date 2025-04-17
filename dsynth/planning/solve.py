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

from dsynth.envs.pickcube_mptest import PickCubeEnvMPTest
from dsynth.envs.pick_to_cart import PickToCartEnv
from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
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
    convert_actor_convex_mesh_to_fcl
)

def solve_panda_pick_cube_test(env: PickCubeEnvMPTest, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolverV2(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    cube_center_T = obb.primitive.transform
    cube_extents = obb.primitive.extents
    cube_pose = sapien.Pose(cube_center_T)
    # cube_pose = env.cube.pose.sp
    planner.add_box_collision(cube_extents, cube_pose, 'cube')

    wall_obb = get_actor_obb(env.wall)
    wall_center_T = wall_obb.primitive.transform
    wall_extents = wall_obb.primitive.extents
    wall_pose = sapien.Pose(wall_center_T)
    # wall_pose = env.wall.pose.sp
    planner.add_box_collision(wall_extents, wall_pose, 'wall')

    # table_collider_obb = get_actor_obb(env.table_collider)
    # # table_collider_center_T = table_collider_obb.primitive.transform
    # table_collider_extents = table_collider_obb.extents
    # # table_collider_pose = sapien.Pose(table_collider_center_T)
    # table_collider_pose = env.table_collider.pose.sp
    # planner.add_box_collision(table_collider_extents, table_collider_pose, 'table_collider')
    

    # table = env.scene.actors['table-workspace']
    # table_mesh = get_component_mesh(
    #     table._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
    #     to_world_frame=True,
    # )
    # assert table_mesh is not None, "can not get actor mesh for table"
    # pts, _ = trimesh.sample.sample_surface(table_mesh, 5000)
    # planner.add_collision_pts(pts, 'table-workspace')

    # trimesh.points.PointCloud(planner.get_all_collision_pts()).show(flags={'axis': True})

    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.remove_collision_pts('cube')
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper()

    # TODO: wrong (no) orientation!!!!
    target_obb = get_actor_obb(env.cube)
    target_extents = target_obb.primitive.extents
    target_center_pose = sapien.Pose(target_obb.primitive.transform)
    # TODO:change to: - no :) 
    # target_extents = target_obb.extents
    # target_center_pose = env.cube.pose.sp
    
    target_center_pose_wrt_tcp = env.agent.tcp.pose.inv() * target_center_pose # seems correct
    tcp_wrt_target = target_center_pose.inv() * env.agent.tcp.pose.sp



    # idx = self.planner.user_link_names.index("panda_hand")
    # planner.planner.planning_world.attach_object(env.cube, planner.robot, -1)
    # planner.planner.update_attached_box(target_extents, 
    #         mplib.Pose(target_center_pose.p, 
    #                    target_center_pose.q), 
    #         link_id=-1)
    
    planner.planner.update_attached_box(target_extents, 
            mplib.Pose(target_center_pose_wrt_tcp.p.cpu().numpy()[0], 
                       target_center_pose_wrt_tcp.q.cpu().numpy()[0]), 
            link_id=-1)
    
    # planner.planner.update_attached_box(target_extents, 
    #         mplib.Pose(tcp_wrt_target.p, 
    #                    tcp_wrt_target.q), 
    #         link_id=-1)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)

    planner.open_gripper()

    planner.close()
    return res



def solve_panda_pick_cube_sapien_planning(env: PickCubeEnvMPTest, seed=None, debug=False, vis=False):
    # raise NotImplementedError
    env.reset(seed=seed)
    env = env.unwrapped
    planner = PandaArmMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        get_fcl_object_name(env.scene.actors['table-workspace']), 'scene-0-panda_wristcam_panda_link0', True
    ) #WTF?
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            "'scene-0-panda_wristcam_panda_rightfinger'", get_fcl_object_name(env.cube), True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            "'scene-0-panda_wristcam_panda_leftfinger'", get_fcl_object_name(env.cube), True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        get_fcl_object_name(env.scene.actors['table-workspace']), get_fcl_object_name(env.cube), True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        get_fcl_object_name(env.table_collider), get_fcl_object_name(env.cube), True
    )


    FINGER_LENGTH = 0.025

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTConnect(reach_pose)
    planner.planner.update_from_simulation()
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper()
    planner.planner.update_from_simulation()
    
    # idx = planner.planner.user_link_names.index("scene-0-panda_wristcam_panda_hand_tcp")
    # planner.planner.planning_world.attach_object(planner.planner.planning_world, env.cube, planner.robot, idx)
    
    
    kwargs = {"name": get_fcl_object_name(env.cube), "art_name": 'scene-0_panda_wristcam_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)
    planner.planner.update_from_simulation()

    planner.open_gripper()

    planner.close()
    return res


def solve_panda_pick_cube_fcl_test(env: PickCubeEnvMPTest, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolverV2(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    wall_obb = get_actor_obb(env.wall)
    wall_center_pose = sapien.Pose(wall_obb.primitive.transform)
    wall_extents = wall_obb.primitive.extents
    wall_fcl = fcl.Box(wall_extents)
    collision_wall = fcl.CollisionObject(wall_fcl, mplib.Pose(p=wall_center_pose.p, q=wall_center_pose.q))
    planner.planner.planning_world.add_object("wall", collision_wall)

    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper()


    target_obb = get_actor_obb(env.cube)
    target_extents = target_obb.primitive.extents
    target_center_pose = sapien.Pose(target_obb.primitive.transform)
    target_fcl = fcl.Box(target_extents)
    # attach_target = fcl.CollisionObject(target_fcl, mplib.Pose(p=target_center_pose.p, q=target_center_pose.q))

    
    target_center_pose_wrt_tcp = env.agent.tcp.pose.inv() * target_center_pose # seems correct
    # tcp_wrt_target = target_center_pose.inv() * env.agent.tcp.pose.sp


    planner.planner.update_attached_object(
        collision_geometry=target_fcl,
        pose=mplib.Pose(p=target_center_pose_wrt_tcp.sp.p, q=target_center_pose_wrt_tcp.sp.q),
        )

    
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)

    planner.open_gripper()

    planner.close()
    return res


def solve_panda_pick_cube_fcl_V2_test(env: PickCubeEnvMPTest, seed=None, debug=False, vis=False):
    env.reset(seed=seed)

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    objects = []
    for entity in [env.wall, env.cube]:
        component = entity._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
        assert component is not None, (
            f"No PhysxRigidBaseComponent found in {entity.name}: "
            f"{entity.components=}"
        )
        if (fcl_obj := SapienPlanningWorld.convert_physx_component(component)) is not None:  # type: ignore
            objects.append(fcl_obj)

    planner = PandaArmMotionPlanningSolverV2(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
        objects=objects
    )
    
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            "panda_rightfinger", get_fcl_object_name(env.cube), True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            "panda_leftfinger", get_fcl_object_name(env.cube), True
        )


    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper()

    kwargs = {"name": get_fcl_object_name(env.cube), "art_name": 'panda', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)

    planner.open_gripper()

    planner.close()
    return res


def solve_panda_ai360(env: PickToCartEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    # planner = FetchArmMotionPlanningSolver(
    #     env,
    #     debug=debug,
    #     vis=vis,
    #     base_pose=env.unwrapped.agent.robot.pose,
    #     visualize_target_grasp_pose=vis,
    #     print_env_info=False,
    # )
    planner = PandaArmMotionPlanningSolverV2(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    ) 

    FINGER_LENGTH = 0.025
    # env = env.unwrapped

    target = env.actors['products'][env.target_product_name]
    goal_pose = env.goal_zone.pose

    # retrieves the object oriented bounding box (trimesh box object)
    if target.get_collision_meshes():  # Ensure it has collision meshes
        obb = get_actor_obb(target, vis=False)  # Should now work correctly
    else:
        print("Error: Target has no collision meshes.")
    # retrieves the object oriented bounding box (trimesh box object)

    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_approaching = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    ee_direction = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    tcp_center = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()

    goal_closing = goal_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    goal_approaching = goal_pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()

    pre_goal_center = goal_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy() - np.array([0.1, -0.2,-0.4])
    goal_center = goal_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy() - np.array([0.0, 0.05, -0.2])

    init_pose = env.agent.build_grasp_pose(target_approaching, target_closing, tcp_center)
    pre_goal_pose = env.agent.build_grasp_pose(-goal_approaching, -goal_closing, pre_goal_center)
    goal_pose = env.agent.build_grasp_pose(-goal_approaching, -goal_closing, goal_center)


    # we can build a simple grasp pose using this information for Panda
    agent_pose = env.agent.robot.get_pose()
    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        target_closing=target_closing,
        ee_direction=ee_direction,
        depth=FINGER_LENGTH,
    )
    height = obb.primitive.extents[2]
    closing, center, approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_screw(reach_pose)

    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_RRTConnect(grasp_pose)
    res = planner.close_gripper()

    target_obb = get_actor_obb(target, vis=False)
    target_extents = target_obb.primitive.extents
    target_center_pose = sapien.Pose(target_obb.primitive.transform)

    
    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #

    lift_pose = grasp_pose * sapien.Pose([0.02, 0., 0.])
    res = planner.move_to_pose_with_screw(lift_pose)


    res = planner.move_to_pose_with_RRTConnect(pre_goal_pose)
    
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #

    res = planner.move_to_pose_with_RRTConnect(goal_pose)
    res = planner.open_gripper()

   

    planner.close()
    return res


def solve_panda_pick_to_cart(env: PickToCartEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    
    target = env.actors['products'][env.target_product_name]
    collision_objects = []
    for name, actor in env.actors['products'].items():
        if name != env.target_product_name: # TODO: remove if
            fcl_obj = convert_actor_convex_mesh_to_fcl(actor)
            collision_objects.append(fcl_obj)
            #====

            # if (fcl_obj := SapienPlanningWorld.convert_physx_component(component)) is not None:  # type: ignore
            #     objects.append(fcl_obj)

    planner = PandaArmMotionPlanningSolverV2(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
        objects=collision_objects
    )

    for name, actor in env.actors['fixtures']['shelves'].items():
        shelf_mesh = env.assets_lib['fixtures.shelf'].trimesh_scene.geometry['object/shelf'].copy()
        T = actor.pose.sp.to_transformation_matrix()
        
        # idk why, but shelves are rotated
        deg90 = 3.14 / 2
        rot_x_90 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(deg90), -np.sin(deg90), 0],
            [0, np.sin(deg90), np.cos(deg90), 0],
            [0, 0, 0, 1]
        ])
        shelf_mesh.apply_transform(T @ rot_x_90)

        pts, _ = trimesh.sample.sample_surface(shelf_mesh, 5000)
        planner.add_collision_pts(pts)

    goal_pose = env.goal_zone.pose

    # retrieves the object oriented bounding box (trimesh box object)
    if target.get_collision_meshes():  # Ensure it has collision meshes
        obb = get_actor_obb(target, vis=False)  # Should now work correctly
    else:
        print("Error: Target has no collision meshes.")
    # retrieves the object oriented bounding box (trimesh box object)

    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_approaching = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    ee_direction = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    tcp_center = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()

    goal_closing = goal_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    goal_approaching = goal_pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()

    pre_goal_center = goal_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy() - np.array([0.1, -0.2,-0.4])
    goal_center = goal_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy() - np.array([0.0, 0.05, -0.2])

    init_pose = env.agent.build_grasp_pose(target_approaching, target_closing, tcp_center)
    pre_goal_pose = env.agent.build_grasp_pose(-goal_approaching, -goal_closing, pre_goal_center)
    goal_pose = env.agent.build_grasp_pose(-goal_approaching, -goal_closing, goal_center)


    # we can build a simple grasp pose using this information for Panda
    agent_pose = env.agent.robot.get_pose()
    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        target_closing=target_closing,
        ee_direction=ee_direction,
        depth=FINGER_LENGTH,
    )
    height = obb.primitive.extents[2]
    closing, center, approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    
    
    target_obb = get_actor_obb(target, vis=False)
    center_T = target_obb.primitive.transform
    target_extents = target_obb.primitive.extents
    target_pose = sapien.Pose(center_T)

    box = trimesh.creation.box(target_extents, transform=target_pose.to_transformation_matrix())
    pts, _ = trimesh.sample.sample_surface(box, 500)

    # all_collision_pts = planner.get_all_collision_pts()
    # all_pts = np.vstack([all_collision_pts, pts])
    # colors = np.zeros((all_pts.shape[0], 4), dtype=np.uint8)
    # colors[:, 3] = 127
    # colors[:len(all_collision_pts), 0] = 255
    # colors[len(all_collision_pts):, 1] = 255

    # trimesh.points.PointCloud(all_pts, colors).show(flags={'axis': True})
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_RRTConnect(reach_pose)
    planner.update_from_simulation()

    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.update_from_simulation()

    fcl_target_obj = convert_actor_convex_mesh_to_fcl(target)
    planner.planner.planning_world.add_object(fcl_target_obj)
    
    kwargs = {"name": get_fcl_object_name(target), "art_name": 'panda', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            "scene_pcd", get_fcl_object_name(target), True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            "panda_rightfinger", get_fcl_object_name(target), True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            "panda_leftfinger", get_fcl_object_name(target), True
        )
    
    res = planner.close_gripper()
    planner.update_from_simulation()

    print("WTF1", target.pose)
    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #

    lift_pose = grasp_pose * sapien.Pose([0.03, 0., 0.])
    res = planner.move_to_pose_with_screw(lift_pose)
    planner.update_from_simulation()
    

    print("WTF2", target.pose)

    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            "scene_pcd", get_fcl_object_name(target), False
        )
    
    # res = planner.move_to_pose_with_RRTConnect(psre_goal_pose)
    
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #

    # planner.planner.planning_world.remove_object(get_fcl_object_name(target))
    # fcl_target_obj = convert_actor_convex_mesh_to_fcl(target)
    # planner.planner.planning_world.add_object(fcl_target_obj)
    # kwargs = {"name": get_fcl_object_name(target), "art_name": 'panda', "link_id": planner.planner.move_group_link_id}
    # planner.planner.planning_world.attach_object(**kwargs)

    res = planner.move_to_pose_with_screw(reach_pose * sapien.Pose([0.03, 0., 0.]))
    planner.update_from_simulation()

    res = planner.move_to_pose_with_RRTConnect(goal_pose)
    planner.update_from_simulation()

    res = planner.open_gripper()

   

    planner.close()
    return res

def solve_panda_pick_to_cart_sapien(env: PickToCartEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    
    target = env.actors['products'][env.target_product_name]
    collision_objects = []
    for name, actor in env.actors['products'].items():
        if name != env.target_product_name: # TODO: remove if
            fcl_obj = convert_actor_convex_mesh_to_fcl(actor)
            collision_objects.append(fcl_obj)
            #====

            # if (fcl_obj := SapienPlanningWorld.convert_physx_component(component)) is not None:  # type: ignore
            #     objects.append(fcl_obj)

    planner = PandaArmMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
        objects=collision_objects
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
            'scene-0_floor_room_0_20', True
        )

    goal_pose = env.goal_zone.pose

    # retrieves the object oriented bounding box (trimesh box object)
    if target.get_collision_meshes():  # Ensure it has collision meshes
        obb = get_actor_obb(target, vis=False)  # Should now work correctly
    else:
        print("Error: Target has no collision meshes.")
    # retrieves the object oriented bounding box (trimesh box object)

    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_approaching = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    ee_direction = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()
    tcp_center = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 3].cpu().numpy()

    goal_closing = goal_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    goal_approaching = goal_pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()

    pre_goal_center = goal_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy() - np.array([0.1, -0.2,-0.4])
    goal_center = goal_pose.to_transformation_matrix()[0, :3, 3].cpu().numpy() - np.array([0.0, -0.05, -0.35])

    init_pose = env.agent.build_grasp_pose(target_approaching, target_closing, tcp_center)
    pre_goal_pose = env.agent.build_grasp_pose(-goal_approaching, -goal_closing, pre_goal_center)
    goal_pose = env.agent.build_grasp_pose(-goal_approaching, -goal_closing, goal_center)


    # we can build a simple grasp pose using this information for Panda
    agent_pose = env.agent.robot.get_pose()
    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        target_closing=target_closing,
        ee_direction=ee_direction,
        depth=FINGER_LENGTH,
    )
    height = obb.primitive.extents[2]
    closing, center, approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    
    
    
    target_obb = get_actor_obb(target, vis=False)
    center_T = target_obb.primitive.transform
    target_extents = target_obb.primitive.extents
    target_pose = sapien.Pose(center_T)

    box = trimesh.creation.box(target_extents, transform=target_pose.to_transformation_matrix())
    pts, _ = trimesh.sample.sample_surface(box, 500)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    res = planner.move_to_pose_with_RRTConnect(reach_pose)
    planner.planner.update_from_simulation()

    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.planner.update_from_simulation()

    res = planner.close_gripper()
    planner.planner.update_from_simulation()

    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-panda_wristcam_panda_leftfinger', get_fcl_object_name(target), True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-panda_wristcam_panda_rightfinger', get_fcl_object_name(target), True
        )
    
    kwargs = {"name": get_fcl_object_name(target), "art_name": 'scene-0_panda_wristcam_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    print("WTF1", target.pose)
    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #

    lift_pose = grasp_pose * sapien.Pose([0.03, 0., 0.])
    res = planner.move_to_pose_with_RRTConnect(lift_pose)
    planner.planner.update_from_simulation()
    

    print("WTF2", target.pose)

    
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #


    res = planner.move_to_pose_with_screw(reach_pose * sapien.Pose([0.03, 0., 0.]))
    planner.planner.update_from_simulation()
    

    planner.planner.update_from_simulation()

    res = planner.move_to_pose_with_RRTConnect(goal_pose)
    planner.planner.update_from_simulation()

    res = planner.open_gripper()

   

    planner.close()
    return res




def solve_fetch_static_pick_cube(env: PickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = FetchStaticArmMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_static_r_wheel_link', 'scene-0_ground_31', True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_static_l_wheel_link', 'scene-0_ground_31', True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_static_base_link', 'scene-0_table-workspace_30', True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_static_laser_link', 'scene-0_table-workspace_30', True
        )


    FINGER_LENGTH = 0.07
    env = env.unwrapped
    
    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.02])

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTConnect(reach_pose)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    planner.move_to_pose_with_screw(grasp_pose)
    planner.planner.update_from_simulation()
    
    res = planner.close_gripper()
    planner.planner.update_from_simulation()
    
    kwargs = {"name": get_fcl_object_name(env.cube), "art_name": 'scene-0_ds_fetch_static_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    planner.move_to_pose_with_screw(reach_pose)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    planner.move_to_pose_with_RRTConnect(goal_pose)
    planner.planner.update_from_simulation()

    planner.render_wait()
    return res
    


def solve_fetch_quasi_static_pick_cube(env: PickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = FetchQuasiStaticArmMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_quasi_static_r_wheel_link', 'scene-0_ground_31', True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_quasi_static_l_wheel_link', 'scene-0_ground_31', True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_quasi_static_base_link', 'scene-0_table-workspace_30', True
        )
    
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_quasi_static_laser_link', 'scene-0_table-workspace_30', True
        )


    FINGER_LENGTH = 0.07
    env = env.unwrapped
    
    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.02])
    
    mask_wo_moving =[True, False, False, False, False, False, False, False, False, False, False, False, False]

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])

    planner.move_to_pose_with_RRTConnect(reach_pose, n_init_qpos=100)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    planner.move_to_pose_with_screw(grasp_pose)
    planner.planner.update_from_simulation()
    
    res = planner.close_gripper()
    planner.planner.update_from_simulation()
    
    kwargs = {"name": get_fcl_object_name(env.cube), "art_name": 'scene-0_ds_fetch_quasi_static_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    planner.move_to_pose_with_screw(reach_pose)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    planner.move_to_pose_with_RRTConnect(goal_pose)
    planner.planner.update_from_simulation()

    planner.render_wait()
    return res

def solve_fetch_pick_cube(env: PickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_r_wheel_link', 'scene-0_ground_31', True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_l_wheel_link', 'scene-0_ground_31', True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_base_link', 'scene-0_table-workspace_30', True
        )
    
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            'scene-0-ds_fetch_laser_link', 'scene-0_table-workspace_30', True
        )


    FINGER_LENGTH = 0.07
    env = env.unwrapped
    
    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.02])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    
    mask_wo_moving =[True, True, True, False, False, False, False, False, False, False, False, False, False, False, False]


    # -------------------------------------------------------------------------- #
    # Drive
    # -------------------------------------------------------------------------- #
    drive_pos = sapien.Pose(
        p = reach_pose.p - [1., 0, 0],
        q = env.agent.tcp.pose.sp.q
    )

    planner.drive_base(drive_pos, reach_pose)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    planner.move_base_forward_and_manipulation(reach_pose, n_init_qpos=100)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    planner.static_manipulation(grasp_pose)
    planner.planner.update_from_simulation()
    
    res = planner.close_gripper()
    planner.planner.update_from_simulation()
    
    kwargs = {"name": get_fcl_object_name(env.cube), "art_name": 'scene-0_ds_fetch_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    planner.static_manipulation(reach_pose)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    planner.static_manipulation(goal_pose)
    planner.planner.update_from_simulation()

    planner.render_wait()
    return res


def solve_fetch_pick_target_object(env: DarkstoreCellBaseEnv, seed=None, debug=False, vis=False):
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

    approaching = np.array([0, 0, -1])
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
    
    grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.02])
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    
    mask_wo_moving =[True, True, True, False, False, False, False, False, False, False, False, False, False, False, False]


    # -------------------------------------------------------------------------- #
    # Drive
    # -------------------------------------------------------------------------- #
    drive_pos = sapien.Pose(
        p = reach_pose.p - [0, 0.6, 0],
        q = env.agent.tcp.pose.sp.q
    )

    planner.drive_base(drive_pos, reach_pose)
    planner.planner.update_from_simulation()
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    planner.static_manipulation(reach_pose)
    # planner.move_base_x_and_manipulation(reach_pose, n_init_qpos=100)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #

    # planner.move_base_x_and_manipulation(grasp_pose)
    planner.static_manipulation(grasp_pose)
    planner.planner.update_from_simulation()
    
    res = planner.close_gripper()
    planner.planner.update_from_simulation()
    
    kwargs = {"name": get_fcl_object_name(target), "art_name": 'scene-0_ds_fetch_1', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0.02, 0., 0.])
    planner.static_manipulation(lift_pose, 200)
    planner.planner.update_from_simulation()

    # -------------------------------------------------------------------------- #
    # Pull
    # -------------------------------------------------------------------------- #

    planner.static_manipulation(reach_pose)
    planner.planner.update_from_simulation()

    planner.render_wait()
    return res
