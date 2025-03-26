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
import mplib
import sapien
import sys
import trimesh
from mplib.collision_detection import fcl
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube
from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils import common

from dsynth.envs.pick_to_cart import PickToCartEnv
from dsynth.planning.motionplanner import PandaArmMotionPlanningSolverV2, FetchArmMotionPlanningSolver


def compute_box_grasp_thin_side_info(
    obb: trimesh.primitives.Box,
    target_closing=None,
    ee_direction=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    inds = np.argsort(extents[:2])
    short_base_side_ind = inds[0]
    long_base_side_ind = inds[1]

    height = extents[2]

    approaching = np.array(T[:3, long_base_side_ind])
    approaching = common.np_normalize_vector(approaching)

    if ee_direction @ approaching < 0:
        approaching = -approaching

    closing = np.array(T[:3, short_base_side_ind])

    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    # Find the origin on the surface
    center = T[:3, 3]
    half_size = extents[long_base_side_ind] / 2
    center = center + approaching * (-half_size + min(depth, half_size))

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info

def solve_panda_ai360(env: PickCubeEnv, seed=None, debug=False, vis=False):
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
    
    for name, actor in env.actors['products'].items():
        if name != env.target_product_name:
            not_collide_obb = get_actor_obb(actor, vis=False)
            center_T = not_collide_obb.primitive.transform
            collision_extents = not_collide_obb.primitive.extents
            collision_pose = sapien.Pose(center_T)
            planner.add_box_collision(collision_extents, collision_pose)
    trimesh.points.PointCloud(planner.all_collision_pts).show()
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

    target_pose = target.pose.sp
    planner.planner.update_attached_box(target_extents, mplib.Pose(target_center_pose.p, target_center_pose.q), link_id=-1)
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


def solve_panda_pick_to_cart(env: PickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = PandaArmMotionPlanningSolverV2(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    ) 

    FINGER_LENGTH = 0.025
    env = env.unwrapped

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
    
    for name, actor in env.actors['products'].items():
        if name != env.target_product_name:
            not_collide_obb = get_actor_obb(actor, vis=False)
            center_T = not_collide_obb.primitive.transform
            collision_extents = not_collide_obb.primitive.extents
            collision_pose = sapien.Pose(center_T)

            # collision_cube = fcl.Box(collision_extents)
            # collision_object = fcl.CollisionObject(collision_cube, mplib.Pose(p=collision_pose.p, q=collision_pose.q))
            # planner.planner.planning_world.add_object(name, collision_object)

            planner.add_box_collision(collision_extents, collision_pose)
    
    for name, actor in env.actors['fixtures']['shelves'].items():
        shelf_mesh = env.assets_lib['fixtures.shelf'].trimesh_scene.geometry['object/shelf'].copy()
        T = actor.pose.sp.to_transformation_matrix()
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
    
    target_obb = get_actor_obb(target, vis=False)
    center_T = target_obb.primitive.transform
    target_extents = target_obb.primitive.extents
    target_pose = sapien.Pose(center_T)

    box = trimesh.creation.box(target_extents, transform=target_pose.to_transformation_matrix())
    pts, _ = trimesh.sample.sample_surface(box, 500)

    all_collision_pts = np.vstack([planner.all_collision_pts, pts])
    colors = np.zeros((all_collision_pts.shape[0], 4), dtype=np.uint8)
    colors[:, 3] = 127
    colors[:len(planner.all_collision_pts), 0] = 255
    colors[len(planner.all_collision_pts):, 1] = 255

    trimesh.points.PointCloud(all_collision_pts, colors).show(flags={'axis': True})
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_RRTConnect(reach_pose)

    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_RRTConnect(grasp_pose)
    res = planner.close_gripper()

    target_obb = get_actor_obb(target, vis=False)
    target_extents = target_obb.primitive.extents * 1.05
    target_center_pose = sapien.Pose(target_obb.primitive.transform)
    target_center_pose_wrt_tcp = env.agent.tcp.pose.inv() * target_center_pose
    tcp_wrt_target = target_center_pose.inv() * env.agent.tcp.pose.sp
    # target_pose = target.pose.sp
    planner.planner.update_attached_box(target_extents, 
            mplib.Pose(target_center_pose_wrt_tcp.p.cpu().numpy()[0], 
                       target_center_pose_wrt_tcp.q.cpu().numpy()[0]), 
            link_id=-1)
    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #

    lift_pose = grasp_pose * sapien.Pose([0.02, 0., 0.])
    res = planner.move_to_pose_with_screw(lift_pose)


    # res = planner.move_to_pose_with_RRTConnect(pre_goal_pose)
    
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #

    res = planner.move_to_pose_with_RRTConnect(goal_pose)
    res = planner.open_gripper()

   

    planner.close()
    return res

def solve_fetch_pick_cube(env: PickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = FetchArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
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

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    # reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    reach_pose = grasp_pose * sapien.Pose([0.30, 0, 0])
    reach_pose = env.agent.tcp.pose * sapien.Pose([0.0, 0, -0.20])
    planner.move_to_pose_with_screw(reach_pose)
    res = planner.close_gripper()
    planner.render_wait()
    return res
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
