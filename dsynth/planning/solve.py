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
import sapien.physx as physx
import sys
import trimesh
from mplib.collision_detection import fcl
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
from dsynth.planning.motionplanner import (
    PandaArmMotionPlanningSolverV2, 
    PandaArmMotionPlanningSapienSolver,
    FetchArmMotionPlanningSolver
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


from mplib.sapien_utils.conversion import convert_object_name
from mplib.collision_detection.fcl import CollisionGeometry
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from sapien import Entity
from sapien.physx import (
    PhysxArticulation,
    PhysxArticulationLinkComponent
)
from typing import Literal, Optional, Sequence, Union

def attach_object(  # type: ignore
    planning_world: SapienPlanningWorld,
    obj: Union[Entity, str],
    articulation: Union[PhysxArticulation, str],
    link: Union[PhysxArticulationLinkComponent, int],
    pose: Optional[mplib.Pose] = None,
    *,
    touch_links: Optional[list[Union[PhysxArticulationLinkComponent, str]]] = None,
    obj_geom: Optional[CollisionGeometry] = None,
) -> None:
    """
    Attaches given non-articulated object to the specified link of articulation.

    Updates ``acm_`` to allow collisions between attached object and touch_links.

    :param obj: the non-articulated object (or its name) to attach
    :param articulation: the planned articulation (or its name) to attach to
    :param link: the link of the planned articulation (or its index) to attach to
    :param pose: attached pose (relative pose from attached link to object).
        If ``None``, attach the object at its current pose.
    :param touch_links: links (or their names) that the attached object touches.
        When ``None``,

        * if the object is not currently attached, touch_links are set to the name
        of articulation links that collide with the object in the current state.

        * if the object is already attached, touch_links of the attached object
        is preserved and ``acm_`` remains unchanged.
    :param obj_geom: a CollisionGeometry object representing the attached object.
        If not ``None``, pose must be not ``None``.

    .. raw:: html

        <details>
        <summary><a>Overloaded
        <code class="docutils literal notranslate">
        <span class="pre">PlanningWorld.attach_object()</span>
        </code>
        methods</a></summary>
    .. automethod:: mplib.PlanningWorld.attach_object
        :no-index:
    .. raw:: html
        </details>
    """
    kwargs = {"name": obj, "art_name": articulation, "link_id": link}
    if pose is not None:
        kwargs["pose"] = pose
    if touch_links is not None:
        kwargs["touch_links"] = [
            l.name if isinstance(l, PhysxArticulationLinkComponent) else l
            for l in touch_links  # noqa: E741
        ]
    if obj_geom is not None:
        kwargs["obj_geom"] = obj_geom

    if isinstance(obj, Entity):
        kwargs["name"] = convert_object_name(obj)
    if isinstance(articulation, PhysxArticulation):
        kwargs["art_name"] = articulation = convert_object_name(articulation)
    if isinstance(link, PhysxArticulationLinkComponent):
        kwargs["link_id"] = (
            planning_world.get_articulation(articulation)
            .get_pinocchio_model()
            .get_link_names()
            .index(link.name)
        )

    planning_world.attach_object(**kwargs)

def solve_panda_pick_cube_sapien_planning(env: PickCubeEnvMPTest, seed=None, debug=False, vis=False):
    raise NotImplementedError
    env.reset(seed=seed)
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

    idx = planner.planner.user_link_names.index("panda_hand_tcp")
    attach_object(planner.planner.planning_world, env.cube, planner.robot, idx)
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)

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
            "panda_rightfinger", "scene-0_cube_18", True
        )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
            "panda_leftfinger", "scene-0_cube_18", True
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

    kwargs = {"name": 'scene-0_cube_18', "art_name": 'panda', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)

    planner.open_gripper()

    planner.close()
    return res



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
