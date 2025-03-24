import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube
import mplib
import sys
from mplib.collision_detection import fcl
sys.path.append('.')
from dsynth.envs.pick_to_cart import PickToCartEnv
from dsynth.planning.motionplanner import PandaArmMotionPlanningSolverV2

OPEN = 1
CLOSED = -1

MP_SOLUTIONS = {
    "PickCube-v1": solvePickCube,
    "StackCube-v1": solveStackCube,
    "PegInsertionSide-v1": solvePegInsertionSide,
    "PlugCharger-v1": solvePlugCharger,
    "PushCube-v1": solvePushCube,
    "PullCubeTool-v1": solvePullCubeTool,
    "LiftPegUpright-v1": solveLiftPegUpright,
    "PullCube-v1": solvePullCube

}




#===========================


import numpy as np
import sapien

from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)



from mani_skill.utils import common
import trimesh

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




#===========================
import mplib

class FetchArmMotionPlanningSolver(PandaArmMotionPlanningSolverV2):
    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        # link_names = [
        #     'torso_lift_link',
        #     'shoulder_pan_link',
        #     'shoulder_lift_link',
        #     'upperarm_roll_link',
        #     'elbow_flex_link',
        #     'forearm_roll_link',
        #     'wrist_flex_link',
        #     'wrist_roll_link',
        #     'gripper_link'
        # ]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        # joint_names = ['torso_lift_joint', 
        #                'shoulder_lift_joint', 
        #                'upperarm_roll_joint', 
        #                'elbow_flex_joint', 
        #                'forearm_roll_joint', 
        #                'wrist_flex_joint', 
        #                'wrist_roll_joint']
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="gripper_link",
            joint_vel_limits=np.ones(11) * self.joint_vel_limits,
            joint_acc_limits=np.ones(11) * self.joint_acc_limits,
            verbose=True
        )
        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
            body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
            gripper = self.env_agent.controller.controllers['gripper'].qpos[0].cpu().numpy()[0]
            base_vel = np.array([0, 0])

            qpos = result["position"][min(i, n_step - 1)]

            qpos_dict = {}
            for idx, q in zip(self.planner.move_group_joint_indices, qpos):
                joint_name = self.planner.user_joint_names[idx]
                qpos_dict[joint_name] = q
            
            for n, joint_name in enumerate(self.env_agent.controller.controllers['arm'].config.joint_names):
                arm_action[n] = qpos_dict[joint_name]
            
            body_action[2] = qpos_dict['torso_lift_joint']

            assert self.control_mode == "pd_joint_pos"
            action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info
    
    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state = CLOSED):
        self.gripper_state = gripper_state
        arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
        base_vel = np.array([0, 0])

        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
            else:
                raise NotImplementedError
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info
    
    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p , q=pose.q)
        result = self.planner.plan_screw(
            mplib.Pose(pose.p, pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            verbose=True,
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                mplib.Pose(pose.p, pose.q),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                # # use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

def solve_panda(env: PickCubeEnv, seed=None, debug=False, vis=False):
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


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    return parser.parse_args()

def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = 'PickToCartEnv'# args.env_id
    # env_id = 'PickCube-v1'
    # env = gym.make(
    #     env_id,
    #     robot_uids='fetch',
    #     obs_mode=args.obs_mode,
    #     control_mode="pd_joint_pos",
    #     render_mode=args.render_mode,
    #     sensor_configs=dict(shader_pack=args.shader),
    #     human_render_camera_configs=dict(shader_pack=args.shader),
    #     viewer_camera_configs=dict(shader_pack=args.shader),
    #     sim_backend=args.sim_backend
    # )
    scene_dir = 'generated_envs/mp_test/'
    scene_dir = 'generated_envs/one_milk/'
    record_dir = scene_dir + '/demos'
    env = gym.make(env_id, 
                   robot_uids='panda_wristcam', 
                   config_dir_path = scene_dir,
                   num_envs=1, 
                   sim_backend=args.sim_backend,
                   control_mode="pd_joint_pos",
                   viewer_camera_configs={'shader_pack': args.shader}, 
                    human_render_camera_configs={'shader_pack': args.shader},
                    sensor_configs={'shader_pack': args.shader},
                #    render_mode="human" if gui else "rgb_array", 
                   render_mode="rgb_array", 
                #    control_mode='pd_ee_delta_pos',
                   enable_shadow=True,
                   obs_mode='rgbd',
                   parallel_in_single_scene = False,
                   )
    # if env_id not in MP_SOLUTIONS:
    #     raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS.keys())}")

    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = RecordEpisode(
        env,
        output_dir=osp.join(record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name, save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        record_reward=False,
        save_on_reset=False
    )
    output_h5_path = env._h5_file.filename
    # solve = MP_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    while True:
        res = solve_panda(env, seed=seed, debug=True, vis=True if args.vis else False)
        # res = solve_fetch_pick_cube(env, seed=seed, debug=True, vis=True if args.vis else False)


        # try:
        # except Exception as e:
        #     print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
        #     res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        successes.append(success)
        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue
        else:
            env.flush_trajectory()
            if args.save_video:
                env.flush_video()
            pbar.update(1)
            pbar.set_postfix(
                dict(
                    success_rate=np.mean(successes),
                    failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                    avg_episode_length=np.mean(solution_episode_lengths),
                    max_episode_length=np.max(solution_episode_lengths),
                    # min_episode_length=np.min(solution_episode_lengths)
                )
            )
            seed += 1
            passed += 1
            if passed == args.num_traj:
                break
    env.close()
    return output_h5_path

def main(args):
    if args.num_procs > 1 and args.num_procs < args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        # Merge trajectory files
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        _main(args)

if __name__ == "__main__":
    # start = time.time()
    mp.set_start_method("spawn")
    main(parse_args())
    # print(f"Total time taken: {time.time() - start}")
