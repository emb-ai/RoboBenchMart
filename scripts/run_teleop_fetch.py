import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv

import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro
from dataclasses import dataclass

from mplib.sapien_utils.srdf_exporter import export_srdf
from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf
from mplib.sapien_utils.conversion import ArticulatedModel

import sys
sys.path.append('.')
from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver
from dsynth.envs import *
from dsynth.robots import *
from dsynth.planning.utils import SapienPlanningWorldV2

#=============================
import numpy as np
import sapien
from sapien import internal_renderer as R
from sapien.utils.viewer.transform_window import TransformWindow
from sapien.utils.viewer.plugin import Plugin

class TransformWindowFetchStatic(TransformWindow):
    def compute_ik(self):
        if (
            self.selected_entity is not None
            and self.ik_enabled
            and self.get_articulation(self.selected_entity)
        ):
            link_idx = self.ik_articulation.get_links().index(
                next(
                    c
                    for c in self.selected_entity.components
                    if isinstance(c, sapien.physx.PhysxArticulationLinkComponent)
                )
            )
            mask = np.array(
                [
                    self.move_group_selection[j]
                    for j in range(len(self.move_group_joints))
                ]
            ).astype(int)
            mask[0] = mask[1] = mask[2] = mask[3] = 0
            pose = self.ik_articulation.pose.inv() * self._gizmo_pose
            result, success, error = self.pinocchio_model.compute_inverse_kinematics(
                link_idx,
                pose,
                initial_qpos=self.ik_articulation.get_qpos(),
                active_qmask=mask,
                max_iterations=100,
            )
            return result, success, error

#=============================



@dataclass
class Args:
    scene_dir: str = None
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickToCartOneProdEnv"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "ds_fetch_basket"
    """The robot to use. Robot setups supported for teleop in this script are ds_fetch_static and ds_fetch_basket_static"""
    record_dir: str = "demos"
    """directory to record the demonstration data and optionally videos"""
    save_video: bool = False
    """whether to save the videos of the demonstrations after collecting them all"""
    viewer_shader: str = "default"
    """the shader to use for the viewer. 'default' is fast but lower-quality shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
    video_saving_shader: str = "default"
    """the shader to use for the videos of the demonstrations. 'minimal' is the fast shader, 'rt' and 'rt-fast' are the ray tracing shaders"""

def parse_args() -> Args:
    return tyro.cli(Args)

def main(args: Args):
    output_dir = f"{args.scene_dir}/demos/teleop/"
    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uid,
        config_dir_path = args.scene_dir,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader)
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system"
    )
    num_trajs = 0
    seed = 0
    env.reset(seed=seed, options={'reconfigure': True})
    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
        code = solve(env, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed, options={'reconfigure': True})
            continue
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=False))
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print(f"Trajectories saved to {h5_file_path}")
    if args.save_video:
        print(f"Saving videos to {output_dir}")

        trajectory_data = h5py.File(h5_file_path)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="none",
            human_render_camera_configs=dict(shader_pack=args.video_saving_shader),
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=True,
            info_on_video=False,
            save_trajectory=False,
            video_fps=30
        )
        for episode in json_data["episodes"]:
            traj_id = f"traj_{episode['episode_id']}"
            data = trajectory_data[traj_id]
            env.reset(**episode["reset_kwargs"])
            env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

            env.base_env.set_state_dict(env_states_list[0])
            for action in np.array(data["actions"]):
                env.step(action)

        trajectory_data.close()
        env.close()
        del env



def solve(env: BaseEnv, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    robot_has_gripper = False
    robot_has_gripper = True
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_acc_limits=0.5,
        joint_vel_limits=0.5,
        disable_actors_collision=True,
    )
    viewer = env.render_human()

    last_checkpoint_state = None
    gripper_open = True
    def select_fetch_hand():
        viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "gripper_link")._objs[0].entity)

    viewer.plugins = viewer.plugins + [TransformWindowFetchStatic()]
    viewer.init_plugins(viewer.plugins)
    
    # select_fetch_hand()

    for plugin in viewer.plugins:
        if isinstance(plugin, TransformWindowFetchStatic):
            transform_window = plugin
    while True:

        transform_window.enabled = True
        # transform_window.update_ghost_objects
        # print(transform_window.ghost_objects, transform_window._gizmo_pose)
        # planner.grasp_pose_visual.set_pose(transform_window._gizmo_pose)

        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open (if there is a gripper)
            u: move the fetch hand up
            j: move the fetch hand down
            arrow_keys: move the fetch hand in the direction of the arrow keys
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost fetch arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data. Save trajectories and optionally videos.
            """)
            pass
        # elif viewer.window.key_press("k"):
        #     print("Saving checkpoint")
        #     last_checkpoint_state = env.get_state_dict()
        # elif viewer.window.key_press("l"):
        #     if last_checkpoint_state is not None:
        #         print("Loading previous checkpoint")
        #         env.set_state_dict(last_checkpoint_state)
        #     else:
        #         print("Could not find previous checkpoint")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        # elif viewer.window.key_press("r"):
        #     viewer.select_entity(None)
        #     return "restart"
        # elif viewer.window.key_press("t"):
        #     # TODO (stao): change from position transform to rotation transform
        #     pass
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g") and robot_has_gripper:
            if gripper_open:
                gripper_open = False
                _, reward, _ ,_, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _ ,_, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
        elif viewer.window.key_press("u"):
            select_fetch_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            select_fetch_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, +0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            select_fetch_hand()
            result = planner.move_forward_delta(delta=-0.05, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_moving_forward(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            select_fetch_hand()
            result = planner.move_forward_delta(delta=0.05, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_moving_forward(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            select_fetch_hand()
            result = planner.rotate_z_delta(delta=-0.05, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_rotation(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            select_fetch_hand()
            result = planner.rotate_z_delta(delta=0.05, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_rotation(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("z"):
            select_fetch_hand()
            result = planner.lift_hand(delta_h=0.05, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_path(result, refine=True)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("x"):
            select_fetch_hand()
            result = planner.lift_hand(delta_h=-0.05, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_path(result, refine=True)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            transform_window.update_ghost_objects()
        if execute_current_pose:
            # z-offset of end-effector gizmo to TCP position is hardcoded for the fetch robot here
            result = planner.move_to_pose_with_screw_static_body(transform_window._gizmo_pose, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_path(result, refine=True)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False
            transform_window.update_ghost_objects()



    return args
if __name__ == "__main__":
    main(parse_args())
