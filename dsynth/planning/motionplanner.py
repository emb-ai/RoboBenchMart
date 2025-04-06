import mplib
import numpy as np
import sapien
import trimesh
import sapien.physx as physx
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.panda.motionplanner import (
    build_panda_gripper_grasp_pose_visual,
    PandaArmMotionPlanningSolver
)
from mplib.collision_detection.fcl import FCLObject
from mplib.sapien_utils.conversion import convert_object_name
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib.pymp import ArticulatedModel

from dsynth.planning.utils import SapienPlanningWorldV2

OPEN = 1
CLOSED = -1

class PandaArmMotionPlanningSolverV2(PandaArmMotionPlanningSolver):
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        objects = [],
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self._sim_scene: sapien.Scene = self.base_env.scene.sub_scenes[0]
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner(objects)
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_panda_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def setup_planner(self, objects = []):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
            objects=objects
        )
        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner
    
    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0, mask=None
    ):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = mplib.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_pose(
            pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            # use_point_cloud=self.use_point_cloud,
            wrt_world=True,
            verbose=True,
            planning_time=2,
            rrt_range=0.1,
            simplify=True,
            mask=mask,
            
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

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
            verbose=True
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
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(t):
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

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose, name='scene_pcd'):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 500)
        if self.all_collision_pts is None:
            self.all_collision_pts = {name: pts}
        else:
            self.all_collision_pts[name] = pts
        self.planner.update_point_cloud(self.all_collision_pts[name], resolution=1e-2, name=name)

    def remove_collision_pts(self, name):
        del self.all_collision_pts[name]
        self.planner.remove_point_cloud(name)

    def add_collision_pts(self, pts: np.ndarray, name='scene_pcd'):
        if self.all_collision_pts is None:
            self.all_collision_pts = {name: pts}
        else:
            # self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
            self.all_collision_pts[name] = pts
        self.planner.update_point_cloud(self.all_collision_pts[name], resolution=1e-2, name=name)
    
    def get_all_collision_pts(self):
        all_points = [pts for pts in self.all_collision_pts.values()]
        return np.vstack(all_points)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass


class PandaArmMotionPlanningSapienSolver(PandaArmMotionPlanningSolverV2):
    def setup_planner(self, objects = []):
        # raise NotImplementedError
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(self._sim_scene, [planned_articulation])
        planner = SapienPlanner(
            planning_world,
            "scene-0-panda_wristcam_panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits
        )
        
        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner

class FetchStaticArmMotionPlanningSapienSolver(PandaArmMotionPlanningSolverV2):
    def setup_planner(self, *args, **kwargs):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(self._sim_scene, [planned_articulation])
        planner = SapienPlanner(
            planning_world,
            "scene-0-ds_fetch_static_gripper_link",
            joint_vel_limits=np.ones(8) * self.joint_vel_limits,
            joint_acc_limits=np.ones(8) * self.joint_acc_limits
        )
        
        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner


        return planner

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
            body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
            gripper = self.env_agent.controller.controllers['gripper'].qpos[0].cpu().numpy()[0]
            
            print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            # print("Arm: ", np.round(self.robot.get_qpos().cpu().numpy()[0][self.env_agent.controller.controllers['arm'].active_joint_indices], 4))
            qpos = result["position"][min(i, n_step - 1)]
            qvel = result["velocity"][min(i, n_step - 1)]

            qpos_dict = {}
            for idx, q in zip(self.planner.move_group_joint_indices, qpos):
                joint_name = self.planner.user_joint_names[idx]
                qpos_dict[joint_name] = q
            print("qpos mp", np.round(qpos, 4))
            for n, joint_name in enumerate(self.env_agent.controller.controllers['arm'].config.joint_names):
                arm_action[n] = qpos_dict[f'scene-0-ds_fetch_static_{joint_name}']
            
            # body_action[2] = qpos_dict['scene-0-ds_fetch_static_torso_lift_joint'] - body_action[2]
            # body_action[2] *= 10.
            body_action[2] = qpos_dict['scene-0-ds_fetch_static_torso_lift_joint']

            base_vel = np.array([0., 0.])
            # base_vel[0] = np.sqrt(qvel[0] ** 2 + qvel[1] ** 2)

            phi = self.robot.get_qpos().cpu().numpy()[0, 2]
            base_vel[0] = qvel[0] * np.cos(phi) + qvel[1] * np.sin(phi)

            base_vel[1] = qvel[2]

            assert self.control_mode == "pd_joint_pos"
            # action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
            action = np.hstack([arm_action, self.gripper_state, body_action])
            print("arm Action:", np.round(arm_action, 4))
            print("body Action:", np.round(body_action, 4))
            # print("base Action:", np.round(base_vel, 4))

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
                # action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
                action = np.hstack([arm_action, self.gripper_state, body_action])
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
