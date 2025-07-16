import mplib
import numpy as np
from collections import deque
import sapien
import trimesh
import sapien.physx as physx
from transforms3d.euler import euler2quat, euler2mat
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

from dsynth.planning.utils import SapienPlanningWorldV2, SapienPlannerV2

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
            mask=mask   
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
        disable_actors_collision=False
    ):
        self.disable_actors_collision = disable_actors_collision
        super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits, objects)
        
    def setup_planner(self, objects = []):
        # raise NotImplementedError
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(self._sim_scene, [planned_articulation], disable_actors_collision=self.disable_actors_collision)
        planner = SapienPlannerV2(
            planning_world,
            "scene-0-panda_wristcam_panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits
        )
        
        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner
    
    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0, mask=None, n_init_qpos=20
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
            n_init_qpos=n_init_qpos
            
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)


class FetchStaticArmMotionPlanningSapienSolver(PandaArmMotionPlanningSapienSolver):
    def setup_planner(self, *args, **kwargs):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(self._sim_scene, [planned_articulation], disable_actors_collision=self.disable_actors_collision)
        planner = SapienPlannerV2(
            planning_world,
            f"scene-0-{self.robot.name}_gripper_link",
            joint_vel_limits=np.ones(8) * self.joint_vel_limits,
            joint_acc_limits=np.ones(8) * self.joint_acc_limits
        )
        
        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner

    def move_to_pose_with_screw_static_body(
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
            masked_joints=[False] + [True] * 11
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                mplib.Pose(pose.p, pose.q),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                masked_joints=[False] + [True] * 11

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
    
    def follow_path(self, result, refine_steps:int = 0, refine: bool = False):
        return self.follow_forward_path_w_refinement(result, refine)
    
    def lift_hand(self, delta_h = 0., dry_run: bool = False, refine_steps: int = 0):
        cur_pose = self.base_env.agent.tcp.pose.sp
        taget_pose = mplib.Pose(p=cur_pose.p + np.array([0., 0., delta_h]),
                                q=cur_pose.q)
        result = self.planner.plan_screw(
            taget_pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            verbose=True
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)


    def follow_forward_path_w_refinement(self, result, refine: bool = False, static=False):
        qpos_final = result["position"][-1]
        qpos_dict_final = {}
        for idx, q in zip(self.planner.move_group_joint_indices, qpos_final):
            joint_name = self.planner.user_joint_names[idx]
            qpos_dict_final[joint_name] = q
            
        n_step = result["position"].shape[0]

        for i in range(n_step):
            arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()

            qpos = result["position"][min(i, n_step - 1)]
            qvel = result["velocity"][min(i, n_step - 1)]

            qpos_dict = {}
            
            for idx, q in zip(self.planner.move_group_joint_indices, qpos):
                joint_name = self.planner.user_joint_names[idx]
                qpos_dict[joint_name] = q

            for n, joint_name in enumerate(self.env_agent.controller.controllers['arm'].config.joint_names):
                arm_action[n] = qpos_dict[f'scene-0-{self.robot.name}_{joint_name}']

            assert self.control_mode == "pd_joint_pos"

            body_action = np.zeros_like(self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy())
            body_action[2] = qpos_dict[f'scene-0-{self.robot.name}_torso_lift_joint']

            # base_action = np.array([0., 0.])
            # base_action[0] =  np.sqrt(qvel[0] ** 2 + qvel[1] ** 2)

            action = np.hstack([arm_action, self.gripper_state, body_action])
            print("arm Action:", np.round(arm_action, 4))
            print("body Action:", np.round(body_action, 4))
            # print("base Action:", np.round(base_action, 4))
            print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        if refine:
            # REFINEMENT!
            passed_refine_steps = 0
            last_lift_poses = deque(maxlen=10)
            last_x_base_poses = deque(maxlen=10)
            last_lift_vels = deque(maxlen=10)
            last_x_base_vels = deque(maxlen=10)
            print("==== REFINEMENT ====")
    
            while not self.check_body_close_to_target(qpos_dict_final):
                if (len(last_lift_vels) > 4 and np.std(last_lift_vels) < 1e-3) \
                        and (len(last_x_base_vels) > 4 and np.std(last_x_base_vels) < 1e-3) \
                        and (len(last_lift_poses) > 4 and np.std(last_lift_poses) < 1e-3) \
                        and (len(last_x_base_poses) > 4 and np.std(last_x_base_poses) < 1e-3):
                    # robot is stuck
                    print("Robot is stuck")
                    break

                body_action = np.zeros_like(self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy())
                body_action[2] = qpos_dict_final[f'scene-0-{self.robot.name}_torso_lift_joint']
                body_action[0] = body_action[1] = 0.

                # base_action = np.array([0., 0.])
                                    
                last_lift_poses.append(self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()[2])
                last_lift_vels.append(self.env_agent.controller.controllers['body'].qvel[0].cpu().numpy()[2])
                
                action = np.hstack([arm_action, self.gripper_state, body_action])
                print("arm Action:", np.round(arm_action, 4))
                print("body Action:", np.round(body_action, 4))
                # print("base Action:", np.round(base_action, 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
                obs, reward, terminated, truncated, info = self.env.step(action)
                passed_refine_steps += 1
                self.elapsed_steps += 1
                if self.print_env_info:
                    print(
                        f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                    )
                if self.vis:
                    self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def check_body_close_to_target(self, target_dict, eps=1e-3):
        body_qpos = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()[2]
        target_lift_joint_height = target_dict[f'scene-0-{self.robot.name}_torso_lift_joint']

        # base_xy = self.env_agent.controller.controllers['base'].qpos[0].cpu().numpy()[0:2]
        # target_base = np.array([
        #     target_dict[f'scene-0-{self.robot.name}_root_x_axis_joint'],
        #     target_dict[f'scene-0-{self.robot.name}_root_y_axis_joint']
        # ])

        robot_qpos = self.robot.get_qpos().cpu().numpy()[0]
        arm_pos = robot_qpos[self.env_agent.controller.controllers['arm'].active_joint_indices.cpu().numpy()]
        target_arm_pos = np.array([
            target_dict[f'scene-0-{self.robot.name}_shoulder_pan_joint'],
            target_dict[f'scene-0-{self.robot.name}_shoulder_lift_joint'],
            target_dict[f'scene-0-{self.robot.name}_upperarm_roll_joint'],
            target_dict[f'scene-0-{self.robot.name}_elbow_flex_joint'],
            target_dict[f'scene-0-{self.robot.name}_forearm_roll_joint'],
            target_dict[f'scene-0-{self.robot.name}_wrist_flex_joint'],
            target_dict[f'scene-0-{self.robot.name}_wrist_roll_joint']
        ])
        return np.allclose(body_qpos, target_lift_joint_height, atol=eps) and \
            np.allclose(arm_pos, target_arm_pos, atol=eps)

    
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
    
class FetchQuasiStaticArmMotionPlanningSapienSolver(PandaArmMotionPlanningSapienSolver):
    def setup_planner(self, *args, **kwargs):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(self._sim_scene, [planned_articulation], disable_actors_collision=self.disable_actors_collision)
        planner = SapienPlannerV2(
            planning_world,
            "scene-0-ds_fetch_quasi_static_gripper_link",
            joint_vel_limits=np.ones(9) * self.joint_vel_limits,
            joint_acc_limits=np.ones(9) * self.joint_acc_limits
        )
        
        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        qpos_final = result["position"][-1]
        qpos_dict_final = {}
        for idx, q in zip(self.planner.move_group_joint_indices, qpos_final):
            joint_name = self.planner.user_joint_names[idx]
            qpos_dict_final[joint_name] = q
            
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()

            qpos = result["position"][min(i, n_step - 1)]

            qpos_dict = {}

            for idx, q in zip(self.planner.move_group_joint_indices, qpos):
                joint_name = self.planner.user_joint_names[idx]
                qpos_dict[joint_name] = q

            for n, joint_name in enumerate(self.env_agent.controller.controllers['arm'].config.joint_names):
                arm_action[n] = qpos_dict[f'scene-0-ds_fetch_quasi_static_{joint_name}']

            assert self.control_mode == "pd_joint_pos"

            body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
            body_action[2] = qpos_dict['scene-0-ds_fetch_quasi_static_torso_lift_joint']
            body_action[0] = body_action[1] = 0.

            base_action = self.env_agent.controller.controllers['base'].qpos[0].cpu().numpy()
            base_action[0] = qpos_dict['scene-0-ds_fetch_quasi_static_root_x_axis_joint']

            action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
            print("arm Action:", np.round(arm_action, 4))
            print("body Action:", np.round(body_action, 4))
            print("base Action:", np.round(base_action, 4))
            print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
            
        # REFINEMENT!
        # We refine only x position and lift at the end of the trajectory
        passed_refine_steps = 0
        last_lift_poses = deque(maxlen=10)
        last_x_base_poses = deque(maxlen=10)
        last_lift_vels = deque(maxlen=10)
        last_x_base_vels = deque(maxlen=10)
        print("==== REFINEMENT ====")
        while not self.check_body_base_close_to_target(qpos_dict_final):
            if (len(last_lift_vels) > 4 and np.std(last_lift_vels) < 1e-3) \
                    and (len(last_x_base_vels) > 4 and np.std(last_x_base_vels) < 1e-3) \
                    and (len(last_lift_poses) > 4 and np.std(last_lift_poses) < 1e-3) \
                    and (len(last_x_base_poses) > 4 and np.std(last_x_base_poses) < 1e-3):
                # robot is stuck
                print("Robot is stuck")
                break

            body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
            body_action[2] = qpos_dict_final['scene-0-ds_fetch_quasi_static_torso_lift_joint']
            body_action[0] = body_action[1] = 0.

            base_action = self.env_agent.controller.controllers['base'].qpos[0].cpu().numpy()
            base_action[0] = qpos_dict_final['scene-0-ds_fetch_quasi_static_root_x_axis_joint']
                                
            last_lift_poses.append(self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()[2])
            last_x_base_poses.append(self.env_agent.controller.controllers['base'].qpos[0].cpu().numpy()[0])

            last_lift_vels.append(self.env_agent.controller.controllers['body'].qvel[0].cpu().numpy()[2])
            last_x_base_vels.append(self.env_agent.controller.controllers['base'].qvel[0].cpu().numpy()[0])

            action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
            print("arm Action:", np.round(arm_action, 4))
            print("body Action:", np.round(body_action, 4))
            print("base Action:", np.round(base_action, 4))
            print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)
            passed_refine_steps += 1
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def check_body_base_close_to_target(self, target_dict, eps=1e-2):
        body_qpos = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()[2]
        target_lift_joint_height = target_dict['scene-0-ds_fetch_quasi_static_torso_lift_joint']

        base_x = self.env_agent.controller.controllers['base'].qpos[0].cpu().numpy()[0]
        target_base_x = target_dict['scene-0-ds_fetch_quasi_static_root_x_axis_joint']
        
        robot_qpos = self.robot.get_qpos().cpu().numpy()[0]
        arm_pos = robot_qpos[self.env_agent.controller.controllers['arm'].active_joint_indices.cpu().numpy()]
        target_arm_pos = np.array([
            target_dict['scene-0-ds_fetch_quasi_static_shoulder_pan_joint'],
            target_dict['scene-0-ds_fetch_quasi_static_shoulder_lift_joint'],
            target_dict['scene-0-ds_fetch_quasi_static_upperarm_roll_joint'],
            target_dict['scene-0-ds_fetch_quasi_static_elbow_flex_joint'],
            target_dict['scene-0-ds_fetch_quasi_static_forearm_roll_joint'],
            target_dict['scene-0-ds_fetch_quasi_static_wrist_flex_joint'],
            target_dict['scene-0-ds_fetch_quasi_static_wrist_roll_joint']
        ])
        return np.allclose(body_qpos, target_lift_joint_height, atol=eps) and \
            np.allclose(base_x, target_base_x, atol=eps) and \
            np.allclose(arm_pos, target_arm_pos, atol=eps)

    
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
        base_action = np.zeros_like(self.env_agent.controller.controllers['base'].qpos[0].cpu().numpy())

        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                # action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
                action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
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

class FetchMotionPlanningSapienSolver(PandaArmMotionPlanningSapienSolver):
    MAX_REFINE_STEPS = 200
    
    def setup_planner(self, *args, **kwargs):
        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(self._sim_scene, [planned_articulation], disable_actors_collision=self.disable_actors_collision)
        planner = SapienPlannerV2(
            planning_world,
            f"scene-0-{self.robot.name}_gripper_link",
            joint_vel_limits=np.ones(11) * self.joint_vel_limits,
            joint_acc_limits=np.ones(11) * self.joint_acc_limits
        )
        
        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner
    
    def rotate_base_z(self, new_direction, n_init_qpos=20, dry_run=False, rotate_recalculation_enabled=True):
        assert new_direction[2] == 0.
        tcp_pose = self.base_env.agent.tcp.pose.sp
        base_link_pose = self.base_env.agent.base_link.pose.sp
        base_x_axis = base_link_pose.to_transformation_matrix()[:3, 0]

        angle = np.arccos(np.dot(new_direction, base_x_axis) / np.linalg.norm(base_x_axis) / np.linalg.norm(new_direction))
        if np.cross(base_x_axis, new_direction)[2] < 0:
            angle = -angle
        
        if np.abs(angle) < 1e-3:
            return self.idle_steps(t=1)

        rotation_wrt_base_link = sapien.Pose(q=euler2quat(0, 0, angle))
        target_tcp_pose = base_link_pose * rotation_wrt_base_link * base_link_pose.inv() * tcp_pose

        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target_tcp_pose)
        target_tcp_pose = mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q)

        result = self.planner.plan_screw(
            mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
        )

        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        
        if not rotate_recalculation_enabled:
            if dry_run:
                return result
            self.render_wait()
            return self.follow_rotation(result)

        self.render_wait()
        res = self.follow_rotation(result)

        result = self.planner.plan_screw(
            mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
        )
        
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1

        if dry_run:
            return result

        return self.follow_rotation(result)
    
    def drive_base(self, target_pos=None, target_view_vec=None):
        if not target_pos is None:
            moving_direction = target_pos - self.base_env.agent.base_link.pose.sp.p
            moving_direction[2] = 0.

            if np.linalg.norm(moving_direction) < 1e-2:
                res = self.idle_steps(t=1)
                self.planner.update_from_simulation()

            else:
                self.rotate_base_z(moving_direction)
                self.planner.update_from_simulation()

                res = self.move_base_forward(target_pos, n_init_qpos=100)
                self.planner.update_from_simulation()
        
        # view_direction = target_view_pos.p - self.base_env.agent.base_link.pose.sp.p
        if not target_view_vec is None:
            res = self.rotate_base_z(target_view_vec)
        return res
    
    def move_base_forward(self, new_base_pose, n_init_qpos=20, dry_run = False):
        tcp_pose = self.base_env.agent.tcp.pose.sp
        base_link_pose = self.base_env.agent.base_link.pose.sp
        delta = new_base_pose - base_link_pose.p
        delta[2] = 0.
        target_tcp_pose = sapien.Pose(p=tcp_pose.p + delta, q=tcp_pose.q)

        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target_tcp_pose)
        target_tcp_pose = mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q)
        result = self.planner.plan_screw(
            mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            # masked_joints=[True, True, True] + [False] * 12
        )
        
        self.render_wait()

        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.follow_moving_forward(result)

        
        result = self.planner.plan_screw(
            mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            # masked_joints=[True, True, True] + [False] * 12
        )

        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        
        if dry_run:
            return result

        return self.follow_moving_forward(result)

    def move_base_x_and_manipulation(self, target_tcp_pose, n_init_qpos=20):
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target_tcp_pose)
        target_tcp_pose = mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q)
       
        move_x_and_manipulate =[False, True, True, False, False, False, False, False, False, False, False, False, False, False, False]
        result = self.planner.plan_pose(
            target_tcp_pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            # use_point_cloud=self.use_point_cloud,
            wrt_world=True,
            verbose=True,
            planning_time=2,
            rrt_range=0.1,
            simplify=True,
            mask=move_x_and_manipulate,
            fixed_joint_indices=[1],
            n_init_qpos=n_init_qpos   
        )

        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()

        res = self.follow_forward_path_w_refinement(result)
        self.planner.update_from_simulation()
        return self.static_manipulation(target_tcp_pose, n_init_qpos=n_init_qpos)


    def static_manipulation(self, target_tcp_pose, n_init_qpos=20, disable_lift_joint: bool = False):
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(sapien.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q))
        target_tcp_pose = mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q)
        only_manipulate =[True, True, True, disable_lift_joint, False, False, False, False, False, False, False, False, False, False, False]
        fixed_joint_indices = [0, 1, 2, 3] if disable_lift_joint else [0, 1, 2]

        result = self.planner.plan_screw(
            mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            masked_joints=~np.array(only_manipulate)
        )

        if result["status"] != "Success":

            result = self.planner.plan_pose(
                target_tcp_pose,
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                # use_point_cloud=self.use_point_cloud,
                wrt_world=True,
                verbose=True,
                planning_time=4,
                rrt_range=0.1,
                simplify=True,
                mask=only_manipulate,
                fixed_joint_indices=fixed_joint_indices,
                n_init_qpos=n_init_qpos   
            )

            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
            
        self.render_wait()

        return self.follow_forward_path_w_refinement(result, refine=True)

    def move_to_pose_with_screw_static_body(
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
            masked_joints=[False, False, False, False] + [True] * 11
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                mplib.Pose(pose.p, pose.q),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                masked_joints=[False, False, False, False] + [True] * 11
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

    def lift_hand(self, delta_h = 0., dry_run: bool = False, refine_steps: int = 0):
        cur_pose = self.base_env.agent.tcp.pose.sp
        taget_pose = mplib.Pose(p=cur_pose.p + np.array([0., 0., delta_h]),
                                q=cur_pose.q)
        result = self.planner.plan_screw(
            taget_pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            verbose=True
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)
    
    def move_forward_delta(self, delta = 0., dry_run: bool = False):
        cur_pose = self.base_env.agent.base_link.pose.sp
        direction = cur_pose.to_transformation_matrix()[:3, 0]
        direction[2] = 0.
        shift = direction * delta
        taget_pose = mplib.Pose(p=cur_pose.p + shift,
                                q=cur_pose.q)
        result = self.move_base_forward(taget_pose.p, dry_run=dry_run)
        return result

    def rotate_z_delta(self, delta = 0., dry_run: bool = False, rotate_recalculation_enabled: bool = True):
        cur_pose = self.base_env.agent.base_link.pose.sp
        direction = cur_pose.to_transformation_matrix()[:3, 0]
        direction[2] = 0.

        rot_matrix = euler2mat(0, 0, delta)

        new_direction = rot_matrix @ direction
        
        result = self.rotate_base_z(new_direction, dry_run=dry_run, rotate_recalculation_enabled=rotate_recalculation_enabled)
        
        return result

    def follow_rotation(self, result, refine_steps: int = 0):
        qpos_final = result["position"][-1]
        qpos_dict_final = {}
        for idx, q in zip(self.planner.move_group_joint_indices, qpos_final):
            joint_name = self.planner.user_joint_names[idx]
            qpos_dict_final[joint_name] = q
        
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
            body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
            body_action[0] = body_action[1] = 0.
            base_action = np.array([0., 0.])

            qvel = result["velocity"][min(i, n_step - 1)]

            base_action[1] = qvel[2]

            action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
            print("base Action:", np.round(base_action, 4))
            print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
            
        return obs, reward, terminated, truncated, info

    def follow_moving_forward(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        base_direction = self.env_agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
        root_to_world = self.env_agent.robot.root_pose.sp.to_transformation_matrix()[:3, :3]
        for i in range(n_step + refine_steps):
            arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
            body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
            body_action[0] = body_action[1] = 0.
            base_action = np.array([0., 0.])

            qvel = result["velocity"][min(i, n_step - 1)]
            base_vel = np.array([qvel[0], qvel[1], 0.])
            base_vel_wrt_world = root_to_world @ base_vel
            is_forward = 1 if np.dot(base_vel_wrt_world, base_direction) > 0 else -1
            base_action[0] = is_forward * np.sqrt(qvel[0] ** 2 + qvel[1] ** 2)

            action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
            print("base Action:", np.round(base_action, 4))
            print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
            
        return obs, reward, terminated, truncated, info

    def follow_path(self, result, refine_steps:int = 0, refine: bool = False):
        return self.follow_forward_path_w_refinement(result, refine)

    def follow_forward_path_w_refinement(self, result, refine: bool = False, static=False):
        qpos_final = result["position"][-1]
        qpos_dict_final = {}
        for idx, q in zip(self.planner.move_group_joint_indices, qpos_final):
            joint_name = self.planner.user_joint_names[idx]
            qpos_dict_final[joint_name] = q
            
        n_step = result["position"].shape[0]

        for i in range(n_step):
            arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()

            qpos = result["position"][min(i, n_step - 1)]
            qvel = result["velocity"][min(i, n_step - 1)]

            qpos_dict = {}
            
            for idx, q in zip(self.planner.move_group_joint_indices, qpos):
                joint_name = self.planner.user_joint_names[idx]
                qpos_dict[joint_name] = q

            for n, joint_name in enumerate(self.env_agent.controller.controllers['arm'].config.joint_names):
                arm_action[n] = qpos_dict[f'scene-0-{self.robot.name}_{joint_name}']

            assert self.control_mode == "pd_joint_pos"

            body_action = np.zeros_like(self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy())
            body_action[2] = qpos_dict[f'scene-0-{self.robot.name}_torso_lift_joint']

            base_action = np.array([0., 0.])
            base_action[0] =  np.sqrt(qvel[0] ** 2 + qvel[1] ** 2)

            action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
            print("arm Action:", np.round(arm_action, 4))
            print("body Action:", np.round(body_action, 4))
            print("base Action:", np.round(base_action, 4))
            print("qpos: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        if refine:
            # REFINEMENT!
            passed_refine_steps = 0
            last_lift_poses = deque(maxlen=10)
            last_x_base_poses = deque(maxlen=10)
            last_lift_vels = deque(maxlen=10)
            last_x_base_vels = deque(maxlen=10)
            print("==== REFINEMENT ====")
    
            while not self.check_body_base_close_to_target(qpos_dict_final):
                if (len(last_lift_vels) > 4 and np.std(last_lift_vels) < 1e-3) \
                        and (len(last_x_base_vels) > 4 and np.std(last_x_base_vels) < 1e-3) \
                        and (len(last_lift_poses) > 4 and np.std(last_lift_poses) < 1e-3) \
                        and (len(last_x_base_poses) > 4 and np.std(last_x_base_poses) < 1e-3):
                    # robot is stuck
                    print("Robot is stuck")
                    break
                if passed_refine_steps > self.MAX_REFINE_STEPS:
                    print("Reached max refining steps!")
                    break

                body_action = np.zeros_like(self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy())
                body_action[2] = qpos_dict_final[f'scene-0-{self.robot.name}_torso_lift_joint']
                body_action[0] = body_action[1] = 0.

                base_action = np.array([0., 0.])
                                    
                last_lift_poses.append(self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()[2])
                last_x_base_poses.append(self.env_agent.controller.controllers['base'].qpos[0].cpu().numpy()[0])

                last_lift_vels.append(self.env_agent.controller.controllers['body'].qvel[0].cpu().numpy()[2])
                last_x_base_vels.append(self.env_agent.controller.controllers['base'].qvel[0].cpu().numpy()[0])

                action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
                print("arm Action:", np.round(arm_action, 4))
                print("body Action:", np.round(body_action, 4))
                print("base Action:", np.round(base_action, 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
                obs, reward, terminated, truncated, info = self.env.step(action)
                passed_refine_steps += 1
                self.elapsed_steps += 1
                if self.print_env_info:
                    print(
                        f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                    )
                if self.vis:
                    self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def check_body_base_close_to_target(self, target_dict, eps=1e-2):
        body_qpos = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()[2]
        target_lift_joint_height = target_dict[f'scene-0-{self.robot.name}_torso_lift_joint']

        base_xy = self.env_agent.controller.controllers['base'].qpos[0].cpu().numpy()[0:2]
        target_base = np.array([
            target_dict[f'scene-0-{self.robot.name}_root_x_axis_joint'],
            target_dict[f'scene-0-{self.robot.name}_root_y_axis_joint']
        ])

        robot_qpos = self.robot.get_qpos().cpu().numpy()[0]
        arm_pos = robot_qpos[self.env_agent.controller.controllers['arm'].active_joint_indices.cpu().numpy()]
        target_arm_pos = np.array([
            target_dict[f'scene-0-{self.robot.name}_shoulder_pan_joint'],
            target_dict[f'scene-0-{self.robot.name}_shoulder_lift_joint'],
            target_dict[f'scene-0-{self.robot.name}_upperarm_roll_joint'],
            target_dict[f'scene-0-{self.robot.name}_elbow_flex_joint'],
            target_dict[f'scene-0-{self.robot.name}_forearm_roll_joint'],
            target_dict[f'scene-0-{self.robot.name}_wrist_flex_joint'],
            target_dict[f'scene-0-{self.robot.name}_wrist_roll_joint']
        ])
        return np.allclose(body_qpos, target_lift_joint_height, atol=eps) and \
            np.allclose(base_xy, target_base, atol=eps) and \
            np.allclose(arm_pos, target_arm_pos, atol=eps)

    def change_gripper_state(self, t=6, gripper_state = OPEN):
        self.gripper_state = gripper_state
        arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
        base_action = np.array([0, 0])

        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                # action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
                action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
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

    def close_gripper(self, t=6):
        return self.change_gripper_state(t=t, gripper_state = CLOSED)
        
    def open_gripper(self, t=6):
        return self.change_gripper_state(t=t, gripper_state = OPEN)
    
    def idle_steps(self, t=20):
        arm_action = self.env_agent.controller.controllers['arm'].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers['body'].qpos[0].cpu().numpy()
        base_action = np.array([0, 0])
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                # action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
                action = np.hstack([arm_action, self.gripper_state, body_action, base_action])
            else:
                raise NotImplementedError
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info
