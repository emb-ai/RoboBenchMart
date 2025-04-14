from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat


import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT

@register_env("PickCubeEnvMPTest", max_episode_steps=50)
class PickCubeEnvMPTest(PickCubeEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    cube_half_sizes = [0.02, 0.02, 0.07]
    wall_half_sizes = [0.02, 0.35, 0.08]
    goal_thresh = 0.025
    FINGER_LENGTH = 0.025
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_box(
            self.scene,
            half_sizes=self.cube_half_sizes,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_sizes[-1]]),
        )
        self.wall = actors.build_box(
            self.scene,
            half_sizes=self.wall_half_sizes,
            color=[0, 0, 1, 1],
            name="wall",
            initial_pose=sapien.Pose(p=[0, 0, self.wall_half_sizes[-1]]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self.table_collider = actors.build_box(
            self.scene,
            half_sizes=[self.table_scene.table_length / 2, self.table_scene.table_width / 2, 0.01],
            color=[1, 0, 0, 1],
            name="table_collider",
            body_type='kinematic',
            initial_pose=sapien.Pose(p=[0.15, 0, -0.01]),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.02
            xyz[:, 0] += 0.15
            xyz[:, 2] = self.cube_half_sizes[-1]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.02 
            goal_xyz[:, 0] -= 0.32
            goal_xyz[:, 2] = 2 * self.cube_half_sizes[-1] - min(self.cube_half_sizes[-1], self.FINGER_LENGTH) + 0.01
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            wall_xyz = torch.zeros((b, 3))
            wall_xyz[:, 0] -= 0.06
            wall_xyz[:, 2] = self.wall_half_sizes[-1]
            self.wall.set_pose(Pose.create_from_pq(wall_xyz))

        if self.robot_uids == "panda_wristcam":
            qpos = np.array(
                [
                    0.0,        
                    np.pi / 10, 
                    0.0,        
                    - 0.8 * np.pi, 
                    0.0,        
                    np.pi * 3 / 4,  
                    np.pi / 4,  
                    0.04,       
                    0.04,       
                ]
            )
            self.agent.reset(qpos)

# TODO (stao): make the build and initialize api consistent with other scenes
class TableSceneBuilderDSynth(TableSceneBuilder):
    def initialize(self, env_idx: torch.Tensor):
        # table_height = 0.9196429
        b = len(env_idx)
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        )
        if self.env.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == "ds_fetch_static":
            qpos = np.array(
                [
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == "ds_fetch_quasi_static":
            qpos = np.array(
                [
                    -1.01,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == "ds_fetch":
            qpos = np.array(
                [
                    -2. - np.random.randn() * 0.5,
                    -1. - np.random.randn() * 0.5,
                    np.random.rand() * 6.2832 - 3.1416,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        else:
            raise NotImplementedError

@register_env("PickCubeEnvDSynth", max_episode_steps=50)
class PickCubeEnvDSynth(PickCubeEnv):
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilderDSynth(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.02
            xyz[:, 0] -= 0.3
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.02 - 0.1
            goal_xyz[:, 0] -= 0.2
            # goal_xyz[:, 1] -= 0.2
            goal_xyz[:, 2] = torch.rand((b)) * 0.1 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))


