import torch
import numpy as np
import os
import sapien
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
import copy

LANGUAGE_INSTRUCTION = 'pick a milk from the shelf and put it on the cart'

@register_env('PickToCartEnv', max_episode_steps=200000)
class PickToCartEnv(DarkstoreCellBaseEnv):

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        self.target_sizes = np.array([0.3, 0.3, 0.3])
        self.target_volume = actors.build_box(
            self.scene,
            half_sizes=list(self.target_sizes/2),
            color=[0, 1, 0, 0.5],
            name="target_box",
            body_type="static",
            add_collision=False,
            initial_pose=self.shopping_cart.pose,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        
        if self.robot_uids == "panda_wristcam":
            qpos = np.array(
                [
                    -0.006,        
                    -1.467,
                    0.012,        
                    -2.823,
                    0.003,        
                    2.928,
                    0.796,
                    0.04,       
                    0.04,       
                ]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0]))

        robot_pose = self.agent.robot.get_pose()
        cart_shift = np.array([0.45, -0.2, 0.])
        cube_shift_up = np.array([0, 0.13, 0.35])
        new_cart_pose_p = robot_pose.p[0].cpu().numpy() + cart_shift 
        
        self.shopping_cart.set_pose(sapien.Pose(p=new_cart_pose_p, q=robot_pose.q[0].numpy()))
        self.target_volume.set_pose(sapien.Pose(p=new_cart_pose_p + cube_shift_up, q=robot_pose.q[0].numpy()))
        # self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0], [0.70710678118, 0, 0, 0.70710678118]))
        self.setup_target_object()


