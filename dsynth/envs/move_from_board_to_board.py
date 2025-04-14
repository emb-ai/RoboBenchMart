import torch
import numpy as np
import os
import sapien
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb

@register_env('MoveFromBoardToBoardEnv', max_episode_steps=200000)
class MoveFromBoardToBoardEnv(DarkstoreCellBaseEnv):

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        target_pose = self.actors['products'][self.target_product_name]['actor'].pose
        target_pose.raw_pose[0,2] += get_actor_obb(self.actors['products'][self.target_product_name]['actor']).extents[2]/2 + 0.397 #height of board
        target_pose.raw_pose[0][3:] = torch.Tensor([1, 0, 0, 0])
        self.target_sizes = get_actor_obb(self.actors['products'][self.target_product_name]['actor']).extents
        self.target_volume = actors.build_box(
            self.scene,
            half_sizes=list(self.target_sizes/2),
            color=[0, 1, 0, 0.5],
            name="target_box",
            body_type="static",
            add_collision=False,
            initial_pose=target_pose,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        
        if self.robot_uids == "panda_wristcam":
            qpos = np.array(
                [
                    0.0,        
                    -np.pi / 6, 
                    0.0,        
                    -np.pi / 3, 
                    0.0,        
                    np.pi / 2,  
                    np.pi / 4,  
                    0.04,       
                    0.04,       
                ]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0]))

        robot_pose = self.agent.robot.get_pose()
        cart_shift = np.array([0.4, -0.2, 0.])
        new_cart_pose_p = robot_pose.p[0].cpu().numpy() + cart_shift 
        
        self.shopping_cart.set_pose(sapien.Pose(p=new_cart_pose_p, q=robot_pose.q[0].numpy()))
        self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0], [0.70710678118, 0, 0, 0.70710678118]))
        self.setup_target_object()
