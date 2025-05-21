import torch
import numpy as np
import os
import sapien
from transforms3d.euler import euler2quat
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from dsynth.scene_gen.arrangements import CELL_SIZE
import copy

LANGUAGE_INSTRUCTION = 'pick a milk from the shelf and put it on the cart'

@register_env('PickToCartStaticEnv', max_episode_steps=200000)
class PickToCartStaticEnv(DarkstoreCellBaseEnv):

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # self._load_shopping_cart(options)
        
        self.target_sizes = np.array([0.3, 0.3, 0.3])
        self.target_volume = actors.build_box(
            self.scene,
            half_sizes=list(self.target_sizes/2),
            color=[0, 1, 0, 0.5],
            name="target_box",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0]),
        )
    
    def _compute_robot_init_pose(self, env_idx = None): #TODO: redo this shit
        target_shelf = 'zone1.shelf1'
        for i in range(len(self.scene_builder.room[0])):
            for j in range(len(self.scene_builder.room[0][i])):
                if self.scene_builder.room[0][i][j] == target_shelf:
                    rot = self.scene_builder.rotations[0][i][j]
                    shelf_i, shelf_j = i, j
                    break
        if rot == 0:
            origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j - 1, 0.]), np.pi / 2, np.array([0, 1, 0])
        if rot == -90:
            origin, angle, direction_to_shelf = np.array([shelf_i - 1, shelf_j, 0.]), 0 , np.array([1, 0, 0])
        if rot == 90:
            origin, angle, direction_to_shelf = np.array([shelf_i + 1, shelf_j, 0.]), np.pi, np.array([-1, 0, 0])
        if rot == 180:
            origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j + 1, 0.]), - np.pi / 2, np.array([0, -1, 0])
        self.direction_to_shelf = direction_to_shelf

        origin = origin * CELL_SIZE
        origin[:2] += CELL_SIZE / 2

        origin += direction_to_shelf * CELL_SIZE * 0.2

        return origin, angle

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

        elif self.robot_uids in ["ds_fetch_basket", "ds_fetch"]:
            qpos = np.array(
                [
                    0,
                    0,
                    0,#np.random.rand() * 6.2832 - 3.1416,
                    0.36,
                    0,
                    0,
                    0,
                    1.4,
                    0,
                    0.76,
                    0,
                    - 2 * np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            robot_origin, robot_angle = self._compute_robot_init_pose()
            self.agent.robot.set_pose(sapien.Pose(p=robot_origin, q=euler2quat(0, 0, robot_angle)))

        robot_pose = self.agent.robot.get_pose()
        cart_shift = sapien.Pose(p=[0.15, 0.25, 0.44])
        # cube_shift_up = np.array([0, 0.13, 0.35])
        new_cart_pose_p = robot_pose * cart_shift 
        
        # self.shopping_cart.set_pose(sapien.Pose(p=new_cart_pose_p, q=robot_pose.q[0].numpy()))
        self.target_volume.set_pose(new_cart_pose_p)
        # self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0], [0.70710678118, 0, 0, 0.70710678118]))
        self.setup_target_object()

@register_env('PickToCartEnv', max_episode_steps=200000)
class PickToCartEnv(PickToCartStaticEnv):
    pass

@register_env('PickToCartStatiсOneProdEnv', max_episode_steps=200000)
class PickToCartStatiсOneProdEnv(PickToCartStaticEnv):
    def setup_target_object(self):
        self.target_product_name = sorted(list(self.actors['products'].keys()))[2] # pick sprite
        obb = get_actor_obb(self.actors['products'][self.target_product_name])
        center = np.array(obb.primitive.transform)[:3, 3]

        self.target_product_marker.set_pose(sapien.Pose(center))

        target_product_name = 'products_hierarchy.' + self.target_product_name.split(':')[0]
        self.target_product_str = self.assets_lib[target_product_name].asset_name

