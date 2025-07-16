from typing import Dict
import itertools
import os
import json
import torch
import numpy as np
from transforms3d import quaternions
import random
import re
import copy
import sapien
from pathlib import Path
import hydra
import pandas as pd
from transforms3d.euler import euler2quat
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.sapien_env import BaseEnv
from dsynth.envs.fixtures.robocasaroom_cont import DarkstoreSceneContinuous
from dsynth.scene_gen.arrangements import CELL_SIZE, DEFAULT_ROOM_HEIGHT
from dsynth.assets.asset import load_assets_lib
from dsynth.scene_gen.utils import flatten_dict
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb

from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv

@register_env('DarkstoreContinuousBaseEnv', max_episode_steps=200000)
class DarkstoreContinuousBaseEnv(DarkstoreCellBaseEnv):
    def _load_scene(self, options: dict):
        BaseEnv._load_scene(self, options)
        self.is_rebuild = True

        self.actors = {
            "fixtures": {
                "shelves" : {},
                "lamps": {},
                "scene_assets": {}
            },
            "products": {}
        }
        self.active_shelves = {}

        self.scene_builder = DarkstoreSceneContinuous(self, config_dir_path=self.config_dir_path)
        self.scene_builder.build()

        actor_names = []
        scene_idxs = []   
        asset_names = []
        product_names = []
        board_idxs = []
        col_idxs = []
        row_idxs = []

        for actor_name, actor in self.actors["products"].items():
            actor_names.append(actor_name)

            assert len(actor._scene_idxs) == 1
            scene_idx = actor._scene_idxs.cpu().numpy()[0]
            scene_idxs.append(scene_idx)

            asset_name = actor_name.replace(f'[ENV#{scene_idx}]_', '')
            asset_name, shelf_idx, board_idx ,col_idx, row_idx = asset_name.split(':')
            asset_names.append(asset_name)
            product_names.append(self.assets_lib['products_hierarchy.' + asset_name].asset_name)

            board_idxs.append(board_idx)
            col_idxs.append(col_idx)
            row_idxs.append(row_idx)

        self.products_df = pd.DataFrame(dict(
                actor_name=actor_names,
                scene_idx=scene_idxs,
                product_name=product_names,
                asset_name=asset_names,
                board_idxs=board_idxs,
                col_idxs=col_idxs,
                row_idxs=row_idxs
            )
        )


        self.products_df.to_csv(self.config_dir_path / 'scene_items.csv')
        print("built")
        print(f"Total {len(self.actors['products'])} products in {self.num_envs} scene(s)")

    def _compute_robot_init_pose(self, env_idx = None):
        origins = []
        angles = []
        directions_to_shelf = []

        for idx in env_idx:
            idx = idx.cpu().item()
            assert len(self.active_shelves[idx]) == 1, "Environments with continuous layout support only one active shelving"

            actor_shelf_name = self.active_shelves[idx][0]
            shelf_pose = self.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp
            shelf_direction = shelf_pose.to_transformation_matrix()[:3, 1]

            direction_to_scene_center = np.array([self.scene_builder.x_size[idx] / 2, self.scene_builder.y_size[idx] / 2, 0.]) - shelf_pose.p
            direction_to_scene_center /= (np.linalg.norm(direction_to_scene_center) + 1e-3)

            if np.dot(direction_to_scene_center, shelf_direction) < 0:
                shelf_direction = -shelf_direction

            origin = shelf_pose.p + 1.5 * shelf_direction
            direction_to_shelf = -shelf_direction

            base_x_axis = np.array([1, 0, 0])
            angle = np.arccos(np.dot(direction_to_shelf, base_x_axis))
            if np.cross(base_x_axis, direction_to_shelf)[2] < 0:
                angle = -angle

            origins.append(origin)
            angles.append(angle)
            directions_to_shelf.append(direction_to_shelf)

        return np.array(origins), np.array(angles), np.array(directions_to_shelf)



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if not self.is_rebuild:
            raise RuntimeError("To reset arrangement use 'reconfigure' flag: env.reset(options={'reconfigure': True})")
        self.is_rebuild = False
        
        self.product_displaced = False
        self.products_initial_poses = {}
        for p, a in self.actors['products'].items():
            self.products_initial_poses[p] = copy.deepcopy(a.pose.raw_pose)


        self.robot_origins, self.robot_angles, self.directions_to_shelf = self._compute_robot_init_pose(env_idx)
        quats = np.array([euler2quat(0, 0, robot_angle) for robot_angle in self.robot_angles])

        if self.robot_uids in ["panda", "panda_wristcam"]:
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
            self.agent.robot.set_pose(Pose.create_from_pq(p=self.robot_origins, q=quats))

        elif self.robot_uids in ["fetch", "ds_fetch", "ds_fetch_basket"]:
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.36,
                    0,
                    0,
                    0,
                    0.75,
                    0,
                    0.81,
                    0,
                    -0.78,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose.create_from_pq(p=self.robot_origins, q=quats))
        elif self.robot_uids in ["ds_fetch_static", "ds_fetch_basket_static"]:
            qpos = np.array(
               [
                    0.36,
                    0,
                    0,
                    0,
                    0.75,
                    0,
                    0.81,
                    0,
                    -0.78,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose.create_from_pq(p=self.robot_origins, q=quats))

        else:
            raise NotImplementedError

