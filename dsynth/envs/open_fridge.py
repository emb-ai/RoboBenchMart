import torch
import numpy as np
import pandas as pd
import os
import sapien
import sapien.physx as physx
from transforms3d.euler import euler2quat
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.structs.pose import Pose
from dsynth.scene_gen.arrangements import CELL_SIZE
from dsynth.scene_gen.utils import flatten_dict
from dsynth.scene_gen.hydra_configs import ShelfType

@register_env('OpenDoorFridgeEnv', max_episode_steps=200000)
class OpenDoorFridgeEnv(DarkstoreCellBaseEnv):
    ROBOT_INIT_POSE_RANDOM_ENABLED = False
    SHELF_TYPE = ShelfType.FRIDGE_FOOD_SHOWCASE
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
    def setup_target_fridge(self, env_idxs):
        self.target_zone_names = {}
        self.target_zone_ids = {}
        self.target_fridge_names = {}
        self.target_fridge_ids = {}
        self.target_actor_name = {}
        
        self.target_showcases_df = None

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()

            scene_shelvings_df = self.shelvings_df[self.shelvings_df['scene_idx'] == scene_idx]
            scene_showcases = scene_shelvings_df[scene_shelvings_df['shelf_type'] == self.SHELF_TYPE.value]
            if len(scene_showcases) == 0:
                raise RuntimeError(f"No showcases found on scene {scene_idx}!")
            target_showcase_name = self._batched_episode_rng[scene_idx].choice(scene_showcases['actor_name'].unique())
            self.target_actor_name[scene_idx] = target_showcase_name
            
            target_showcase_df = scene_showcases[scene_showcases['actor_name'] == target_showcase_name]
            assert len(target_showcase_df) == 1
            self.target_zone_names[scene_idx] = target_showcase_df['zone_name'].array[0]
            self.target_zone_ids[scene_idx] = target_showcase_df['zone_id'].array[0]
            self.target_fridge_names[scene_idx] = target_showcase_df['shelf_name'].array[0]
            self.target_fridge_ids[scene_idx] = target_showcase_df['shelf_id'].array[0]

            if self.target_showcases_df is None:
                self.target_showcases_df = target_showcase_df
            else:
                self.target_showcases_df = pd.concat([self.target_showcases_df, target_showcase_df])
            
    def _compute_robot_init_pose(self, env_idx = None):
        origins = []
        init_cells = []
        angles = []
        directions_to_shelf = []

        for idx in env_idx:
            idx = idx.cpu().item()
            scene_target_products = self.target_showcases_df[self.target_showcases_df['scene_idx'] == idx].reset_index()
            shelf_i, shelf_j = scene_target_products['i'][0], scene_target_products['j'][0]
            rot = self.scene_builder.rotations[idx][shelf_i][shelf_j]

            if rot == 0:
                origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j - 1, 0.]), np.pi / 2, np.array([0., 1., 0.])
            if rot == -90:
                origin, angle, direction_to_shelf = np.array([shelf_i - 1, shelf_j, 0.]), 0 , np.array([1., 0., 0.])
            if rot == 90:
                origin, angle, direction_to_shelf = np.array([shelf_i + 1, shelf_j, 0.]), np.pi, np.array([-1., 0., 0.])
            if rot == 180:
                origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j + 1, 0.]), - np.pi / 2, np.array([0., -1., 0.])
            
            # self.target_drive_position = origin.copy() + direction_to_shelf * CELL_SIZE * 0.2
            
            init_cell = np.array([origin[0], origin[1]])
            origin = origin * CELL_SIZE
            origin[:2] += CELL_SIZE / 2

            # move to the left door
            perp_direction = np.cross(direction_to_shelf, [0, 0, 1])
            origin += -0.7 * perp_direction + 0.5 * direction_to_shelf

            if self.ROBOT_INIT_POSE_RANDOM_ENABLED:
                # base movement enabled, add initial pose randomization
                perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

                delta_par = self._batched_episode_rng[idx].rand() * CELL_SIZE * 0.4
                delta_perp = (self._batched_episode_rng[idx].rand() - 0.5) * 2 * CELL_SIZE * 0.4

                origin += - direction_to_shelf * delta_par + perp_direction * delta_perp

                angle += (self._batched_episode_rng[idx].rand() - 0.5) * np.pi / 4

            origins.append(origin)
            init_cells.append(init_cell)
            angles.append(angle)
            directions_to_shelf.append(direction_to_shelf)

        return np.array(origins), np.array(init_cells), np.array(angles), np.array(directions_to_shelf)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.setup_target_fridge(env_idx)
        self.setup_language_instructions(env_idx)

        b = len(env_idx)
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

        elif self.robot_uids in ["ds_fetch_basket", "ds_fetch", "fetch"]:
            qpos = np.array(
                [
                    0,
                    0,
                    0,
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
            self.robot_origins, self.init_cells, self.robot_angles, self.directions_to_shelf = self._compute_robot_init_pose(env_idx)
            quats = np.array([euler2quat(0, 0, robot_angle) for robot_angle in self.robot_angles])
            self.agent.robot.set_pose(Pose.create_from_pq(p=self.robot_origins, q=quats))
        elif self.robot_uids in ["ds_fetch_static", "ds_fetch_basket_static"]:
            qpos = np.array(
                [
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
            self.robot_origins, self.init_cells, self.robot_angles, self.directions_to_shelf = self._compute_robot_init_pose(env_idx)
            quats = np.array([euler2quat(0, 0, robot_angle) for robot_angle in self.robot_angles])
            self.agent.robot.set_pose(Pose.create_from_pq(p=self.robot_origins, q=quats))
       

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(f'open the fridge')
    