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
from dsynth.envs.darkstore_cont_base import DarkstoreContinuousBaseEnv
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.structs.pose import Pose
from dsynth.scene_gen.arrangements import CELL_SIZE
from dsynth.scene_gen.utils import flatten_dict
from dsynth.scene_gen.hydra_configs import ShelfType

@register_env('OpenDoorFridgeEnv', max_episode_steps=200000)
class OpenDoorFridgeEnv(DarkstoreCellBaseEnv):
    ROBOT_INIT_POSE_RANDOM_ENABLED = True
    SHELF_TYPE = ShelfType.FRIDGE_FOOD_SHOWCASE
    SUCCESS_THRESH_ANGLE = 0.2
    DOOR_NAMES = ['first', 'second', 'third', 'fourth']
    DOOR_NAMES_2_IDX = {name: i + 1 for i, name in enumerate(DOOR_NAMES)}
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
    def setup_target_fridge(self, env_idxs):
        self.target_zone_names = {}
        self.target_zone_ids = {}
        self.target_fridge_names = {}
        self.target_fridge_ids = {}
        self.target_actor_name = {}
        self.target_door_names = {}
        
        self.target_showcases_df = None

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()

            scene_shelvings_df = self.shelvings_df[self.shelvings_df['scene_idx'] == scene_idx]
            scene_showcases = scene_shelvings_df[scene_shelvings_df['shelf_type'] == self.SHELF_TYPE.value]
            if len(scene_showcases) == 0:
                raise RuntimeError(f"No showcases found on scene {scene_idx}!")
            target_showcase_name = self._batched_episode_rng[scene_idx].choice(sorted(scene_showcases['actor_name'].unique()))
            self.target_actor_name[scene_idx] = target_showcase_name
            self.target_door_names[scene_idx] = self._batched_episode_rng[scene_idx].choice(self.DOOR_NAMES)
            
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
            # origin += -0.3 * perp_direction + 0.6 * direction_to_shelf
            # angle = 4.2560362
            # origin = np.array([1.288, 2.497, 0.0])
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
            door_name = self.target_door_names[scene_idx]
            self.language_instructions.append(f'open the {door_name} fridge')

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.2, 0.2, 4], [5, 5, 2])
        pose = sapien_utils.look_at([3, 3, 2], [0, 0, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def evaluate(self):
        is_door_opened = []
        for scene_idx in range(self.num_envs):
            target_showcase_name = self.target_actor_name[scene_idx]
            showcase_actor = self.actors['fixtures']['shelves'][target_showcase_name]
            num_door = self.DOOR_NAMES_2_IDX[self.target_door_names[scene_idx]]
            is_door_opened.append(
                torch.abs(showcase_actor.joints_map[f'door{num_door}_link_joint'].qpos - 1.57) < self.SUCCESS_THRESH_ANGLE
            )
        is_door_opened = torch.cat(is_door_opened)
        
        is_robot_static = self.agent.is_static(0.2)

        return {
            "is_door_opened" : is_door_opened,
            "is_robot_static" : is_robot_static,
            "success": is_door_opened & is_robot_static
        }    
    

@register_env('OpenDoorShowcaseContEnv', max_episode_steps=200000)
class OpenDoorShowcaseContEnv(DarkstoreContinuousBaseEnv):
    ROBOT_INIT_POSE_RANDOM_ENABLED = True
    SUCCESS_THRESH_ANGLE = 0.2
    DOOR_NAMES = ['first', 'second', 'third', 'fourth']
    DOOR_NAMES_2_IDX = {name: i + 1 for i, name in enumerate(DOOR_NAMES)}
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
    def setup_target_objects(self, env_idxs):
        # no guarantees that target shelving is exactly a showcase - TODO
        self.target_actor_name = {}
        self.target_door_names = {}

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()

            assert len(self.active_shelves[scene_idx]) == 1
            self.target_actor_name[scene_idx] = self.active_shelves[scene_idx][0]
            
            self.target_door_names[scene_idx] = self._batched_episode_rng[scene_idx].choice(self.DOOR_NAMES)

    def _compute_robot_init_pose(self, env_idx = None):
        robot_origins, robot_angles, directions_to_shelf = super()._compute_robot_init_pose(env_idx)
        for idx in env_idx:
            robot_origins[idx] -= 0.15 * directions_to_shelf[idx]
            if self.ROBOT_INIT_POSE_RANDOM_ENABLED:
                # base movement enabled, add initial pose randomization
                batched_rng = self._batched_episode_rng
                if self.extra_robot_pose_randomization:
                    batched_rng = self._batched_init_pose_rng
                idx = idx.cpu().item()
                direction_to_shelf = directions_to_shelf[idx]
                perp_direction = np.cross(direction_to_shelf, [0, 0, 1])


                delta_par = batched_rng[idx].rand() * 1.55 * 0.4
                delta_perp = (batched_rng[idx].rand() - 0.5) * 2 * 1.55 * 0.4


                robot_origins[idx] += - direction_to_shelf * delta_par + perp_direction * delta_perp
                # robot_origins[idx] += -direction_to_shelf * delta_par + perp_direction * delta_perp
                robot_angles[idx] += (batched_rng[idx].rand() - 0.5) * np.pi / 4

        return robot_origins, robot_angles, directions_to_shelf

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            door_name = self.target_door_names[scene_idx]
            self.language_instructions.append(f'open the {door_name} door of showcase')

    def evaluate(self):
        is_door_opened = []
        for scene_idx in range(self.num_envs):
            target_showcase_name = self.target_actor_name[scene_idx]
            showcase_actor = self.actors['fixtures']['shelves'][target_showcase_name]
            num_door = self.DOOR_NAMES_2_IDX[self.target_door_names[scene_idx]]
            is_door_opened.append(
                torch.abs(showcase_actor.joints_map[f'door{num_door}_link_joint'].qpos - 1.57) < self.SUCCESS_THRESH_ANGLE
            )
        is_door_opened = torch.cat(is_door_opened)
        
        is_robot_static = self.agent.is_static(0.2)

        return {
            "is_door_opened" : is_door_opened,
            "is_robot_static" : is_robot_static,
            "success": is_door_opened & is_robot_static
        }   
    

@register_env('CloseDoorShowcaseContEnv', max_episode_steps=200000)
class CloseDoorShowcaseContEnv(OpenDoorShowcaseContEnv):
    def setup_target_objects(self, env_idxs):
        super().setup_target_objects(env_idxs)

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            showcase_actor = self.target_actor_name[scene_idx]
            num_door = self.DOOR_NAMES_2_IDX[self.target_door_names[scene_idx]]
            qpos_new = np.zeros((4,))
            qpos_new[num_door - 1] = 1.4 + (self._batched_episode_rng[scene_idx].random() - 0.5) / 0.5 * 0.08
            self.actors['fixtures']['shelves'][showcase_actor].set_qpos(qpos_new)

    def evaluate(self):
        is_door_closed = []
        for scene_idx in range(self.num_envs):
            target_showcase_name = self.target_actor_name[scene_idx]
            showcase_actor = self.actors['fixtures']['shelves'][target_showcase_name]
            num_door = self.DOOR_NAMES_2_IDX[self.target_door_names[scene_idx]]
            is_door_closed.append(
                torch.abs(showcase_actor.joints_map[f'door{num_door}_link_joint'].qpos - 0.0) < self.SUCCESS_THRESH_ANGLE
            )
        is_door_closed = torch.cat(is_door_closed)
        
        is_robot_static = self.agent.is_static(0.2)

        return {
            "is_door_closed" : is_door_closed,
            "is_robot_static" : is_robot_static,
            "success": is_door_closed & is_robot_static
        }
    
    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            door_name = self.target_door_names[scene_idx]
            self.language_instructions.append(f'close the door of the showcase')


@register_env('OpenDoorFridgeContEnv', max_episode_steps=200000)
class OpenDoorFridgeContEnv(OpenDoorShowcaseContEnv):
    ROBOT_INIT_POSE_RANDOM_ENABLED = True
    SUCCESS_THRESH_ANGLE = 0.1

    def setup_target_objects(self, env_idxs):
        # no guarantees that target shelving is exactly a fridge - TODO
        super().setup_target_objects(env_idxs)

    def _compute_robot_init_pose(self, env_idx = None):
        robot_origins, robot_angles, directions_to_shelf = DarkstoreContinuousBaseEnv._compute_robot_init_pose(self, env_idx)
        for idx in env_idx:
            # robot_origins[idx] -= 0.15 * directions_to_shelf[idx]
            if self.ROBOT_INIT_POSE_RANDOM_ENABLED:
                # base movement enabled, add initial pose randomization
                batched_rng = self._batched_episode_rng
                if self.extra_robot_pose_randomization:
                    batched_rng = self._batched_init_pose_rng

                idx = idx.cpu().item()
                direction_to_shelf = directions_to_shelf[idx]
                perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

                delta_par = batched_rng[idx].rand() * 0.2
                delta_perp = (batched_rng[idx].rand() - 0.5) * 0.5

                robot_origins[idx] += direction_to_shelf * delta_par + perp_direction * delta_perp
                robot_angles[idx] += (batched_rng[idx].rand() - 0.5) * np.pi / 4

        return robot_origins, robot_angles, directions_to_shelf

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
    
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            target_showcase_name = self.target_actor_name[scene_idx]
            fridge_actor = self.actors['fixtures']['shelves'][target_showcase_name]
            fridge_actor.set_pose(fridge_actor.pose * sapien.Pose(q=euler2quat(0, 0, 3.14)))

    def evaluate(self):
        is_door_opened = []
        for scene_idx in range(self.num_envs):
            target_showcase_name = self.target_actor_name[scene_idx]
            showcase_actor = self.actors['fixtures']['shelves'][target_showcase_name]
            is_door_opened.append(
                torch.abs(showcase_actor.joints_map[f'right_cover_joint'].qpos - 0.624) < self.SUCCESS_THRESH_ANGLE
            )
        is_door_opened = torch.cat(is_door_opened)
        
        is_robot_static = self.agent.is_static(0.2)

        return {
            "is_door_opened" : is_door_opened,
            "is_robot_static" : is_robot_static,
            "success": is_door_opened & is_robot_static
        }   

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            door_name = self.target_door_names[scene_idx]
            self.language_instructions.append(f'open the fridge')


@register_env('CloseDoorFridgeContEnv', max_episode_steps=200000)
class CloseDoorFridgeContEnv(OpenDoorFridgeContEnv):
    def setup_target_objects(self, env_idxs):
        super().setup_target_objects(env_idxs)

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            frdidge_actor = self.target_actor_name[scene_idx]
            qpos_new = np.zeros((1,))
            qpos_new[0] = 0.6 + (self._batched_episode_rng[scene_idx].random() - 0.5) / 0.5 * 0.02
            self.actors['fixtures']['shelves'][frdidge_actor].set_qpos(qpos_new)

    def evaluate(self):
        is_door_closed = []
        for scene_idx in range(self.num_envs):
            target_showcase_name = self.target_actor_name[scene_idx]
            showcase_actor = self.actors['fixtures']['shelves'][target_showcase_name]
            is_door_closed.append(
                torch.abs(showcase_actor.joints_map[f'right_cover_joint'].qpos - 0.0) < self.SUCCESS_THRESH_ANGLE
            )
        is_door_closed = torch.cat(is_door_closed)
        
        is_robot_static = self.agent.is_static(0.2)

        return {
            "is_door_closed" : is_door_closed,
            "is_robot_static" : is_robot_static,
            "success": is_door_closed & is_robot_static
        }  
    
    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            door_name = self.target_door_names[scene_idx]
            self.language_instructions.append(f'close the fridge')