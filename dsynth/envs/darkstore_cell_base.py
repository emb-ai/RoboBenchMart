from typing import Dict
import itertools
import os
import json
import torch
import numpy as np
from transforms3d import quaternions
import random
import copy
import sapien
from pathlib import Path
import hydra
import pandas as pd
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.sapien_env import BaseEnv
from dsynth.envs.fixtures.robocasaroom import DarkstoreScene
from dsynth.scene_gen.arrangements import CELL_SIZE, DEFAULT_ROOM_HEIGHT
from dsynth.assets.asset import load_assets_lib
from dsynth.scene_gen.utils import flatten_dict
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb


@register_env('DarkstoreCellBaseEnv', max_episode_steps=200000)
class DarkstoreCellBaseEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    NUM_MARKERS = 100

    def __init__(self, *args, 
                 config_dir_path,
                 user_target_product_name=None,
                 robot_uids="panda_wristcam",
                 markers_enabled=True,
                 hidden_objects_enabled=False,
                 all_static=False,
                 **kwargs):
        self.config_dir_path = Path(config_dir_path)
        self.is_rebuild = False

        self.markers_enabled = markers_enabled

        # hidden objects are broken in GPU simulation (https://github.com/haosulab/ManiSkill/issues/1134)
        self.hidden_objects_enabled = hidden_objects_enabled

        self.all_static = all_static
        with hydra.initialize_config_dir(config_dir=str(self.config_dir_path.absolute()), version_base=None):
            self.cfg = hydra.compose(config_name='input_config')

        self.assets_lib = flatten_dict(load_assets_lib(self.cfg.assets), sep='.')

        self.actors = {
            "fixtures": {
                "shelves" : {},
                "lamps": {},
                "scene_assets": {}
            },
            "products": {}
        }

        self.user_target_product_name = user_target_product_name
        
        self.target_product_str = ''
        self.language_instructions = []
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.7, 1.8, 1.15], [1.2, 2.2, 1.2])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.2, 0.2, 4], [5, 5, 2])
        pose = sapien_utils.look_at([3, 3, 3], [0, 0, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        ps = torch.zeros((self.num_envs, 3), device=self.device)
        ps[:, 0] = -0.615
        super()._load_agent(options, Pose.create_from_pq(p=ps))

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.is_rebuild = True

        self.actors = {
            "fixtures": {
                "shelves" : {},
                "lamps": {},
                "scene_assets": {}
            },
            "products": {}
        }
        self.products2shelves = {}

        self.scene_builder = DarkstoreScene(self, config_dir_path=self.config_dir_path)
        self.scene_builder.build()

        self.room = self.scene_builder.room
        self.ds_names = self.scene_builder.ds_names

        self.shelves_placement = []
        for room in self.room:
            self.shelves_placement.append({})
            for i, j in itertools.product(range(len(room)), range(len(room[0]))):
                if room[i][j] != 0:
                    zone_name, shelf_name = room[i][j].split('.')
                    if not zone_name in self.shelves_placement[-1]:
                        self.shelves_placement[-1][zone_name] = {}
                    assert not shelf_name in self.shelves_placement[-1][zone_name], "Duplicate names of shelves found"
                    self.shelves_placement[-1][zone_name][shelf_name] = (i, j)

        actor_names = []
        scene_idxs = []
        shelf_ids = []
        shelf_names = []
        zone_ids = []
        zone_names = []
        asset_names = []
        product_names = []
        i = []
        j = []

        for actor_name, actor in self.actors["products"].items():
            actor_names.append(actor_name)

            assert len(actor._scene_idxs) == 1
            scene_idx = actor._scene_idxs.cpu().numpy()[0]
            scene_idxs.append(scene_idx)

            zone_id, shelf_id = self.products2shelves[actor_name]
            zone_ids.append(zone_id)
            shelf_ids.append(shelf_id)
            i.append(self.shelves_placement[scene_idx][zone_id][shelf_id][0])
            j.append(self.shelves_placement[scene_idx][zone_id][shelf_id][1])
            
            zone_name = self.ds_names[scene_idx][zone_id]['zone_name']
            zone_names.append(zone_name)

            shelf_name = self.ds_names[scene_idx][zone_id]['shelf_names'][shelf_id]
            shelf_names.append(shelf_name)

            asset_name = actor_name.replace(f'[ENV#{scene_idx}]_', '')
            asset_names.append(asset_name)
            product_names.append(self.assets_lib['products_hierarchy.' + asset_name.split(':')[0]].asset_name)

        self.products_df = pd.DataFrame(dict(
                actor_name=actor_names,
                scene_idx=scene_idxs,
                zone_id=zone_ids,
                zone_name=zone_names,
                shelf_id=shelf_ids,
                shelf_name=shelf_names,
                product_name=product_names,
                asset_name=asset_names,
                i=i,
                j=j
            )
        )
        self.products_df.to_csv(self.config_dir_path / 'scene_items.csv')
        print(self.products_df)
        print("built")
        print(f"Total {len(self.actors['products'])} products in {self.num_envs} scene(s)")

    def _load_lighting(self, options: dict):
        """Overrides default _load_lighting to avoid loading defauls. The actual lighting is set in dsynth/envs/fixtures/robocasaroom.py"""
        pass

    # def _get_lamps_coords(self):
    #     lamps_coords = []

    #     # TODO: max number of light sources can be reached
    #     for i, j in itertools.product(range(self.x_cells), range(self.y_cells // 2)):
    #         x = CELL_SIZE / 2 + 2 * CELL_SIZE * i
    #         y = CELL_SIZE / 2 + 2 * CELL_SIZE * j
    #         lamps_coords.append((x, y))
        
    #     return lamps_coords

    # def _load_lighting(self, options: dict):
    #     """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""

    #     shadow = self.enable_shadow
    #     self.scene.set_ambient_light([0.4, 0.4, 0.4])
    #     lamp_height = self.assets_lib['fixtures.lamp'].extents[2]
    #     for x, y in self.lamps_coords:
    #         # I have no idea what inner_fov and outer_fov mean :/
    #         self.scene.add_spot_light([x, y, self.height - lamp_height], [0, 0, -1], inner_fov=10, outer_fov=20, color=[20, 20, 20], shadow=shadow)

    # def _load_lighting(self, options: dict):
    #     """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""
    #     # pass
    #     shadow = self.enable_shadow
    #     self.scene.set_ambient_light([0.3, 0.3, 0.3])
    #     self.scene.add_directional_light(
    #         [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
    #     )
    #     self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    # def _load_lamps(self, options: dict):
    #     self.actors["fixtures"]["lamps"] = {}
    #     for n, (x, y) in enumerate(self.lamps_coords):
    #         pose = sapien.Pose(p=[x, y, self.height], q=[1, 0, 0, 0])
    #         lamp = self.assets_lib['fixtures.lamp'].ms_build_actor(f'lamp_{n}', self.scene, pose=pose)
    #         self.actors["fixtures"]["lamps"][f'lamp_{n}'] = lamp

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if not self.is_rebuild:
            raise RuntimeError("To reset arrangement use 'reconfigure' flag: env.reset(options={'reconfigure': True})")
        self.is_rebuild = False
        
        self.product_displaced = False
        self.products_initial_poses = {}
        for p, a in self.actors['products'].items():
            self.products_initial_poses[p] = copy.deepcopy(a.pose.raw_pose)
        if self.robot_uids == "fetch":
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
                    np.pi / 4,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            # self.agent.robot.set_pose(sapien.Pose([0.5, 0.5, 0.0]))
            self.agent.robot.set_pose(sapien.Pose([1.0, 0.5, 0.0]))
            # self._load_shopping_cart(options)
        elif self.robot_uids == "panda_wristcam":
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
            self.agent.robot.set_pose(sapien.Pose([0.0, 0.0, 0.0]))

        elif self.robot_uids in ["ds_fetch", "ds_fetch_basket"]:
            qpos = np.array(
                [
                 0,
                    0,
                    1.57,#np.random.rand() * 6.2832 - 3.1416,
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
            self.agent.robot.set_pose(sapien.Pose([3.7, 1, 0]))
        elif self.robot_uids in ["ds_fetch_static", "ds_fetch_basket_static"]:
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
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([3.7, 1, 0]))

        else:
            raise NotImplementedError

    def hide_object(self, actor):
        if self.hidden_objects_enabled:
            self._hidden_objects.append(actor)

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([7, 7, 7], [5, 5, 2])
        pose = sapien_utils.look_at([-1, 0.3, 1.2], [1, 2, 1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.9, 1.4, 1.3], [0.8, 1.8, 1.05])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]
    
    def _get_obs_extra(self, info: Dict):
        inst_encoded = [np.frombuffer(language_instruction.encode('utf8'), dtype=np.uint8) for language_instruction in self.language_instructions]
        max_length = len(max(inst_encoded, key=lambda x: len(x)))
        mask = np.ones((len(inst_encoded), max_length), dtype=bool)
        for i in range(len(inst_encoded)):
            mask[i][len(inst_encoded[i]):max_length] = False
            inst_encoded[i] = inst_encoded[i].tolist() + [0] * (max_length - len(inst_encoded[i]))
        inst_encoded = np.array(inst_encoded, dtype=np.uint8)
        
        obs = {
            'language_instruction_bytes': inst_encoded,
            'language_instruction_mask': mask
        }

        return obs

