from typing import Dict
import itertools
import os
import json
import torch
import numpy as np
from transforms3d import quaternions
import random
import sapien
from pathlib import Path
import hydra
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.sapien_env import BaseEnv
from dsynth.envs.fixtures.robocasaroom import DarkstoreScene
from dsynth.scene_gen.arrangements import CELL_SIZE, DEFAULT_ROOM_HEIGHT
from dsynth.assets.asset import load_assets_lib
from dsynth.scene_gen.utils import flatten_dict
from mani_skill.utils.structs.pose import Pose


@register_env('DarkstoreCellBaseEnv', max_episode_steps=200000)
class DarkstoreCellBaseEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]


    def __init__(self, *args, 
                 config_dir_path,
                 robot_uids="panda_wristcam",
                 build_config_idxs=None,
                #  style_ids = 0, 
                #  mapping_file=None,
                 **kwargs):
        self.config_dir_path = Path(config_dir_path)
        self.build_config_idxs = build_config_idxs

        with hydra.initialize_config_dir(config_dir=str(self.config_dir_path.absolute()), version_base=None):
            cfg = hydra.compose(config_name='input_config')

        self.assets_lib = flatten_dict(load_assets_lib(cfg.assets), sep='.')

        self.actors = {
            "fixtures": {
                "shelves" : {},
                "lamps": {},
                "scene_assets": {}
            },
            "products": {}
        }

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
        self.actors = {
            "fixtures": {
                "shelves" : {},
                "lamps": {},
                "scene_assets": {}
            },
            "products": {}
        }
        self.scene_builder = DarkstoreScene(self, config_dir_path=self.config_dir_path)

        # if self.build_config_idxs is None:
        build_config_idxs = []
        for i in range(self.num_envs):
            # Total number of configs is 10 * 12 = 120
            config_idx = self._batched_episode_rng[i].randint(0, self.scene_builder.num_generated_scenes * 12)
            build_config_idxs.append(config_idx)
        
        self.build_config_idxs = build_config_idxs
                
        self.scene_builder.build(self.build_config_idxs)
        # self.scene_builder.load_scene_from_json(self.json_file_path)

        # self._load_lamps(options)

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
    
    def _process_string(self, s):
        if '_' in s:
            return s.split('_',1)[0] + '.obj'
        if '.' in s:
            return s.split('.',1)[0] + '.obj'
        return s + '.obj'

    def _temp_process_string(self, s):
        for i, char in enumerate(s):
            if char in "_." or char.isdigit():
                return s[:i] + ".obj"
        return s + ".obj"

    
    def _add_noise(self, p, max_noise = 1e-4):
        new_p = [0] * len(p)
        for i in range(len(p)):
            new_p[i] = p[i] + random.randrange(-max_noise, max_noise)
        return new_p


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

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
            
        else:
            raise NotImplementedError

    def _get_obs_extra(self, info: Dict):
        return dict()
