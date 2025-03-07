from typing import Dict
import itertools
import os
import json
import torch
import numpy as np
from transforms3d import quaternions
import random
import sapien
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.sapien_env import BaseEnv
from dsynth.envs.fixtures.robocasaroom import RoomFromRobocasa
from dsynth.scene_gen.arrangements import CELL_SIZE, DEFAULT_ROOM_HEIGHT


def get_arena_data(x_cells=4, y_cells=5, height = DEFAULT_ROOM_HEIGHT):
    x_size = x_cells * CELL_SIZE
    y_size = y_cells * CELL_SIZE
    return {
        'meta': {
            'x_cells': x_cells,
            'y_cells': y_cells,
            'x_size': x_size,
            'y_size': y_size,
            'height': height
        },
        'arena_config': {
            'room': {
                'walls': [
                    {'name': 'wall', 'type': 'wall', 'size': [x_size / 2, height / 2, 0.02], 'pos': [x_size / 2, y_size, height / 2]}, 
                    {'name': 'wall_backing', 'type': 'wall', 'backing': True, 'backing_extended': [True, False], 'size': [x_size / 2, height / 2, 0.1], 'pos': [x_size / 2, y_size, height / 2]}, 
                    
                    {'name': 'wall_front', 'type': 'wall', 'wall_side' : 'front', 'size': [x_size / 2, height / 2, 0.02], 'pos': [x_size / 2, 0, height / 2]}, 
                    {'name': 'wall_front_backing', 'type': 'wall', 'wall_side' : 'front', 'backing': True, 'size': [x_size / 2, height / 2, 0.1], 'pos': [x_size / 2, 0, height / 2]}, 
                    
                    {'name': 'wall_left', 'type': 'wall', 'wall_side': 'left', 'size': [y_size / 2, height / 2, 0.02], 'pos': [0, y_size / 2, height / 2]}, 
                    {'name': 'wall_left_backing', 'type': 'wall', 'wall_side': 'left', 'backing': True, 'size': [y_size / 2, height / 2, 0.1], 'pos': [0, y_size / 2, height / 2]}, 
                    
                    {'name': 'wall_right', 'type': 'wall', 'wall_side': 'right', 'size': [y_size / 2, height / 2, 0.02], 'pos': [x_size, y_size / 2, height / 2]}, 
                    {'name': 'wall_right_backing', 'type': 'wall', 'wall_side': 'right', 'backing': True, 'size': [y_size / 2, height / 2, 0.1], 'pos': [x_size, y_size / 2, height / 2]}
                ], 
                'floor': [
                    {'name': 'floor', 'type': 'floor', 'size': [x_size / 2, y_size / 2, 0.02], 'pos': [x_size / 2, y_size / 2, 0.0]}, 
                    {'name': 'floor_backing', 'type': 'floor', 'backing': True, 'size': [x_size / 2, y_size / 2, 0.1], 'pos': [x_size / 2, y_size / 2, 0.0]},
                    # {'name': 'ceiling', 'type': 'floor', 'size': [x_size / 2, y_size / 2, 0.02], 'pos': [x_size / 2, y_size / 2, height]}, 
                    {'name': 'ceiling_backing', 'type': 'floor', 'backing': True, 'size': [x_size / 2, y_size / 2, 0.02], 'pos': [x_size / 2, y_size / 2, height + 4 * 0.02]}
                ]
            }
        }
    }


@register_env('DarkstoreCellBaseEnv', max_episode_steps=200000)
class DarkstoreCellBaseEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is just a very smart environment for goida transformation from ss
    """
    IMPORTED_SS_SCENE_SHIFT = np.array([CELL_SIZE / 2, CELL_SIZE / 2, 0])

    def __init__(self, *args, 
                 assets_lib,
                 robot_uids="panda_wristcam", 
                 scene_json=None, 
                 arena_config = None, 
                 meta = None, 
                 style_ids = 0, 
                #  mapping_file=None,
                 **kwargs):
        self.style_ids = style_ids
        self.arena_config = arena_config
        self.json_file_path = scene_json
        self.assets_lib = assets_lib
        self.x_cells = meta['x_cells']
        self.y_cells = meta['y_cells']
        self.x_size = meta['x_size']
        self.y_size = meta['y_size']
        self.height = meta['height']

        self.lamps_coords = self._get_lamps_coords()
        self.actors = {
            "fixtures": {
                "shelves" : {},
                "lamps": {}
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
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = RoomFromRobocasa(self, arena_config=self.arena_config)
        self.scene_builder.build(self.style_ids)
        self._load_scene_from_json(options)

        self._load_lamps(options)

    def _get_lamps_coords(self):
        lamps_coords = []

        # TODO: max number of light sources can be reached
        for i, j in itertools.product(range(self.x_cells), range(self.y_cells)):
            x = CELL_SIZE / 2 + CELL_SIZE * i
            y = CELL_SIZE / 2 + CELL_SIZE * j
            lamps_coords.append((x, y))
        
        return lamps_coords

    def _load_lighting(self, options: dict):
        """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""

        shadow = self.enable_shadow
        self.scene.set_ambient_light([0.4, 0.4, 0.4])
        lamp_height = self.assets_lib['fixtures.lamp'].extents[2]
        for x, y in self.lamps_coords:
            # I have no idea what inner_fov and outer_fov mean :/
            self.scene.add_spot_light([x, y, self.height - lamp_height], [0, 0, -1], inner_fov=10, outer_fov=20, color=[20, 20, 20], shadow=shadow)

    def _load_lamps(self, options: dict):
        self.actors["fixtures"]["lamps"] = {}
        for n, (x, y) in enumerate(self.lamps_coords):
            pose = sapien.Pose(p=[x, y, self.height], q=[1, 0, 0, 0])
            lamp = self.assets_lib['fixtures.lamp'].ms_build_actor(f'lamp_{n}', self.scene, pose=pose)
            self.actors["fixtures"]["lamps"][f'lamp_{n}'] = lamp
    
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

    def _get_absolute_matrix(self, node, nodes_dict):
        current_matrix = np.array(node[2]["matrix"])
        parent_name = node[0]
        while parent_name != "world":
            parent_node = nodes_dict[parent_name]
            parent_matrix = np.array(parent_node[2]["matrix"])
            current_matrix = parent_matrix @ current_matrix
            parent_name = parent_node[0]
        return current_matrix

    def _get_pq(self, matrix, origin):
        matrix = np.array(matrix)
        q = quaternions.mat2quat(matrix[:3,:3])
        p = matrix[:-1, 3] - origin
        return p, q
    
    def _add_noise(self, p, max_noise = 1e-4):
        new_p = [0] * len(p)
        for i in range(len(p)):
            new_p[i] = p[i] + random.randrange(-max_noise, max_noise)
        return new_p

    
    def _load_scene_from_json(self, options: dict):
        super()._load_scene(options)

        # scale = np.array(options.get("scale", [0.3, 0.3, 0.3]))
        origin = - self.IMPORTED_SS_SCENE_SHIFT#np.array(options.get("origin", [0.0, 1.0, 0.0]))

        with open(self.json_file_path, "r") as f:
            data = json.load(f)

        nodes_dict = {}
        for node in data["graph"]:
            nodes_dict[node[1]] = node

        for node in data["graph"]:
            parent_name, obj_name, props = node
            if '/' not in obj_name:
                abs_matrix = self._get_absolute_matrix(node, nodes_dict)
                p, q = self._get_pq(abs_matrix, origin)
                pose = sapien.Pose(p=p, q=q)
                if 'SHELF' in obj_name:
                    actor = self.assets_lib['fixtures.shelf'].ms_build_actor(obj_name, self.scene, pose=pose)
                    self.actors["fixtures"]["shelves"][obj_name] = {"actor" : actor, "p" : p, "q" : q}
                    continue

                asset_name = f'products_hierarchy.{obj_name.split(":")[0]}'
                actor = self.assets_lib[asset_name].ms_build_actor(obj_name, self.scene, pose=pose)
                self.actors["products"][obj_name] = {"actor" : actor, "p" : p, "q" : q}

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
            self.agent.robot.set_pose(sapien.Pose([0.5, 0.5, 0.0]))
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
