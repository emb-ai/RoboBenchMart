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

    def __init__(self, *args, 
                 config_dir_path,
                 target_product_name=None,
                 robot_uids="panda_wristcam",
                #  style_ids = 0, 
                #  mapping_file=None,
                 **kwargs):
        self.config_dir_path = Path(config_dir_path)
        # self.build_config_idxs = build_config_idxs

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

        self.target_product_name = target_product_name
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
        self.scene_builder.build()
        print("built")

        self.target_product_marker = actors.build_sphere(
            self.scene,
            radius=0.05,
            color=[0, 1, 0, 1],
            name="target_product",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        self._load_shopping_cart(options)
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

        elif self.robot_uids == "ds_fetch":
            qpos = np.array(
                [
                #     # -2. - np.random.randn() * 0.5,
                #     # -1. - np.random.randn() * 0.5,
                #     0,
                #     0,
                #     1.57,#np.random.rand() * 6.2832 - 3.1416,
                #     0.386,
                #     0,
                #     0,
                #     0,
                #      -np.pi / 4,
                #     0,
                #     np.pi / 4,
                #     0,
                #     np.pi / 3,
                #     0,
                #     0.015,
                #     0.015,
                # ]

                 0,
                    0,
                    1.57,#np.random.rand() * 6.2832 - 3.1416,
                    0.386,
                    0,
                    0,
                    0,
                    1.517,
                    0,
                    0,
                    0,
                    - 2 * np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([3.7, 1, 0]))

            # self.ground.set_collision_group_bit(
            #     group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            # )
        else:
            raise NotImplementedError

    def _get_obs_extra(self, info: Dict):
        return dict()

    def evaluate(self):
        target_pos = torch.tensor(self.target_volume.pose.p, dtype=torch.float32)
        target_pos[0][2] -= self.target_sizes[2]/2
        tolerance = torch.tensor(self.target_sizes/2, dtype=torch.float32)
        target_product_pos = self.actors['products'][self.target_product_name].pose.p
        
        is_obj_placed = torch.all(
            (target_product_pos >= (target_pos - tolerance)) & 
            (target_product_pos <= (target_pos + tolerance)),
            dim=-1
        )
        #print("is_obj_placed", is_obj_placed, target_pos, target_product_pos, tolerance)

        is_robot_static = self.agent.is_static(0.2)
        if not self.product_displaced:
            for p, a in self.actors['products'].items():
                if p != self.target_product_name:
                    if not torch.all(torch.isclose(a.pose.raw_pose, self.products_initial_poses[p], rtol=0.1, atol=0.1)):
                        self.product_displaced = True
                        self.target_volume = actors.build_box(
                            self.scene,
                            half_sizes=list(self.target_sizes/2),
                            color=[1, 0, 0, 0.9],
                            name="target_box_red",
                            body_type="static",
                            add_collision=False,
                            initial_pose=self.target_volume.pose,
                        )
                        break
        
        is_object_grasped = self.agent.is_grasping(self.actors['products'][self.target_product_name])

        print("is_obj_placed", is_obj_placed.item(), "product_displaced", self.product_displaced, "is_object_grasped", is_object_grasped.item())
        return {
            "first" : target_product_pos,
            "second" : target_pos,
            "third" : target_pos - tolerance,
            "fourth" : target_pos + tolerance,
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_object_grasped": is_object_grasped,
            "product_displaced": self.product_displaced
        }


    # def _get_obs_extra(self, info: Dict):
    #     """Get task-relevant extra observations. Usually defined on a task by task basis"""
    #     lang_task = dict(language_instruction = str.encode(LANGUAGE_INSTRUCTION))
    #     return lang_task
    
    def _load_shopping_cart(self, options: dict):
        # recommended to use shift = (0,0.5,0)
        # print(self.unwrapped.agent.robot.get_pose())
        if not hasattr(self, 'shopping_cart'):
            shopping_cart_asset = "/mnt/disk2tb/soshin/repo/darkstore_synthesizer/assets/smallShoppingCart2.glb" #os.path.join(self.assets_dir, "smallShoppingCart2.glb")
            
            if not os.path.exists(shopping_cart_asset):
                print(f"Shopping cart asset not found: {shopping_cart_asset}")
            else:
                builder = self.scene.create_actor_builder()
                builder.add_visual_from_file(filename=shopping_cart_asset, scale=np.array([1.0, 1.0, 1.0]))
                builder.add_nonconvex_collision_from_file(filename=shopping_cart_asset, scale=np.array([1.0, 1.0, 1.0]))
                shopping_cart_pose = sapien.Pose(p=[11.0, 10.0, 0.0], q=np.array([1, 0, 0, 0]))
                builder.set_initial_pose(shopping_cart_pose)
                self.shopping_cart = builder.build_static(name="shopping_cart")
                #self.actors.append(self.shopping_cart)

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
    
    def setup_target_object(self):
        self.target_product_name = 'food.dairy_products.milk:1:1:52'
        # self.target_product_name = 'food.dairy_products.milk:1:1:24'
        if self.target_product_name is None:
            random_product_int = random.randint(0, len(self.actors['products']))
            self.target_product_name = list(self.actors['products'].keys())[random_product_int]
            print("Target product selected randomly")
        obb = get_actor_obb(self.actors['products'][self.target_product_name])
        center = np.array(obb.primitive.transform)[:3, 3]

        self.target_product_marker.set_pose(sapien.Pose(center))

        # goal_pose = self.actors["fixtures"]["scene_assets"][f'[ENV#0]_cart'].pose
        # goal_obb = self.actors["fixtures"]["scene_assets"][f'[ENV#0]_cart'].get_collision_meshes()[0].bounding_box_oriented
        # goal_center = np.array(goal_obb.primitive.transform)[:3, 3]

        # self.goal_zone.set_pose(sapien.Pose(goal_center))