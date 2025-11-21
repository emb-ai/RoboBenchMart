from typing import Dict, Union
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
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils, common
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.sapien_env import BaseEnv
from dsynth.envs.fixtures.robocasaroom import DarkstoreScene
from dsynth.scene_gen.arrangements import CELL_SIZE, DEFAULT_ROOM_HEIGHT
from dsynth.assets.asset import load_assets_lib
from dsynth.scene_gen.utils import flatten_dict
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG


@register_env('DarkstoreCellBaseEnv', max_episode_steps=200000)
class DarkstoreCellBaseEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    NUM_MARKERS = 100

    def __init__(self, *args, 
                 config_dir_path,
                 user_target_product_name=None,
                 robot_uids="panda_wristcam",
                 markers_enabled=False,
                 hidden_objects_enabled=True,
                 all_static=False,
                 **kwargs):
        self.config_dir_path = Path(config_dir_path)
        self.is_rebuild = False

        self.markers_enabled = markers_enabled
        self.extra_robot_pose_randomization = False
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

    def set_init_pose_rand_seed(self, seed: Union[int, list[int]] = 0):
        if self.extra_robot_pose_randomization:
            if not np.iterable(seed):
                seed = [seed]
            self._init_pose_rand_seed = common.to_numpy(seed, dtype=np.int64)
            if len(self._init_pose_rand_seed) == 1 and self.num_envs > 1:
                self._init_pose_rand_seed = np.concatenate((self._init_pose_rand_seed, np.random.RandomState(self._init_pose_rand_seed[0]).randint(2**31, size=(self.num_envs - 1,))))
            self._batched_init_pose_rng = BatchedRNG.from_seeds(self._init_pose_rand_seed, backend=self._batched_rng_backend)

    def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        if options is None:
            options = dict()
        reconfigure = options.get("reconfigure", False)
        robot_init_pose_seed = options.get("robot_init_pose_seed", None)
        if robot_init_pose_seed is not None:
            self.extra_robot_pose_randomization = True

        reconfigure = reconfigure or (
            self._reconfig_counter == 0 and self.reconfiguration_freq != 0
        )
        if "env_idx" in options:
            env_idx = options["env_idx"]
            if len(env_idx) != self.num_envs and reconfigure:
                raise RuntimeError("Cannot do a partial reset and reconfigure the environment. You must do one or the other.")
        else:
            env_idx = torch.arange(0, self.num_envs, device=self.device)

        self._set_main_rng(seed)

        if reconfigure:
            self._set_episode_rng(seed if seed is not None else self._batched_main_rng.randint(2**31), env_idx)
            if self.extra_robot_pose_randomization:
                self.set_init_pose_rand_seed(robot_init_pose_seed)
            
            with torch.random.fork_rng():
                torch.manual_seed(seed=self._episode_seed[0])
                self._reconfigure(options)
                self._after_reconfigure(options)
            # Set the episode rng again after reconfiguration to guarantee seed reproducibility
            self._set_episode_rng(self._episode_seed, env_idx)
        else:
            self._set_episode_rng(seed, env_idx)
            if self.extra_robot_pose_randomization:
                self.set_init_pose_rand_seed(robot_init_pose_seed)
            

        # TODO (stao): Reconfiguration when there is partial reset might not make sense and certainly broken here now.
        # Solution to resolve that would be to ensure tasks that do reconfigure more than once are single-env only / cpu sim only
        # or disable partial reset features explicitly for tasks that have a reconfiguration frequency
        self.scene._reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.scene._reset_mask[env_idx] = True
        self._elapsed_steps[env_idx] = 0

        self._clear_sim_state()
        if self.reconfiguration_freq != 0:
            self._reconfig_counter -= 1

        if self.agent is not None:
            self.agent.reset()

        if seed is not None or self._enhanced_determinism:
            with torch.random.fork_rng():
                torch.manual_seed(self._episode_seed[0])
                self._initialize_episode(env_idx, options)
        else:
            self._initialize_episode(env_idx, options)
        # reset the reset mask back to all ones so any internal code in maniskill can continue to manipulate all scenes at once as usual
        self.scene._reset_mask = torch.ones(
            self.num_envs, dtype=bool, device=self.device
        )
        if self.gpu_sim_enabled:
            # ensure all updates to object poses and configurations are applied on GPU after task initialization
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

        # we reset controllers here because some controllers depend on the agent/articulation qpos/poses
        if self.agent is not None:
            if isinstance(self.agent.controller, dict):
                for controller in self.agent.controller.values():
                    controller.reset()
            else:
                self.agent.controller.reset()

        info = self.get_info()
        obs = self.get_obs(info)

        info["reconfigure"] = reconfigure
        return obs, info



    @property
    def _default_sensor_configs(self):
        # pose = sapien_utils.look_at([0.7, 1.8, 1.15], [1.2, 2.2, 1.2])
        # return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]
        return []

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

        actor_names = []
        scene_idxs = []
        zone_ids = []
        zone_names = []
        shelf_ids = []
        shelf_names = []
        shelf_types = []
        i = []
        j = []

        for actor_name, actor in self.actors['fixtures']['shelves'].items():
            actor_names.append(actor_name)

            assert len(actor._scene_idxs) == 1
            scene_idx = actor._scene_idxs.cpu().numpy()[0]
            scene_idxs.append(scene_idx)

            zone_id, shelf_id = re.sub(r"\[ENV#\d\]_SHELF_\d+_", "", actor_name).split('.')
            zone_ids.append(zone_id)
            shelf_ids.append(shelf_id)

            i.append(self.shelves_placement[scene_idx][zone_id][shelf_id][0])
            j.append(self.shelves_placement[scene_idx][zone_id][shelf_id][1])

            zone_name = self.ds_names[scene_idx][zone_id]['zone_name']
            zone_names.append(zone_name)

            shelf_name = self.ds_names[scene_idx][zone_id]['shelf_names'][shelf_id]
            shelf_names.append(shelf_name)

            shelf_type = self.ds_names[scene_idx][zone_id]['shelf_types'][shelf_id]
            shelf_types.append(shelf_type)

        self.shelvings_df = pd.DataFrame(dict(
                actor_name=actor_names,
                scene_idx=scene_idxs,
                zone_id=zone_ids,
                zone_name=zone_names,
                shelf_id=shelf_ids,
                shelf_name=shelf_names,
                shelf_type=shelf_types,
                i=i,
                j=j
            )
        )


        self.products_df.to_csv(self.config_dir_path / 'scene_items.csv')
        self.shelvings_df.to_csv(self.config_dir_path / 'shelvings.csv')
        print(self.products_df)
        print("built")
        print(f"Total {len(self.actors['products'])} products in {self.num_envs} scene(s)")

    def _load_lighting(self, options: dict):
        """Lighting is additionally set in dsynth/envs/fixtures/robocasaroom.py"""
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=2048,
    
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def store_products_init_poses(self):
        self.product_displaced = False
        self.products_initial_poses = {}
        for p, a in self.actors['products'].items():
            self.products_initial_poses[p] = copy.deepcopy(a.pose.raw_pose)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if not self.is_rebuild:
            raise RuntimeError("To reset arrangement use 'reconfigure' flag: env.reset(options={'reconfigure': True})")
        self.is_rebuild = False
        
        self.store_products_init_poses()
        self.setup_language_instructions(env_idx)

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
    
    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(f'do nothing')
    
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

