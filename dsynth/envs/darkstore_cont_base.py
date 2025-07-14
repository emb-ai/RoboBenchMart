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
        self.products2shelves = {}

        self.scene_builder = DarkstoreSceneContinuous(self, config_dir_path=self.config_dir_path)
        self.scene_builder.build()

        # self.shelves_placement = []
        # for room in self.room:
        #     self.shelves_placement.append({})
        #     for i, j in itertools.product(range(len(room)), range(len(room[0]))):
        #         if room[i][j] != 0:
        #             zone_name, shelf_name = room[i][j].split('.')
        #             if not zone_name in self.shelves_placement[-1]:
        #                 self.shelves_placement[-1][zone_name] = {}
        #             assert not shelf_name in self.shelves_placement[-1][zone_name], "Duplicate names of shelves found"
        #             self.shelves_placement[-1][zone_name][shelf_name] = (i, j)

        # actor_names = []
        # scene_idxs = []
        # shelf_ids = []
        # shelf_names = []
        # zone_ids = []
        # zone_names = []
        # asset_names = []
        # product_names = []
        # i = []
        # j = []

        # for actor_name, actor in self.actors["products"].items():
        #     actor_names.append(actor_name)

        #     assert len(actor._scene_idxs) == 1
        #     scene_idx = actor._scene_idxs.cpu().numpy()[0]
        #     scene_idxs.append(scene_idx)

        #     zone_id, shelf_id = self.products2shelves[actor_name]
        #     zone_ids.append(zone_id)
        #     shelf_ids.append(shelf_id)
        #     i.append(self.shelves_placement[scene_idx][zone_id][shelf_id][0])
        #     j.append(self.shelves_placement[scene_idx][zone_id][shelf_id][1])
            
        #     zone_name = self.ds_names[scene_idx][zone_id]['zone_name']
        #     zone_names.append(zone_name)

        #     shelf_name = self.ds_names[scene_idx][zone_id]['shelf_names'][shelf_id]
        #     shelf_names.append(shelf_name)

        #     asset_name = actor_name.replace(f'[ENV#{scene_idx}]_', '')
        #     asset_names.append(asset_name)
        #     product_names.append(self.assets_lib['products_hierarchy.' + asset_name.split(':')[0]].asset_name)

        # self.products_df = pd.DataFrame(dict(
        #         actor_name=actor_names,
        #         scene_idx=scene_idxs,
        #         zone_id=zone_ids,
        #         zone_name=zone_names,
        #         shelf_id=shelf_ids,
        #         shelf_name=shelf_names,
        #         product_name=product_names,
        #         asset_name=asset_names,
        #         i=i,
        #         j=j
        #     )
        # )

        # actor_names = []
        # scene_idxs = []
        # zone_ids = []
        # zone_names = []
        # shelf_ids = []
        # shelf_names = []
        # shelf_types = []
        # i = []
        # j = []

        # for actor_name, actor in self.actors['fixtures']['shelves'].items():
        #     actor_names.append(actor_name)

        #     assert len(actor._scene_idxs) == 1
        #     scene_idx = actor._scene_idxs.cpu().numpy()[0]
        #     scene_idxs.append(scene_idx)

        #     zone_id, shelf_id = re.sub(r"\[ENV#\d\]_SHELF_\d+_", "", actor_name).split('.')
        #     zone_ids.append(zone_id)
        #     shelf_ids.append(shelf_id)

        #     i.append(self.shelves_placement[scene_idx][zone_id][shelf_id][0])
        #     j.append(self.shelves_placement[scene_idx][zone_id][shelf_id][1])

        #     zone_name = self.ds_names[scene_idx][zone_id]['zone_name']
        #     zone_names.append(zone_name)

        #     shelf_name = self.ds_names[scene_idx][zone_id]['shelf_names'][shelf_id]
        #     shelf_names.append(shelf_name)

        #     shelf_type = self.ds_names[scene_idx][zone_id]['shelf_types'][shelf_id]
        #     shelf_types.append(shelf_type)

        # self.shelvings_df = pd.DataFrame(dict(
        #         actor_name=actor_names,
        #         scene_idx=scene_idxs,
        #         zone_id=zone_ids,
        #         zone_name=zone_names,
        #         shelf_id=shelf_ids,
        #         shelf_name=shelf_names,
        #         shelf_type=shelf_types,
        #         i=i,
        #         j=j
        #     )
        # )


        # self.products_df.to_csv(self.config_dir_path / 'scene_items.csv')
        # self.shelvings_df.to_csv(self.config_dir_path / 'shelvings.csv')
        # print(self.products_df)
        print("built")
        print(f"Total {len(self.actors['products'])} products in {self.num_envs} scene(s)")
