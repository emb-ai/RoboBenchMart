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
from dsynth.envs.darkstore_cont_base import DarkstoreContinuousBaseEnv

from dsynth.envs.pick_to_basket import *


def get_composite_pick_task(env_name, *envs):
    @register_env(env_name, max_episode_steps=200000)
    class CompositePickTask(*envs):
        TASK_ENVS = envs
        CUR_TASK_IDX = 0
        INSTRUCTION_MAX_LEN = 256

        def setup_target_objects(self, env_idxs):
            self.target_items_names = []
            self.TASK_ENVS[0].setup_target_objects(self, env_idxs)

        def evaluate(self):
            if self.num_envs > 1:
                raise NotImplementedError("Only one scene is supported by now")
            
            is_success = torch.tensor([True])
            succeses = {f"task_{i}": torch.tensor([True]) if i < self.CUR_TASK_IDX else torch.tensor([False]) \
                        for i in range(len(self.TASK_ENVS))}
            
            if self.CUR_TASK_IDX >= len(self.TASK_ENVS):
                # all tasks are solved
                succeses['success'] = is_success
                return succeses
            
            cur_task_success = self.TASK_ENVS[self.CUR_TASK_IDX].evaluate(self)['success']
            if cur_task_success:
                # next task
                self.CUR_TASK_IDX += 1
                self.target_items_names.append(self.TARGET_PRODUCT_NAME)
                if self.CUR_TASK_IDX < len(self.TASK_ENVS):
                    # setup target product name
                    self.TARGET_PRODUCT_NAME = self.TASK_ENVS[self.CUR_TASK_IDX].TARGET_PRODUCT_NAME

                    # setup target products
                    self.TASK_ENVS[self.CUR_TASK_IDX].setup_target_objects(self, torch.range(0, self.num_envs - 1))

                    # update init product positions
                    self.TASK_ENVS[self.CUR_TASK_IDX].store_products_init_poses(self, exclude_items_names=self.target_items_names)

                    #update language instruction
                    self.TASK_ENVS[self.CUR_TASK_IDX].setup_language_instructions(self, torch.range(0, self.num_envs - 1))

            succeses['success'] = torch.tensor([False])

            return succeses
        
        def _get_obs_extra(self, info: Dict):
            inst_encoded = [np.frombuffer(language_instruction.encode('utf8'), dtype=np.uint8) for language_instruction in self.language_instructions]

            # pad to fixed (large enough) length
            max_length = self.INSTRUCTION_MAX_LEN
            
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
    
    return CompositePickTask

PickNiveaFantaEnv = get_composite_pick_task('PickNiveaFantaEnv', PickToBasketContNiveaEnv, PickToBasketContFantaEnv)
PickNiveaFantaStarsEnv = get_composite_pick_task('PickNiveaFantaStarsEnv', PickToBasketContNiveaEnv, PickToBasketContFantaEnv, PickToBasketContStarsEnv)

