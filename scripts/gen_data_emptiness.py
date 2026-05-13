import sapien
import sys
import json
import gymnasium as gym
import torch
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import time 
import hydra
import cv2
import json
import numpy as np

import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode

import sys 
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *
from dsynth.auxiliary.empty_space_detection import detect_free_spaces
from dsynth.annotate_obs import prepare_observations


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value

def parse_args():
    parser = argparse.ArgumentParser(
        description="Использование: python script.py <путь_к_JSON_файлу> <путь_к_assets> <style id (0-11)> [mapping_file]"
    )
    parser.add_argument("scene_dir", help="Путь к директории с JSON конфигом сцены")
    parser.add_argument("-e", "--env-id", type=str, default="DarkstoreContinuousBaseEnv", help=f"Environment to run")
    parser.add_argument("-s", "--seed", type=int, nargs='+', default=0)
    parser.add_argument('--episode_length', type=int, default=10)
    parser.add_argument('--video',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    return args

def main(args):

    scene_dir = Path(args.scene_dir)
    
    env = gym.make('RandomItemDeletionEnv', 
                   robot_uids='ds_fetch_basket', 
                   config_dir_path = args.scene_dir,
                   num_envs=1, 
                    sensor_configs={'shader_pack': 'default'},
                   render_mode="rgb_array", 
                #    render_mode="rgb_array", 
                   control_mode='pd_ee_delta_pose',
                   enable_shadow=True,
                   sim_config={'spacing': 20},
                   obs_mode='rgb+segmentation',
                   sim_backend='auto',
                   parallel_in_single_scene = False,
                   )

    output_dir = Path(args.scene_dir) / 'emptiness_data'
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    descriptions_dir = output_dir / 'descriptions'
    descriptions_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in range(10):
        env.reset(seed=seed, options={'reconfigure': True})
        
        sample_name = f'seed={seed}'
        obs = prepare_observations(env)
        obs_free = detect_free_spaces(env)
        cv2.imwrite(str(images_dir / f'{sample_name}_im.jpg'), obs['combined']['image'])
        cv2.imwrite(str(images_dir / f'{sample_name}_annot.jpg'), obs['combined']['annotated_image'])

        data = {
            'scene_description': obs['scene_description'],
            'items_bboxes': obs['bboxes'],
            'free_spaces': obs_free['spaces_description'],
            'bboxes': obs_free['bboxes']
        }
        with open(descriptions_dir / f'{sample_name}.json', "w") as f:
            json.dump(to_jsonable(data), f, indent=4)

    env.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)