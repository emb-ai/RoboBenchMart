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

import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode

import sys 
sys.path.append('.')
from dsynth.envs.darkstore_cell_base import get_arena_data
from dsynth.envs.pick_to_cart import PickToCartEnv
from dsynth.assets.asset import load_assets_lib
from dsynth.scene_gen.utils import flatten_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Использование: python script.py <путь_к_JSON_файлу> <путь_к_assets> <style id (0-11)> [mapping_file]"
    )
    parser.add_argument("scene_dir", help="Путь к директории с JSON конфигом сцены")
    parser.add_argument("--style_id", type=int, default=0, help="Style id (0-11)")
    parser.add_argument('--shader',
                        default='default',
                        const='default',
                        nargs='?',
                        choices=['rt', 'rt-fast', 'default', 'minimal'],)
    parser.add_argument('--gui',
                        action='store_true',
                        default=False)
    parser.add_argument('--episode_length', type=int, default=10)
    parser.add_argument('--video',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    return args

def main(args):

    scene_dir = Path(args.scene_dir)
    yaml_config_path = scene_dir / 'input_config.yaml'
    json_file_path = scene_dir / 'scene_config.json'
    style_id = args.style_id
    gui = args.gui

    with hydra.initialize(config_path='../' + str(scene_dir), version_base=None):
        cfg = hydra.compose(config_name='input_config.yaml')
    assets_lib = flatten_dict(load_assets_lib(cfg.assets), sep='.')
    
    with open(json_file_path, "r") as f: # big_scene , one_shelf_many_milk_scene , customize
        data = json.load(f)

    n = data['meta']['n']
    m = data['meta']['m']
    arena_data = get_arena_data(x_cells=n, y_cells=m)

    env = gym.make('PickToCartEnv', 
                   robot_uids='fetch', 
                   scene_json = json_file_path,
                   assets_lib = assets_lib,
                   style_ids = [style_id], 
                   num_envs=1, 
                   viewer_camera_configs={'shader_pack': args.shader}, 
                    human_render_camera_configs={'shader_pack': args.shader},
                #    render_mode="human" if gui else "rgb_array", 
                   render_mode="human", 
                #    control_mode='pd_ee_delta_pos',
                   enable_shadow=True,
                #    obs_mode='rgbd',
                   parallel_in_single_scene = False,
                   **arena_data)

    new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    video_path = scene_dir / f"./videos_style={style_id}_shader={args.shader}"
    env = RecordEpisode(
        env,
        output_dir=video_path,
        trajectory_name=new_traj_name,
        save_video=args.video,
        video_fps=30,
        avoid_overwriting_video=True
    )

    print("Video path:", video_path)
    print("Trajectoty name:", new_traj_name)

    # step through the environment with random actions
    obs, _ = env.reset()


    viewer = env.render()
    if isinstance(viewer, sapien.utils.Viewer):
        viewer.paused = False
    # env.render()

    action = torch.zeros_like(torch.from_numpy(env.action_space.sample()))
    # action[-1] = -1

    for i in tqdm(range(args.episode_length)):
        # action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # rgb_image = obs["sensor_data"]["base_camera"]["rgb"]
        # rgb_image = rgb_image.permute((0, 3, 1, 2))

        if gui:
            env.render_human()

    # render wait
    if gui:
        viewer = env.render_human()
        while True:
            if viewer.closed:
                exit()
            if viewer.window.key_down("c"):
                break
            env.render_human()

    env.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)