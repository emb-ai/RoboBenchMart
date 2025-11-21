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
from dsynth.envs import *
from dsynth.robots import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Использование: python script.py <путь_к_JSON_файлу> <путь_к_assets> <style id (0-11)> [mapping_file]"
    )
    parser.add_argument("scene_dir", help="Путь к директории с JSON конфигом сцены")
    parser.add_argument("-e", "--env-id", type=str, default="DarkstoreContinuousBaseEnv", help=f"Environment to run")
    parser.add_argument("-r", "--robot-uids", type=str, default="ds_fetch_basket", help=f"Robot id")
    parser.add_argument("-n", "--num-envs", type=int, default=1, help=f"Number of scenes")
    parser.add_argument("-s", "--seed", type=int, nargs='+', default=0)
    parser.add_argument('--shader',
                        default='default',
                        const='default',
                        nargs='?',
                        choices=['rt', 'rt-fast', 'rt-med', 'default', 'minimal'],)
    parser.add_argument('--sensor_shader',
                        default='minimal',
                        const='minimal',
                        nargs='?',
                        choices=['rt', 'rt-fast', 'rt-med', 'default', 'minimal'],)
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
    gui = args.gui
    parallel_in_single_scene = args.num_envs > 1 and gui
    env = gym.make(args.env_id, 
                   robot_uids=args.robot_uids, 
                   config_dir_path = args.scene_dir,
                   num_envs=args.num_envs, 
                   viewer_camera_configs={'shader_pack': args.shader}, 
                    human_render_camera_configs={'shader_pack': args.shader},
                    sensor_configs={'shader_pack': args.sensor_shader},
                   render_mode="human" if gui else "rgb_array", 
                #    render_mode="rgb_array", 
                   control_mode=None,
                   enable_shadow=True,
                   sim_config={'spacing': 20},
                   obs_mode='none' if gui else "rgbd",
                   sim_backend='auto',
                   parallel_in_single_scene = parallel_in_single_scene,
                   )

    new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    video_path = scene_dir / f"./videos_seed={args.seed}_shader={args.shader}_sensor_shader={args.sensor_shader}"
    env = RecordEpisode(
        env,
        output_dir=video_path,
        trajectory_name=new_traj_name,
        save_video=args.video,
        video_fps=30,
        avoid_overwriting_video=True,
        max_steps_per_video=10
    )

    print("Video path:", video_path)
    print("Trajectoty name:", new_traj_name)

    env.reset(seed=args.seed, options={'reconfigure': True})

    if gui:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = True
        # env.render()


    for i in tqdm(range(args.episode_length)):
        action = torch.from_numpy(env.action_space.sample())
        # action = torch.zeros_like(torch.from_numpy(env.action_space.sample()))
        obs, reward, terminated, truncated, info = env.step(action)
        print(info)
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