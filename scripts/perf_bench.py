import itertools
import gymnasium as gym
import torch
from tqdm import tqdm
from pathlib import Path
from pynvml import *

import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode

import sys 
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

SCENES_DIR = {
    'high': 'demo_envs/perf_test_milk_only_high',
    'high_downscaled': 'demo_envs/perf_test_milk_only_high_downscaled',
    'med': 'demo_envs/perf_test_milk_only_med',
    'med_downscaled': 'demo_envs/perf_test_milk_only_med_downscaled',
    'low': 'demo_envs/perf_test_milk_only_low',
    'low_downscaled': 'demo_envs/perf_test_milk_only_low_downscaled'
}

def test(
        clutered: str = 'high', 
        shader: str = 'default', 
        num_envs: int = 4,
        downscaled: bool = False,
        all_static: bool = False,
        episode_length: int = 50,
        runs: int = 10
        ):
    h = nvmlDeviceGetHandleByIndex(0)

    key = f'{clutered}_downscaled' if downscaled else f'{clutered}'
    scene_dir = SCENES_DIR[key]

    env = gym.make('NavMoveToZoneEnv', 
                   robot_uids='ds_fetch_basket', 
                   config_dir_path = scene_dir,
                   num_envs=num_envs, 
                   viewer_camera_configs={'shader_pack': shader}, 
                   human_render_camera_configs={'shader_pack': shader},
                   sensor_configs={'shader_pack': shader},
                   render_mode="rgb_array",
                   control_mode=None,
                   enable_shadow=True,
                   sim_config={'spacing': 10},
                   obs_mode='rgbd',
                   sim_backend='gpu',
                   parallel_in_single_scene = False,
                   all_static=all_static
                   )
    gpu_usage_means = []
    elapsed_times = []
    for _ in range(runs):
    # step through the environment with random actions
        obs, _ = env.reset(seed=42, options={'reconfigure': True})

        action = torch.from_numpy(env.action_space.sample())

        gpu_usage = []

        with tqdm(range(episode_length)) as pbar:
            for i in pbar:
                # action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                info = nvmlDeviceGetMemoryInfo(h)
                cur_usage = info.used / info.total
                gpu_usage.append(cur_usage)

            elapsed_time = pbar.format_dict['elapsed']
        
        gpu_usage_means.append(np.array(gpu_usage).mean())
        elapsed_times.append(elapsed_time)
    
    env.close()

    return {
        "elapsed_time_mean": np.mean(elapsed_time),
        "elapsed_time_std": np.std(elapsed_time),
        "gpu_mem_mean": np.mean(gpu_usage_means),
        "gpu_mem_std": np.std(gpu_usage_means)
    }

def main():
    bench_configs = [
        {
            'clutered': 'low',
            'shader': 'default',
            'num_envs': 1,
            'all_static': False,
            'downscaled': False,
        },
        {
            'clutered': 'low',
            'shader': 'default',
            'num_envs': 2,
            'all_static': False,
            'downscaled': False,
        }
    ]
    bench_configs = []
    options_shader = ['default', 'rt-fast', 'rt-med']
    options_all_static = [False, True]
    options_downscaled = [False, True]
    options_cluttered = ['low', 'med', 'high']
    options_num_envs = [1, 2, 4, 8]

    for options in itertools.product(options_shader, options_all_static, options_downscaled, options_cluttered, options_num_envs):
        shader, all_static, downscaled, cluttered, num_envs = options
        bench_configs.append({
            'clutered': cluttered,
            'shader': shader,
            'num_envs': num_envs,
            'all_static': all_static,
            'downscaled': downscaled,
        })
    print(bench_configs)
    for config in bench_configs:
        res = test(**config)
        print("====================")
        print(config)
        print(res)


if __name__ == '__main__':
    nvmlInit()
    main()