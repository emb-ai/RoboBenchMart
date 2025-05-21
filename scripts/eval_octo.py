import multiprocessing as mp
from functools import partial
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from typing import Optional

from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories

from octo.model.octo_model_pt import OctoModelPt
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper, ResizeImageWrapper, TemporalEnsembleWrapper
from octo.utils.train_utils_pt import tree_map, _np2pt

import sys
sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

from dsynth.planning import MP_SOLUTIONS


OPEN = 1
CLOSED = -1

class MyRecordEpisode(RecordEpisode):
    @property
    def h5_file(self):
        return self._h5_file

class OctoEnv(gym.Wrapper):
    """
    Performs receding horizon control. The policy returns `pred_horizon` actions and
    we execute `exec_horizon` of them.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        new_obs = {}
        new_obs['image_primary'] = obs['sensor_data']['left_base_camera_link']['rgb'].detach().cpu().numpy()[0]
        new_obs['image_secondary'] = obs['sensor_data']['right_base_camera_link']['rgb'].detach().cpu().numpy()[0]
        new_obs['image_wrist'] = obs['sensor_data']['fetch_hand']['rgb'].detach().cpu().numpy()[0]

        return new_obs, reward, done, trunc, info
    
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        new_obs = {}
        new_obs['image_primary'] = obs['sensor_data']['left_base_camera_link']['rgb'].detach().cpu().numpy()[0]
        new_obs['image_secondary'] = obs['sensor_data']['right_base_camera_link']['rgb'].detach().cpu().numpy()[0]
        new_obs['image_wrist'] = obs['sensor_data']['fetch_hand']['rgb'].detach().cpu().numpy()[0]

        return new_obs, info

    def get_task(self):
        task = {
            "language_instruction": self.env.language_instruction,
            "goal": {}
        }

        return task


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickToCartEnv", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument("--scene-dir", type=str)
    parser.add_argument("--finetuned-path", type=str)
    parser.add_argument("--device", type=str, default='cuda:0')
    # parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=50, help="Number of trajectories to generate.")
    parser.add_argument("-m", "--max-horizon", type=int, default=500)

    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    # parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    return parser.parse_args()

def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    model = OctoModelPt.load_pretrained(args.finetuned_path)['octo_model']
    model.to(args.device)

    image_shapes = {key.replace('image_', ''): val.shape[-2:] for key, val in model.example_batch['observation'].items() if 'image_' in key}

    env_id = args.env_id
    scene_dir = args.scene_dir
    record_dir = args.scene_dir + '/evaluations'
    env = gym.make(env_id, 
                    robot_uids='ds_fetch',
                   config_dir_path = scene_dir,
                   num_envs=1, 
                   control_mode="pd_joint_pos",
                   viewer_camera_configs={'shader_pack': args.shader}, 
                    human_render_camera_configs={'shader_pack': args.shader},
                    sensor_configs={'shader_pack': args.shader},
                #    render_mode="human" if gui else "rgb_array", 
                   render_mode="rgb_array", 
                   enable_shadow=True,
                   obs_mode='rgbd',
                   parallel_in_single_scene = False,
                   )
    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = MyRecordEpisode(
        env,
        output_dir=osp.join(record_dir, "octo_pt"),
        trajectory_name=new_traj_name, save_video=args.save_video,
        source_type="octo_pt model",
        source_desc="official motion planning solution from dsynth contributors",
        video_fps=30,
        record_reward=False,
        save_on_reset=False
    )

    env = OctoEnv(env)
    env = ResizeImageWrapper(env, 
        resize_size = image_shapes,
        augmented_keys = ()
    )
    env = HistoryWrapper(env, horizon=2)
    env = RHCWrapper(env, exec_horizon=1)
    policy_fn = partial(
        model.sample_actions,
        unnormalization_statistics=model.dataset_statistics["action"],
        generator=torch.Generator(args.device).manual_seed(0),
    )
    env = TemporalEnsembleWrapper(env, pred_horizon = 4, exp_weight=0.1)


    output_h5_path = env.h5_file.filename
    
    print(f"Octo evaluation on {env_id}")
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0

    for _ in range(args.num_traj):
        obs, info = env.reset(seed=seed, options={'reconfigure': True})
        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=[language_instruction], device=args.device)

        for i in range(args.max_horizon):
            obs['timestep_pad_mask'] = obs['timestep_pad_mask'].astype(np.bool_)
            obs = _np2pt(obs, args.device)
            
            actions = policy_fn(tree_map(lambda x: x[None], obs), task)
            actions = actions[0]
            actions[:, 7] = -(actions[:, 7] - 0.5) * 2

            obs, reward, done, trunc, info = env.step(actions)
            if args.vis:
                env.render_human()

            if done or trunc:
                break
        
        success = info["success"][0].item()
        elapsed_steps = info["elapsed_steps"][0].item()
        solution_episode_lengths.append(elapsed_steps)

        successes.append(success)

        env.flush_trajectory()
        if args.save_video:
            env.flush_video()
        pbar.update(1)
        pbar.set_postfix(
            dict(
                success_rate=np.mean(successes),
                failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                avg_episode_length=np.mean(solution_episode_lengths),
                max_episode_length=np.max(solution_episode_lengths),
                # min_episode_length=np.min(solution_episode_lengths)
            )
        )
        seed += 1

    env.close()
    return output_h5_path

def main(args):
    if args.num_procs > 1 and args.num_procs < args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        # Merge trajectory files
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        _main(args)

if __name__ == "__main__":
    # start = time.time()
    mp.set_start_method("spawn")
    main(parse_args())
    # print(f"Total time taken: {time.time() - start}")
