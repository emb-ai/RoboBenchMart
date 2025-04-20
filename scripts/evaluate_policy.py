import sys
import gymnasium as gym
import torch
import sapien
from tqdm import tqdm
from pathlib import Path

import sys 
sys.path.append('.')
from dsynth.envs.pick_to_cart import PickToCartEnv
from dsynth.envs.move_from_board_to_board import MoveFromBoardToBoardEnv
from dsynth.envs.pick_from_floor import PickFromFloorEnv
from dsynth.envs.pick_from_cart import PickFromCartEnv
from dsynth.envs.place_on_top import PlaceOnTopEnv
from mani_skill.utils.wrappers.record import RecordEpisode
import h5py

class DarkStoreTask(object):
    def __init__(self, **kwargs):
        self.env = gym.make(self.env_name, 
                        robot_uids='panda_wristcam', 
                        config_dir_path = self.config_dir_path,
                        target_product_name=self.target_product_name,
                        num_envs=1, 
                        viewer_camera_configs={'shader_pack': 'default'}, 
                        human_render_camera_configs={'shader_pack': 'default'},
                        render_mode="rgb_array", 
                        control_mode='pd_joint_pos',
                        enable_shadow=True,
                        obs_mode='rgbd',
                        parallel_in_single_scene = False,
                        **kwargs
                        )
        self.env = RecordEpisode(
            self.env,
            output_dir=f"videos/{self.reference_trajectory.split('/')[1].split('.')[0]}",
            trajectory_name="trajectory",
            save_video=True,
            info_on_video=False,
            save_trajectory=False,
            video_fps=30,
        )
        self.env.reset(options={'reconfigure': True})
    
    def evaluate(self):
        with h5py.File(self.reference_trajectory, 'r') as file:
            actions = file['actions']
            for i in tqdm(range(len(actions))):
                obs, reward, terminated, truncated, info = self.env.step(actions[i])
                self.env.render_human()
            self.env.close()

class PickToCartTask(DarkStoreTask):
    def __init__(self, instruction, target_product_name):
        self.instruction = instruction
        self.target_product_name = target_product_name
        self.reference_trajectory = "teleop_demo_actions/pick_to_cart.h5"
        self.env_name = "PickToCartEnv"
        self.config_dir_path = Path("demo_envs/env_milk_3")
        super().__init__()

class PickFromCartTask(DarkStoreTask):
    def __init__(self, instruction, target_product_name):
        self.instruction = instruction
        self.target_product_name = target_product_name
        self.reference_trajectory = "teleop_demo_actions/pick_from_cart.h5"
        self.env_name = "PickFromCartEnv"
        self.config_dir_path = Path("demo_envs/env_all_items_together")
        super().__init__()

class PickFromFloorTask(DarkStoreTask):
    def __init__(self, instruction, target_product_name):
        self.instruction = instruction
        self.target_product_name = target_product_name
        self.reference_trajectory = "teleop_demo_actions/pick_from_floor.h5"
        self.env_name = "PickFromFloorEnv"
        self.config_dir_path = Path("demo_envs/env_milk_3")
        super().__init__()

class PlaceOnTopTask(DarkStoreTask):
    def __init__(self, instruction, target_product_name, on_top_of_product_name):
        self.instruction = instruction
        self.target_product_name = target_product_name
        self.reference_trajectory = "teleop_demo_actions/place_on_top.h5"
        self.env_name = "PlaceOnTopEnv"
        self.config_dir_path = Path("demo_envs/env_all_items_together_top")
        super().__init__(on_top_of_product_name=on_top_of_product_name)

class MoveFromBoardToBoardTask(DarkStoreTask):
    def __init__(self, instruction, target_product_name):
        self.instruction = instruction
        self.target_product_name = target_product_name
        self.reference_trajectory = "teleop_demo_actions/move_from_board_to_board.h5"
        self.env_name = "MoveFromBoardToBoardEnv"
        self.config_dir_path = Path("demo_envs/env_milk_3")
        super().__init__()

def main():
    tasks = [
        PickToCartTask("pick a milk from the shelf and put it in the cart", "food.dairy_products.milk:1:1:1"),
        PickFromCartTask("pick up the milk from the cart and put it back on the shelf", "food.dairy_products.milk:1:1:20"),
        PickFromFloorTask("pick up the fallen milk from the floor and put it on the second board", "food.dairy_products.milk:1:1:1"),
        PlaceOnTopTask("pick up the fanta can and place it on top of the mountain dew can", "food.drinks.fanta:1:1:0", on_top_of_product_name="food.drinks.mountainDew:1:1:0"),
        MoveFromBoardToBoardTask("move a milk from the second board to the third board", "food.dairy_products.milk:1:1:1"),
    ]
    for task_num, task in enumerate(tasks):
        task.evaluate()

if __name__ == '__main__':
    main()