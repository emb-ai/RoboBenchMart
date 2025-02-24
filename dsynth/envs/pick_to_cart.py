import torch
import numpy as np
import os
import sapien
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv

LANGUAGE_INSTRUCTION = 'pick a milk from the shelf and put it on the cart'

@register_env('PickToCartEnv', max_episode_steps=200000)
class PickToCartEnv(DarkstoreCellBaseEnv):

    # def evaluate(self):
    #     target_pos = torch.tensor(self.target_volume.pose.p, dtype=torch.float32)
    #     target_half_extents = torch.tensor(self.cube_half_size * 1, dtype=torch.float32)
    #     milk_pos = self.actors["objects"]["milk_1_1_0"][0]['actor'].pose.p
        
    #     is_obj_placed = torch.all(
    #         (milk_pos >= (target_pos - target_half_extents)) & 
    #         (milk_pos <= (target_pos + target_half_extents)),
    #         dim=-1
    #     )

    #     is_robot_static = self.agent.is_static(0.2)
    #     return {
    #         "first" : milk_pos,
    #         "second" : target_pos,
    #         "third" : target_pos - target_half_extents,
    #         "fourth" : target_pos + target_half_extents,
    #         "success": is_obj_placed & is_robot_static,
    #         "is_obj_placed": is_obj_placed,
    #         "is_robot_static": is_robot_static,
    #     }

        
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._load_shopping_cart(options)
    
    # def _get_obs_extra(self, info: Dict):
    #     """Get task-relevant extra observations. Usually defined on a task by task basis"""
    #     lang_task = dict(language_instruction = str.encode(LANGUAGE_INSTRUCTION))
    #     return lang_task
    
    def _load_shopping_cart(self, options: dict):
        # recommended to use shift = (0,0.5,0)
        # print(self.unwrapped.agent.robot.get_pose())
        if not hasattr(self, 'shopping_cart'):
            shopping_cart_asset = os.path.join(self.assets_dir, "smallShoppingCart2.glb")
            self.cube_half_size = 0.2
            
            if not os.path.exists(shopping_cart_asset):
                print(f"Shopping cart asset not found: {shopping_cart_asset}")
            else:
                builder = self.scene.create_actor_builder()
                builder.add_visual_from_file(filename=shopping_cart_asset, scale=np.array([1.0, 1.0, 1.0]))
                builder.add_nonconvex_collision_from_file(filename=shopping_cart_asset, scale=np.array([1.0, 1.0, 1.0]))
                shopping_cart_pose = sapien.Pose(p=[11.0, 10.0, 0.0], q=np.array([1, 0, 0, 0]))
                builder.set_initial_pose(shopping_cart_pose)
                self.shopping_cart = builder.build_static(name="shopping_cart")
                # self.actors.append(self.shopping_cart)
                # self.target_volume = actors.build_cube(
                #     self.scene,
                #     half_size=self.cube_half_size,
                #     color=[0, 0, 0, 0],
                #     name="cube",
                #     body_type="static",
                #     add_collision=False,
                #     initial_pose=shopping_cart_pose,
                # )
        
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        
        if self.robot_uids == "panda_wristcam":
            qpos = np.array(
                [
                    np.pi / 2,        
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
            self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0]))
            
        robot_pose = self.agent.robot.get_pose()
        cart_shift = np.array([0.4, -0.2, 0.])
        cube_shift_up = np.array([0, 0, 0.4])
        new_cart_pose_p = robot_pose.p[0].numpy() + cart_shift 
        
        self.shopping_cart.set_pose(sapien.Pose(p=new_cart_pose_p, q=robot_pose.q[0].numpy()))
        # self.target_volume.set_pose(sapien.Pose(p=new_cart_pose_p + cube_shift_up, q=robot_pose.q[0].numpy()))


