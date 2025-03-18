import torch
import numpy as np
import os
import sapien
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
LANGUAGE_INSTRUCTION = 'pick a milk from the shelf and put it on the cart'

@register_env('PickToCartEnv', max_episode_steps=200000)
class PickToCartEnv(DarkstoreCellBaseEnv):
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._load_shopping_cart(options)

        self.target_product_marker = actors.build_sphere(
            self.scene,
            radius=0.05,
            color=[0, 1, 0, 1],
            name="target_product",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        self.goal_zone = actors.build_sphere(
            self.scene,
            radius=0.15,
            color=[1, 0, 0, 1],
            name="goal_zone",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

    
    # def _get_obs_extra(self, info: Dict):
    #     """Get task-relevant extra observations. Usually defined on a task by task basis"""
    #     lang_task = dict(language_instruction = str.encode(LANGUAGE_INSTRUCTION))
    #     return lang_task
    
    def _load_shopping_cart(self, options: dict):
        # recommended to use shift = (0,0.5,0)
        # print(self.unwrapped.agent.robot.get_pose())
        p = np.array([11.0, 10.0, 0.0])
        q = np.array([1, 0, 0, 0])
        pose = sapien.Pose(p=p, q=q)
        
        for scene_idx, build_config_idx in enumerate(self.build_config_idxs):
            actor = self.assets_lib['scene_assets.shoppingCart'].ms_build_actor(f'[ENV#{scene_idx}]_cart', self.scene, pose=pose, scene_idxs=[scene_idx])
            self.actors["fixtures"]["scene_assets"][f'[ENV#{scene_idx}]_cart'] = actor
            # self.actors["fixtures"]["scene_assets"][f'[ENV#{scene_idx}]_taget_cube'] = actors.build_cube(
            #     self.scene,
            #     half_size=self.cube_half_size,
            #     color=[0, 0, 0, 0],
            #     name="cube",
            #     body_type="static",
            #     add_collision=False,
            #     scene_idxs=[scene_idx],
            #     initial_pose=pose,
            # )
        
    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([7, 7, 7], [5, 5, 2])
        pose = sapien_utils.look_at([-1, 0.3, 1.2], [1, 2, 1])
        pose = sapien_utils.look_at([-0.5, 1.0, 1.2], [1.5, 2.7, 1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.9, 1.4, 1.3], [0.8, 1.8, 1.05])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]
    
    def setup_target_object(self, env_idx):
        # TODO: redo
        # pass
        target_name = 'food.dairy_products.milkCarton' # TODO: choose random
        tcp_p = self.agent.tcp.pose.p
        
        # find closest object to gripper
        dists = {}
        for actor_name, actor in self.actors['products'].items():
            if target_name in actor_name:
                p = actor.pose.p
                dists[actor_name] = (p - tcp_p).pow(2).sum().sqrt()
        self.target_product_name = min(dists, key=dists.get)
        
        obb = get_actor_obb(self.actors['products'][self.target_product_name])
        center = np.array(obb.primitive.transform)[:3, 3]

        self.target_product_marker.set_pose(sapien.Pose(center))


        # goal_obb = get_actor_obb(self.actors["fixtures"]["scene_assets"][f'[ENV#0]_cart'])
        # goal_center = np.array(goal_obb.primitive.transform)[:3, 3]
        # goal_pose = self.actors["fixtures"]["scene_assets"][f'[ENV#0]_cart'].pose
        goal_obb = self.actors["fixtures"]["scene_assets"][f'[ENV#0]_cart'].get_collision_meshes()[0].bounding_box_oriented
        goal_center = np.array(goal_obb.primitive.transform)[:3, 3]

        self.goal_zone.set_pose(sapien.Pose(goal_center))

    def evaluate(self):
        goal_pos = self.goal_zone.pose.p
        # target_object = self.target_product_marker.pose.p
        obb = get_actor_obb(self.actors['products'][self.target_product_name])
        target_object_center = np.array(obb.primitive.transform)[:3, 3]
        is_obj_placed = (goal_pos - target_object_center).pow(2).sum().sqrt().item() < 0.15

        is_robot_static = self.agent.is_static(0.2)
        return {
            "goal_pos" : goal_pos,
            "target_object" : target_object_center,
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
        }



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
        new_cart_pose_p = robot_pose.p[0].cpu().numpy() + cart_shift 
        pose = sapien.Pose(p=new_cart_pose_p, q=robot_pose.q[0].numpy())
    
        for scene_idx, build_config_idx in enumerate(env_idx):
            self.actors["fixtures"]["scene_assets"][f'[ENV#{scene_idx}]_cart'].set_pose(pose)
            # self.actors["fixtures"]["scene_assets"][f'[ENV#{scene_idx}]_taget_cube'].set_pose(pose)

        self.setup_target_object(env_idx)



