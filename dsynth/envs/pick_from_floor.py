import torch
import numpy as np
import os
import sapien
from transforms3d.euler import euler2quat

from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.utils.structs.pose import Pose

from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from dsynth.envs.darkstore_cont_base import  DarkstoreContinuousBaseEnv
from dsynth.envs.pick_to_basket import PickToBasketContEnv
@register_env('PickFromFloorEnv', max_episode_steps=200000)
class PickFromFloorEnv(DarkstoreCellBaseEnv):

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._load_shopping_cart(options)
        
        target_pose = self.actors['products'][self.target_product_name].pose
        target_pose.raw_pose[0,2] += get_actor_obb(self.actors['products'][self.target_product_name]).extents[2]/2
        target_pose.raw_pose[0][3:] = torch.Tensor([1, 0, 0, 0])
        self.target_sizes = get_actor_obb(self.actors['products'][self.target_product_name]).extents
        self.target_volume = actors.build_box(
            self.scene,
            half_sizes=list(self.target_sizes/2),
            color=[0, 1, 0, 0.5],
            name="target_box",
            body_type="static",
            add_collision=False,
            initial_pose=target_pose,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        
        if self.robot_uids == "panda_wristcam":
            qpos = np.array(
                [
                    0.0,        
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
        cart_shift = np.array([1.4, -0.2, 0.])
        new_cart_pose_p = robot_pose.p[0].cpu().numpy() + cart_shift 
        
        self.shopping_cart.set_pose(sapien.Pose(p=new_cart_pose_p, q=robot_pose.q[0].numpy()))
        self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0], [0.70710678118, 0, 0, 0.70710678118]))
        self.actors['products'][self.target_product_name].pose = sapien.Pose(p=[0.8, 1.5, 0.0] , q=[0.498, -0.502, -0.498, -0.502])
        self.setup_target_object()


@register_env('PickFromFloorContEnv', max_episode_steps=200000)
class PickFromFloorContEnv(PickToBasketContEnv):
    ROBOT_INIT_POSE_RANDOM_ENABLED = True

    def setup_target_objects(self, env_idxs):
        self.target_product_names = {}
        self.fallen_items = {}
        self.target_pose = {}
        self.target_products_df = None
        
        # if self.markers_enabled:
        #     target_volumes_iterator = {key: iter(val) for key, val in self.target_volumes.items()}

        self.target_product_names = {idx: self.TARGET_PRODUCT_NAME for idx in range(self.num_envs)}

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            scene_prducts_df = self.products_df[self.products_df['scene_idx'] == scene_idx]
            
            if self.TARGET_PRODUCT_NAME is None:
                product_name = self._batched_episode_rng[scene_idx].choice(sorted(scene_prducts_df['product_name'].unique()))
                self.target_product_names[scene_idx] = product_name
        
            else:
                product_name = self.TARGET_PRODUCT_NAME
                if not self.TARGET_PRODUCT_NAME in scene_prducts_df['product_name'].unique():
                    raise RuntimeError(f"Product {self.TARGET_PRODUCT_NAME} is not present on scene #{scene_idx}")
            
            target_products_df = scene_prducts_df[scene_prducts_df['product_name'] == product_name]
            target_products_df = target_products_df[target_products_df['row_idxs'] == '0']
            fallen_actor_id = self._batched_episode_rng[scene_idx].choice(sorted(target_products_df['actor_name'].unique()))
            
            self.fallen_items[scene_idx] = fallen_actor_id
            
            actor = self.actors['products'][fallen_actor_id]

            self.target_pose[scene_idx] = actor.pose

            if self.markers_enabled:
                self.target_volumes[scene_idx][0].set_pose(actor.pose)
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        # place object on the ground
        for idx in env_idx:
            idx = idx.cpu().item()
            actor_shelf_name = self.active_shelves[idx][0]
            direction_to_shelf = self.directions_to_shelf[idx]
            shelf_pose = self.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp

            fall_position = shelf_pose.p - 1.4 * direction_to_shelf
            perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

            delta_par = 0.3 + self._batched_episode_rng[idx].rand() * 0.4
            delta_perp = (self._batched_episode_rng[idx].rand() - 0.5) * 0.5


            actor = self.actors['products'][self.fallen_items[idx]]
            extents = get_actor_obb(actor).primitive.extents

            fall_position += direction_to_shelf * delta_par + perp_direction * delta_perp
            fall_position[2] = np.max(extents[:2]) / 2 + 1e-2
            
            angle = (self._batched_episode_rng[idx].rand() - 0.5) * 3.14 / 2

            actor.set_pose(sapien.Pose(p=fall_position, q=euler2quat(-1.57, 1.57, angle)))
            if self.markers_enabled:
                self.target_markers[idx][0].set_pose(sapien.Pose(p=fall_position))

    def evaluate(self):
        target_pos = self.calc_target_pose().p 
        tolerance = torch.tensor([self.TARGET_POS_THRESH, self.TARGET_POS_THRESH, self.TARGET_POS_THRESH]).to(self.device)
        
        is_obj_placed = []

        for scene_idx in range(self.num_envs):
            target_product_pos = self.actors['products'][self.fallen_items[scene_idx]].pose.p
            scene_is_obj_placed = torch.all(
                (target_product_pos >= (target_pos[scene_idx] - tolerance)) & 
                (target_product_pos <= (target_pos[scene_idx] + tolerance)),
                dim=-1
            )
            
            is_obj_placed.append(scene_is_obj_placed)

        is_obj_placed = torch.cat(is_obj_placed)
        
        is_robot_static = self.agent.is_static(0.2)
        
        is_non_target_produncts_replaced = torch.zeros_like(is_robot_static, dtype=bool)

        for scene_idx in range(self.num_envs):
            scene_products_df = self.products_df[self.products_df['scene_idx'] == scene_idx]

            non_target_actors = set(scene_products_df['actor_name']) - set([self.fallen_items[scene_idx]])
            
            for actor_name in non_target_actors:
                actor = self.actors['products'][actor_name]
                if not torch.all(torch.isclose(actor.pose.raw_pose, self.products_initial_poses[actor_name], rtol=0.1, atol=0.1)):
                    is_non_target_produncts_replaced[scene_idx] = True

                    if self.markers_enabled:
                        # make marker red if non-target product moved
                        render_component = self.target_volumes[scene_idx][0]._objs[0].find_component_by_type(
                            sapien.pysapien.render.RenderBodyComponent
                        )
                        render_component.render_shapes[0].material.base_color = [1.0, 0.0, 0.0, 0.5]
                    break

        return {
            "is_obj_placed" : is_obj_placed,
            "is_robot_static" : is_robot_static,
            "is_non_target_produncts_displaced" : is_non_target_produncts_replaced,
            "success": is_obj_placed & is_robot_static & (~is_non_target_produncts_replaced),
        }
    
    def calc_target_pose(self):
        target_pose_p = [self.target_pose[scene_idx].sp.p for scene_idx in range(self.num_envs)]
        target_pose_q = [self.target_pose[scene_idx].sp.q for scene_idx in range(self.num_envs)]
        return Pose.create_from_pq(p=target_pose_p, q=target_pose_q)
    
    def _after_simulation_step(self):
        pass


@register_env('PickFromFloorBeansContEnv', max_episode_steps=200000)
class PickFromFloorBeansContEnv(PickFromFloorContEnv):
    TARGET_PRODUCT_NAME = 'Heinz Beans in a rich tomato sauce'

@register_env('PickFromFloorSlamContEnv', max_episode_steps=200000)
class PickFromFloorSlamContEnv(PickFromFloorContEnv):
    TARGET_PRODUCT_NAME = 'SLAM luncheon meat'

@register_env('PickFromFloorTestContEnv', max_episode_steps=200000)
class PickFromFloorTestContEnv(PickFromFloorContEnv):
    # TARGET_PRODUCT_NAME = 'Duff Beer Can'
    TARGET_PRODUCT_NAME = 'Fanta Sabor Naranja 2L'