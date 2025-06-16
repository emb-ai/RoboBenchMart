import torch
import numpy as np
import os
import sapien
import sapien.physx as physx
from transforms3d.euler import euler2quat
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.structs.pose import Pose
from dsynth.scene_gen.arrangements import CELL_SIZE
import copy

@register_env('PickToCartStaticEnv', max_episode_steps=200000)
class PickToCartStaticEnv(DarkstoreCellBaseEnv):
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # self._load_shopping_cart(options)
        
        self.target_sizes = np.array([0.3, 0.3, 0.3])
        self.target_volume = actors.build_box(
            self.scene,
            half_sizes=list(self.target_sizes/2),
            color=[0, 1, 0, 0.5],
            name="target_box",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0]),
        )

        self.target_markers = {}
        for n_env in range(self.num_envs):
            self.target_markers[n_env] = []
            for i in range(self.NUM_MARKERS):
                self.target_markers[n_env].append(
                                actors.build_sphere(
                                    self.scene,
                                    radius=0.05,
                                    color=[0, 1, 0, 1],
                                    name=f"target_product_{n_env}_{i}",
                                    body_type="kinematic",
                                    add_collision=False,
                                    initial_pose=sapien.Pose(p=[0., 0., 0.]),
                                    scene_idxs=[n_env]
                                )
                            )
                self.hide_object(self.target_markers[n_env][-1])

        self.hide_object(self.target_volume)
    
    def _compute_robot_init_pose(self, env_idx = None): #TODO: redo this shit
        target_shelf = 'zone1.shelf1' # TODO: redo 
        for i in range(len(self.scene_builder.room[0])):
            for j in range(len(self.scene_builder.room[0][i])):
                if self.scene_builder.room[0][i][j] == target_shelf:
                    rot = self.scene_builder.rotations[0][i][j]
                    shelf_i, shelf_j = i, j
                    break
        if rot == 0:
            origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j - 1, 0.]), np.pi / 2, np.array([0, 1, 0])
        if rot == -90:
            origin, angle, direction_to_shelf = np.array([shelf_i - 1, shelf_j, 0.]), 0 , np.array([1, 0, 0])
        if rot == 90:
            origin, angle, direction_to_shelf = np.array([shelf_i + 1, shelf_j, 0.]), np.pi, np.array([-1, 0, 0])
        if rot == 180:
            origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j + 1, 0.]), - np.pi / 2, np.array([0, -1, 0])
        self.direction_to_shelf = direction_to_shelf

        origin = origin * CELL_SIZE
        origin[:2] += CELL_SIZE / 2

        origin += direction_to_shelf * CELL_SIZE * 0.2

        return origin, angle

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        if self.robot_uids == "panda_wristcam":
            qpos = np.array(
                [
                    -0.006,        
                    -1.467,
                    0.012,        
                    -2.823,
                    0.003,        
                    2.928,
                    0.796,
                    0.04,       
                    0.04,       
                ]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0]))

        elif self.robot_uids in ["ds_fetch_basket", "ds_fetch", "fetch"]:
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.36,
                    0,
                    0,
                    0,
                    1.4,
                    0,
                    0.76,
                    0,
                    - 2 * np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            robot_origin, robot_angle = self._compute_robot_init_pose()
            self.agent.robot.set_pose(sapien.Pose(p=robot_origin, q=euler2quat(0, 0, robot_angle)))
        elif self.robot_uids in ["ds_fetch_static", "ds_fetch_basket_static"]:
            qpos = np.array(
                [
                    0.36,
                    0,
                    0,
                    0,
                    1.4,
                    0,
                    0.76,
                    0,
                    - 2 * np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            robot_origin, robot_angle = self._compute_robot_init_pose()
            self.agent.robot.set_pose(sapien.Pose(p=robot_origin, q=euler2quat(0, 0, robot_angle)))

        robot_pose = self.agent.robot.get_pose()
        cart_shift = sapien.Pose(p=[0.3, 0.25, 0.14])
        # cube_shift_up = np.array([0, 0.13, 0.35])
        new_cart_pose_p = robot_pose * cart_shift 
        
        # self.shopping_cart.set_pose(sapien.Pose(p=new_cart_pose_p, q=robot_pose.q[0].numpy()))
        self.target_volume.set_pose(new_cart_pose_p)
        # self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0], [0.70710678118, 0, 0, 0.70710678118]))
        self.setup_target_object()
        self.language_instruction = f'pick {self.target_product_str} and put to the basket'

@register_env('PickToCartStaticOneProdEnv', max_episode_steps=200000)
class PickToCartStaticOneProdEnv(PickToCartStaticEnv):
    TARGET_PRODUCT_NAME = 'sprite'
    def setup_target_object(self):
        self.target_product_names = {}
    
        for key, val in self.assets_lib.items():
            if 'products_hierarchy.' in key:
                if val.asset_name == self.TARGET_PRODUCT_NAME:
                    prod_name = key.replace('products_hierarchy.', '')
                    break
        
        target_markers_iterator = {key: iter(val) for key, val in self.target_markers.items()}
        for actor_name, actor in self.actors['products'].items():
            if prod_name in actor_name:
                # select only 4th in each column - they are near the edge
                if int(actor_name.split(':')[-1]) % 4 == 0:
                    assert len(actor._scene_idxs) == 1
                    scene_idx = actor._scene_idxs.cpu().numpy()[0]
                    if not scene_idx in self.target_product_names:
                        self.target_product_names[scene_idx] = [] 
                    self.target_product_names[scene_idx].append(actor_name)

                    try:
                        target_marker = next(target_markers_iterator[scene_idx])
                    except StopIteration:
                        raise RuntimeError(f"Number of target objects exceeds number of markers ({self.NUM_MARKERS})")
                    target_marker.set_pose(actor.pose)

        self.target_product_str = self.TARGET_PRODUCT_NAME

    # def evaluate(self):
    #     target_pos = torch.tensor(self.target_volume.pose.p, dtype=torch.float32)
    #     # target_pos[0][2] -= self.target_sizes[2]/2
    #     tolerance = torch.tensor(self.target_sizes/2, dtype=torch.float32)

    #     is_obj_placed = False
    #     for target_product_name in self.target_product_names:
    #         target_product_pos = self.actors['products'][target_product_name].pose.p
    #         is_obj_placed = torch.all(
    #             (target_product_pos >= (target_pos - tolerance)) & 
    #             (target_product_pos <= (target_pos + tolerance)),
    #             dim=-1
    #         )
    #         if is_obj_placed:
    #             break
    #     #print("is_obj_placed", is_obj_placed, target_pos, target_product_pos, tolerance)

    #     is_robot_static = self.agent.is_static(0.2)
    #     if not self.product_displaced:
    #         for p, a in self.actors['products'].items():
    #             if not p in self.target_product_names:
    #                 if not torch.all(torch.isclose(a.pose.raw_pose, self.products_initial_poses[p], rtol=0.1, atol=0.1)):
    #                     self.product_displaced = True
    #                     self.target_volume = actors.build_box(
    #                         self.scene,
    #                         half_sizes=list(self.target_sizes/2),
    #                         color=[1, 0, 0, 0.9],
    #                         name="target_box_red",
    #                         body_type="static",
    #                         add_collision=False,
    #                         initial_pose=self.target_volume.pose,
    #                     )
    #                     break
    #     for target_product_name in self.target_product_names:
    #         is_object_grasped = self.agent.is_grasping(self.actors['products'][target_product_name])

    #     # print("is_obj_placed", is_obj_placed.item(), "product_displaced", self.product_displaced, "is_object_grasped", is_object_grasped.item())
    #     return {
    #         "first" : target_product_pos,
    #         "second" : target_pos,
    #         "third" : target_pos - tolerance,
    #         "fourth" : target_pos + tolerance,
    #         "success": is_obj_placed & is_robot_static,
    #         "is_obj_placed": is_obj_placed,
    #         "is_robot_static": is_robot_static,
    #         "is_object_grasped": is_object_grasped,
    #         "product_displaced": self.product_displaced
    #     }


@register_env('PickToCartOneProdEnv', max_episode_steps=200000)
class PickToCartOneProdEnv(PickToCartStaticOneProdEnv):
    def _compute_robot_init_pose(self, env_idx = None): #TODO: redo this shit
        target_shelf = 'zone1.shelf1'
        for i in range(len(self.scene_builder.room[0])):
            for j in range(len(self.scene_builder.room[0][i])):
                if self.scene_builder.room[0][i][j] == target_shelf:
                    rot = self.scene_builder.rotations[0][i][j]
                    shelf_i, shelf_j = i, j
                    break
        if rot == 0:
            origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j - 1, 0.]), np.pi / 2, np.array([0, 1, 0])
        if rot == -90:
            origin, angle, direction_to_shelf = np.array([shelf_i - 1, shelf_j, 0.]), 0 , np.array([1, 0, 0])
        if rot == 90:
            origin, angle, direction_to_shelf = np.array([shelf_i + 1, shelf_j, 0.]), np.pi, np.array([-1, 0, 0])
        if rot == 180:
            origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j + 1, 0.]), - np.pi / 2, np.array([0, -1, 0])
        origin = origin * CELL_SIZE
        origin[:2] += CELL_SIZE / 2

        perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

        delta_par = self._batched_episode_rng[0].rand() * CELL_SIZE * 0.3
        delta_perp = (self._batched_episode_rng[0].rand() - 0.5) * 2 * CELL_SIZE * 0.3

        self.target_drive_position = origin.copy() + direction_to_shelf * CELL_SIZE * 0.2
        self.robot_target_angle = angle
        
        origin += - direction_to_shelf * delta_par + perp_direction * delta_perp

        angle += (self._batched_episode_rng[0].rand() - 0.5) * np.pi / 4

        self.direction_to_shelf = direction_to_shelf

        return origin, angle
    
    def _after_simulation_step(self):
        # move target volume with robot
        robot_pose = self.agent.base_link.pose
        cart_shift = sapien.Pose(p=[0.3, 0.25, 0.14])
        new_cart_pose_p = robot_pose * cart_shift
        self.target_volume.set_pose(new_cart_pose_p)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.language_instruction = f'move to the shelf and pick {self.target_product_str} and put to the basket'