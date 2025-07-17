import torch
import numpy as np
import pandas as pd
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
from dsynth.envs.darkstore_cont_base import DarkstoreContinuousBaseEnv
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.structs.pose import Pose
from dsynth.scene_gen.arrangements import CELL_SIZE
import copy

@register_env('PickToBasketEnv', max_episode_steps=200000)
class PickToBasketEnv(DarkstoreCellBaseEnv):
    TARGET_PRODUCT_NAME = None
    ROBOT_INIT_POSE_RANDOM_ENABLED = True
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
        self.target_sizes = np.array([0.3, 0.3, 0.3])

        if self.markers_enabled:
            self.target_volumes = {}
            for n_env in range(self.num_envs):
                self.target_volumes[n_env] = actors.build_box(
                    self.scene,
                    half_sizes=list(self.target_sizes/2),
                    color=[0, 1, 0, 0.5],
                    name=f"target_box_{n_env}",
                    body_type="kinematic",
                    add_collision=False,
                    scene_idxs=[n_env],
                    initial_pose=sapien.Pose(p=[0, 0, 0]),
                )
                self.hide_object(self.target_volumes[n_env])
        
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
    
    def setup_target_objects(self, env_idxs):
        self.target_product_names = {}
        self.target_zones = {}
        self.target_shelves = {}
        self.target_products_df = None
        
        if self.markers_enabled:
            target_markers_iterator = {key: iter(val) for key, val in self.target_markers.items()}

        self.target_product_names = {idx: self.TARGET_PRODUCT_NAME for idx in range(self.num_envs)}

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            scene_prducts_df = self.products_df[self.products_df['scene_idx'] == scene_idx]
            
            if self.TARGET_PRODUCT_NAME is None:
                # select random zone, shelf and product
                zone_id = self._batched_episode_rng[scene_idx].choice(sorted(scene_prducts_df['zone_id'].unique()))
                self.target_zones[scene_idx] = zone_id

                zone_products_df = scene_prducts_df[scene_prducts_df['zone_id'] == zone_id]
                shelf_id = self._batched_episode_rng[scene_idx].choice(sorted(zone_products_df['shelf_id'].unique()))
                self.target_shelves[scene_idx] = shelf_id

                shelf_products_df = zone_products_df[zone_products_df['shelf_id'] == shelf_id]
                product_name = self._batched_episode_rng[scene_idx].choice(sorted(shelf_products_df['product_name'].unique()))
                self.target_product_names[scene_idx] = product_name

                if self.target_products_df is None:
                    self.target_products_df = shelf_products_df[shelf_products_df['product_name'] == product_name]
                else:
                    self.target_products_df = pd.concat([self.target_products_df,
                        shelf_products_df[shelf_products_df['product_name'] == product_name]
                                                      ])
            else:
                # select random zone and shelf with self.TARGET_PRODUCT_NAME

                if not self.TARGET_PRODUCT_NAME in scene_prducts_df['product_name'].unique():
                    raise RuntimeError(f"Product {self.TARGET_PRODUCT_NAME} is not present on scene #{scene_idx}")
                
                zones_w_target_product = scene_prducts_df[scene_prducts_df['product_name'] == self.TARGET_PRODUCT_NAME]
                zone_id = self._batched_episode_rng[scene_idx].choice(sorted(zones_w_target_product['zone_id'].unique()))
                self.target_zones[scene_idx] = zone_id

                shelves_w_target_zone = zones_w_target_product[zones_w_target_product['zone_id'] == zone_id]
                shelf_id = self._batched_episode_rng[scene_idx].choice(sorted(shelves_w_target_zone['shelf_id'].unique()))
                self.target_shelves[scene_idx] = shelf_id

                if self.target_products_df is None:
                    self.target_products_df = shelves_w_target_zone[shelves_w_target_zone['shelf_id'] == shelf_id]
                else:
                    self.target_products_df = pd.concat([self.target_products_df,
                        shelves_w_target_zone[shelves_w_target_zone['shelf_id'] == shelf_id]
                    ])

            if self.markers_enabled:
                target_products = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
                for actor_name in target_products['actor_name']:

                    # select only 4th in each column - they are near the edge
                    if int(actor_name.split(':')[-1]) % 4 == 0: # TODO: redo
                        actor = self.actors['products'][actor_name]
                        try:
                            target_marker = next(target_markers_iterator[scene_idx])
                        except StopIteration:
                            raise RuntimeError(f"Number of target objects exceeds number of markers ({self.NUM_MARKERS}) for scene #{scene_idx}")
                        target_marker.set_pose(actor.pose)

    def _compute_robot_init_pose(self, env_idx = None):
        origins = []
        init_cells = []
        angles = []
        directions_to_shelf = []

        for idx in env_idx:
            idx = idx.cpu().item()
            scene_target_products = self.target_products_df[self.target_products_df['scene_idx'] == idx].reset_index()
            shelf_i, shelf_j = scene_target_products['i'][0], scene_target_products['j'][0]
            rot = self.scene_builder.rotations[idx][shelf_i][shelf_j]

            if rot == 0:
                origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j - 1, 0.]), np.pi / 2, np.array([0, 1, 0])
            if rot == -90:
                origin, angle, direction_to_shelf = np.array([shelf_i - 1, shelf_j, 0.]), 0 , np.array([1, 0, 0])
            if rot == 90:
                origin, angle, direction_to_shelf = np.array([shelf_i + 1, shelf_j, 0.]), np.pi, np.array([-1, 0, 0])
            if rot == 180:
                origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j + 1, 0.]), - np.pi / 2, np.array([0, -1, 0])
            
            # self.target_drive_position = origin.copy() + direction_to_shelf * CELL_SIZE * 0.2
            
            init_cell = np.array([origin[0], origin[1]])
            origin = origin * CELL_SIZE
            origin[:2] += CELL_SIZE / 2

            if self.ROBOT_INIT_POSE_RANDOM_ENABLED:
                # base movement enabled, add initial pose randomization
                perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

                delta_par = self._batched_episode_rng[idx].rand() * CELL_SIZE * 0.4
                delta_perp = (self._batched_episode_rng[idx].rand() - 0.5) * 2 * CELL_SIZE * 0.4

                origin += - direction_to_shelf * delta_par + perp_direction * delta_perp

                angle += (self._batched_episode_rng[idx].rand() - 0.5) * np.pi / 4

            origins.append(origin)
            init_cells.append(init_cell)
            angles.append(angle)
            directions_to_shelf.append(direction_to_shelf)

        return np.array(origins), np.array(init_cells), np.array(angles), np.array(directions_to_shelf)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.setup_target_objects(env_idx)
        self.setup_language_instructions(env_idx)

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
            self.robot_origins, self.init_cells, self.robot_angles, self.directions_to_shelf = self._compute_robot_init_pose(env_idx)
            quats = np.array([euler2quat(0, 0, robot_angle) for robot_angle in self.robot_angles])
            self.agent.robot.set_pose(Pose.create_from_pq(p=self.robot_origins, q=quats))
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
            self.robot_origins, self.init_cells, self.robot_angles, self.directions_to_shelf = self._compute_robot_init_pose(env_idx)
            quats = np.array([euler2quat(0, 0, robot_angle) for robot_angle in self.robot_angles])
            self.agent.robot.set_pose(Pose.create_from_pq(p=self.robot_origins, q=quats))

        if self.markers_enabled:
            target_pose = self.calc_target_pose()
            for scene_idx in env_idx:
                scene_idx = scene_idx.cpu().item()
                self.target_volumes[scene_idx].set_pose(
                    Pose.create_from_pq(p=target_pose.p[scene_idx],
                                        q=target_pose.q[scene_idx])
                )
    
    def evaluate(self):
        target_pos = self.calc_target_pose().p 
        target_pos[:, 2] -= self.target_sizes[2] / 2
        tolerance = torch.tensor(self.target_sizes / 2, dtype=torch.float32).to(self.device)
        is_obj_placed = []

        for scene_idx in range(self.num_envs):
            scene_is_obj_placed = False
            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            for actor_name in scene_target_products_df['actor_name']:
                target_product_pos = self.actors['products'][actor_name].pose.p
                scene_is_obj_placed = torch.all(
                    (target_product_pos >= (target_pos[scene_idx] - tolerance)) & 
                    (target_product_pos <= (target_pos[scene_idx] + tolerance)),
                    dim=-1
                )
                if scene_is_obj_placed:
                    break
            
            is_obj_placed.append(scene_is_obj_placed)

        is_obj_placed = torch.cat(is_obj_placed)
        
        is_robot_static = self.agent.is_static(0.2)

        is_non_target_produncts_replaced = torch.zeros_like(is_robot_static, dtype=bool)

        for scene_idx in range(self.num_envs):
            scene_products_df = self.products_df[self.products_df['scene_idx'] == scene_idx]

            # to speed up evaluation only check products from the target shelf
            scene_products_df = scene_products_df[scene_products_df['shelf_id'] == self.target_shelves[scene_idx]]

            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            non_target_actors = set(scene_products_df['actor_name']) - set(scene_target_products_df['actor_name'])
            
            for actor_name in non_target_actors:
                actor = self.actors['products'][actor_name]
                if not torch.all(torch.isclose(actor.pose.raw_pose, self.products_initial_poses[actor_name], rtol=0.1, atol=0.1)):
                    is_non_target_produncts_replaced[scene_idx] = True

                    if self.markers_enabled:
                        # make marker red if non-target product moved
                        render_component = self.target_volumes[scene_idx]._objs[0].find_component_by_type(
                            sapien.pysapien.render.RenderBodyComponent
                        )
                        render_component.render_shapes[0].material.base_color = [1.0, 0.0, 0.0, 0.5]

                    break


        return {
            "is_obj_placed" : is_obj_placed,
            "is_robot_static" : is_robot_static,
            "is_non_target_produncts_displaced" : is_non_target_produncts_replaced,
            "success": is_obj_placed & is_robot_static & (~is_non_target_produncts_replaced),
            # "success": is_obj_placed & is_robot_static,
        }

    def calc_target_pose(self):
        robot_pose = self.agent.base_link.pose
        basket_shift = Pose.create_from_pq(p=[[0.3, 0.25, 0.14]] * self.num_envs)
        return robot_pose * basket_shift 
       

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(f'move to the shelf and pick {self.target_product_names[scene_idx]} and put to the basket')

    def _after_simulation_step(self):
        #does not work on gpu sim
        if self.markers_enabled:
            target_pose = self.calc_target_pose()
            for scene_idx in range(self.num_envs):
                self.target_volumes[scene_idx].set_pose(
                    Pose.create_from_pq(p=target_pose.p[scene_idx],
                                        q=target_pose.q[scene_idx])
                )
            # self.target_volume.set_pose(target_pose)

@register_env('PickToBasketStaticEnv', max_episode_steps=200000)
class PickToBasketStaticEnv(PickToBasketEnv):
    ROBOT_INIT_POSE_RANDOM_ENABLED = False
@register_env('PickToBasketSpriteEnv', max_episode_steps=200000)
class PickToBasketSpriteEnv(PickToBasketEnv):
    TARGET_PRODUCT_NAME = 'sprite'

@register_env('PickToBasketStaticSpriteEnv', max_episode_steps=200000)
class PickToBasketStaticSpriteEnv(PickToBasketEnv):
    TARGET_PRODUCT_NAME = 'sprite'
    ROBOT_INIT_POSE_RANDOM_ENABLED = False

@register_env('PickToBasketContEnv', max_episode_steps=200000)
class PickToBasketContEnv(DarkstoreContinuousBaseEnv):
    TARGET_PRODUCT_NAME = None
    ROBOT_INIT_POSE_RANDOM_ENABLED = True

    TARGET_POS_THRESH = 0.2
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
        self.target_sizes = np.array([0.3, 0.3, 0.3])

        if self.markers_enabled:
            self.target_volumes = {}
            for n_env in range(self.num_envs):
                self.target_volumes[n_env] = actors.build_box(
                    self.scene,
                    half_sizes=list(self.target_sizes/2),
                    color=[0, 1, 0, 0.5],
                    name=f"target_box_{n_env}",
                    body_type="kinematic",
                    add_collision=False,
                    scene_idxs=[n_env],
                    initial_pose=sapien.Pose(p=[0, 0, 0]),
                )
                self.hide_object(self.target_volumes[n_env])
        
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
    
    def setup_target_objects(self, env_idxs):
        self.target_product_names = {}
        self.target_products_df = None
        
        if self.markers_enabled:
            target_markers_iterator = {key: iter(val) for key, val in self.target_markers.items()}

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
            
            if self.target_products_df is None:
                self.target_products_df = scene_prducts_df[scene_prducts_df['product_name'] == product_name]
            else:
                self.target_products_df = pd.concat([self.target_products_df,
                    scene_prducts_df[scene_prducts_df['product_name'] == product_name]
                                                    ])
            
            if self.markers_enabled:
                target_products = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
                for actor_name in target_products['actor_name']:
                    actor = self.actors['products'][actor_name]
                    try:
                        target_marker = next(target_markers_iterator[scene_idx])
                    except StopIteration:
                        raise RuntimeError(f"Number of target objects exceeds number of markers ({self.NUM_MARKERS}) for scene #{scene_idx}")
                    target_marker.set_pose(actor.pose)

    def _compute_robot_init_pose(self, env_idx = None):
        robot_origins, robot_angles, directions_to_shelf = super()._compute_robot_init_pose(env_idx)
        for idx in env_idx:
            if self.ROBOT_INIT_POSE_RANDOM_ENABLED:
                # base movement enabled, add initial pose randomization
                idx = idx.cpu().item()
                direction_to_shelf = directions_to_shelf[idx]
                perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

                delta_par = self._batched_episode_rng[idx].rand() * 0.2
                delta_perp = (self._batched_episode_rng[idx].rand() - 0.5) * 0.5

                robot_origins[idx] += -direction_to_shelf * delta_par + perp_direction * delta_perp
                robot_angles[idx] += (self._batched_episode_rng[idx].rand() - 0.5) * np.pi / 4

        return robot_origins, robot_angles, directions_to_shelf
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if self.robot_uids in ["fetch", "ds_fetch", "ds_fetch_basket"]:
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.34,
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

    def evaluate(self):
        target_pos = self.calc_target_pose().p 
        # target_pos[:, 2] -= self.target_sizes[2] / 2
        # tolerance = torch.tensor(self.target_sizes / 2, dtype=torch.float32).to(self.device)
        tolerance = torch.tensor([self.TARGET_POS_THRESH, self.TARGET_POS_THRESH, self.TARGET_POS_THRESH]).to(self.device)
        is_obj_placed = []

        for scene_idx in range(self.num_envs):
            scene_is_obj_placed = False
            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            for actor_name in scene_target_products_df['actor_name']:
                target_product_pos = self.actors['products'][actor_name].pose.p
                scene_is_obj_placed = torch.all(
                    (target_product_pos >= (target_pos[scene_idx] - tolerance)) & 
                    (target_product_pos <= (target_pos[scene_idx] + tolerance)),
                    dim=-1
                )
                if scene_is_obj_placed:
                    break
            
            is_obj_placed.append(scene_is_obj_placed)

        is_obj_placed = torch.cat(is_obj_placed)
        
        is_robot_static = self.agent.is_static(0.2)

        is_non_target_produncts_replaced = torch.zeros_like(is_robot_static, dtype=bool)

        for scene_idx in range(self.num_envs):
            scene_products_df = self.products_df[self.products_df['scene_idx'] == scene_idx]

            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            non_target_actors = set(scene_products_df['actor_name']) - set(scene_target_products_df['actor_name'])
            
            for actor_name in non_target_actors:
                actor = self.actors['products'][actor_name]
                if not torch.all(torch.isclose(actor.pose.raw_pose, self.products_initial_poses[actor_name], rtol=0.1, atol=0.1)):
                    is_non_target_produncts_replaced[scene_idx] = True

                    if self.markers_enabled:
                        # make marker red if non-target product moved
                        render_component = self.target_volumes[scene_idx]._objs[0].find_component_by_type(
                            sapien.pysapien.render.RenderBodyComponent
                        )
                        render_component.render_shapes[0].material.base_color = [1.0, 0.0, 0.0, 0.5]

                    break


        return {
            "is_obj_placed" : is_obj_placed,
            "is_robot_static" : is_robot_static,
            "is_non_target_produncts_displaced" : is_non_target_produncts_replaced,
            "success": is_obj_placed & is_robot_static & (~is_non_target_produncts_replaced),
            # "success": is_obj_placed & is_robot_static,
        }

    def calc_target_pose(self):
        robot_pose = self.agent.base_link.pose
        basket_shift = Pose.create_from_pq(p=[[0.3, 0.25, 0.14]] * self.num_envs)
        return robot_pose * basket_shift 
       

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(f'move to the shelf and pick {self.target_product_names[scene_idx]} and put to the basket')

    def _after_simulation_step(self):
        #does not work on gpu sim
        if self.markers_enabled:
            target_pose = self.calc_target_pose()
            for scene_idx in range(self.num_envs):
                self.target_volumes[scene_idx].set_pose(
                    Pose.create_from_pq(p=target_pose.p[scene_idx],
                                        q=target_pose.q[scene_idx])
                )
            # self.target_volume.set_pose(target_pose)

@register_env('PickToBasketContNiveaEnv', max_episode_steps=200000)
class PickToBasketContNiveaEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Nivea Body Milk'

@register_env('PickToBasketContStarsEnv', max_episode_steps=200000)
class PickToBasketContStarsEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Nestle Honey Stars'
    TARGET_POS_THRESH = 0.25

@register_env('PickToBasketContFantaEnv', max_episode_steps=200000)
class PickToBasketContFantaEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Fanta Sabor Naranja 2L'