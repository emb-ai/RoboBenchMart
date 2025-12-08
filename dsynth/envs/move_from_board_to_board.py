import torch
import numpy as np
import os
import sapien
from transforms3d import euler
import pandas as pd
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from dsynth.envs.pick_to_basket import PickToBasketContEnv

from dsynth.scene_gen.arrangements import CELL_SIZE
from transforms3d.euler import euler2quat

@register_env('MoveFromBoardToBoardEnv', max_episode_steps=200000)
class MoveFromBoardToBoardEnv(DarkstoreCellBaseEnv):
    TARGET_PRODUCT_NAME = None
    ROBOT_INIT_POSE_RANDOM_ENABLED = True

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.target_sizes = np.array([0.3, 0.3, 0.3])
        
        if self.markers_enabled:
            self.target_markers = {}
            self.target_volume = {}
            for n_env in range(self.num_envs):
                self.target_markers[n_env] = []
                self.target_volumes[n_env] = []
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
                    self.target_volumes[n_env].append(actors.build_box(
                        self.scene,
                        half_sizes=list(self.target_sizes/2),
                        color=[0, 1, 0, 0.5],
                        name=f"target_box_{n_env}_{i}",
                        body_type="kinematic",
                        add_collision=False,
                        scene_idxs=[n_env],
                        initial_pose=sapien.Pose(p=[0, 0, 0]),
                    ))
                    self.hide_object(self.target_volumes[n_env])
                    self.hide_object(self.target_markers[n_env][-1])


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.setup_target_objects(env_idx)
        self.setup_language_instructions(env_idx)
        
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
        
        elif self.robot_uids in ["ds_fetch_basket", "ds_fetch", "fetch"]:
            qpos = np.array(
                [
                    0,
                    0,
                    0,#np.random.rand() * 6.2832 - 3.1416,
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

    def setup_target_objects(self, env_idxs):
        self.target_product_names = {}
        self.target_zones = {}
        self.target_shelves = {}
        self.target_products_df = None
        
        if self.markers_enabled:
            target_markers_iterator = {key: iter(val) for key, val in self.target_markers.items()}
            target_volumes_iterator = {key: iter(val) for key, val in self.target_volumes.items()}

        self.target_product_names = {idx: self.TARGET_PRODUCT_NAME for idx in range(self.num_envs)}

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            scene_prducts_df = self.products_df[self.products_df['scene_idx'] == scene_idx]
            
            if self.TARGET_PRODUCT_NAME is None:
                # select random zone, shelf and product
                zone_id = self._batched_episode_rng[scene_idx].choice(scene_prducts_df['zone_id'].unique())
                self.target_zones[scene_idx] = zone_id

                zone_products_df = scene_prducts_df[scene_prducts_df['zone_id'] == zone_id]
                shelf_id = self._batched_episode_rng[scene_idx].choice(zone_products_df['shelf_id'].unique())
                self.target_shelves[scene_idx] = shelf_id

                shelf_products_df = zone_products_df[zone_products_df['shelf_id'] == shelf_id]
                product_name = self._batched_episode_rng[scene_idx].choice(shelf_products_df['product_name'].unique())
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
                zone_id = self._batched_episode_rng[scene_idx].choice(zones_w_target_product['zone_id'].unique())
                self.target_zones[scene_idx] = zone_id

                shelves_w_target_zone = zones_w_target_product[zones_w_target_product['zone_id'] == zone_id]
                shelf_id = self._batched_episode_rng[scene_idx].choice(shelves_w_target_zone['shelf_id'].unique())
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
                            target_volume = next(target_volumes_iterator[scene_idx])
                        except StopIteration:
                            raise RuntimeError(f"Number of target objects exceeds number of markers ({self.NUM_MARKERS}) for scene #{scene_idx}")
                        target_marker.set_pose(actor.pose)

                        t_vol_p = actor.pose.p.clone()
                        t_vol_p[scene_idx, 2] += self.get_interboard_height()
                        target_volume_pose = Pose.create_from_pq(p = t_vol_p, q=actor.pose.q)
                        target_volume.set_pose(target_volume_pose)


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

                delta_par = self._batched_episode_rng[idx].rand() * CELL_SIZE * 0.3
                delta_perp = (self._batched_episode_rng[idx].rand() - 0.5) * 2 * CELL_SIZE * 0.3

                origin += - direction_to_shelf * delta_par + perp_direction * delta_perp

                angle += (self._batched_episode_rng[idx].rand() - 0.5) * np.pi / 4
            else:
                # move base closer to the shelf for static manipulation
                origin = origin + direction_to_shelf * CELL_SIZE * 0.2

            origins.append(origin)
            init_cells.append(init_cell)
            angles.append(angle)
            directions_to_shelf.append(direction_to_shelf)

        return np.array(origins), np.array(init_cells), np.array(angles), np.array(directions_to_shelf)

    def get_interboard_height(self):
        #height of board
        return 0.397

    def calc_target_pose(self, actor_name):
        t_pose = self.actors['products'][actor_name].pose.p.clone()
        t_pose.p[:, 2] += self.get_interboard_height()
        return t_pose

    def evaluate(self):
        tolerance = torch.tensor(self.target_sizes / 2, dtype=torch.float32).to(self.device)
        is_obj_placed = []

        for scene_idx in range(self.num_envs):
            scene_is_obj_placed = False
            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            for actor_name in scene_target_products_df['actor_name']:
                target_pos = self.calc_target_pose().p 
                target_pos[:, 2] -= self.target_sizes[2] / 2
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
                        for n_env in range(self.num_envs):
                            render_component = self.target_volumes[scene_idx][n_env]._objs[0].find_component_by_type(
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

@register_env('MoveFromBoardToBoardStaticEnv', max_episode_steps=200000)
class MoveFromBoardToBoardStaticEnv(MoveFromBoardToBoardEnv):
    ROBOT_INIT_POSE_RANDOM_ENABLED = False
@register_env('MoveFromBoardToBoardStaticOneProdEnv', max_episode_steps=200000)
class MoveFromBoardToBoardStaticOneProdEnv(MoveFromBoardToBoardEnv):
    TARGET_PRODUCT_NAME = 'sprite'
    ROBOT_INIT_POSE_RANDOM_ENABLED = False


@register_env('MoveFromBoardToBoardContEnv', max_episode_steps=200000)
class MoveFromBoardToBoardContEnv(PickToBasketContEnv):
    """
    **Task Description:**
    Approach the shelf and pick up the item specified by `TARGET_PRODUCT_NAME`, placing it one board higher (target board).
    Note: `TARGET_PRODUCT_NAME` must be specified. The robot is spawned in close proximity to the shelf.
    It is assumed that there is a free space on a target board.

    **Randomizations:**
    - scene layout, object arrangement, wall and floor textures
    - initial robot position, if `ROBOT_INIT_POSE_RANDOM_ENABLED` is enabled (True by default)

    **Success Conditions:**
    - any product item with the name `TARGET_PRODUCT_NAME` is within `TARGET_POS_THRESH` Euclidean distance of the goal position
    - other items remain untouched (their positions change by no more than 0.1 m)
    - the robot is static (q velocity < 0.2)
    """
    def setup_target_objects(self, env_idxs):
        if self.TARGET_PRODUCT_NAME is None:
            raise NotImplementedError # target object must be specified manually
        
        super().setup_target_objects(env_idxs)
        
        if self.markers_enabled:
            target_volumes_iterator = {key: iter(val) for key, val in self.target_volumes.items()}

            for scene_idx in env_idxs:
                scene_idx = scene_idx.cpu().item()
                target_products = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
                for actor_name in target_products['actor_name']:
                    target_pose = self.calc_target_pose(actor_name)
                    try:
                        target_volume = next(target_volumes_iterator[scene_idx])
                    except StopIteration:
                        raise RuntimeError(f"Number of target objects exceeds number of markers ({self.NUM_MARKERS}) for scene #{scene_idx}")
                    target_volume.set_pose(target_pose)

    def get_interboard_height(self):
        # clearance between two boards
        return 0.397

    def calc_target_pose(self, actor_name):
        t_pose = self.actors['products'][actor_name].pose
        p = t_pose.p
        p[:, 2] += self.get_interboard_height()
        target_pose = Pose.create_from_pq(p=p, q=t_pose.q)
        return target_pose
    
    def evaluate(self):
        tolerance = torch.tensor(self.target_sizes / 2, dtype=torch.float32).to(self.device)
        is_obj_placed = []

        for scene_idx in range(self.num_envs):
            scene_is_obj_placed = False
            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            for actor_name in scene_target_products_df['actor_name']:
                target_pos = self.products_initial_poses[actor_name][:, :3].clone()
                target_pos[:, 2] += self.get_interboard_height()
                target_pos[:, 2] -= self.target_sizes[2] / 2
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
            # "success": is_obj_placed & is_robot_static,
        }
    
    def _after_simulation_step(self):
        pass

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(f'pick {self.TARGET_PRODUCT_NAME} and place on empty board')

MFBTB_DOC_STRING="""**Task Description:**
**Task Description:**
Approach the shelf and pick up any item with the name `{product_name}`, placing it one board higher (target board).
It is assumed that there is a free space on a target board.

**Randomizations:**
- scene layout, object arrangement, wall and floor textures
- initial robot position, if `ROBOT_INIT_POSE_RANDOM_ENABLED` is enabled (True by default)

**Success Conditions:**
- any product item with the name `{product_name}` is within `TARGET_POS_THRESH` Euclidean distance of the goal position
- other items remain untouched (their positions change by no more than 0.1 m)
- the robot is static (q velocity < 0.2)
"""

# train items
@register_env('MoveFromBoardToBoardVanishContEnv', max_episode_steps=200000)
class MoveFromBoardToBoardVanishContEnv(MoveFromBoardToBoardContEnv):
    TARGET_PRODUCT_NAME = 'Vanish Stain Remover'

MoveFromBoardToBoardVanishContEnv.__doc__ = MFBTB_DOC_STRING.format(product_name='Vanish Stain Remover')

@register_env('MoveFromBoardToBoardNestleContEnv', max_episode_steps=200000)
class MoveFromBoardToBoardNestleContEnv(MoveFromBoardToBoardContEnv):
    TARGET_PRODUCT_NAME = 'Nestle Fitness Chocolate Cereals'

MoveFromBoardToBoardNestleContEnv.__doc__ = MFBTB_DOC_STRING.format(product_name='Nestle Fitness Chocolate Cereals')

@register_env('MoveFromBoardToBoardDuffContEnv', max_episode_steps=200000)
class MoveFromBoardToBoardDuffContEnv(MoveFromBoardToBoardContEnv):
    TARGET_PRODUCT_NAME = 'Duff Beer Can'

MoveFromBoardToBoardDuffContEnv.__doc__ = MFBTB_DOC_STRING.format(product_name='Duff Beer Can')


# unseen test items
@register_env('MoveFromBoardToBoardFantaContEnv', max_episode_steps=200000)
class MoveFromBoardToBoardFantaContEnv(MoveFromBoardToBoardContEnv):
    TARGET_PRODUCT_NAME = 'Fanta Sabor Naranja 2L'

MoveFromBoardToBoardFantaContEnv.__doc__ = MFBTB_DOC_STRING.format(product_name='Fanta Sabor Naranja 2L')

@register_env('MoveFromBoardToBoardNiveaContEnv', max_episode_steps=200000)
class MoveFromBoardToBoardNiveaContEnv(MoveFromBoardToBoardContEnv):
    TARGET_PRODUCT_NAME = 'Nivea Body Milk'

MoveFromBoardToBoardNiveaContEnv.__doc__ = MFBTB_DOC_STRING.format(product_name='Nivea Body Milk')

