import torch
import numpy as np
import os
import sapien
from transforms3d import euler
import itertools 
from transforms3d.euler import euler2quat
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from dsynth.scene_gen.arrangements import CELL_SIZE
from dsynth.scene_gen.utils import find_paths

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    multiply = 1 / (norm + 1e-5)
    return multiply * vec

@register_env('NavMoveToZoneEnv', max_episode_steps=200000)
class NavMoveToZoneEnv(DarkstoreCellBaseEnv):
    TARGET_ZONE_NAME = None

    INIT_POSE_GEN_MAX_TRIES = 20
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)

        if self.markers_enabled:
            self.target_volumes = {}
            for n_env in range(self.num_envs):
                self.target_volumes[n_env] = []
                for i in range(self.NUM_MARKERS):
                    self.target_volumes[n_env].append(
                                    actors.build_box(
                                        self.scene,
                                        half_sizes=(0.2, 0.2, 0.2),
                                        color=[0, 1, 0, 0.5],
                                        name=f"target_box_{n_env}_{i}",
                                        body_type="kinematic",
                                        add_collision=False,
                                        initial_pose=sapien.Pose(p=[0., 0., 0.]),
                                        scene_idxs=[n_env]
                                    )
                                )
                    self.hide_object(self.target_volumes[n_env][-1])

    def setup_target_zone(self, env_idxs):
        # pick random zone
        self.target_cells = {}
        self.target_directions = {}
        self.target_zone_names = {}
        self.target_zone_ids = {}

        self.target_zone_names = {idx: self.TARGET_ZONE_NAME for idx in range(self.num_envs)}

        # pick random target zone
        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            scene_prducts_df = self.products_df[self.products_df['scene_idx'] == scene_idx]

            if self.TARGET_ZONE_NAME is None:
                zone_name = self._batched_episode_rng[scene_idx].choice(scene_prducts_df['zone_name'].unique())
                self.target_zone_names[scene_idx] = zone_name

            zone_scene_prducts_df = self.products_df[self.products_df['zone_name'] == self.target_zone_names[scene_idx]]
            if len(zone_scene_prducts_df) == 0:
                raise RuntimeError(f"No zones with name {self.target_zone_names[scene_idx]}!")
            
            zone_id = zone_scene_prducts_df['zone_id'].unique()
            if len(zone_id) > 1:
                raise RuntimeError(f"More than one zones with name {self.target_zone_names[scene_idx]}!")
            zone_id = zone_id[0]

            # define target cells and view directions
            scene_target_cells = []
            scene_target_directions = []

            shelves_placement = self.shelves_placement[scene_idx]
            room = self.room[scene_idx]
            for shelf_name, coords in shelves_placement[zone_id].items():
                i, j = coords
                rot = self.scene_builder.rotations[scene_idx][i][j]
                if rot == 0:
                    cell_coords, direction_to_shelf = np.array([i, j - 1, 0.]), np.array([0, 1, 0])
                if rot == -90:
                    cell_coords, direction_to_shelf = np.array([i - 1, j, 0.]), np.array([1, 0, 0])
                if rot == 90:
                    cell_coords, direction_to_shelf = np.array([i + 1, j, 0.]), np.array([-1, 0, 0])
                if rot == 180:
                    cell_coords, direction_to_shelf = np.array([i, j + 1, 0.]), np.array([0, -1, 0])
                scene_target_cells.append(cell_coords)
                scene_target_directions.append(direction_to_shelf)

                opposite_cell = cell_coords + 2 * direction_to_shelf
                opposite_cell = opposite_cell.astype(np.int32)
                if opposite_cell[0] >= 0 and opposite_cell[0] < len(room) and \
                    opposite_cell[1] >= 0 and opposite_cell[1] < len(room[0]) and \
                    room[opposite_cell[0]][opposite_cell[1]] == 0:
                        scene_target_cells.append(opposite_cell)
                        scene_target_directions.append(-direction_to_shelf)
            self.target_cells[scene_idx] = scene_target_cells
            self.target_directions[scene_idx] = scene_target_directions


    def _compute_robot_init_pose(self, env_idx = None):
        origin = []
        angle = []
        for idx in env_idx:
            room = self.room[idx]
            shelves_placement = self.shelves_placement[idx]
            all_cells = list(itertools.product(range(len(room)), range(len(room[0]))))
            occupied_cells = []
            for zone_name in shelves_placement.keys():
                for shelf_name in shelves_placement[zone_name].keys():
                    occupied_cells.append(shelves_placement[zone_name][shelf_name])
            free_cells = sorted(list(set(all_cells) - set(occupied_cells)))

            is_success = False
            for _ in range(self.INIT_POSE_GEN_MAX_TRIES):
                init_cell_idx = self._batched_episode_rng[idx].randint(0, len(free_cells))
                init_cell = free_cells[init_cell_idx]
                paths = find_paths(room, (0, 0), init_cell)
                # if init_cell == (0, 0)
                if len(paths) > 0:
                    is_success = True
                    break
            
            if not is_success:
                raise RuntimeError("Can't generate init pose!")

            origin.append(
                np.array([init_cell[0] * CELL_SIZE + CELL_SIZE / 2, 
                    init_cell[1] * CELL_SIZE + CELL_SIZE / 2, 0])
            )
            
            angle.append(
                self._batched_episode_rng[idx].rand() * 2 * np.pi
            )

        return np.array(origin), np.array(angle)
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        robot_xyz_origins, robot_angles = self._compute_robot_init_pose(env_idx)
        self.setup_target_zone(env_idx)

        self.language_instructions = [f'move to {zone_name}' for scene_idx, zone_name in self.target_zone_names.items()]
        
        target_volumes_iterators = {key: iter(val) for key, val in self.target_volumes.items()}
        if self.markers_enabled:
            for n_env in range(self.num_envs):
                target_cells = self.target_cells[n_env]
                for target_cell in target_cells:
                    target_p = target_cell * CELL_SIZE + np.array([CELL_SIZE / 2, CELL_SIZE / 2, 0])
                    try:
                        target_volume = next(target_volumes_iterators[n_env])
                    except StopIteration:
                        raise RuntimeError(f"Number of target objects exceeds number of markers ({self.NUM_MARKERS})")
                    target_volume.set_pose(sapien.Pose(p=target_p))
        
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
        
        elif self.robot_uids in ["ds_fetch", "ds_fetch_basket"]:
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
            robot_quats_origin = np.array([euler2quat(0, 0, angle) for angle in robot_angles])
            self.agent.robot.set_pose(Pose.create_from_pq(p=robot_xyz_origins, q=robot_quats_origin))




    def evaluate(self):
        EPS = 0.11
        is_target_in_view = []
        is_robot_placed = []
        for scene_idx in range(self.num_envs):
            robot_view_vec = self.agent.base_link.pose.to_transformation_matrix()[scene_idx, :3, 0]
            robot_cur_pose = self.agent.base_link.pose.p[scene_idx]
            robot_cur_cell = (int(robot_cur_pose[0] // CELL_SIZE), int(robot_cur_pose[1] // CELL_SIZE))

            is_target_in_view.append(False)
            is_robot_placed.append(False)
            for cell, direction in zip(self.target_cells[scene_idx], self.target_directions[scene_idx]):
                if robot_cur_cell == (int(cell[0]), int(cell[1])):
                    is_robot_placed[-1] = True
                    if np.abs(np.dot(normalize_vector(direction), 
                                    normalize_vector(robot_view_vec.cpu().numpy())) - 1) < EPS:
                        is_target_in_view[-1] = True
                        break

        is_robot_static = self.agent.is_static(0.2)
        
        is_robot_placed = torch.tensor(is_robot_placed, device=self.device, dtype=bool)
        is_target_in_view = torch.tensor(is_target_in_view, device=self.device, dtype=bool)

        # print("is_robot_placed", is_robot_placed, "is_target_in_view", is_target_in_view, "product_displaced", self.product_displaced)
        return {
            "success": is_robot_placed & is_target_in_view,
            "is_robot_placed": is_robot_placed,
            "is_target_in_view": is_target_in_view,
            "is_robot_static": is_robot_static
        }


    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.2, 0.2, 4], [5, 5, 2])
        pose = sapien_utils.look_at([0.1, 0.1, 2.75], [1.8, 1.8, 0])
        return CameraConfig(
            "render_camera", 
            pose=pose,
            width=512, 
            height=512, 
            fov=np.pi / 2, 
            near=0.01, 
            far=100, 
        )
    
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.1, 0.1, 2.75], [1.8, 1.8, 0])
        return [CameraConfig("base_camera", pose, 512, 512, np.pi / 2, 0.01, 100)]

