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
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb

from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from dsynth.scene_gen.arrangements import CELL_SIZE
from dsynth.scene_gen.utils import find_paths

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    multiply = 1 / (norm + 1e-5)
    return multiply * vec

@register_env('NavMoveToZoneEnv', max_episode_steps=200000)
class NavMoveToZoneEnv(DarkstoreCellBaseEnv):
    INIT_POSE_GEN_MAX_TRIES = 20
    def _load_scene(self, options: dict):
        super()._load_scene(options)

    def setup_target_object(self):
        # pick random zone
        zone_names = list(self.shelves_placement.keys())
        zone_idx = self._batched_episode_rng[0].randint(0, len(zone_names))
        target_zone = zone_names[zone_idx]

        self.target_zone_name = self.scene_builder.ds_names[0][target_zone]['zone_name']

        # define target cells and view directions to them
        self.target_cells = []
        self.target_directions = []
        for shelf_name, coords in self.shelves_placement[target_zone].items():
            i, j = coords
            rot = self.scene_builder.rotations[0][i][j]
            if rot == 0:
                cell_coords, direction_to_shelf = np.array([i, j - 1, 0.]), np.array([0, 1, 0])
            if rot == -90:
                cell_coords, direction_to_shelf = np.array([i - 1, j, 0.]), np.array([1, 0, 0])
            if rot == 90:
                cell_coords, direction_to_shelf = np.array([i + 1, j, 0.]), np.array([-1, 0, 0])
            if rot == 180:
                cell_coords, direction_to_shelf = np.array([i, j + 1, 0.]), np.array([0, -1, 0])
            self.target_cells.append(cell_coords)
            self.target_directions.append(direction_to_shelf)

            opposite_cell = cell_coords + 2 * direction_to_shelf
            opposite_cell = opposite_cell.astype(np.int32)
            if opposite_cell[0] >= 0 and opposite_cell[0] < len(self.room) and \
                opposite_cell[1] >= 0 and opposite_cell[1] < len(self.room[0]) and \
                self.room[opposite_cell[0]][opposite_cell[1]] == 0:
                    self.target_cells.append(opposite_cell)
                    self.target_cells.append(-direction_to_shelf)


    def _compute_robot_init_pose(self, env_idx = None):
        all_cells = list(itertools.product(range(len(self.room)), range(len(self.room[0]))))
        occupied_cells = []
        for zone_name in self.shelves_placement.keys():
            for shelf_name in self.shelves_placement[zone_name].keys():
                occupied_cells.append(self.shelves_placement[zone_name][shelf_name])
        free_cells = sorted(list(set(all_cells) - set(occupied_cells)))

        is_success = False
        for _ in range(self.INIT_POSE_GEN_MAX_TRIES):
            init_cell_idx = self._batched_episode_rng[0].randint(0, len(free_cells))
            init_cell = free_cells[init_cell_idx]
            paths = find_paths(self.room, (0, 0), init_cell)
            # if init_cell == (0, 0)
            if len(paths) > 0:
                is_success = True
                break
        
        if not is_success:
            raise RuntimeError("Can't generate init pose!")

        origin = np.array([init_cell[0] * CELL_SIZE + CELL_SIZE / 2, 
                  init_cell[1] * CELL_SIZE + CELL_SIZE / 2, 0])
        
        angle = self._batched_episode_rng[0].rand() * 2 * np.pi

        return origin, angle
    
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
            robot_origin, robot_angle = self._compute_robot_init_pose()
            self.agent.robot.set_pose(sapien.Pose(p=robot_origin, q=euler2quat(0, 0, robot_angle)))

        self.setup_target_object()

        self.language_instruction = f'move to {self.target_zone_name}'

        self.target_volumes = []
        for i, target_cell in enumerate(self.target_cells):
            target_p = target_cell * CELL_SIZE + np.array([CELL_SIZE / 2, CELL_SIZE / 2, 0])
            target_pose = sapien.Pose(p=target_p)
            self.target_volumes.append(actors.build_box(
                    self.scene,
                    half_sizes=(0.2, 0.2, 0.2),
                    color=[0, 1, 0, 0.5],
                    name=f"target_box_{i}",
                    body_type="kinematic",
                    add_collision=False,
                    initial_pose=target_pose,
                )
            )
            self._hidden_objects.append(self.target_volumes[-1])

    def evaluate(self):
        EPS = 0.11
        robot_view_vec = self.agent.base_link.pose.to_transformation_matrix()[0, :3, 0]
        robot_cur_pose = self.agent.base_link.pose.sp.p
        robot_cur_cell = (int(robot_cur_pose[0] // CELL_SIZE), int(robot_cur_pose[1] // CELL_SIZE))

        is_target_in_view = False
        is_robot_placed = False
        for cell, direction in zip(self.target_cells, self.target_directions):
            if robot_cur_cell == (int(cell[0]), int(cell[1])):
                is_robot_placed = True
                if np.abs(np.dot(normalize_vector(direction), 
                                 normalize_vector(robot_view_vec)) - 1) < EPS:
                    is_target_in_view = True
                    break

        is_robot_static = self.agent.is_static(0.2)
        if not self.product_displaced:
            for p, a in self.actors['products'].items():
                if not torch.all(torch.isclose(a.pose.raw_pose, self.products_initial_poses[p], rtol=0.1, atol=0.1)):
                    self.product_displaced = True
                    break
        
        is_robot_placed = torch.tensor([is_robot_placed])
        is_target_in_view = torch.tensor([is_target_in_view])
        product_displaced = torch.tensor([self.product_displaced])

        print("is_robot_placed", is_robot_placed, "is_target_in_view", is_target_in_view, "product_displaced", self.product_displaced)
        return {
            "success": is_robot_placed & is_target_in_view,
            "is_robot_placed": is_robot_placed,
            "is_target_in_view": is_target_in_view,
            "is_robot_static": is_robot_static,
            "product_displaced": product_displaced
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

