import torch
import numpy as np
import os
import sapien
from transforms3d import euler

from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from dsynth.scene_gen.arrangements import CELL_SIZE
from transforms3d.euler import euler2quat

@register_env('MoveFromBoardToBoardStaticEnv', max_episode_steps=200000)
class MoveFromBoardToBoardStaticEnv(DarkstoreCellBaseEnv):

    def _load_scene(self, options: dict):
        super()._load_scene(options)

    def _compute_robot_init_pose(self, env_idx = None): #TODO: redo this shit
        target_shelf = 'zone1.shelf1'
        for i in range(len(self.scene_builder.room[0])):
            for j in range(len(self.scene_builder.room[0][i])):
                if self.scene_builder.room[0][i][j] == target_shelf:
                    rot = self.scene_builder.rotations[0][i][j]
                    shelf_i, shelf_j = i, j
                    break
        if rot == 0:
            origin, angle = np.array([shelf_i, shelf_j - 1, 0.]), np.pi / 2
        if rot == -90:
            origin, angle = np.array([shelf_i - 1, shelf_j, 0.]), 0 
        if rot == 90:
            origin, angle = np.array([shelf_i + 1, shelf_j, 0.]), np.pi
        if rot == 180:
            origin, angle = np.array([shelf_i, shelf_j + 1, 0.]), - np.pi / 2
        origin = origin * CELL_SIZE
        origin[:2] += CELL_SIZE / 2
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
        
        elif self.robot_uids == "ds_fetch":
            qpos = np.array(
                [
                 0.2 * CELL_SIZE,
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

        robot_pose = self.agent.robot.get_pose()
        cart_shift = np.array([0.4, -0.2, 0.])
        new_cart_pose_p = robot_pose.p[0].cpu().numpy() + cart_shift 
        
        self.setup_target_object()

        self.language_instruction = f'move {self.target_product_str} one board higher'

        target_pose = self.actors['products'][self.target_product_name].pose
        target_pose.raw_pose[0, 2] += get_actor_obb(self.actors['products'][self.target_product_name]).extents[2]/2 + 0.397 #height of board
        target_pose.raw_pose[0][3:] = torch.Tensor([1, 0, 0, 0])
        self.target_sizes = get_actor_obb(self.actors['products'][self.target_product_name]).extents
        # self.target_volume.set_pose(target_pose)
        # self.target_volume.remove_from_scene()
        self.target_volume = actors.build_box(
            self.scene,
            half_sizes=list(self.target_sizes/2),
            color=[0, 1, 0, 0.5],
            name="target_box",
            body_type="static",
            add_collision=False,
            initial_pose=target_pose,
        )

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