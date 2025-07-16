import os
from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import Dict, Any, IO, BinaryIO, Union, Literal, Optional, List, Sequence
from omegaconf import DictConfig, OmegaConf
from transforms3d import quaternions
import torch
import numpy as np

import scene_synthesizer as synth
from scene_synthesizer import utils
import sapien
from mani_skill.envs.scene import ManiSkillScene

logger = logging.getLogger(__name__)

ASSET_TYPE_MAPPING = {
    "MeshAsset": synth.assets.MeshAsset,
    "USDAsset": synth.assets.USDAsset,
    "URDFAsset": synth.assets.URDFAsset,
    "Asset": synth.Asset
}

@dataclass
class Asset:
    asset_file_path: Union[str, os.PathLike, BinaryIO, IO[bytes]]
    ss_asset_type: Any = 'Asset'
    ss_params: Dict = field(default_factory = lambda: {})
    ms_is_static: bool = False
    ms_is_nonconvex_collision: bool = False

    disable_caching: bool = False
    _ss_asset: Any = None
    _trimesh_scene: Any = None
    _ms_scale: Any = None
    _ms_origin: Any = None

    asset_name: str = 'asset'

    def __post_init__(self):
        if not Path(self.asset_file_path).exists():
            logger.warning(f'Asset path {str(self.asset_file_path)} does not exist!')
    

    @property
    def ss_asset(self):
        if self._ss_asset is not None:
            return self._ss_asset
        
        contructor = ASSET_TYPE_MAPPING.get(self.ss_asset_type, None)
        if contructor is None:
            raise ValueError(f"Wrong asset type: {self.ss_asset_type}, possible values: {list(ASSET_TYPE_MAPPING.keys())}")
        
        ss_asset = contructor(self.asset_file_path, **self.ss_params)
        if not self.disable_caching:
            self._ss_asset = ss_asset

        return ss_asset
    
    @property
    def trimesh_scene(self):
        if self._trimesh_scene is not None:
            return self._trimesh_scene
        
        trimesh_scene = self.ss_asset.as_trimesh_scene()
        if not self.disable_caching:
            self._trimesh_scene = trimesh_scene

        return trimesh_scene
    
    @property
    def extents(self):
        return self.trimesh_scene.extents
    
    def scale_and_transform(self):
        use_collision_geometry = self.ss_params.get('use_collision_geometry', True)
        trimesh_scene = self.ss_asset._as_trimesh_scene(
            namespace="", use_collision_geometry=use_collision_geometry
        )
        trimesh_scene = utils.normalize_and_bake_scale(trimesh_scene)
        scale = self.ss_asset._get_scale(raw_extents=trimesh_scene.extents)
        
        scaled_scene = utils.scaled_trimesh_scene(trimesh_scene, scale=scale)
        center_mass = utils.center_mass(trimesh_scene=scaled_scene, node_names=scaled_scene.graph.nodes_geometry)
        origin = self.ss_asset._get_origin_transform(
            bounds=scaled_scene.bounds,
            center_mass= center_mass,
            centroid=scaled_scene.centroid,
        )
        return scale, origin
    
    @property
    def ms_scale(self):
        if self._ms_scale is not None:
            return self._ms_scale
        
        scale, origin = self.scale_and_transform()
        if not self.disable_caching:
            self._ms_scale, self._ms_origin = scale, origin

        return scale
    
    @property
    def ms_origin(self):
        if self._ms_origin is not None:
            return self._ms_origin

        scale, origin = self.scale_and_transform()
        if not self.disable_caching:
            self._ms_scale, self._ms_origin = scale, origin

        return origin
    
    def ms_build_actor(
        self,
        obj_name: str,
        scene: ManiSkillScene,
        pose: Optional[sapien.Pose] = None,
        T: Optional[np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float32]]] = None,
        scene_idxs: Optional[Union[List[int], Sequence[int], torch.Tensor, np.ndarray]] = None,
        force_static: bool = False
    ):
        assert (pose is not None) != (T is not None), "Actor pose or (exclusive) transform must be specified"
        
        if T is None:
            T = pose.to_transformation_matrix()

        T = T @ self.ms_origin

        p = T[:-1, 3]
        q = quaternions.mat2quat(T[:3,:3])
        pose = sapien.Pose(p = p, q = q)
        
        # ms_scale, origin = self.ms_scale_and_transform
        scale = np.array([self.ms_scale, self.ms_scale, self.ms_scale])

        if self.ss_asset_type == 'URDFAsset':
            return self.load_actor_as_urdf(obj_name, scene, pose, scale, scene_idxs)

        builder = scene.create_actor_builder()
        builder.set_scene_idxs(scene_idxs)
        builder.add_visual_from_file(filename=self.asset_file_path, scale=scale)
        builder.set_initial_pose(pose)
        if self.ms_is_nonconvex_collision:
            builder.add_nonconvex_collision_from_file(filename=self.asset_file_path, scale=scale)
        else:
            builder.add_convex_collision_from_file(filename=self.asset_file_path, scale=scale)
        

        if self.ms_is_static or force_static:
            actor = builder.build_static(name=obj_name)
        else:
            actor = builder.build(name=obj_name)
            if actor.get_mass().item() > 0.7:
                actor.set_mass(0.7)

        return actor

    def load_actor_as_urdf(self, obj_name: str, scene: ManiSkillScene, pose: sapien.Pose, scale, scene_idxs):
        loader = scene.create_urdf_loader()
        loader.scale = scale[0]
        articulation_builders = loader.parse(self.asset_file_path)["articulation_builders"]
        builder = articulation_builders[0]
        builder.initial_pose = pose
        builder.set_scene_idxs(scene_idxs)
        return builder.build(name=obj_name)

def load_assets_lib(products_hierarchy_dict: DictConfig, disable_caching=False):
    assets_dict = {}
    products_dict = OmegaConf.to_container(products_hierarchy_dict, resolve = True)
    if 'asset_file_path' in products_dict.keys():
        return Asset(**products_dict, disable_caching = disable_caching)
    for key, val in products_dict.items():
        if not isinstance(val, Dict):
            assets_dict[key] = val
        else:
            assets_dict[key] = load_assets_lib(products_hierarchy_dict[key])
    return assets_dict


if __name__ == '__main__':
    a = {
        'asset_file_path': 'sasha_assets/milkHandle.glb',
        'ss_params': {
            'height': 0.25,
            'up': [0, 1, 0],
            'front': [0, 0, -1] ,
            'origin': ["left", "bottom", "com"]
            # 'up': [0, 1, 0], 
            # 'front': [0, 0, -1], 
            # 'origin': ["left", "bottom", "com"]
        }
    }
    a = Asset(**a)
    scale = a.scale

    print(a)