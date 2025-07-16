import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)
import hydra
import random
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
import os
from pathlib import Path
import json
import sys
import numpy as np
from dataclasses import dataclass, asdict
import scene_synthesizer as synth

import trimesh

sys.path.append('.')
from dsynth.scene_gen.scene_generator import SceneGenerator, product_filling_from_shelf_config
# from dsynth.scene_gen.layouts.layout_generator import LAYOUT_CONTINUOUS_TO_CLS
from dsynth.scene_gen.utils import flatten_dict
# from dsynth.scene_gen.hydra_configs import DsContinuousConfig, ShelfConfig
from dsynth.assets.asset import load_assets_lib
from dsynth.assets.ss_assets import DefaultShelf
from dsynth.scene_gen.arrangements import set_shelf, add_objects_to_shelf_v2
from hydra import compose, initialize
from omegaconf import OmegaConf

def preprocess():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="assets/assets")
    # cfg = OmegaConf.to_container(cfg, resolve = True)

    output_dir = Path("assets/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)
    # print(cfg['assets']['products_hierarchy'])
    # prod_conf = cfg['assets']['products_hierarchy']['food']

    product_assets_lib = flatten_dict(load_assets_lib(cfg.assets), sep='.')

    for prod_id, asset in product_assets_lib.items():
        if prod_id == 'assets_dir_path':
            continue
        print(asset)
        scene = asset.trimesh_scene

        if Path(asset.asset_file_path).suffix != '.glb':
            continue

        output_path = output_dir / Path(asset.asset_file_path).name
        exported = trimesh.exchange.gltf.export_glb(scene)
        # scene.show()
        with open(output_path, 'wb') as f:
            f.write(exported)
        

            



if __name__ == '__main__':
    preprocess()