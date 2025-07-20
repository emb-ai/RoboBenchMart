import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)
import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
import os
from pathlib import Path
import json
import sys
import numpy as np
import random
import scene_synthesizer as synth

sys.path.append('.')
from dsynth.scene_gen.scene_generator import SceneGenerator, product_filling_from_shelf_config
from dsynth.scene_gen.layouts.layout_generator import LAYOUT_TYPES_TO_CLS
from dsynth.scene_gen.utils import flatten_dict
from dsynth.scene_gen.hydra_configs import DsConfig, ShelfConfig
from dsynth.assets.asset import load_assets_lib
from hydra import compose, initialize
from dsynth.assets.ss_assets import DefaultShelf
from dsynth.scene_gen.arrangements import set_shelf, add_objects_to_shelf_v2

cs = ConfigStore.instance()
cs.store(name="base_shelf_config", node=ShelfConfig)

OUTPUT_PATH = 'generated_envs'
CONF_PATH = '../conf'

with initialize(version_base=None, config_path=CONF_PATH ):
    assets_cfg = compose(config_name="assets/assets_downscaled")

@hydra.main(version_base=None, config_name="assets/shelf_fake_1.yaml", config_path=str(Path(CONF_PATH) / "shelves"))
def main(shelf_cfg) -> None:
    seed_arrangement = 42
    log.info(OmegaConf.to_yaml(shelf_cfg))
    product_assets_lib = flatten_dict(load_assets_lib(assets_cfg.assets), sep='.')
    filling, shelf_name, shelf_type = product_filling_from_shelf_config(shelf_cfg, list(product_assets_lib.keys()), rng=random.Random(seed_arrangement))
    # print(filling, shelf_name, shelf_type)

    scene = synth.Scene()
    
    shelf_asset_name = shelf_cfg.shelf_asset
    
    if shelf_asset_name is None:
        shelf = DefaultShelf
        shelf_asset_name = 'fixtures.shelf'
    else:
        shelf = product_assets_lib[shelf_asset_name].ss_asset
    
    support_data = set_shelf(
        scene,
        shelf,
        0,
        0.274,
        0,
        f'SHELF_{shelf_name}',
        f'support_SHELF_{shelf_name}',
    )
    # scene.show_supports()
    add_objects_to_shelf_v2(
                scene,
                0,
                filling,
                product_assets_lib,
                support_data,
                shelf_cfg.x_gap,
                shelf_cfg.y_gap,
                shelf_cfg.delta_x,
                shelf_cfg.delta_y,
                shelf_cfg.start_point_x,
                shelf_cfg.start_point_y,
                shelf_cfg.filling_type,
                seed_arrangement,
                shelf_cfg.noise_std_x,
                shelf_cfg.noise_std_y,
                shelf_cfg.rotation_lower,
                shelf_cfg.rotation_upper,
            )
    # scene.show()
    out_name = f'assets/fake_shelves/{shelf_cfg.name}.glb'
    scene.export(out_name)
    print(f"Write to {out_name}")

if __name__ == "__main__":
    main()