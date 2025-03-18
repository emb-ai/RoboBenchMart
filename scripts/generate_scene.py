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

sys.path.append('.')
from dsynth.scene_gen.scene_generator import SceneGenerator
from dsynth.scene_gen.layouts.layout_generator import LAYOUT_TYPES_TO_CLS
from dsynth.scene_gen.utils import flatten_dict
from dsynth.scene_gen.hydra_configs import DsConfig, ShelfConfig
from dsynth.assets.asset import load_assets_lib


cs = ConfigStore.instance()
cs.store(group="shelves", name="base_shelf_config", node=ShelfConfig)
cs.store(group="ds", name="main_darkstore_config_base", node=DsConfig)

OUTPUT_PATH = 'generated_envs'

@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    product_assets_lib = flatten_dict(load_assets_lib(cfg.assets.products_hierarchy), sep='.')

    if cfg.ds.output_dir is not None:
        output_dir = Path(cfg.ds.output_dir)
        output_dir.mkdir(parents=True, exist_ok=cfg.ds.rewrite)
    else:
        output_dir = Path(OUTPUT_PATH) / 'env'

        i = 2
        while output_dir.exists() and not cfg.ds.rewrite:
            output_dir = Path(OUTPUT_PATH) / f'env({i})'
            i += 1
        output_dir.mkdir(parents=True, exist_ok=cfg.ds.rewrite)

    log.info(f"Write results to: {output_dir}")

    layout_gen_cls = LAYOUT_TYPES_TO_CLS[cfg.ds.layout_gen_type]
    layout_generator = layout_gen_cls(sizes_nm=(cfg.ds.size_n, cfg.ds.size_m), 
                   start_coords=(cfg.ds.entrance_coords_x, cfg.ds.entrance_coords_y))

    scene_gen = SceneGenerator(
        layout_generator = layout_generator,
        product_assets_lib = product_assets_lib,
        darkstore_arrangement_cfg=cfg.ds.zones,
        num_scenes=cfg.ds.num_scenes,
        num_workers=cfg.ds.num_workers,
        output_dir=output_dir,
        randomize_arrangements=cfg.ds.randomize_arrangements,
        randomize_layout=cfg.ds.randomize_layout,
        random_seed=cfg.ds.random_seed
    )
    results = scene_gen.generate()
    results = np.array(results)

    with open(output_dir / "input_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    if np.all(results):
        log.info(f"Done")
    elif np.all(~results):
        log.info(f"All generations are failed")
    else:
        log.info(f"Not all generations are sucessful: {results}")
if __name__ == "__main__":
    main()