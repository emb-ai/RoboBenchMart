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

sys.path.append('.')
from dsynth.scene_gen.arrangements import shelf_placement_v2
from dsynth.scene_gen.layouts.random_connectivity import add_many_zones, get_orientation
from dsynth.scene_gen.utils import flatten_dict
from dsynth.scene_gen.hydra_configs import product_filling_from_darkstore_config, Config, ShelfConfig
from dsynth.assets.asset import load_assets_lib


cs = ConfigStore.instance()
cs.store(group="shelves", name="base_shelf_config", node=ShelfConfig)
cs.store(group="ds", name="main_darkstore_config_base", node=Config)

OUTPUT_PATH = 'generated_envs'

@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg: Config) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    product_assets_lib = flatten_dict(load_assets_lib(cfg.assets.products_hierarchy), sep='.')

    product_filling = product_filling_from_darkstore_config(cfg.ds, list(product_assets_lib.keys()))
    zones_dict = {key: list(val.keys()) for key, val in product_filling.items()}
    product_filling_flattened = flatten_dict(product_filling, sep='.')

    n, m = cfg.ds.size_n, cfg.ds.size_m
    x, y = cfg.ds.entrance_coords_x, cfg.ds.entrance_coords_y
    mat = [[0] * m for _ in range(n)]
    is_gen, room = add_many_zones((x, y), mat, zones_dict)
    if not is_gen:
        log.error(f"Can't generate!")
        exit(-1)
    is_rotate = get_orientation((x, y), room)


    scene_meta = shelf_placement_v2(product_filling_flattened, room, is_rotate, product_assets_lib, cfg.ds.show)

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

    with open(output_dir / "scene_config.json", "w") as f:
        json.dump(scene_meta, f, indent=4)

    with open(output_dir / "input_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    log.info(f"Done")

if __name__ == "__main__":
    main()