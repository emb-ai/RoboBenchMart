import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
import os
from pathlib import Path
import json
import sys

sys.path.append('.')
from dsynth.scene_gen.arrangements import load_assets, shelf_placement_v2
from dsynth.scene_gen.layouts.random_connectivity import add_many_zones, get_orientation
from dsynth.scene_gen.utils import flatten_dict
from dsynth.scene_gen.hydra_configs import product_filling_from_darkstore_config, Config, ALL_PRODUCTS_FLATTENED, ShelfConfig
from dsynth.assets import ASSETS_PATH


cs = ConfigStore.instance()
cs.store(group="shelves", name="base_shelf_config", node=ShelfConfig)
cs.store(name="main_config_base", node=Config)

OUTPUT_PATH = 'generated_envs'

@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    product_filling = product_filling_from_darkstore_config(cfg, list(ALL_PRODUCTS_FLATTENED.keys()))
    zones_dict = {key: list(val.keys()) for key, val in product_filling.items()}
    product_filling_flattened = flatten_dict(product_filling, sep='/')

    n, m = cfg.size_n, cfg.size_m
    x, y = cfg.entrance_coords_x, cfg.entrance_coords_y
    mat = [[0] * m for _ in range(n)]
    is_gen, room = add_many_zones((x, y), mat, zones_dict)
    if not is_gen:
        print(f"Can't generate!")
        exit(-1)
    is_rotate = get_orientation((x, y), room)

    product_models = load_assets(ALL_PRODUCTS_FLATTENED)

    scene_meta = shelf_placement_v2(product_filling_flattened, room, is_rotate, product_models, cfg.show)

    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=cfg.rewrite)
    else:
        output_dir = Path(OUTPUT_PATH) / 'env'

        i = 2
        while output_dir.exists() and not cfg.rewrite:
            output_dir = Path(OUTPUT_PATH) / f'env({i})'
            i += 1
        output_dir.mkdir(parents=True, exist_ok=cfg.rewrite)

    print(f"Write results to: {output_dir}")

    with open(output_dir / "scene_config.json", "w") as f:
        json.dump(scene_meta, f, indent=4)

    with open(output_dir / "input_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    print(f"Done")

if __name__ == "__main__":
    main()