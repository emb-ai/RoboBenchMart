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

sys.path.append('.')
from dsynth.scene_gen.scene_generator import SceneGenerator, product_filling_from_shelf_config
from dsynth.scene_gen.layouts.layout_generator import LAYOUT_CONTINUOUS_TO_CLS
from dsynth.scene_gen.utils import flatten_dict
from dsynth.scene_gen.hydra_configs import DsContinuousConfig, ShelfConfig
from dsynth.assets.asset import load_assets_lib
from dsynth.assets.ss_assets import DefaultShelf
from dsynth.scene_gen.arrangements import set_shelf, add_objects_to_shelf_v2

cs = ConfigStore.instance()
cs.store(group="shelves", name="base_shelf_config", node=ShelfConfig)
cs.store(group="ds_continuous", name="main_darkstore_continuous_config_base", node=DsContinuousConfig)

OUTPUT_PATH = 'generated_envs'

@hydra.main(version_base=None, config_name="config_continuous", config_path="../conf")
def main(cfg) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    product_assets_lib = flatten_dict(load_assets_lib(cfg.assets), sep='.')

    if cfg.ds_continuous.output_dir is not None:
        output_dir = Path(cfg.ds_continuous.output_dir)
        output_dir.mkdir(parents=True, exist_ok=cfg.ds_continuous.rewrite)
    else:
        output_dir = Path(OUTPUT_PATH) / 'env'

        i = 2
        while output_dir.exists() and not cfg.ds_continuous.rewrite:
            output_dir = Path(OUTPUT_PATH) / f'env({i})'
            i += 1
        output_dir.mkdir(parents=True, exist_ok=cfg.ds_continuous.rewrite)

    log.info(f"Write results to: {output_dir}")

    with open(output_dir / "input_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    
        
    layout_gen_cls = LAYOUT_CONTINUOUS_TO_CLS[cfg.ds_continuous.layout_gen_type]
    scene_sizes = (cfg.ds_continuous.size_x, cfg.ds_continuous.size_y)
    layout_generator = layout_gen_cls(sizes_xy=scene_sizes,
                                                product_assets_lib=product_assets_lib,
                                                cfg=cfg,
                                                rng=random.Random(cfg.ds_continuous.random_seed))

    layout_data = layout_generator(name=f'{cfg.ds_continuous}_seed{cfg.ds_continuous.random_seed}')
    # layout_data = [asdict(fixture_data) for fixture_data in layout_data]
    results = dict(layout_data=layout_data, 
                   size_x=cfg.ds_continuous.size_x, 
                   size_y=cfg.ds_continuous.size_y)

    with open(Path(output_dir) / 'layout_data.json', "w") as f:
        json.dump(results, f, indent=4)

    for active_fixture_cfg in cfg.ds_continuous.active_shelvings_list:
        filling, shelf_name, shelf_type = product_filling_from_shelf_config(active_fixture_cfg, list(product_assets_lib.keys()), rng=random.Random(42))
        
        scene = synth.Scene()
        shelf_name = active_fixture_cfg.name
        shelf_asset_name = active_fixture_cfg.shelf_asset

        # shelf_asset_name = None
        if shelf_asset_name is None:
            shelf = DefaultShelf
            shelf_asset_name = 'fixtures.shelf'
        else:
            shelf = product_assets_lib[shelf_asset_name].ss_asset
        # shelf = DefaultShelf
        support_data = set_shelf(
            scene,
            shelf,
            0,
            0,
            0,
            f'SHELF_{shelf_name}',
            f'support_SHELF_{shelf_name}',
        )
        # scene.show_supports()
        
        # trimesh_scene = product_assets_lib[shelf_asset_name].trimesh_scene
        # trimesh_scene.show(flags={'axis': True})
        # trimesh_scene = DefaultShelf.as_trimesh_scene()
        # trimesh_scene.show(flags={'axis': True})

        add_objects_to_shelf_v2(
                    scene,
                    0,
                    filling,
                    product_assets_lib,
                    support_data,
                    active_fixture_cfg.x_gap,
                    active_fixture_cfg.y_gap,
                    active_fixture_cfg.delta_x,
                    active_fixture_cfg.delta_y,
                    active_fixture_cfg.start_point_x,
                    active_fixture_cfg.start_point_y,
                    active_fixture_cfg.filling_type
                )
        # scene.show()
        
        json_str = synth.exchange.export.export_json(scene, include_metadata=False)
        data = json.loads(json_str)
        del data["geometry"]
        output_name = f'{shelf_name}.json'
        with open(Path(output_dir) / output_name, "w") as f:
            json.dump(data, f, indent=4)
    

    log.info(f"Done")
    # if np.all(results):
    #     log.info(f"Done")
    # elif np.all(~results):
    #     log.info(f"All generations are failed")
    # else:
    #     log.info(f"Not all generations are sucessful: {results}")
if __name__ == "__main__":
    main()