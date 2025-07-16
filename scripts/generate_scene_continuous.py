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
from dsynth.scene_gen.scene_generator import SceneGeneratorContinuous, product_filling_from_shelf_config
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

    generator = SceneGeneratorContinuous(cfg, output_dir)
    results = generator.generate()
        
    if np.all(results):
        log.info(f"Done")
    elif np.all(~results):
        log.info(f"All generations are failed")
    else:
        log.info(f"Not all generations are sucessful: {results}")

if __name__ == "__main__":
    main()