import logging
log = logging.getLogger(__name__)
import os
import random
from functools import partial
import traceback
from multiprocessing import Pool
import json
from pathlib import Path
import hashlib
from tqdm import tqdm
from typing import Optional, Union, BinaryIO, IO, Dict, Tuple
import numpy as np
import scene_synthesizer as synth

from omegaconf import DictConfig
from omegaconf import OmegaConf
from dsynth.scene_gen.layouts.layout_generator import LayoutGeneratorBase
from dsynth.scene_gen.hydra_configs import ShelfConfig, FillingType
from dsynth.scene_gen.arrangements import shelf_placement_v2
from dsynth.scene_gen.utils import flatten_dict, ProductnameIteratorInfinite, ProductnameIterator
from dsynth.scene_gen.layouts.layout_generator import LAYOUT_CONTINUOUS_TO_CLS
from dsynth.scene_gen.hydra_configs import DsContinuousConfig, ShelfConfig
from dsynth.assets.asset import load_assets_lib
from dsynth.assets.ss_assets import DefaultShelf
from dsynth.scene_gen.arrangements import set_shelf, add_objects_to_shelf_v2

import os
os.environ['PYTHONHASHSEED'] = '42'

class SceneGenerator:
    def __init__(self, 
                 layout_generator: LayoutGeneratorBase,
                 product_assets_lib: Dict,
                 darkstore_arrangement_cfg: DictConfig,
                 num_scenes: int,
                 num_workers: int = 1,
                 output_dir: Optional[Union[str, os.PathLike, BinaryIO, IO[bytes]]] = None,
                 randomize_layout: bool = False,
                 randomize_arrangements: bool = True,
                 random_seed: int = 42,
                 show: bool = False
                 ):
        self.num_workers = num_workers

        config_hash = hashlib.sha1(OmegaConf.to_yaml(darkstore_arrangement_cfg).encode()).hexdigest()[-8:]


        seeds_layout = [random_seed] * num_scenes
        if randomize_layout:
            seeds_layout = np.arange(num_scenes) + random_seed
        
        seeds_arrangements = [random_seed] * num_scenes
        if randomize_arrangements:
            seeds_arrangements = np.arange(num_scenes) + random_seed

        self.task_params = []
        for n, (seed_layout, seed_arr) in enumerate(zip(seeds_layout, seeds_arrangements)):
            self.task_params.append({
                'layout_gen_params': {},
                'seed_layout': seed_layout,
                'seed_arrangement': seed_arr,
                'output_name': f'scene_config_{config_hash}_{n}.json'
            })

        self.generate_routine = partial(
            _generate_routine,
            layout_generator = layout_generator,
            product_assets_lib = product_assets_lib,
            darkstore_arrangement_cfg = darkstore_arrangement_cfg,
            output_dir = output_dir,
            show = show
        )
    
    def generate(self):
        if self.num_workers == 1:
            results = []
            for task_param in tqdm(self.task_params):
                results.append(self.generate_routine(task_param))
            # results = list(map(self.generate_routine, self.task_params))
        else:
            with Pool(self.num_workers) as p:
                total_samples = len(self.task_params)
                results = list(tqdm(p.imap(self.generate_routine, self.task_params), total=total_samples))
        return results


class SceneGeneratorContinuous:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = Path(output_dir)

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        random_seed = cfg.ds_continuous.random_seed
        num_scenes = cfg.ds_continuous.num_scenes

        self.num_workers = cfg.ds_continuous.num_workers

        seeds_layout = [random_seed] * num_scenes
        if cfg.ds_continuous.randomize_layout:
            seeds_layout = np.arange(num_scenes) + random_seed
        
        seeds_arrangements = [random_seed] * num_scenes
        if cfg.ds_continuous.randomize_arrangements:
            seeds_arrangements = np.arange(num_scenes) + random_seed
        
        with open(self.output_dir / f"input_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        self.task_params = []
        for n, (seed_layout, seed_arr) in enumerate(zip(seeds_layout, seeds_arrangements)):
            self.task_params.append({
                'scene_id': n,
                'cfg': cfg,
                'output_dir': self.output_dir,
                'seed_layout_gen': seed_layout,
                'seed_arrangement_gen': seed_arr
            })

    def generate(self):
        if self.num_workers == 1:
            results = []
            for task_param in tqdm(self.task_params):
                results.append(_generate_continuous_routine(task_param))
        else:
            with Pool(self.num_workers) as p:
                total_samples = len(self.task_params)
                results = list(tqdm(p.imap(_generate_continuous_routine, self.task_params), total=total_samples))
        return results

    

def _generate_continuous_routine(task_params):
    scene_id = task_params['scene_id']
    cfg = task_params['cfg']
    output_dir = task_params['output_dir']
    seed_layout_gen = task_params['seed_layout_gen']
    seed_arrangement_gen = task_params['seed_arrangement_gen']
    config_hash = hashlib.sha1(OmegaConf.to_yaml(cfg).encode()).hexdigest()[-8:]

    scene_name = f'{cfg.ds_continuous.name}_{config_hash}_{scene_id}'

    try:
        layout_gen_cls = LAYOUT_CONTINUOUS_TO_CLS[cfg.ds_continuous.layout_gen_type]

        layout_generator = layout_gen_cls(name=scene_name,
            cfg=cfg,
            rng=random.Random(seed_layout_gen),
        )

        layout_data = layout_generator()
        if layout_data is None:
            log.error(f"Failed to generate {scene_name}!")
            return False
        
        fake_arrangements_mapping = OmegaConf.to_container(cfg.ds_continuous.fake_arrangements_mapping, resolve = True)
        results = dict(layout_data=layout_data, 
                    size_x=cfg.ds_continuous.size_x, 
                    size_y=cfg.ds_continuous.size_y,
                    fake_arrangements_mapping=fake_arrangements_mapping)
        
        results['hash'] = config_hash
        results['scene_id'] = scene_id
        results['scene_name'] = scene_name

        with open(Path(output_dir) / f'layout_data_{scene_name}.json', "w") as f:
            json.dump(results, f, indent=4)

        product_assets_lib = flatten_dict(load_assets_lib(cfg.assets, disable_caching=True), sep='.')

        for shelvings_list in [cfg.ds_continuous.active_shelvings_list, cfg.ds_continuous.active_wall_shelvings_list]:
            for active_fixture_cfg in shelvings_list:
                filling, shelf_name, shelf_type = product_filling_from_shelf_config(active_fixture_cfg, 
                                        list(product_assets_lib.keys()), rng=random.Random(seed_arrangement_gen)
                                        )
                
                scene = synth.Scene()
                shelf_name = f'{scene_name}_{active_fixture_cfg.name}'
                shelf_asset_name = active_fixture_cfg.shelf_asset

                if shelf_asset_name is None:
                    shelf = DefaultShelf
                    shelf_asset_name = 'fixtures.shelf'
                else:
                    shelf = product_assets_lib[shelf_asset_name].ss_asset

                support_data = set_shelf(
                    scene,
                    shelf,
                    0,
                    0,
                    0,
                    f'SHELF_{shelf_name}',
                    f'support_SHELF_{shelf_name}',
                )

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
                            active_fixture_cfg.filling_type,
                            seed_arrangement_gen,
                            active_fixture_cfg.noise_std_x,
                            active_fixture_cfg.noise_std_y,
                            active_fixture_cfg.rotation_lower,
                            active_fixture_cfg.rotation_upper,
                        )
                
                json_str = synth.exchange.export.export_json(scene, include_metadata=False)
                data = json.loads(json_str)
                del data["geometry"]
                output_name = f'{shelf_name}.json'
                with open(Path(output_dir) / output_name, "w") as f:
                    json.dump(data, f, indent=4)

        return True
    
    except Exception as e:
        log.error(f"Failed to generate {scene_name}! Unexpected error: {e}")
        traceback.print_exc()
        return False


def _generate_routine(
    task_params: Tuple,
    layout_generator: LayoutGeneratorBase,
    product_assets_lib: Dict,
    darkstore_arrangement_cfg: DictConfig,
    output_dir: Optional[Union[str, os.PathLike, BinaryIO, IO[bytes]]] = None,
    show: bool = False
):
    layout_gen_params = task_params['layout_gen_params']
    seed_layout = task_params['seed_layout']
    seed_arrangement = task_params['seed_arrangement']
    output_name = task_params['output_name']
    
    product_filling, ds_names = product_filling_from_darkstore_config(
        darkstore_arrangement_cfg.zones, 
        list(product_assets_lib.keys()), 
        rng=random.Random(seed_arrangement)
    )

    zones_dict = {key: list(val.keys()) for key, val in product_filling.items()}
    product_filling_flattened = flatten_dict(product_filling, sep='.')

    layout_generator.rng = random.Random(seed_layout)
    layout_data = layout_generator(**layout_gen_params, zones_dict=zones_dict, darkstore_arrangement_cfg=darkstore_arrangement_cfg)
    if layout_data is None:
        log.error(f"Can't generate {output_name}!")
        return False

    scene_meta = shelf_placement_v2(
        product_filling_flattened=product_filling_flattened,
        product_assets_lib=product_assets_lib, 
        is_showed=show,
        darkstore_cfg=darkstore_arrangement_cfg,
        **layout_data
        )
    
    scene_meta["meta"]["ds_names"] = ds_names
    
    if output_dir is not None:
        with open(Path(output_dir) / output_name, "w") as f:
            json.dump(scene_meta, f, indent=4)
        return True
    else:
        return scene_meta

def product_filling_from_shelf_config(shelf_config: ShelfConfig, all_product_names, rng):
    assert 0 <= shelf_config.start_filling_board <= shelf_config.end_filling_from_board <= shelf_config.num_boards
    shelf_name = shelf_config.name
    shelf_type = shelf_config.shelf_type.name

    # all_product_names = ['products_hierarchy.' + name for name in all_product_names]

    filling = [[] for _ in range(shelf_config.start_filling_board)]

    if '_INFINITE' in str(shelf_config.filling_type):
        product_iterator = ProductnameIteratorInfinite(shelf_config.queries, all_product_names, rng=rng)
    else:
        product_iterator = ProductnameIterator(shelf_config.queries, all_product_names, rng=rng)


    if shelf_config.filling_type == FillingType.FULL_AUTO:
        product = next(product_iterator) #pick first suitable product
        for _ in range(shelf_config.start_filling_board, shelf_config.end_filling_from_board):
            filling.append([product for _ in range(shelf_config.num_products_per_board)])
    
    elif shelf_config.filling_type in [FillingType.BOARDWISE_AUTO, FillingType.BOARDWISE_AUTO_INFINITE]:
        for _ in range(shelf_config.start_filling_board, shelf_config.end_filling_from_board):
            try:
                product = next(product_iterator)
            except StopIteration:
                break
            filling.append([product for _ in range(shelf_config.num_products_per_board)])
    
    elif shelf_config.filling_type in [FillingType.BLOCKWISE_AUTO, FillingType.BLOCKWISE_AUTO_INFINITE]:
        cur_board = shelf_config.start_filling_board
        cur_product = next(product_iterator)
        left_products_to_put = shelf_config.num_products_per_block
        left_space_on_board = shelf_config.num_products_per_board
        while True:
            num_products = min(left_products_to_put, left_space_on_board)
            if len(filling) <= cur_board:
                filling.append([cur_product for _ in range(num_products)])
            else:
                filling[cur_board].extend([cur_product for _ in range(num_products)])
            left_space_on_board -= num_products
            left_products_to_put -= num_products

            if left_space_on_board <= 0:
                cur_board += 1
                left_space_on_board = shelf_config.num_products_per_board
            if left_products_to_put <= 0:
                try:
                    cur_product = next(product_iterator)
                except StopIteration:
                    break
                left_products_to_put = shelf_config.num_products_per_block
            
            if cur_board >= shelf_config.end_filling_from_board:
                break

    elif shelf_config.filling_type == FillingType.BOARDWISE_COLUMNS:
        board_product_numcol = OmegaConf.to_container(shelf_config['board_product_numcol'])
        # filling = shelf_config['board_product_numcol']
        for board_idx in range(shelf_config.start_filling_board, shelf_config.end_filling_from_board):
            if not board_idx in board_product_numcol:
                filling.append([])
                continue
            cur_board_arrangement = list(board_product_numcol[board_idx].items())
            rng.shuffle(cur_board_arrangement)

            filling.append([f'{key}:{val}' for key, val in cur_board_arrangement])
        
    for _ in range(shelf_config.end_filling_from_board, shelf_config.num_boards):
        filling.append([])
    
    if shelf_config.shuffle_boards:
        # shuffle only non-empty boards
        non_empty_boards_idxs = [i for i in range(len(filling)) if len(filling[i]) > 0]
        non_empty_boards_filling = [filling[i] for i in range(len(filling)) if len(filling[i]) > 0]
        rng.shuffle(non_empty_boards_filling)
        for i, board_filling in zip(non_empty_boards_idxs, non_empty_boards_filling):
            filling[i] = board_filling

    if shelf_config.shuffle_items_on_board:
        for i in range(len(filling)):
            rng.shuffle(filling[i])

    return filling, shelf_name, shelf_type


def product_filling_from_zone_config(zone_config, all_product_names, rng):
    filling = {}
    if 'name' in zone_config:
        zone_names = {'zone_name': zone_config['name'], 'shelf_names': {}, 'shelf_types': {}}
    else:
        zone_names = {'zone_name': 'Unnamed', 'shelf_names': {}, 'shelf_types': {}}
        
    for key, val in zone_config.items():
        if key != 'name':
            filling[key], \
            zone_names['shelf_names'][key], \
            zone_names['shelf_types'][key] = product_filling_from_shelf_config(val, all_product_names, rng)
    return filling, zone_names

def product_filling_from_darkstore_config(darkstore_config: DictConfig, all_product_names, rng):
    filling = {}
    ds_names = {}
    for zone_name, zone_config in darkstore_config.items():
        filling[zone_name], ds_names[zone_name] = product_filling_from_zone_config(zone_config, all_product_names, rng)
    return filling, ds_names

