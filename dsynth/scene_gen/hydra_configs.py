from dataclasses import dataclass, field
from enum import Enum
from typing import List, Any, Dict
import re
import random
from itertools import cycle

from omegaconf import DictConfig, OmegaConf, MISSING
import hydra
from hydra.core.config_store import ConfigStore

from dsynth.scene_gen.utils import ProductnameIterator, ProductnameIteratorInfinite, flatten_dict

class FillingType(Enum):
    BLOCKWISE_AUTO = 'BLOCKWISE_AUTO'
    BLOCKWISE_AUTO_INFINITE = 'BLOCKWISE_AUTO_INFINITE'
    BOARDWISE_AUTO = 'BOARDWISE_AUTO'
    BOARDWISE_AUTO_INFINITE = 'BOARDWISE_AUTO_INFINITE'
    FULL_AUTO = 'FULL_AUTO'
    LISTED = 'LISTED'


@dataclass
class ShelfConfig:
    name: str
    filling_type: FillingType = FillingType.FULL_AUTO
    queries: List[str] = field(default_factory=lambda: [])

    num_products_per_block: int = 7
    num_products_per_board: int = 10
    start_filling_board: int = 0
    end_filling_from_board: int = 5

    is_dynamic: bool = True

    num_boards: int = 5

def product_filling_from_shelf_config(shelf_config: ShelfConfig, all_product_names):
    assert 0 <= shelf_config.start_filling_board <= shelf_config.end_filling_from_board <= shelf_config.num_boards

    filling = [[] for _ in range(shelf_config.start_filling_board)]

    if '_INFINITE' in str(shelf_config.filling_type):
        product_iterator = ProductnameIteratorInfinite(shelf_config.queries, all_product_names)
    else:
        product_iterator = ProductnameIterator(shelf_config.queries, all_product_names)


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
            

    for _ in range(shelf_config.end_filling_from_board, shelf_config.num_boards):
        filling.append([])
    return filling

def product_filling_from_zone_config(zone_config, all_product_names):
    filling = {}
    for shelf_name, shelf_config in zone_config.items():
        filling[shelf_name] = product_filling_from_shelf_config(shelf_config, all_product_names)
    return filling

@dataclass    
class Config:
    name: str 
    size_n: int = MISSING
    size_m: int = MISSING
    entrance_coords_x: int = 0
    entrance_coords_y: int = 0
    zones: Dict = MISSING
    
    output_dir: Any = None
    rewrite: bool = False
    show: bool = False
    layout_gen: str = 'random_connectivity'

def product_filling_from_darkstore_config(darkstore_config: Config, all_product_names):
    filling = {}
    for zone_name, zone_config in darkstore_config.zones.items():
        filling[zone_name] = product_filling_from_zone_config(zone_config, all_product_names)
    return filling

