from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
from omegaconf import MISSING


class LayoutGenType(Enum):
    DEFAULT = 'DEFAULT'
    CONNECTED_ZONES = 'CONNECTED_ZONES'

class FillingType(Enum):
    BLOCKWISE_AUTO = 'BLOCKWISE_AUTO'
    BLOCKWISE_AUTO_INFINITE = 'BLOCKWISE_AUTO_INFINITE'
    BOARDWISE_AUTO = 'BOARDWISE_AUTO'
    BOARDWISE_AUTO_INFINITE = 'BOARDWISE_AUTO_INFINITE'
    FULL_AUTO = 'FULL_AUTO'
    LISTED = 'LISTED'
    BOARDWISE_COLUMNS = 'BOARDWISE_COLUMNS'


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

    board_product_numcol: Dict[int, Dict[str, int]] = field(default_factory=lambda: {})
    x_gap: float = 0.002
    y_gap: float = 0.002
    delta_x: float = 0.
    delta_y: float = 0.
    start_point_x: float = -1.
    start_point_y: float = -1.

@dataclass    
class DsConfig:
    name: str 
    size_n: int = MISSING
    size_m: int = MISSING
    entrance_coords_x: int = 0
    entrance_coords_y: int = 0
    zones: Dict = MISSING
    
    num_scenes: int = 1
    num_workers: int = 1
    output_dir: Optional[str] = None
    rewrite: bool = False
    show: bool = False
    layout_gen_type: LayoutGenType = LayoutGenType.CONNECTED_ZONES
    randomize_layout: bool = False
    randomize_arrangements: bool = True
    random_seed: int = 42

