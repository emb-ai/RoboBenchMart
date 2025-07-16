from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from omegaconf import MISSING


class LayoutGenType(Enum):
    DEFAULT = 'DEFAULT'
    CONNECTED_ZONES = 'CONNECTED_ZONES'
    FIXED_LAYOUT = 'FIXED_LAYOUT'

class LayoutContGenType(Enum):
    FIXED_LAYOUT = 'FIXED_LAYOUT'
    PROCEDURAL_TENSOR_FIELD = 'PROCEDURAL_TENSOR_FIELD'
    PROCEDURAL_TENSOR_FIELD_HORIZONTAL = 'PROCEDURAL_TENSOR_FIELD_HORIZONTAL'
    PROCEDURAL_TENSOR_FIELD_VERTICAL = 'PROCEDURAL_TENSOR_FIELD_VERTICAL'

class FillingType(Enum):
    BLOCKWISE_AUTO = 'BLOCKWISE_AUTO'
    BLOCKWISE_AUTO_INFINITE = 'BLOCKWISE_AUTO_INFINITE'
    BOARDWISE_AUTO = 'BOARDWISE_AUTO'
    BOARDWISE_AUTO_INFINITE = 'BOARDWISE_AUTO_INFINITE'
    FULL_AUTO = 'FULL_AUTO'
    LISTED = 'LISTED'
    BOARDWISE_COLUMNS = 'BOARDWISE_COLUMNS'

class ShelfType(Enum):
    SHELF = 'SHELF'
    FRIDGE_FOOD_SHOWCASE = 'FRIDGE_FOOD_SHOWCASE'
    FRIDGE_GLASS_TOP = 'FRIDGE_GLASS_TOP'
    SMALL_SHELF_ONE_SIDED = 'SMALL_SHELF_ONE_SIDED'
    SMALL_SHELF_TWO_SIDED = 'SMALL_SHELF_TWO_SIDED'


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
    shuffle_boards: bool = False
    board_product_numcol: Dict[int, Dict[str, int]] = field(default_factory=lambda: {})
    x_gap: float = 0.002
    y_gap: float = 0.002
    delta_x: float = 0.
    delta_y: float = 0.
    start_point_x: float = -1.
    start_point_y: float = -1.

    shelf_asset: Optional[str] = None
    shelf_type: ShelfType = ShelfType.SHELF

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

    layout: Any = None
    rotations: Any = None

@dataclass
class DsContinuousConfig:
    name: str
    size_x: float = MISSING
    size_y: float = MISSING

    # active shelves (fridges, etc.) used in tasks
    active_shelvings_list: List[ShelfConfig] = field(default_factory=lambda: [])
    active_wall_shelvings_list: List[ShelfConfig] = field(default_factory=lambda: [])

    # passive scene assets
    inactive_shelvings_list: List[str] = field(default_factory=lambda: [])
    inactive_wall_shelvings_list: List[str] = field(default_factory=lambda: [])
    scene_fixtures_list: List[str] = field(default_factory=lambda: [])

    num_scenes: int = 1
    num_workers: int = 1
    output_dir: Optional[str] = None
    rewrite: bool = False

    show: bool = False
    layout_gen_type: LayoutContGenType = LayoutContGenType.PROCEDURAL_TENSOR_FIELD
    randomize_layout: bool = False
    randomize_arrangements: bool = True
    random_seed: int = 42
    max_tries: int = 20

    tf_blending_decay: float = 12.
    inactive_wall_shelvings_occupancy_width: float = 0.4
    inactive_shelvings_occupancy_width: float = 0.6
    inactive_shelvings_skip_prob: float = 0.0
    inactive_shelvings_passage_width: float = 1.5

