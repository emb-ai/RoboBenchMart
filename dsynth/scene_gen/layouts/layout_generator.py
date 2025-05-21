import logging
log = logging.getLogger(__name__)
import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict

from dsynth.scene_gen.layouts.random_connectivity import add_many_zones, get_orientation
from dsynth.scene_gen.hydra_configs import LayoutGenType

class LayoutGeneratorBase(ABC):
    def __init__(self, 
                 sizes_nm: Tuple[int], 
                 start_coords: Tuple[int], 
                 rng: random.Random = random.Random(42),
                 max_tries = 5,
                 ):
        assert len(sizes_nm) == 2
        assert len(start_coords) == 2
        self.sizes_nm = sizes_nm
        self.start_coords = start_coords
        self.rng = rng
        self.max_tries = max_tries

    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class RandomConnectedZones(LayoutGeneratorBase):
    def __call__(self, *args, zones_dict: Dict = dict(), **kwargs):
        n, m = self.sizes_nm
        x, y = self.start_coords
        for n_try in range(self.max_tries):
            mat = [[0] * m for _ in range(n)]
            if n_try > 0:
                log.error(f"Can't generate! Retry...[{n_try + 1}/{self.max_tries}]")

            is_gen, room = add_many_zones((x, y), mat, zones_dict, self.rng)
            if is_gen:
                break

        if not is_gen:
            log.error(f"Can't generate! Layout generation failed!")
            return None
        
        rotations = get_orientation((x, y), room)

        return {
            "darkstore": room,
            "rotations": rotations
        }
    
LAYOUT_TYPES_TO_CLS = {
    LayoutGenType.CONNECTED_ZONES: RandomConnectedZones,
    LayoutGenType.DEFAULT: None,

}