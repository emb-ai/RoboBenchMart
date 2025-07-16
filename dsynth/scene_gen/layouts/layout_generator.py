import logging
log = logging.getLogger(__name__)
import random
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dsynth.scene_gen.layouts.random_connectivity import add_many_zones, get_orientation
from dsynth.scene_gen.hydra_configs import LayoutGenType, LayoutContGenType
from dsynth.scene_gen.utils import RectFixture, check_collisions
import dsynth.scene_gen.layouts.tensor_field as tfield
class LayoutGeneratorBase(ABC):
    def __init__(self, 
                 sizes_nm: Tuple[int], 
                 start_coords: Tuple[int] = (0, 0), 
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
    
class FixedLayout(LayoutGeneratorBase):
    def __call__(self, *args, zones_dict: Dict = dict(), darkstore_arrangement_cfg: Dict = dict(), **kwargs):
        assert 'layout' in darkstore_arrangement_cfg
        assert 'rotations' in darkstore_arrangement_cfg
        n, m = self.sizes_nm
        darkstore = [[0] * m for _ in range(n)]
        rotations = [[0] * m for _ in range(n)]
        darkstore_dict = OmegaConf.to_container(darkstore_arrangement_cfg['layout'], resolve = True)
        rotations_dict = OmegaConf.to_container(darkstore_arrangement_cfg['rotations'], resolve = True)
        assert n == len(darkstore_dict.keys())
        assert n == len(rotations_dict.keys())
        # convert to list of lists
        for i in darkstore_dict.keys():
            assert len(darkstore_dict[i]) == m
            assert len(rotations_dict[i]) == m
            darkstore[i] = darkstore_dict[i]
            rotations[i] = rotations_dict[i]

        return {
            "darkstore": darkstore,
            "rotations": rotations
        }
    
LAYOUT_TYPES_TO_CLS = {
    LayoutGenType.CONNECTED_ZONES: RandomConnectedZones,
    LayoutGenType.DEFAULT: None,
    LayoutGenType.FIXED_LAYOUT: FixedLayout,
}

class TensorFieldLayout(LayoutGeneratorBase):
    def __init__(self, 
                 sizes_xy: Tuple[int], 
                 product_assets_lib,
                 cfg,
                 name='tf_layout',
                 rng: random.Random = random.Random(42),
                 max_tries = 20,
                 
                 skip_object_prob: float = 0.0,
                 inactive_wall_shelvings_occupancy_width = 0.4,
                 inactive_shelvings_occupancy_width = 0.6,
                 inactive_shelvings_skip_prob = 0.0,
                 inactive_shelvings_passage_width = 1.5
                 ):
        assert len(sizes_xy) == 2
        self.product_assets_lib = product_assets_lib
        self.cfg = cfg
        self.size_x = sizes_xy[0]
        self.size_y = sizes_xy[1]
        self.rng = rng
        self.max_tries = max_tries
        self.skip_object_prob = skip_object_prob
        self.all_fixtures = []
        self.name = name

        self.inactive_wall_shelvings_occupancy_width = inactive_wall_shelvings_occupancy_width
        self.inactive_shelvings_occupancy_width = inactive_shelvings_occupancy_width
        self.inactive_shelvings_skip_prob = inactive_shelvings_skip_prob
        self.inactive_shelvings_passage_width = inactive_shelvings_passage_width

    def _all_fixtures_list(self):
        all_fixtures = []
        for key, val in self.all_fixtures.items():
            all_fixtures.extend(val)
        return all_fixtures

    def __call__(self, *args, **kwargs):
        self.all_fixtures = {
            "service": [],
            "scene_fixtures": [],
            "inactive_wall_shelvings": [],
            "active_wall_shelvings": [],
            "inactive_shelvings": [],
            "active_shelvings": []
        }

        self.all_fixtures['service'].append(
            RectFixture('blocked_area',
                        x=self.size_x,
                        y=self.size_y - 2,
                        l=2, w=2, )
        )

        self.place_fixtures()
        self.place_inactive_wall_shelvings()
        self.place_active_wall_shelvings()
        self.place_inactive_shelvings()
        self.place_active_shelvings()

        return self.rect_fixture2dict(self.all_fixtures)

    def rect_fixture2dict(self, rect_fixture_dict: list):
        return {key: [asdict(r) for r in rect_fixture_list] for key, rect_fixture_list in rect_fixture_dict.items()}
    
    def place_fixtures(self):
        scene_fixtures_list = self.cfg.ds_continuous.scene_fixtures_list
        for asset_name in scene_fixtures_list:
            rect = RectFixture.make_from_asset(self.product_assets_lib[asset_name], name=f'scene_fixture:{asset_name}',
                                               occupancy_width=0.0, 
                                        x=0., y=0., asset_name=asset_name)
            success = False
            for _ in range(self.max_tries):
                if rect.is_valid(self.size_x, self.size_y) and not check_collisions(rect, self._all_fixtures_list()):
                    self.all_fixtures['scene_fixtures'].append(rect)
                    success = True
                    break
                rect.x = self.rng.uniform(0.0, self.size_x)
                rect.y = self.rng.uniform(0.0, self.size_y)
            if not success:
                log.warning('Failed to place scene fixture')
    
    def place_inactive_wall_shelvings(self):
        inactive_wall_shelvings_list = self.cfg.ds_continuous.inactive_wall_shelvings_list
        half_perimeter = self.size_x + self.size_y
        perimeter_points = np.linspace(0, half_perimeter, 40)
        self.rng.shuffle(perimeter_points)
        for asset_name in inactive_wall_shelvings_list:
            rect = RectFixture.make_from_asset(self.product_assets_lib[asset_name], name=f'inactive_wall_shelving:{asset_name}',
                                               occupancy_width=self.inactive_wall_shelvings_occupancy_width, 
                                        x=0., y=0., asset_name=asset_name)
            success = False

            for point in perimeter_points:
                if point <= self.size_x:
                    x = point
                    y = self.size_y
                    rect.orientation = 'horizontal'
                elif self.size_x < point <= self.size_x + self.size_y:
                    x = 0
                    y = point - self.size_x
                    rect.orientation = 'vertical'
                else:
                    raise RuntimeError
                # elif self.size_x + self.size_y < point <= 2 * self.size_x + self.size_y:
                #     x = point - self.size_x - self.size_y
                #     y = self.size_y
                #     rect.orientation = 'horizontal'
                # else:
                #     x = 0
                #     y = point - 2 * self.size_x - self.size_y
                #     rect.orientation = 'vertical'
                
                if rect.orientation == 'horizontal':
                    if y == self.size_y:
                        y -= rect.w / 2 + rect.occupancy_width + 1e-2
                if rect.orientation == 'vertical':
                    if x == 0:
                        x += rect.w / 2 + rect.occupancy_width + 1e-2

                rect.x = x
                rect.y = y
                if rect.is_valid(self.size_x, self.size_y) and not check_collisions(rect, self._all_fixtures_list()):
                    success = True
                    self.all_fixtures['inactive_wall_shelvings'].append(rect)
                    break
                
            if not success:
                log.warning('Failed to place scene fixture')
    
    
    def place_active_wall_shelvings(self):
        active_wall_shelvings = self.cfg.ds_continuous.active_wall_shelvings_list
        return []
    
    def place_inactive_shelvings(self):
        res = []
        inactive_shelvings_list = self.cfg.ds_continuous.inactive_shelvings_list


        # asset_name = inactive_shelvings_list[0]
        sample_rects = []
        for asset_name in inactive_shelvings_list:
            rect = RectFixture.make_from_asset(self.product_assets_lib[asset_name], name=f'inactive_shelving:{asset_name}',
                                                occupancy_width=self.inactive_shelvings_occupancy_width, 
                                            x=0., y=0., asset_name=asset_name)
            sample_rects.append(rect)
        tf = tfield.TensorField(self.size_x + 1, self.size_y + 1) # TODO: redo
        tf.add_boundary()
        tf.add_fixture_list(self._all_fixtures_list())
        tf.calculate_field(decay=12.)

        res = tfield.place_shelves(tf,
                             sample_rects,
                             passage_width=self.inactive_shelvings_passage_width,
                             skip_shelf_prob=self.inactive_shelvings_skip_prob,
                             scene_fixtures=self._all_fixtures_list()
                             )
        self.all_fixtures['inactive_shelvings'] = res

        fig, ax = tf.vis_field()
        for fixture in self._all_fixtures_list():
            fixture.draw(ax[1], show_occupancy=False)
        fig.savefig(Path(self.cfg.ds_continuous.output_dir) / f'{self.name}_tf.jpg')

        return res
    
    def place_active_shelvings(self):
        res = []
        active_shelvings_list = self.cfg.ds_continuous.active_shelvings_list
        assert len(active_shelvings_list) <= 1
        active_fixture = active_shelvings_list[0]

        asset_name = active_fixture['shelf_asset']
        to_be_replaced_idxs = []
        for i, fixture in enumerate(self.all_fixtures['inactive_shelvings']):
            if fixture.asset_name == asset_name:
                to_be_replaced_idxs.append(i)

        to_be_replaced_shelf_idx = self.rng.choice(to_be_replaced_idxs)
        active_shelf = self.all_fixtures['inactive_shelvings'].pop(to_be_replaced_shelf_idx)
        active_shelf.name = active_fixture.name
        
        self.all_fixtures['active_shelvings'].append(active_shelf)



LAYOUT_CONTINUOUS_TO_CLS = {
    LayoutContGenType.PROCEDURAL_TENSOR_FIELD: TensorFieldLayout,
}
