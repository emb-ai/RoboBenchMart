import json
import numpy as np
import sapien
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from transforms3d.euler import euler2quat
import json
import sapien
from transforms3d.euler import euler2quat

from mani_skill.utils.registration import register_env
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb


from dsynth.envs.fixtures.robocasaroom import _get_absolute_matrix, _get_pq
from dsynth.envs.darkstore_cont_base import DarkstoreContinuousBaseEnv
from dsynth.envs.fixtures.robocasaroom_cont import DarkstoreSceneContinuous
from dsynth.assets.ss_assets import WIDTH, DEPTH

class DarkstoreSceneContinuousWithRandomItemDeletion(DarkstoreSceneContinuous):
    def __init__(self, *args, skip_item_prob=0.4, config_dir_path=None, **kwargs):
        super().__init__(*args, config_dir_path=config_dir_path, **kwargs)
        self.skip_item_prob = skip_item_prob
        
    def load_arrangement_from_json(self, scene_idx, scene_data):
        active_fixtures_categories = ["active_shelvings",
            "active_wall_shelvings"
            ]
        inactive_fixtures_categories = [
            "inactive_shelvings",
            "inactive_wall_shelvings",
            "scene_fixtures"
            ]
        
        if len(scene_data['layout_data']["active_shelvings"]) > 0:
            # pick random suitable inactive shelf and replace it to active
            active_shelving = scene_data['layout_data']["active_shelvings"][0]

            inactive_shelvings = scene_data['layout_data']['inactive_shelvings']
            to_be_replaced_idxs = []
            for i, fixture in enumerate(inactive_shelvings):
                if fixture['asset_name'] == active_shelving['asset_name']:
                    to_be_replaced_idxs.append(i)
            if len(to_be_replaced_idxs) < 1:
                return RuntimeError("No suitable shelvings!")
            to_be_replaced_shelf_idx = self.env._batched_episode_rng[scene_idx].choice(to_be_replaced_idxs)
            active_shelf = inactive_shelvings.pop(to_be_replaced_shelf_idx)
            
            active_shelving['x'] = active_shelf['x']
            active_shelving['y'] = active_shelf['y']
            active_shelving['orientation'] = active_shelf['orientation']

            scene_data['layout_data']["inactive_shelvings"] = inactive_shelvings
            scene_data['layout_data']["active_shelvings"][0] = active_shelving



        fake_shelfs_mapping = scene_data['fake_arrangements_mapping']
        for fixture_category in inactive_fixtures_categories:
            for i, inactive_fixture in enumerate(scene_data['layout_data'][fixture_category]):
                shelf_id = inactive_fixture['asset_name']
                fixture_name = inactive_fixture['name']
                if shelf_id in fake_shelfs_mapping:
                    fake_shelves_id = fake_shelfs_mapping[shelf_id]
                    shelf_id = self.env._batched_episode_rng[scene_idx].choice(fake_shelves_id)


                item_name = f'[ENV#{scene_idx}]_inactive_{fixture_name}_{i}_{shelf_id}'
                p = np.array([inactive_fixture['x'], inactive_fixture['y'], 0.])
                angle = 0.
                if inactive_fixture['orientation'] == 'vertical':
                    angle = 3.14 / 2.
                q = euler2quat(0, 0, angle)
                pose = sapien.Pose(p=p, q=q)

                asset = self.env.assets_lib[shelf_id]
                asset.ms_is_nonconvex_collision = False
                actor = asset.ms_build_actor(item_name, self.env.scene, pose=pose, scene_idxs=[scene_idx])
                self.env.actors["fixtures"]["shelves"][item_name] = actor
        
        self.env.active_shelves[scene_idx] = []

        for fixture_category in active_fixtures_categories:
            for i, active_fixture in enumerate(scene_data['layout_data'][fixture_category]):
                shelf_id = active_fixture['asset_name']
                fixture_name = active_fixture['name']
                item_name = f'[ENV#{scene_idx}]_active_{fixture_name}_{i}'
                p = np.array([active_fixture['x'], active_fixture['y'], 0.])
                angle = 0.
                if active_fixture['orientation'] == 'vertical':
                    angle = 3.14 / 2.
                q = euler2quat(0, 0, angle)
                pose = sapien.Pose(p=p, q=q)

                # rotate shelf to center of the scene
                shelf_direction = pose.to_transformation_matrix()[:3, 1]
                direction_to_scene_center = np.array([self.x_size[scene_idx] / 2, self.y_size[scene_idx] / 2, 0.]) - pose.p
                direction_to_scene_center /= (np.linalg.norm(direction_to_scene_center) + 1e-3)
                if np.dot(direction_to_scene_center, shelf_direction) > 0:
                    pose = pose * sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, 3.14))

                asset = self.env.assets_lib[shelf_id]
                asset.ms_is_nonconvex_collision = True
                actor = asset.ms_build_actor(item_name, self.env.scene, pose=pose, scene_idxs=[scene_idx])
                self.env.actors["fixtures"]["shelves"][item_name] = actor
                self.env.active_shelves[scene_idx].append(item_name)

                with open(Path(self.config_dir_path) / f'{fixture_name}.json') as f:
                    shelf_arrangement = json.load(f)
                
                nodes_dict = {}
                for node in shelf_arrangement["graph"]:
                    nodes_dict[node[1]] = node

                for node in shelf_arrangement["graph"]:
                    parent_name, obj_name, props = node
                    if '/' not in obj_name and 'SHELF' not in obj_name:
                        abs_matrix = _get_absolute_matrix(node, nodes_dict)
                        p, q = _get_pq(abs_matrix, [0., 0., 0.])
                        prod_pose = pose * sapien.Pose(p=p, q=q)
                        asset_name = f'products_hierarchy.{obj_name.split(":")[0]}'
                        item_name = f'[ENV#{scene_idx}]_{obj_name}'
                        if self.env._batched_episode_rng[scene_idx].random() < self.skip_item_prob:
                            continue
                        actor = self.env.assets_lib[asset_name].ms_build_actor(item_name, self.env.scene, pose=prod_pose, scene_idxs=[scene_idx])
                        self.env.actors["products"][item_name] = actor
                


@register_env('RandomItemDeletionEnv', max_episode_steps=200000)
class RandomItemDeletionEnv(DarkstoreContinuousBaseEnv):
    def _load_scene(self, options: dict):
        BaseEnv._load_scene(self, options)
        self.is_rebuild = True
        
        self.target_sizes = np.array([0.3, 0.3, 0.3])
        self.build_markers()

        self.actors = {
            "fixtures": {
                "shelves" : {},
                "lamps": {},
                "scene_assets": {}
            },
            "products": {}
        }
        self.active_shelves = {}

        self.scene_builder = DarkstoreSceneContinuousWithRandomItemDeletion(self, skip_item_prob=0.2, config_dir_path=self.config_dir_path)
        self.scene_builder.build()

        actor_names = []
        scene_idxs = []   
        asset_names = []
        product_names = []
        board_idxs = []
        col_idxs = []
        row_idxs = []

        for actor_name, actor in self.actors["products"].items():
            actor_names.append(actor_name)

            assert len(actor._scene_idxs) == 1
            scene_idx = actor._scene_idxs.cpu().numpy()[0]
            scene_idxs.append(scene_idx)

            asset_name = actor_name.replace(f'[ENV#{scene_idx}]_', '')
            asset_name, shelf_idx, board_idx ,col_idx, row_idx = asset_name.split(':')
            asset_names.append(asset_name)
            product_names.append(self.assets_lib['products_hierarchy.' + asset_name].asset_name)

            board_idxs.append(board_idx)
            col_idxs.append(col_idx)
            row_idxs.append(row_idx)

        self.products_df = pd.DataFrame(dict(
                actor_name=actor_names,
                scene_idx=scene_idxs,
                product_name=product_names,
                asset_name=asset_names,
                board_idxs=board_idxs,
                col_idxs=col_idxs,
                row_idxs=row_idxs
            )
        )

        self.products_df.to_csv(self.config_dir_path / 'scene_items.csv')

        self.update_human_camera()

        print("built")
        print(f"Total {len(self.actors['products'])} products in {self.num_envs} scene(s)")
 