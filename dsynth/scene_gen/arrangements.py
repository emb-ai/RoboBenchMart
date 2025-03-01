import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer.assets import TrimeshSceneAsset
from dsynth.assets.ss_assets import DefaultShelf
from dsynth.scene_gen.utils import PositionIteratorPI
from scene_synthesizer import utils
from shapely.geometry import Point
import trimesh.transformations as tra
import json
import sys
import argparse
import trimesh
import os

CELL_SIZE = 1.55
DEFAULT_ROOM_HEIGHT = 2.7

def get_assets_dict(assets_path):
    with open(f"{assets_path}/assets.json", "r") as f:
        assets_config = json.load(f)

    asset_type_mapping = {
        "MeshAsset": synth.assets.MeshAsset,
        "USDAsset": synth.assets.USDAsset,
    }

    assets_dict = {}

    for name, params in assets_config.items():
        asset_type_str = params.pop("asset_type")
        asset_constructor = asset_type_mapping.get(asset_type_str)
        if asset_constructor is None:
            raise ValueError(f"Unknown asset type: {asset_type_str}")

        file_path = os.path.join(assets_path, params.pop("filename"))

        asset_obj = asset_constructor(file_path, **params)

        assets_dict[name] = asset_obj

    return assets_dict



def set_shelf(
    scene, shelf, x: float, y: float, rotation: bool, name: str, support_name: str
):
    if not (rotation):
        scene.add_object(
            shelf,
            name,
            transform=np.dot(
                tra.translation_matrix((x, y, 0.0)),
                tra.rotation_matrix(np.radians(90), [0, 0, 1]),
            ),
        )
    else:
        scene.add_object(shelf, name, transform=tra.translation_matrix((x, y, 0.0)))
    support_data = scene.label_support(
        label=support_name,
        obj_ids=[name],
        min_area=0.05,
        gravity=np.array([0, 0, -1]),
    )
    return support_data

def add_objects_to_shelf(
    scene,
    product_names: dict,
    cnt_boards: int,
    product_on_board: list[list],
    suf: str,
    cnt_prod_on_board: int,
    support_data,
    is_pi: bool = False
):
    for num_board in range(cnt_boards):
        for elem in product_on_board[num_board]:
            scene.place_objects(
                obj_id_iterator=utils.object_id_generator(
                    f"{elem}_" + suf + f"_{num_board}_"
                ),
                obj_asset_iterator=tuple(product_names[elem] for _ in range(cnt_prod_on_board)),
                # obj_support_id_iterator=scene.support_generator(f'support{cnt}'),
                obj_support_id_iterator=utils.cycle_list(support_data, [num_board]),
                obj_position_iterator=PositionIteratorPI(step_x=1, step_y=1) if is_pi else utils.PositionIteratorGrid(
                    step_x=0.2,
                    step_y=0.1,
                    noise_std_x=0.01,
                    noise_std_y=0.01,
                    direction="x",
                ),
                obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
            )



def shelf_placement(
        product_names,
        num_boards,
        num_products_per_board,
        darkstore: list[list],
        is_rotate: list[list],
        random_shelfs: list[list[list]] = None,
        is_showed: bool = False,
        is_pi: bool = False):
    n, m = len(darkstore), len(darkstore[0])
    cells = []
    for i in range(n):
        for j in range(m):
            if darkstore[i][j]:
                cells.append(i * m + j)
    scene = synth.Scene()
    shelf = DefaultShelf
    cnt = 0
    it = 0
    for x in range(n):
        for y in range(m):
            if not darkstore[x][y]:
                cnt += 1
                continue
            support_data = set_shelf(
                scene,
                shelf,
                x * 1.55,
                y * 1.55,
                is_rotate[x][y],
                f"shelf{cnt}",
                f"support{cnt}",
            )
            add_objects_to_shelf(
                scene,
                product_names,
                num_boards,
                random_shelfs[it] if random_shelfs else [[darkstore[x][y]]] * num_boards,
                str(cnt),
                num_products_per_board,
                support_data,
                is_pi
            )
            cnt += 1
            it += 1

    if is_showed:
        scene.colorize()
        # scene.colorize(specific_objects={f"shelf{i}": [123, 123, 123] for i in cells})
        scene.show()
    json_str = synth.exchange.export.export_json(scene, include_metadata=False)

    data = json.loads(json_str)
    del data["geometry"]
    data["meta"] = {"n": n, "m": m}

    return data
    # with open(f"myscene_{n}_{m}.json", "w") as f:
    #     # f.write(json_str)
    #     json.dump(data, f, indent=4)


def one_shelf_placement_with(
        product_names,
        num_boards,
        num_products_per_board,
        products_on_boards: list[list]):
    scene = synth.Scene()
    shelf = DefaultShelf
    support_data = set_shelf(
        scene,
        shelf,
        0,
        0,
        False,
        f"shelf",
        f"support",
    )
    add_objects_to_shelf(
        scene,
        product_names,
        num_boards,
        products_on_boards,
        'try',
        num_products_per_board,
        support_data,
    )
    scene.colorize()
    scene.show()


def one_shelf_placement_with_diff_of_one_board(
    num_boards,
    set_of_products_on_each_boards: list[tuple],
    suf: str = 'diff'
    ):
    scene = synth.Scene()
    shelf = DefaultShelf
    support_data = set_shelf(
        scene,
        shelf,
        0,
        0,
        False,
        f"shelf",
        f"support",
    )
    for num_board in range(num_boards):
        scene.place_objects(
            obj_id_iterator=utils.object_id_generator(
                f"products_" + suf + f"_{num_board}_"
            ),
            obj_asset_iterator=set_of_products_on_each_boards[num_board],
            # obj_support_id_iterator=scene.support_generator(f'support{cnt}'),
            obj_support_id_iterator=utils.cycle_list(support_data, [num_board]),
            obj_position_iterator=utils.PositionIteratorGrid(
                step_x=0.2,
                step_y=0.1,
                noise_std_x=0.01,
                noise_std_y=0.01,
                direction="x",
            ),
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
        )
    scene.colorize()
    scene.show()




