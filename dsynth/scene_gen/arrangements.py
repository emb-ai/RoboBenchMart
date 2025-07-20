import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer.assets import TrimeshSceneAsset
from dsynth.assets.ss_assets import DefaultShelf
from dsynth.scene_gen.utils import PositionIteratorPI, PositionIteratorGridColumns, object_id_generator
from scene_synthesizer import utils
from shapely.geometry import Point
import trimesh.transformations as tra
import dataclasses
import json
import sys
import argparse
import trimesh
import os
from dsynth.scene_gen.hydra_configs import FillingType

CELL_SIZE = 1.55
DEFAULT_ROOM_HEIGHT = 2.7


def set_shelf(
    scene, shelf, x: float, y: float, rotation_angle: bool, name: str, support_name: str
):
    scene.add_object(
            shelf,
            name,
            transform=np.dot(
                tra.translation_matrix((x, y, 0.0)),
                tra.rotation_matrix(np.radians(rotation_angle), [0, 0, 1]),
            ),
        )
    support_data = scene.label_support(
        erosion_distance=0.05,
        label=support_name,
        obj_ids=[name],
        min_area=0.05,
        gravity=np.array([0, 0, -1]),
    )
    
    support_data = sorted(support_data, key=lambda x: x.transform[2, 3])

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

def add_objects_to_shelf_v2(
    scene,
    shelf_cnt,
    product_placement: dict,
    product_assets_lib,
    support_data,
    x_gap,
    y_gap,
    delta_x,
    delta_y,
    start_point_x,
    start_point_y,
    filling_type,
    seed,
    noise_std_x=0.,
    noise_std_y=0.,
    rotation_lower=0.,
    rotation_upper=0.,
):
    if filling_type == FillingType.BOARDWISE_COLUMNS:
        for board_idx, board_arrangement in enumerate(product_placement):
            start_point = np.array([start_point_x, start_point_y])
            board_arrangement = [(arr_value.split(':')[0], int(arr_value.split(':')[1])) for arr_value in board_arrangement]
            for product, num_col in board_arrangement:
                obj = product_assets_lib['products_hierarchy.' + product].ss_asset
                dims = obj.get_extents()

                bounds = support_data[0].polygon.bounds
                max_obj = int(np.ceil((bounds[3] - bounds[1]) / min(dims[0], dims[1])) * num_col) #upperbound on how many objects can fit
                obj_position_iterator = PositionIteratorGridColumns(obj_width=dims[0], 
                                                                      obj_depth=dims[1], 
                                                                      x_gap=x_gap, 
                                                                      y_gap=y_gap, 
                                                                      delta_x=delta_x,
                                                                      delta_y=delta_y,
                                                                      current_point=start_point, 
                                                                      num_cols = num_col,
                                                                      seed=seed,
                                                                      noise_std_x=noise_std_x,
                                                                      noise_std_y=noise_std_y)
                scene.place_objects(
                    obj_id_iterator=object_id_generator(f"{product}:" + f"{shelf_cnt}:{board_idx}:", obj_position_iterator),
                    obj_asset_iterator=(obj for _ in range(max_obj)), 
                    obj_support_id_iterator=utils.cycle_list(support_data, [board_idx]),
                    obj_position_iterator=obj_position_iterator,
                    obj_orientation_iterator=utils.orientation_generator_uniform_around_z(rotation_lower, rotation_upper, seed=seed),
                )
    else:
        for num_board, board in enumerate(product_placement):
            pos_iter = utils.PositionIteratorGrid(
                        step_x=delta_x,
                        step_y=delta_y,
                        noise_std_x=0.001,
                        noise_std_y=0.001,
                        direction="y",
                    )
            for cnt, elem_name in enumerate(board):
                scene.place_objects(
                    obj_id_iterator=utils.object_id_generator(
                        f"{elem_name}:" + f"{shelf_cnt}:{num_board}:{cnt}:"
                    ),
                    obj_asset_iterator=tuple([product_assets_lib[elem_name].ss_asset]),
                    # obj_support_id_iterator=scene.support_generator(f'support{cnt}'),
                    obj_support_id_iterator=utils.cycle_list(support_data, [num_board]),
                    obj_position_iterator=pos_iter,
                    obj_orientation_iterator=utils.orientation_generator_uniform_around_z(0, upper= 3.14 / 20),
                )


def shelf_placement_v2(
        product_filling_flattened,
        darkstore: list[list],
        rotations: list[list],
        product_assets_lib,
        darkstore_cfg,
        is_showed: bool = False,
    ):
    n, m = len(darkstore), len(darkstore[0])
    cells = []
    for i in range(n):
        for j in range(m):
            if darkstore[i][j] != 0:
                cells.append(i * m + j)
    scene = synth.Scene()
    cnt = 0
    it = 0
    for x in range(n):
        for y in range(m):
            shelf_name = darkstore[x][y]
            if shelf_name == 0:
                cnt += 1
                continue
            zone_id, shelf_id = shelf_name.split('.')
            shelf_asset_name = darkstore_cfg.zones[zone_id][shelf_id].shelf_asset
            if shelf_asset_name is None:
                shelf = DefaultShelf
                shelf_asset_name = 'fixtures.shelf' # corresponds to the default shelf
            else:
                shelf = product_assets_lib[shelf_asset_name].ss_asset
            support_data = set_shelf(
                scene,
                shelf,
                x * 1.55,
                y * 1.55,
                rotations[x][y],
                f'SHELF_{cnt}_{shelf_name}',
                f'support_SHELF_{cnt}_{shelf_name}',
            )
            z_name = shelf_name.split(".")[0]
            s_name = shelf_name.split(".")[1]
            add_objects_to_shelf_v2(
                scene,
                cnt,
                product_filling_flattened[shelf_name],
                product_assets_lib,
                support_data,
                darkstore_cfg.zones[z_name][s_name].x_gap,
                darkstore_cfg.zones[z_name][s_name].y_gap,
                darkstore_cfg.zones[z_name][s_name].delta_x,
                darkstore_cfg.zones[z_name][s_name].delta_y,
                darkstore_cfg.zones[z_name][s_name].start_point_x,
                darkstore_cfg.zones[z_name][s_name].start_point_y,
                darkstore_cfg.zones[z_name][s_name].filling_type
            )
            cnt += 1
            it += 1

    if is_showed:
        # scene.colorize()
        # scene.colorize(specific_objects={f"shelf{i}": [123, 123, 123] for i in cells})
        scene.show()
    json_str = synth.exchange.export.export_json(scene, include_metadata=False)

    data = json.loads(json_str)
    del data["geometry"]
    if type(product_filling_flattened) == list:
        data["meta"] = {"n": n, "m": m, "room": darkstore, "rotations": rotations, "filling": product_filling_flattened}
    else:
        data["meta"] = {"n": n, "m": m, "room": darkstore, "rotations": rotations}
    return data

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




