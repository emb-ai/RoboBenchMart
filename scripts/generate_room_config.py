import sys
sys.path.append('.')
from dsynth.scene_gen.arrangements import get_assets_dict, shelf_placement
from dsynth.scene_gen.layouts.random_connectivity import add_many_products, get_orientation
import json
import sys
import argparse
import os
from pathlib import Path
from time import gmtime, strftime

# CONST
COUNT_OF_PRODUCT_ON_SHELF = 2
BOARDS = 5
COUNT_OF_PRODUCT_ON_BOARD = 1
WIDTH = 1.517
DEPTH = 0.5172


OUTPUT_PATH = 'generated_envs'
ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")


class UserError(Exception):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск генерации сцены")
    parser.add_argument(
        "--input",
        default="configs/input.json",
        help="Путь к JSON-файлу с входными данными (по умолчанию: models/input.json)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показать сцену после обработки"
    )
    parser.add_argument(
        "--pi",
        action="store_true",
        help="Показать сцену с определённой П-образной расстановкой предметов"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output path"
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Rewrite output dir"
    )
    parser.add_argument(
        "--assets_dir",
        default=ASSETS_PATH,
    )

    product_names = get_assets_dict(ASSETS_PATH)

    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=args.rewrite)
    else:
        output_dir = Path(OUTPUT_PATH) / 'env'

        i = 2
        while output_dir.exists() and not args.rewrite:
            output_dir = Path(OUTPUT_PATH) / f'env({i})'
            i += 1
        output_dir.mkdir(parents=True, exist_ok=args.rewrite)

    with open(args.input, "r") as f:
        data = json.load(f)

    n, m = data["room_size"]
    x, y = data["door_coords"]

    mat = data.get("blocked_matrix", [[0] * m for _ in range(n)])
    shelfname_to_cnt = data.get("shelfname_to_cnt", {'milk': 1, 'baby': 0, 'cereal': 0})
    n_products_on_shelf = data.get("n_products_on_shelf", COUNT_OF_PRODUCT_ON_BOARD)
    is_gen, room = add_many_products((x, y), mat, shelfname_to_cnt)
    is_rotate = get_orientation((x, y), room)

    if not is_gen:
        raise UserError("retry to generate a scene")

    scene_meta = shelf_placement(product_names, BOARDS, n_products_on_shelf, room, is_rotate, data['random_shelfs'], args.show, args.pi)

    print(f"Writing to {str(output_dir / 'scene_config.json')}...")
    
    with open(output_dir / "scene_config.json", "w") as f:
        json.dump(scene_meta, f, indent=4)

    print("DONE")

