# Assets for scene_synthesiser

import numpy as np
import trimesh.transformations as tra
import trimesh
from scene_synthesizer.assets import TrimeshSceneAsset



# CONST
BOARDS = 5
WIDTH = 1.517
DEPTH = 0.5172


class ShelfAsset(TrimeshSceneAsset):
    """A shelf asset."""

    def __init__(
        self,
        width,
        depth,
        height,
        num_boards,
        board_thickness=0.03,
        backboard_thickness=0.0,
        num_vertical_boards=0,
        num_side_columns=2,
        column_thickness=0.03,
        bottom_board=True,
        cylindrical_columns=True,
        shift_bottom=0.0,
        shift_top=0.0,
        **kwargs,
    ):
        boards = []
        board_names = []
        if backboard_thickness > 0:
            back = trimesh.primitives.Box(
                extents=[width, backboard_thickness, height],
                transform=tra.translation_matrix(
                    [0, depth / 2.0 + backboard_thickness / 2.0, height / 2.0]
                ),
            )
            boards.append(back)
            board_names.append("back")

        min_z = +float("inf")
        max_z = -float("inf")
        cnt = 0
        for h in np.linspace(
            shift_bottom + board_thickness / 2.0,
            height - board_thickness / 2.0 - shift_top,
            num_boards,
        ):
            if h == shift_bottom + board_thickness / 2.0 and not bottom_board:
                continue

            boards.append(
                trimesh.primitives.Box(
                    extents=[width, depth, board_thickness],
                    transform=tra.translation_matrix([0, 0, h]),
                )
            )
            board_names.append(f"board_{cnt}")
            cnt += 1

            min_z = min(min_z, h)
            max_z = max(max_z, h)

        cnt = 0
        for v in np.linspace(-width / 2.0, width / 2.0, num_vertical_boards + 2)[1:-1]:
            boards.append(
                trimesh.primitives.Box(
                    extents=[board_thickness, depth, max_z - min_z],
                    transform=tra.translation_matrix(
                        [v, 0, min_z + (max_z - min_z) / 2.0]
                    ),
                )
            )
            board_names.append(f"separator_{cnt}")
            cnt += 1

        int_num_side_columns = 1 if np.isinf(num_side_columns) else num_side_columns
        offset = depth / 2.0 if int_num_side_columns == 1 else 0.0
        for j in range(2):
            cnt = 0
            for c in np.linspace(-depth / 2.0, depth / 2.0, int_num_side_columns):
                if cylindrical_columns:
                    column = trimesh.primitives.Cylinder(
                        radius=column_thickness,
                        height=height,
                        transform=tra.translation_matrix(
                            [-width / 2.0 + j * width, c + offset, height / 2.0]
                        ),
                    )
                else:
                    column = trimesh.primitives.Box(
                        extents=[
                            column_thickness,
                            depth if np.isinf(num_side_columns) else column_thickness,
                            height,
                        ],
                        transform=tra.translation_matrix(
                            [-width / 2.0 + j * width, c + offset, height / 2.0]
                        ),
                    )
                boards.append(column)
                board_names.append(f"post_{j}_{cnt}")
                cnt += 1

        scene = trimesh.Scene()
        for mesh, name in zip(boards, board_names):
            scene.add_geometry(
                geometry=mesh,
                geom_name=name,
                node_name=name,
            )

        super().__init__(scene=scene, **kwargs)

DefaultShelf = ShelfAsset(
    width=WIDTH,
    depth=DEPTH,
    height=2.0,
    board_thickness=0.05135,
    num_boards=BOARDS,
    num_side_columns=2,
    bottom_board=True,
    cylindrical_columns=False,
    num_vertical_boards=0,
    shift_bottom=0.131952 - 0.05135 / 2,
    shift_top=0.2288 + 0.05135 / 2,
)

