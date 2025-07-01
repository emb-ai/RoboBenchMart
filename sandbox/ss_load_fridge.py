import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer import utils
import trimesh.transformations as tra


def try1():
    s = synth.Scene()

    can = synth.assets.MeshAsset(
        'assets/fantaNaranja1p35l.glb',
        scale=0.05,
        origin=("bottom", "bottom", "bottom"),
        # transform=np.array([
        #     [1, 0, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 0, 1]
        # ]),

    )

    # fridge = synth.assets.URDFAsset(
    #     'assets/fridge_urdf/nestle.urdf',
    #     scale=0.066,
    # )
    fridge = synth.assets.MeshAsset(
        'assets/fridge_urdf/body.obj',
        scale=0.066,
    )

    s.add_object(
        fridge,
        'fridge',
    )

    s.label_support(
        label="support",
        min_area=0.05,
        gravity=np.array([0, 0, -1]),
    )
    s.show_supports()


    # print(s.is_articulated())
    # print(s.get_joint_names())
    # s.random_configurations()
    # s.colorize()

def try2():
    s = synth.Scene()

    can = synth.assets.MeshAsset(
        'assets/fantaNaranja1p35l.glb',
        scale=0.05,
        origin=("bottom", "bottom", "bottom"),
        # transform=np.array([
        #     [1, 0, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 0, 1]
        # ]),

    )

    fridge = synth.assets.URDFAsset(
        'assets/food_showcase/urdf/food_showcase.urdf',
    )

    s.add_object(
        fridge,
        'fridge',
    )
    s.random_configurations()
    s.show()
    s.label_support(
        label="support",
        min_area=0.05,
        gravity=np.array([0, 0, -1]),
    )
    s.show_supports()

def try3():
    s = synth.Scene()

    can = synth.assets.MeshAsset(
        'assets/fantaNaranja1p35l.glb',
        scale=0.05,
        origin=("bottom", "bottom", "bottom"),
        # transform=np.array([
        #     [1, 0, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 0, 1]
        # ]),

    )

    shelf = synth.assets.MeshAsset(
        'assets/store_mini_shelf_one_sided2.glb',
        origin=("center", "bottom", "center"),
        up=(0, 1, 0),
        front=(0, 0, -1),
        # scale=10,
    )

    s.add_object(
        shelf,
        'shelf',
    )
    # s.random_configurations()
    # s.show()
    support_data = s.label_support(
        label="support",
        min_area=0.0,
        gravity_tolerance=0.5,
        erosion_distance=0.01,
        gravity=np.array([0, 0, -1]),
    )
    s.show_supports()


try2()