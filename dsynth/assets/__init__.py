import os
from dataclasses import dataclass 
from typing import Any

ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")

def set_asset_path(new_asset_path):
    global ASSETS_PATH
    ASSETS_PATH = new_asset_path

@dataclass
class AssetMetainfo:
    asset_type: str
    filename: str
    scale: str
    origin: Any


# TODO: read this dict from somewhere
ALL_ASSETS = {
    'assets_path': ASSETS_PATH,
    'products_hierarchy':{
        'food': {
            'grocery':{
                'cereal': AssetMetainfo(
                    asset_type = 'MeshAsset',
                    filename = 'cereals.glb',
                    scale = 1,
                    origin = ['com', 'com', 'bottom']
                ),
                'baby': AssetMetainfo(
                    asset_type = 'USDAsset',
                    filename = 'baby.usdc',
                    scale = 1,
                    origin = ['com', 'com', 'bottom']
                ),
                'cerealsForSS': AssetMetainfo(
                    asset_type = 'MeshAsset',
                    filename = 'cereals.glb',
                    scale = 1,
                    origin = ['com', 'com', 'bottom']
                )
            },
            'dairy_products': {
                'milk': AssetMetainfo(
                    asset_type = 'MeshAsset',
                    filename = 'milk.glb',
                    scale = 1.1,
                    origin = ['com', 'com', 'bottom']
                ),
            },
            'drinks': {
                'CokeBottle': AssetMetainfo(
                    asset_type = 'USDAsset',
                    filename = 'CokeBottle.usdc',
                    scale = 0.9,
                    origin = ['com', 'com', 'bottom']
                ),
                'coke': AssetMetainfo(
                    asset_type = 'USDAsset',
                    filename = 'coke.usdc',
                    scale = 1,
                    origin = ['com', 'com', 'bottom']
                ),
            },
            'fruits': {
                'banana': AssetMetainfo(
                    asset_type = 'USDAsset',
                    filename = 'banana.usdc',
                    scale = 1.5,
                    origin = ['com', 'com', 'bottom']
                ),
            }
        },
    },
    'service':{
        'shelf': {
            'filename': 'shelf_woodenglb.glb'
        },
        'shoppingCart': {
            'filename': 'smallShoppingCart.glb'
        },
        'lamp':{
            'filename': 'lamp1.glb'
        }
    }

}

