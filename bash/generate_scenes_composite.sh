#! /bin/bash

WORKERS=2
DATA_PATH='.'

python scripts/generate_scene_continuous.py ds_continuous=composite_pick_to_basket ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/composite_pick_to_basket

python scripts/generate_scene_continuous.py ds_continuous=composite_pick_from_showcase ds_continuous.num_workers=$WORKERS \
assets=assets_downscaled \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/composite_pick_from_showcase