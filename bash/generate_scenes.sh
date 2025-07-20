#! /bin/bash

WORKERS=2
DATA_PATH=/home/jovyan/shares/SR006.nfs2/data/dsynth

python scripts/generate_scene_continuous.py ds_continuous=pick_to_basket_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=pick_to_basket_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_nestle_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_nestle_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_vanish_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_vanish_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_duff_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_duff_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=open_showcase ds_continuous.num_workers=$WORKERS \
assets=assets_downscaled \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=open_fridge ds_continuous.num_workers=$WORKERS \
assets=assets_downscaled \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge
