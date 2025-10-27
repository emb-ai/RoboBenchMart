#! /bin/bash

WORKERS=2
DATA_PATH='.'

python scripts/generate_scene_continuous.py ds_continuous=open_showcase ds_continuous.num_workers=$WORKERS \
assets=assets_downscaled \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_showcase

python scripts/generate_scene_continuous.py ds_continuous=open_fridge ds_continuous.num_workers=$WORKERS \
assets=assets_downscaled \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/open_fridge

python scripts/generate_scene_continuous.py ds_continuous=pick_to_basket_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/pick_to_basket

python scripts/generate_scene_continuous.py ds_continuous=pick_to_basket_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/pick_to_basket

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_nestle_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/move_from_board_to_board_nestle

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_nestle_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/move_from_board_to_board_nestle

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_vanish_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/move_from_board_to_board_vanish

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_vanish_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/move_from_board_to_board_vanish

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_duff_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/move_from_board_to_board_duff

python scripts/generate_scene_continuous.py ds_continuous=move_from_board_to_board_duff_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/move_from_board_to_board_duff

python scripts/generate_scene_continuous.py ds_continuous=pick_from_floor_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/pick_from_floor

python scripts/generate_scene_continuous.py ds_continuous=pick_from_floor_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/pick_from_floor


python scripts/generate_scene_continuous.py ds_continuous=pick_from_floor_1 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/pick_from_floor

python scripts/generate_scene_continuous.py ds_continuous=pick_from_floor_2 ds_continuous.num_workers=$WORKERS \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/pick_from_floor

python scripts/generate_scene_continuous.py ds_continuous=close_showcase ds_continuous.num_workers=$WORKERS \
assets=assets_downscaled \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/close_showcase

python scripts/generate_scene_continuous.py ds_continuous=close_fridge ds_continuous.num_workers=$WORKERS \
assets=assets_downscaled \
assets.assets_dir_path=$DATA_PATH/assets \
ds_continuous.output_dir=$DATA_PATH/demo_envs/close_fridge
