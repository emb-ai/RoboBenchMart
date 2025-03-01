#!/bin/bash

FLAGS="--rewrite" # Add --show for visualization

python scripts/generate_room_config.py --input configs/many_objects.json --output_dir generated_envs/many_objects $FLAGS
python scripts/generate_room_config.py --input configs/pi_arrangement.json --output_dir generated_envs/pi_arrangement --pi $FLAGS
python scripts/generate_room_config.py --input configs/random_shelf.json --output_dir generated_envs/random_shelf $FLAGS
python scripts/generate_room_config.py --input configs/small_room.json --output_dir generated_envs/small_room $FLAGS


SIM_FLAGS="--shader rt-fast" # use --shader default to disable shader; use --gui to open GUI
python scripts/show_env_in_sim.py generated_envs/many_objects $SIM_FLAGS
python scripts/show_env_in_sim.py generated_envs/pi_arrangement $SIM_FLAGS
python scripts/show_env_in_sim.py generated_envs/random_shelf $SIM_FLAGS
python scripts/show_env_in_sim.py generated_envs/small_room $SIM_FLAGS
