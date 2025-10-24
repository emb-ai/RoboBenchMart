#! /bin/bash

NUM_TRAJ=248
NUM_PROCS=4
DATA_PATH='.'

python scripts/run_mp.py -e OpenDoorShowcaseContEnv --scene-dir \
$DATA_PATH/demo_envs/open_showcase --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name open_showcase_248traj_4workers  

python scripts/run_mp.py -e OpenDoorFridgeContEnv --scene-dir \
$DATA_PATH/demo_envs/open_fridge --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name open_fridge_248traj_4workers  

python scripts/run_mp.py -e CloseDoorFridgeContEnv --scene-dir \
$DATA_PATH/demo_envs/close_fridge --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name close_fridge_248traj_4workers 

python scripts/run_mp.py -e CloseDoorShowcaseContEnv --scene-dir \
$DATA_PATH/demo_envs/close_showcase --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name close_showcase_248traj_4workers  

python scripts/run_mp.py -e MoveFromBoardToBoardDuffContEnv --scene-dir \
$DATA_PATH/demo_envs/move_from_board_to_board_duff --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name move_from_board_to_board_duff_248traj_4workers  

python scripts/run_mp.py -e MoveFromBoardToBoardNestleContEnv --scene-dir \
$DATA_PATH/demo_envs/move_from_board_to_board_nestle --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name move_from_board_to_board_nestle_248traj_4workers  

python scripts/run_mp.py -e MoveFromBoardToBoardVanishContEnv --scene-dir \
$DATA_PATH/demo_envs/move_from_board_to_board_vanish --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name move_from_board_to_board_vanish_248traj_4workers  

python scripts/run_mp.py -e PickFromFloorBeansContEnv --scene-dir \
$DATA_PATH/demo_envs/pick_from_floor --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name pick_from_floor_beans_248traj_4workers  

python scripts/run_mp.py -e PickFromFloorSlamContEnv --scene-dir \
$DATA_PATH/demo_envs/pick_from_floor --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name pick_from_floor_slam_248traj_4workers  

python scripts/run_mp.py -e PickToBasketContFantaEnv --scene-dir \
$DATA_PATH/demo_envs/pick_to_basket --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name pick_to_basket_fanta_248traj_4workers  

python scripts/run_mp.py -e PickToBasketContNiveaEnv --scene-dir \
$DATA_PATH/demo_envs/pick_to_basket --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name pick_to_basket_nivea_248traj_4workers  

python scripts/run_mp.py -e PickToBasketContStarsEnv --scene-dir \
$DATA_PATH/demo_envs/pick_to_basket --only-count-success --num-procs $NUM_PROCS --num-traj $NUM_TRAJ \
--traj-name pick_to_basket_stars_248traj_4workers  


