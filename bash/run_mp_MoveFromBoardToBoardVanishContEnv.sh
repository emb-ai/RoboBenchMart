#! /bin/bash

DATA_PATH=/home/jovyan/shares/SR006.nfs2/data/dsynth

python scripts/run_mp.py -e MoveFromBoardToBoardVanishContEnv --scene-dir \
$DATA_PATH/demo_envs/move_from_board_to_board_vanish --only-count-success --num-procs 4 --num-traj 250 \
--traj-name move_from_board_to_board_vanish_250traj_4workers  


