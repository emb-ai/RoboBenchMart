#! /bin/bash

DATA_PATH=/home/jovyan/shares/SR006.nfs2/data/dsynth

python scripts/run_mp.py -e CloseDoorFridgeContEnv --scene-dir \
$DATA_PATH/demo_envs/close_fridge --only-count-success --num-procs 4 --num-traj 248 \
--traj-name close_fridge_248traj_4workers  


