#! /bin/bash

DATA_PATH=/home/jovyan/shares/SR006.nfs2/data/dsynth

python scripts/run_mp.py -e PickFromFloorBeansContEnv --scene-dir \
$DATA_PATH/demo_envs/pick_from_floor --only-count-success --num-procs 4 --num-traj 248 \
--traj-name pick_from_floor_beans_248traj_4workers  


