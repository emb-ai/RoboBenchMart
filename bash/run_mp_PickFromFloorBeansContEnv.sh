#! /bin/bash

DATA_PATH=/home/jovyan/shares/SR006.nfs2/data/dsynth

python scripts/run_mp.py -e PickFromFloorBeansContEnv --scene-dir \
$DATA_PATH/demo_envs/pick_from_floor --only-count-success --num-procs 4 --num-traj 250 \
--traj-name pick_from_floor_beans_250traj_4workers  


