#! /bin/bash

DATA_PATH=/home/jovyan/shares/SR006.nfs2/data/dsynth

python scripts/run_mp.py -e CloseDoorShowcaseContEnv --scene-dir \
$DATA_PATH/demo_envs/close_showcase --only-count-success --num-procs 4 --num-traj 248 \
--traj-name close_showcase_248traj_4workers  


