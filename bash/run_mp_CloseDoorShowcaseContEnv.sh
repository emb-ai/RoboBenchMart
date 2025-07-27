#! /bin/bash

DATA_PATH=/home/jovyan/shares/SR006.nfs2/data/dsynth

python scripts/run_mp.py -e CloseDoorShowcaseContEnv --scene-dir \
$DATA_PATH/demo_envs/close_showcase --only-count-success --num-procs 4 --num-traj 250 \
--traj-name close_showcase_250traj_4workers  


