#! /bin/bash

DATA_PATH=/home/jovyan/shares/SR006.nfs2/data/dsynth

python scripts/run_mp.py -e OpenDoorShowcaseContEnv --scene-dir \
demo_envs/open_showcase --only-count-success --num-procs 4 --num-traj 248 \
--traj-name open_showcase_248traj_4workers  

python scripts/run_mp.py -e OpenDoorFridgeContEnv --scene-dir \
demo_envs/open_fridge --only-count-success --num-procs 4 --num-traj 248 \
--traj-name open_fridge_248traj_4workers  

