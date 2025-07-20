#! /bin/bash

DATA_PATH=/home/jovyan/shares/SR006.nfs2/data/dsynth

python scripts/run_mp.py -e PickToBasketContFantaEnv --scene-dir \
$DATA_PATH/demo_envs/pick_to_basket --only-count-success --num-procs 4 --num-traj 250 \
--traj-name pick_to_basket_fanta_250traj_4workers  


