#! /bin/bash

NET_PARAMS="--host=localhost --port=8000"
EVAL_PARAMS="--max-horizon 1000 --num-traj 30 --save-video"
# ENV_PARAMS="MS_ASSET_DIR=/mnt/disk2tb/maniskill/"
ENV_PARAMS=""

# =========================================================
# pick to basket 2 items
# =========================================================

$ENV_PARAMS python scripts/eval_policy_composite_client.py $NET_PARAMS \
--env-id PickNiveaFantaEnv \
--scene-dir demo_envs/composite_pick_to_basket --eval-subdir octo_jax_1000K_nivea_fanta $EVAL_PARAMS

# =========================================================
# pick to basket 3 items
# =========================================================

$ENV_PARAMS python scripts/eval_policy_composite_client.py $NET_PARAMS \
--env-id PickNiveaFantaStarsEnv \
--scene-dir demo_envs/composite_pick_to_basket --eval-subdir octo_jax_1000K_nivea_fanta_stars $EVAL_PARAMS

# =========================================================
# open showcae, pick item, close showcase
# =========================================================

$ENV_PARAMS python scripts/eval_policy_composite_client.py $NET_PARAMS \
--env-id OpenPickDuffCloseEnv \
--scene-dir demo_envs/composite_pick_from_showcase --eval-subdir octo_jax_1000K_pick_from_showcase $EVAL_PARAMS


