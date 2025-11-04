#! /bin/bash

# Required: --model {octo|pi0|pi05}
MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required (octo|pi0|pi05)" >&2
  exit 1
fi

if [[ "$MODEL" != "octo" && "$MODEL" != "pi0" && "$MODEL" != "pi05" ]]; then
  echo "Error: --model must be one of: octo, pi0, pi05" >&2
  exit 1
fi

EVAL_SCRIPT="scripts/eval_policy_composite_client.py"

RUN_TS=$(date +%Y%m%d_%H%M%S)
SUBDIR_PREFIX="${MODEL}_"
SUBDIR_SUFFIX="_${RUN_TS}"

NUM_TRAJ=30

NET_PARAMS="--host=localhost --port=8000"
EVAL_PARAMS="--max-horizon 1000 --num-traj 30 --save-video"
# ENV_PARAMS="MS_ASSET_DIR=/mnt/disk2tb/maniskill/"
ENV_PARAMS=""

# =========================================================
# pick to basket 2 items
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--env-id PickNiveaFantaEnv \
--scene-dir demo_envs/composite_pick_to_basket \
--eval-subdir ${SUBDIR_PREFIX}composite_pick_to_basket_nivea_fanta${SUBDIR_SUFFIX} $EVAL_PARAMS

# =========================================================
# pick to basket 3 items
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--env-id PickNiveaFantaStarsEnv \
--scene-dir demo_envs/composite_pick_to_basket \
--eval-subdir ${SUBDIR_PREFIX}composite_pick_to_basket_fanta_stars${SUBDIR_SUFFIX} $EVAL_PARAMS

# =========================================================
# open showcae, pick item, close showcase
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--env-id OpenPickDuffCloseEnv \
--scene-dir demo_envs/composite_pick_from_showcase \
--eval-subdir ${SUBDIR_PREFIX}composite_pick_from_showcase${SUBDIR_SUFFIX} $EVAL_PARAMS
