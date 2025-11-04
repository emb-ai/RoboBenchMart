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

EVAL_SCRIPT="scripts/eval_policy_client.py"

RUN_TS=$(date +%Y%m%d_%H%M%S)
SUBDIR_PREFIX="${MODEL}_"
SUBDIR_SUFFIX="_${RUN_TS}"

NUM_TRAJ=30

NET_PARAMS="--host=localhost --port=8000"
EVAL_PARAMS_BOARD="--max-horizon 750 --num-traj ${NUM_TRAJ} --save-video"
EVAL_PARAMS_FRIDGE="--max-horizon 500 --num-traj ${NUM_TRAJ} --save-video"
EVAL_PARAMS_SHOWC="--max-horizon 1000 --num-traj ${NUM_TRAJ} --save-video"
EVAL_PARAMS_FLOOR="--max-horizon 750 --num-traj ${NUM_TRAJ} --save-video"
EVAL_PARAMS_BASKET="--max-horizon 600 --num-traj ${NUM_TRAJ} --save-video"

#TRAIN
# =========================================================
# move_from_board_to_board
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_duff \
--json-path demo_envs/move_from_board_to_board_duff/demos/motionplanning/move_from_board_to_board_duff_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}board_duff_train${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_nestle \
--json-path demo_envs/move_from_board_to_board_nestle/demos/motionplanning/move_from_board_to_board_nestle_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}board_nestle_train${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_vanish \
--json-path demo_envs/move_from_board_to_board_vanish/demos/motionplanning/move_from_board_to_board_vanish_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}board_vanish_train${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD
# =========================================================
# open fridge
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/open_fridge/ \
--json-path demo_envs/open_fridge/demos/motionplanning/open_fridge_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}open_fridge_train${SUBDIR_SUFFIX} $EVAL_PARAMS_FRIDGE
# =========================================================
# close fridge
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/close_fridge/ \
--json-path demo_envs/close_fridge/demos/motionplanning/close_fridge_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}close_fridge_train${SUBDIR_SUFFIX} $EVAL_PARAMS_FRIDGE
# =========================================================
# open showcase
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/open_showcase/ \
--json-path demo_envs/open_showcase/demos/motionplanning/open_showcase_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}open_showcase_train${SUBDIR_SUFFIX} $EVAL_PARAMS_SHOWC
# =========================================================
# close showcase
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/close_showcase/ \
--json-path demo_envs/close_showcase/demos/motionplanning/close_showcase_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}close_showcase_train${SUBDIR_SUFFIX} $EVAL_PARAMS_SHOWC
# =========================================================
# pick from floor
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_from_floor/ \
--json-path demo_envs/pick_from_floor/demos/motionplanning/pick_from_floor_beans_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}floor_beans_train${SUBDIR_SUFFIX} $EVAL_PARAMS_FLOOR
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_from_floor/ \
--json-path demo_envs/pick_from_floor/demos/motionplanning/pick_from_floor_slam_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}floor_slam_train${SUBDIR_SUFFIX} $EVAL_PARAMS_FLOOR
# =========================================================
# pick to basket
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_fanta_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}basket_fanta_train${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_nivea_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}basket_nivea_train${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_stars_250traj_4workers.json \
--eval-subdir ${SUBDIR_PREFIX}basket_stars_train${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET

#ROBO (train seeds, but randomize robot init pose)
# =========================================================
# move_from_board_to_board
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_duff \
--json-path demo_envs/move_from_board_to_board_duff/demos/motionplanning/move_from_board_to_board_duff_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}board_duff_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_nestle \
--json-path demo_envs/move_from_board_to_board_nestle/demos/motionplanning/move_from_board_to_board_nestle_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}board_nestle_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_vanish \
--json-path demo_envs/move_from_board_to_board_vanish/demos/motionplanning/move_from_board_to_board_vanish_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}board_vanish_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD
# =========================================================
# open fridge
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/open_fridge/ \
--json-path demo_envs/open_fridge/demos/motionplanning/open_fridge_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}open_fridge_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_FRIDGE
# =========================================================
# close fridge
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/close_fridge/ \
--json-path demo_envs/close_fridge/demos/motionplanning/close_fridge_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}close_fridge_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_FRIDGE
# =========================================================
# open showcase
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/open_showcase/ \
--json-path demo_envs/open_showcase/demos/motionplanning/open_showcase_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}open_showcase_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_SHOWC
# =========================================================
# close showcase
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/close_showcase/ \
--json-path demo_envs/close_showcase/demos/motionplanning/close_showcase_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}close_showcase_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_SHOWC
# =========================================================
# pick from floor
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_from_floor/ \
--json-path demo_envs/pick_from_floor/demos/motionplanning/pick_from_floor_beans_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}floor_beans_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_FLOOR
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_from_floor/ \
--json-path demo_envs/pick_from_floor/demos/motionplanning/pick_from_floor_slam_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}floor_slam_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_FLOOR
# =========================================================
# pick to basket
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_fanta_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}basket_fanta_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_nivea_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}basket_nivea_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET
python "$EVAL_SCRIPT" $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_stars_250traj_4workers.json \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}basket_stars_robo${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET




#UNSEEN
# =========================================================
# move_from_board_to_board
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e MoveFromBoardToBoardDuffContEnv \
--scene-dir demo_envs/test_unseen_scenes_move_from_board_to_board_duff \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}board_duff_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD --start-seed 42000
python "$EVAL_SCRIPT" $NET_PARAMS \
-e MoveFromBoardToBoardNestleContEnv \
--scene-dir demo_envs/test_unseen_scenes_move_from_board_to_board_nestle \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}board_nestle_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD --start-seed 42000
python "$EVAL_SCRIPT" $NET_PARAMS \
-e MoveFromBoardToBoardVanishContEnv \
--scene-dir demo_envs/test_unseen_scenes_move_from_board_to_board_vanish \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}board_vanish_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD --start-seed 42000
# =========================================================
# open fridge
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e OpenDoorFridgeContEnv \
--scene-dir demo_envs/open_fridge/ \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}open_fridge_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_FRIDGE --start-seed 42000
# =========================================================
# close fridge
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e CloseDoorFridgeContEnv \
--scene-dir demo_envs/close_fridge/ \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}close_fridge_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_FRIDGE --start-seed 42000
# =========================================================
# open showcase
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e OpenDoorShowcaseContEnv \
--scene-dir demo_envs/open_showcase/ \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}open_showcase_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_SHOWC --start-seed 42000
# =========================================================
# close showcase
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e CloseDoorShowcaseContEnv \
--scene-dir demo_envs/close_showcase// \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}close_showcase_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_SHOWC --start-seed 42000
# =========================================================
# pick from floor
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e PickFromFloorBeansContEnv \
--scene-dir demo_envs/test_unseen_scenes_pick_from_floor \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}floor_beans_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_FLOOR --start-seed 42000
python "$EVAL_SCRIPT" $NET_PARAMS \
-e PickFromFloorSlamContEnv \
--scene-dir demo_envs/test_unseen_scenes_pick_from_floor \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}floor_slam_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_FLOOR --start-seed 42000
# =========================================================
# pick to basket
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e PickToBasketContFantaEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}basket_fanta_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET --start-seed 42000
python "$EVAL_SCRIPT" $NET_PARAMS \
-e PickToBasketContNiveaEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}basket_nivea_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET --start-seed 42000
python "$EVAL_SCRIPT" $NET_PARAMS \
-e PickToBasketContStarsEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}basket_stars_uns${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET --start-seed 42000




# OOD
# =========================================================
# move_from_board_to_board
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e MoveFromBoardToBoardNiveaContEnv \
--scene-dir demo_envs/test_unseen_items_move_from_board_to_board_nivea \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}board_nivea_ood${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD --start-seed 42000
python "$EVAL_SCRIPT" $NET_PARAMS \
-e MoveFromBoardToBoardFantaContEnv \
--scene-dir demo_envs/test_unseen_items_move_from_board_to_board_fanta \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}board_fanta_ood${SUBDIR_SUFFIX} $EVAL_PARAMS_BOARD --start-seed 42000
# =========================================================
# pick from floor
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e PickFromFloorFantaContEnv \
--scene-dir demo_envs/test_unseen_items_pick_from_floor \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}floor_fanta_ood${SUBDIR_SUFFIX} $EVAL_PARAMS_FLOOR --start-seed 42000
python "$EVAL_SCRIPT" $NET_PARAMS \
-e PickFromFloorDuffContEnv \
--scene-dir demo_envs/test_unseen_items_pick_from_floor \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}floor_duff_ood${SUBDIR_SUFFIX} $EVAL_PARAMS_FLOOR --start-seed 42000
# =========================================================
# pick to basket
# =========================================================
python "$EVAL_SCRIPT" $NET_PARAMS \
-e PickToBasketContNestleEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}basket_nestle_ood${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET --start-seed 42000
python "$EVAL_SCRIPT" $NET_PARAMS \
-e PickToBasketContSlamEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--robot-init-pose-start-seed 10000 --eval-subdir ${SUBDIR_PREFIX}basket_slam_ood${SUBDIR_SUFFIX} $EVAL_PARAMS_BASKET --start-seed 42000


































































