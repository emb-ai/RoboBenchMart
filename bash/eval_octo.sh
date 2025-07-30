#! /bin/bash

NET_PARAMS="--host=localhost --port=8000"
EVAL_PARAMS="--max-horizon 500 --num-traj 30 --save-video"

# =========================================================
# move_from_board_to_board
# =========================================================

## item: duff
### train seeds
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_duff \
--json-path demo_envs/move_from_board_to_board_duff/demos/motionplanning/move_from_board_to_board_duff_250traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
--eval-subdir octo_jax_1000K_train $EVAL_PARAMS

### train seeds, but randomize robot init pose
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_duff \
--json-path demo_envs/move_from_board_to_board_duff/demos/motionplanning/move_from_board_to_board_duff_250traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
--robot-init-pose-start-seed 1000 --eval-subdir octo_jax_1000K_rand_init_pose $EVAL_PARAMS

### unseen scenes
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e MoveFromBoardToBoardDuffContEnv \
--scene-dir demo_envs/test_unseen_scenes_move_from_board_to_board_duff \
--eval-subdir octo_jax_1000K_unseen_scenes $EVAL_PARAMS



## item: nestle
### train seeds
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_nestle \
--json-path demo_envs/move_from_board_to_board_nestle/demos/motionplanning/move_from_board_to_board_nestle_250traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
--eval-subdir octo_jax_1000K_train $EVAL_PARAMS

### train seeds, but randomize robot init pose
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_nestle \
--json-path demo_envs/move_from_board_to_board_nestle/demos/motionplanning/move_from_board_to_board_nestle_250traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
--robot-init-pose-start-seed 1000 --eval-subdir octo_jax_1000K_rand_init_pose $EVAL_PARAMS

### unseen scenes
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e MoveFromBoardToBoardNestleContEnv \
--scene-dir demo_envs/test_unseen_scenes_move_from_board_to_board_nestle \
--eval-subdir octo_jax_1000K_unseen_scenes $EVAL_PARAMS


## item: vanish
### train seeds
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_vanish \
--json-path demo_envs/move_from_board_to_board_vanish/demos/motionplanning/move_from_board_to_board_vanish_250traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
--eval-subdir octo_jax_1000K_train $EVAL_PARAMS

### train seeds, but randomize robot init pose
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/move_from_board_to_board_vanish \
--json-path demo_envs/move_from_board_to_board_vanish/demos/motionplanning/move_from_board_to_board_vanish_250traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
--robot-init-pose-start-seed 1000 --eval-subdir octo_jax_1000K_rand_init_pose $EVAL_PARAMS

### unseen scenes
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e MoveFromBoardToBoardVanishContEnv \
--scene-dir demo_envs/test_unseen_scenes_move_from_board_to_board_vanish \
--eval-subdir octo_jax_1000K_unseen_scenes $EVAL_PARAMS



## OOD item: NiveaBodyMilk
python scripts/eval_policy_client.py $NET_PARAMS \
-e MoveFromBoardToBoardNiveaContEnv \
--scene-dir demo_envs/test_unseen_items_move_from_board_to_board_nivea \
--eval-subdir octo_jax_1000K_ood_items $EVAL_PARAMS



## OOD item: FantaSaborNaranja2L
python scripts/eval_policy_client.py $NET_PARAMS \
-e MoveFromBoardToBoardFantaContEnv \
--scene-dir demo_envs/test_unseen_items_move_from_board_to_board_fanta \
--eval-subdir octo_jax_1000K_ood_items $EVAL_PARAMS


# =========================================================
# pick to basket
# =========================================================

## item: nivea
### train seeds
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_nivea_250traj_4workers.rgb.pd_joint_pos.physx_cpu.json \
--eval-subdir octo_jax_1000K_train_nivea $EVAL_PARAMS

### train seeds, but randomize robot init pose
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_nivea_250traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
--robot-init-pose-start-seed 1000 --eval-subdir octo_jax_1000K_rand_init_pose_nivea $EVAL_PARAMS

### unseen scenes
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e PickToBasketContNiveaEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--eval-subdir octo_jax_1000K_unseen_scenes_nivea $EVAL_PARAMS


## item: fanta

MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_fanta_250traj_4workers.rgb.pd_joint_pos.physx_cpu.json \
--eval-subdir octo_jax_1000K_train_fanta $EVAL_PARAMS

### train seeds, but randomize robot init pose
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_fanta_250traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
--robot-init-pose-start-seed 1000 --eval-subdir octo_jax_1000K_rand_init_pose_fanta $EVAL_PARAMS

### unseen scenes
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e PickToBasketContFantaEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--eval-subdir octo_jax_1000K_unseen_scenes_fanta $EVAL_PARAMS



## item: stars

MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_stars_250traj_4workers.rgb.pd_joint_pos.physx_cpu.json \
--eval-subdir octo_jax_1000K_train_stars $EVAL_PARAMS

### train seeds, but randomize robot init pose
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_to_basket \
--json-path demo_envs/pick_to_basket/demos/motionplanning/pick_to_basket_stars_250traj_4workers.rgbd.pd_joint_pos.physx_cpu.json \
--robot-init-pose-start-seed 1000 --eval-subdir octo_jax_1000K_rand_init_pose_stars $EVAL_PARAMS

### unseen scenes
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e PickToBasketContStarsEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--eval-subdir octo_jax_1000K_unseen_scenes_stars $EVAL_PARAMS



## OOD item: NestleFitnessChocolateCereals
python scripts/eval_policy_client.py $NET_PARAMS \
-e PickToBasketContNestleEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--eval-subdir octo_jax_1000K_ood_items_nestle $EVAL_PARAMS



## OOD item: SlamLuncheonMeat
python scripts/eval_policy_client.py $NET_PARAMS \
-e PickToBasketContSlamEnv \
--scene-dir demo_envs/test_unseen_items_pick_to_basket \
--eval-subdir octo_jax_1000K_ood_items_slam $EVAL_PARAMS

# =========================================================
# pick from floor
# =========================================================

## item: beans

MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_from_floor/ \
--json-path demo_envs/pick_from_floor/demos/motionplanning/pick_from_floor_beans_250traj_4workers.rgb.pd_joint_pos.physx_cpu.json \
--eval-subdir octo_jax_1000K_train_beans $EVAL_PARAMS

### train seeds, but randomize robot init pose
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_from_floor/ \
--json-path demo_envs/pick_from_floor/demos/motionplanning/pick_from_floor_beans_250traj_4workers.rgb.pd_joint_pos.physx_cpu.json \
--robot-init-pose-start-seed 1000 --eval-subdir octo_jax_1000K_rand_init_pose_beans $EVAL_PARAMS

### unseen scenes
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e PickFromFloorBeansContEnv \
--scene-dir demo_envs/test_unseen_scenes_pick_from_floor \
--eval-subdir octo_jax_1000K_unseen_scenes_beans $EVAL_PARAMS

## item: slam

MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_from_floor/ \
--json-path demo_envs/pick_from_floor/demos/motionplanning/pick_from_floor_slam_250traj_4workers.rgb.pd_joint_pos.physx_cpu.json \
--eval-subdir octo_jax_1000K_train_slam $EVAL_PARAMS

### train seeds, but randomize robot init pose
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
--scene-dir demo_envs/pick_from_floor/ \
--json-path demo_envs/pick_from_floor/demos/motionplanning/pick_from_floor_slam_250traj_4workers.rgb.pd_joint_pos.physx_cpu.json \
--robot-init-pose-start-seed 1000 --eval-subdir octo_jax_1000K_rand_init_pose_slam $EVAL_PARAMS

### unseen scenes
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e PickFromFloorSlamContEnv \
--scene-dir demo_envs/test_unseen_scenes_pick_from_floor \
--eval-subdir octo_jax_1000K_unseen_scenes_slam $EVAL_PARAMS


## OOD item: FantaSaborNaranja2L
python scripts/eval_policy_client.py $NET_PARAMS \
-e PickFromFloorFantaContEnv \
--scene-dir demo_envs/test_unseen_items_pick_from_floor \
--eval-subdir octo_jax_1000K_ood_items_fanta $EVAL_PARAMS



## OOD item: DuffBeerCan
python scripts/eval_policy_client.py $NET_PARAMS \
-e PickFromFloorDuffContEnv \
--scene-dir demo_envs/test_unseen_items_pick_from_floor \
--eval-subdir octo_jax_1000K_ood_items_duff $EVAL_PARAMS


# =========================================================
# open fridge
# =========================================================

MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e OpenDoorFridgeContEnv \
--scene-dir demo_envs/open_fridge/ \
--eval-subdir octo_jax_1000K $EVAL_PARAMS

# =========================================================
# open showcase
# =========================================================

MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e OpenDoorShowcaseContEnv \
--scene-dir demo_envs/open_showcase/ \
--eval-subdir octo_jax_1000K $EVAL_PARAMS

# =========================================================
# close fridge
# =========================================================
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e CloseDoorFridgeContEnv \
--scene-dir demo_envs/close_fridge/ \
--eval-subdir octo_jax_1000K $EVAL_PARAMS

# =========================================================
# close showcase
# =========================================================
MS_ASSET_DIR=/mnt/disk2tb/maniskill/ python scripts/eval_policy_client.py $NET_PARAMS \
-e CloseDoorShowcaseContEnv \
--scene-dir demo_envs/close_showcase// \
--eval-subdir octo_jax_1000K $EVAL_PARAMS