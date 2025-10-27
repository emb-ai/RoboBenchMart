# Darkstore Synthesizer

![teaser](docs/assets/teaser3.jpg)

## Installation

### Prerequisites 

ManiSkill simulator requires Vulkan API to be installed.

```bash
sudo apt-get install libvulkan1
```

To test your vulkan installation:
```bash
sudo apt install vulkan-tools
vulkaninfo
```

See more for [troubleshooting](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting).

### Installation from github

```bash
git clone https://gitlab.2a2i.org/cv/robo/darkstore-synthesizer
cd darkstore-synthesizer
conda create -n dsynth python=3.10
conda activate dsynth
pip install -r requirements.txt
pip install mplib==0.2.1
```

Test your ManiSkill installation:
```bash
python -m mani_skill.examples.demo_random_action
```

### Copy assets to directory

All assets are stored in `/home/jovyan/shares/SR006.nfs2/data/dsynth/assets`.
Copy or link this directory to `assets/` directory.
Also assets available via GDrive https://drive.google.com/file/d/1u3z320yyQ_Ad6BzcNSvLVJ7mneeLCZdx.

## Sample Scene

Generate Simple scene

```bash
python scripts/generate_scene_continuous.py ds_continuous=small_scene
```

The default saving directory is `generated_envs/`, however you can change it using `ds_continuous.output_dir=<YOUR_PATH>`.

Vizualize generated env using SAPIEN viewer:

```bash
python scripts/show_env_in_sim.py generated_envs/ds_small_scene/ --gui
```

## Teleoperation

You can use teleoperation for recording demonstration trajectories.

```bash
python scripts/run_teleop_fetch.py --scene-dir generated_envs/ds_small_scene/
```

## Tutorials

You can open [example notebook](notebooks/tutorial.ipynb) to find out more about scene generation, importing scenes to ManiSkill and motion planning. 

## Training dataset generation

### Training scenes generation

First, generate training scenes:

```bash
bash bash/generate_scenes.sh
```

### Collecting Demo Trajectories

Then run Motion Planning to collect raw .h5 trajectories without visual observations in training environments:

```bash
bash bash/run_mp_all.sh
```

Motion Planning generation is very time-consuming process.
We recommend to launch per-environment scripts `bash/run_mp_CloseDoorFridgeContEnv.sh`, `bash/run_mp_MoveFromBoardToBoardVanishContEnv.sh`, etc. in parallel to speed up the trajectory generation.

Next we have to replay all trajectories to write visual observations.

```bash
bash bash/replay.sh
```

To convert data to RLDS format please refer to this [repo](https://github.com/emb-ai/DsynthAtomicTasks_rlds_builder).

## Evaluation

Generate test scenes: 

```bash
bash bash/generate_test_scenes.sh
```

### Octo evaluation

Follow [original installation](https://github.com/octo-models/octo) instructions to set up environment with Octo.

Launch Octo server (in Octo environment):

```bash
python scripts/octo_server.py --finetuned-path <PATH_TO_OCTO_WEIGHTS>
```

Run evaluation script (in `dsynth` environment):

```bash
bash bash/eval_octo.py
```

### Pi0 evaluation

WIP

## Atomic PnP Tasks

Train/test item distribution

<table>
<tr>
<th>

</th>
<th>PickToBasket</th>
<th>MoveFromBoardToBoard</th>
<th>PickFromFloor</th>
</tr>
<tr>
<td>Train items</td>
<td>

* NiveaBodyMilk
* NestleHoneyStars
* FantaSaborNaranja2L
</td>
<td>

* NestleFitnessChocolateCereals
* DuffBeerCan
* VanishStainRemover
</td>
<td>

* HeinzBeansInARichTomatoSauce
* SlamLuncheonMeat
</td>
</tr>
<tr>
<td>OOD test items</td>
<td>

* NestleFitnessChocolateCereals
* SlamLuncheonMeat
</td>
<td>

* NiveaBodyMilk
* FantaSaborNaranja2L

</td>
<td>

* FantaSaborNaranja2L
* DuffBeerCan
</td>
</tr>
<tr>
<td>#layouts</td>
<td>20</td>
<td>10x3</td>
<td>10</td>
</tr>
<tr>
<td>#trajs</td>
<td>248x3</td>
<td>248x3</td>
<td>248x3</td>
</tr>
</table>

### PickToBasket

**Task Description:**
Approach the shelf and pick up any item with specified name, placing it into the basket attached to the Fetch robot.
The robot is spawned in close proximity to the shelf.

<details>
  <summary>Click to reveal</summary>

#### Train environments

Environments: `PickToBasketContNiveaEnv`, `PickToBasketContStarsEnv`, `PickToBasketContFantaEnv`.

Scene configs: `conf/pick_to_basket_1`, `conf/pick_to_basket_2`.

#### Test environments

Environments: `PickToBasketContNestleEnv`, `PickToBasketContSlamEnv`, `PickToBasketContDuffEnv`.

Scene configs: `conf/test_unseen_scenes_pick_to_basket_1`, `conf/test_unseen_scenes_pick_to_basket_2`,
`conf/test_unseen_items_pick_to_basket_1`, `conf/test_unseen_items_pick_to_basket_2`.

</details>

### PickFromFloor

**Task Description:**
Approach to the shelf, pick the fallen item and place it on the shelf.
The robot is spawned in close proximity to the shelf. The goal position for the fallen item is its original location on the shelf.

<details>
  <summary>Click to reveal</summary>

#### Train environments

Environments: `PickFromFloorBeansContEnv`, `PickFromFloorSlamContEnv`.

Scene configs: `conf/pick_from_floor_1`, `conf/pick_from_floor_2`.

#### Test environments

Environments: `PickFromFloorFantaContEnv`, `PickFromFloorDuffContEnv`.

Scene configs: `conf/test_unseen_scenes_pick_from_floor_1`, `conf/test_unseen_scenes_pick_from_floor_2`,
`conf/test_unseen_items_pick_from_floor_1`, `conf/test_unseen_items_pick_from_floor_2`.

</details>

### MoveFromBoardToBoard

**Task Description:**
Approach the shelf and pick up any item with the specified name, placing it one board higher (target board).
It is assumed that there is a free space on a target board.

<details>
  <summary>Click to reveal</summary>

#### Train environments

Environments: `MoveFromBoardToBoardVanishContEnv`, `MoveFromBoardToBoardNestleContEnv`, `MoveFromBoardToBoardDuffContEnv`.

Scene configs: `conf/move_from_board_to_board_nestle_1`, `conf/move_from_board_to_board_nestle_2`, `conf/move_from_board_to_board_vanish_1`, `conf/move_from_board_to_board_vanish_2`, `conf/move_from_board_to_board_duff_1`, `conf/move_from_board_to_board_duff_2`.

#### Test environments

Environments: `MoveFromBoardToBoardFantaContEnv`, `MoveFromBoardToBoardNiveaContEnv`.

Scene configs: `conf/test_unseen_scenes_move_from_board_to_board_duff_1`, `conf/test_unseen_scenes_move_from_board_to_board_duff_2`, `conf/test_unseen_scenes_move_from_board_to_board_nestle_1`, `conf/test_unseen_scenes_move_from_board_to_board_nestle_2`, `conf/test_unseen_scenes_move_from_board_to_board_vanish_1`, `conf/test_unseen_scenes_move_from_board_to_board_vanish_2`, `conf/test_unseen_items_move_from_board_to_board_nivea_1`, `conf/test_unseen_items_move_from_board_to_board_nivea_2`, `conf/test_unseen_items_move_from_board_to_board_fanta_1`, `conf/test_unseen_items_move_from_board_to_board_fanta_2`.

</details>

## Opening and Closing tasks

### OpenDoorShowcase

**Task Description:**
Approach the showcase and open the specified (`first`, `second`, `third`, `fourth`) door of the showcase.
The robot is spawned in close proximity to the showcase.

<details>
  <summary>Click to reveal</summary>

#### Train/Test environments

Environments: `OpenDoorShowcaseContEnv`.

Scene configs:. `conf/open_showcase`.

</details>

### CloseDoorShowcase

**Task Description:**
Approach the showcase and close the opened door of the showcase.
The robot is spawned in close proximity to the showcase.

<details>
  <summary>Click to reveal</summary>

#### Train/Test environments

Environments: `CloseDoorShowcaseContEnv`.

Scene configs:. `conf/close_showcase`.

</details>


### OpenDoorFridge

**Task Description:**
Approach the fridge and open the door.
The robot is spawned in close proximity to the fridge.

<details>
  <summary>Click to reveal</summary>

#### Train/Test environments

Environments: `OpenDoorFridgeContEnv`.

Scene configs:. `conf/open_fridge`.

</details>

### CloseDoorFridge

**Task Description:**
Approach the fridge and close the door.
The robot is spawned in close proximity to the fridge.

<details>
  <summary>Click to reveal</summary>

#### Train/Test environments

Environments: `CloseDoorFridgeContEnv`.

Scene configs:. `conf/close_fridge`.

</details>