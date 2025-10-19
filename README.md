# Darkstore Synthesizer

## Quickstart

You can open [example notebook](notebooks/dsynth_scengen.ipynb) in [Google Colab](https://colab.research.google.com/) to test basic usage.

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
Also assets available via GDrive https://drive.google.com/file/d/1RhBw9HfoHm6uvxrFC9hxYq0JB1FRI_d-

## Sample Scene

Generate Simple scene

```bash
python scripts/generate_scene_continuous.py ds_continuous=small_scene
```

The default saving directory is `genereted_envs/`, however you can change it using `ds_continuous.output_dir=<YOUR_PATH>`.

Vizualize generated env using SAPIEN viewer:

```bash
python scripts/show_env_in_sim.py generated_envs/ds_small_scene/ --gui
```

## Teleoperation

You can use teleoperation for recording demonstration trajectories.

```bash
python scripts/run_teleop_fetch.py --scene-dir generated_envs/ds_small_scene/
```

## Training dataset generation

First, generate training scenes:

```bash
bash bash/generate_scenes.sh
```

Then run Motion Planning to collect trajectories in training environments:

```bash
bash bash/run_mp_all.sh
```

To convert data ro RLDS format please reffer to this [repo](https://github.com/emb-ai/DsynthAtomicTasks_rlds_builder).

## Evaluation

Gnereate test scenes:

```bash
bash bash/generate_test_scenes.sh
```

### Octo evaluation

### Pi0 evaluation

## Atomic PnP Tasks

Item train distribution

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

#### Train envs

Envs: `PickToBasketContNiveaEnv`, `PickToBasketContStarsEnv`, `PickToBasketContFantaEnv`

Scene configs: `conf/pick_to_basket_1`, `conf/pick_to_basket_2`.


### PickFromFloor


### MoveFromBoardToBoard

WIP


## Tutorials

* Custom layouts and scenes
* Motion Planning for Fetch robot for custom tasks
