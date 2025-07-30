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

Vizualize generated env in SAPIEN viewer:

```bash
python scripts/show_env_in_sim.py generated_envs/ds_small_scene/ --gui
```

## Teleoperation

You can use teleoperation for recording demonstration trajectories.

```bash
python scripts/run_teleop_fetch.py --scene-dir generated_envs/ds_small_scene/
```

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

Scene generation

```shellscript
python scripts/generate_scene_continuous.py ds_continuous=pick_to_basket_1 

python scripts/generate_scene_continuous.py ds_continuous=pick_to_basket_2 
```

Motion Planning

```shellscript
python scripts/run_mp.py -e PickToBasketContNiveaEnv --scene-dir demo_envs/pick_to_basket --only-count-success --num-procs 4 --num-traj 248 --traj-name pick_to_basket_nivea_248traj_4workers
```

```shellscript
python scripts/run_mp.py -e PickToBasketContStarsEnv --scene-dir demo_envs/pick_to_basket --only-count-success --num-procs 4 --num-traj 248 --traj-name pick_to_basket_stars_248traj_4workers  
```

```shellscript
python scripts/run_mp.py -e PickToBasketContFantaEnv --scene-dir demo_envs/pick_to_basket --only-count-success --num-procs 4 --num-traj 248 --traj-name pick_to_basket_fanta_248traj_4workers
```



### PickFromFloor

WIP

### MoveFromBoardToBoard

WIP


## Scene generation and environment visualization

### Scene generation

To generate layout and object arrangement use input configurations from `conf/` or make your own:

```bash
python scripts/generate_scene.py ds=config
```

Add `ds.show=true` flag to visualize generated arrangement.


## Visualize generated scene in ManiSkill

To show generated scene in ManiSkill:
```bash
python scripts/show_env_in_sim.py my_save_path
```

Rendered video and trajectory are stored in the same directory `my_save_path`.
To open SAPIEN GUI window use `--gui` flag.


## Assets

All existing assets and their paths are specified in `conf/assets/assets.yaml` configuration.
Fill free to modify it or make your own configuration.

```yaml
assets_dir_path: assets
products_hierarchy:
  food:
    grocery:
      baby:
        ss_asset_type: MeshAsset
        asset_file_path: ${assets.assets_dir_path}/...
        ss_params:
          scale : 1
          origin : ['com', 'com', 'bottom']
      cerealescornflakescora:
        asset_file_path: ${assets.assets_dir_path}/...
        ss_params:
          height: 0.25 
          up: [0, 1, 0] 
    dairy_products:
      milk:
        asset_file_path: ${assets.assets_dir_path}/...
...
```

We can observe, that structure of this file also dictates strict hierarchy between products.
All products are combined in categories such as `grocery`, `dairy_products`, etc.
Categories are combined in super-category `food`.

## More about configs

### Shelves

Basic configs are shelves' configs stored in `conf/shelves`.
They all are inherited from `base_shelf_config` configuration desctibed in `dsynth/scene_gen/hydra_configs.py` in the class `ShelfConfig`.
Most of the fields are self-explanatory, but pay attention to the two main variables: `queries` and `filling_type`.
The first one dictates **what** to put on the shelf's boards, namely list of regular expressions describing the set of desired products:

```yaml
defaults:
  - base_shelf_config
  - _self_
  
name: shelf_1
queries:
- food.dairy_products #Place all objects from category 'dairy_products' ('food.dairy_products.milk, 'food.dairy_products.milkCarton', etc.)

- food.grocery.baby #One particular product

- food #All available food
```

All fitting product names are put in one list and wrapped into iterator.

The second variable `filling_type` tells **how** queried product should be arranged on the shelf.
They are possible variants:

* `BOARDWISE_AUTO` Sequentially put every queried object on its shelf. If iterator raises `StopIteration`, procedure stops.
* `BOARDWISE_AUTO_INFINITE` The same as above, but iterator is cyclic. Procedure stops when all allowed boards are filled. 
* `BLOCKWISE_AUTO` Sequentially put every queried object in a groups (blocks). The size of each group is bounded with parameter `num_products_per_block`. You can place different groups of objects on the one board if `num_products_per_block` < `num_products_per_board`.
* `BLOCKWISE_AUTO_INFINITE` The same as above, but iterator is cyclic.
* `FULL_AUTO` Fill all boards on the shelf with one (first) queried object. 

### Zones

**Zone** is a group of shelves placed next to each other during procedural generation.
Zone configurations are just lists of included shelves. 

```yaml
defaults:
  - /shelves@shelf1: grocery #Shelf in conf/shelves/grocery.yaml
  - /shelves@shelf2: drinks  #Shelf in conf/shelves/drinks.yaml
```

Stored in `conf/zones`.

### Zone Lists

Accordingly, all zones presented on the scene are stored zone lists configs:

```yaml
defaults:
  - /zones@zone1: milk_zone #Zone in conf/zones/milk_zone.yaml
  - /zones@zone2: grocery   #Zone in conf/zones/grocery.yaml
```

Stored in `conf/ds` next to darkstore configurations.

### Darkstore Configuration

Includes its zone list and other important meta information, like scene name, and its size:

```yaml
defaults:
  - main_darkstore_config_base
  - /ds@zones: zones_list
  - _self_


name: ds
size_n: 3
size_m: 2
```

Stored in `conf/ds`.

### Default config

Default configuration `conf/config.yaml` includes darkstore and assets configurations.

```yaml
defaults:
  - ds: config
  - assets: assets
  - _self_
```
