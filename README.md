# darkstore_synthesizer

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
```

Test your ManiSkill installation:
```bash
python -m mani_skill.examples.demo_random_action
```

### Copy assets to directory

All assets are stored in `/home/jovyan/shares/SR006.nfs2/data/dsynth/assets`.
Copy or link this directory to `assets/` directory.
Also assets available via GDrive https://drive.google.com/file/d/1RhBw9HfoHm6uvxrFC9hxYq0JB1FRI_d-

## Scene generation and environment visualization

To generate layout and object arrangement use input configurations from `config/` or make your own:

```bash
python scripts/generate_room_config.py --input configs/many_objects.json --output_dir my_save_path
```

Add `--show` flag to visualize generated arrangement.

To show generated scene in ManiSkill:
```bash
python scripts/show_env_in_sim.py my_save_path
```

Rendered video and trajectory are stored in the same directory `my_save_path`.
To open SAPIEN GUI window use `--gui` flag.
Press `c` button to close GUI and store the recorded trajectory.
