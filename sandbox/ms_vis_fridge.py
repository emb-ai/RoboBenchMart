import gymnasium as gym
from tqdm import tqdm

import sapien
import mani_skill.envs
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
import sys
sys.path.append('.')
from dsynth.assets.asset import Asset

@register_env("BuggyPush", max_episode_steps=50)
class BuggyPush(PushCubeEnv):
    def _load_scene(self, options: dict):
        super()._load_scene(options)
       
        # load ycb object
        model_id='002_master_chef_can'
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{model_id}",
        )
        builder.set_initial_pose(sapien.Pose(p = [0.2, 0.2, 0.071]))
        self.can = builder.build(name=model_id)

        # hide goal_region
        # self._hidden_objects.append(self.goal_region)

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.066
        articulation_builders = loader.parse("assets/fridge_urdf/nestle.urdf")["articulation_builders"]
        builder = articulation_builders[0]
        builder.initial_pose = sapien.Pose(p=[3.0, 3.0, -0.92])
        self.cabinet = builder.build(name="nestle_fridge")
        
        loader = self.scene.create_urdf_loader()
        loader.scale = 1
        articulation_builders = loader.parse("assets/food_showcase/urdf/food_showcase.urdf")["articulation_builders"]
        builder = articulation_builders[0]
        builder.initial_pose = sapien.Pose(p=[-3.0, -3.0, 3.0])
        self.fridge = builder.build(name="fridge")
        
        self._cabinets = [self.cabinet, self.fridge]


        shelf_asset = Asset(
            'assets/store_mini_shelf_one_sided.glb',
            ss_params=dict(
                origin=("center", "bottom", "center"),
                up=(0, 1, 0),
                front=(0, 0, -1)
            ),
            asset_name='shelf'
        )
        shelf_asset.trimesh_scene.show(flags={'axis': True})

        self.mini_shelf = shelf_asset.ms_build_actor(
            'mini_shelf', 
            self.scene,
            pose=sapien.Pose(p=[5.0, -5.0, 0.0]),
            force_static=True
        )

        shelf_asset = Asset(
            'assets/store_mini_shelf_two_sided.glb',
            ss_params=dict(
                origin=("center", "bottom", "center"),
                up=(0, 1, 0),
                front=(0, 0, -1)
            ),
            asset_name='two_sided_shelf'
        )
        shelf_asset.trimesh_scene.show(flags={'axis': True})

        self.two_sided_shelf = shelf_asset.ms_build_actor(
            'two_sided_shelf', 
            self.scene,
            pose=sapien.Pose(p=[5.0, -6.0, 0.0]),
            force_static=True
        )

        # builder = self.scene.create_actor_builder()
        # builder.add_visual_from_file(filename='assets/store_mini_shelf_one_sided.glb', scale=[1, 1, 1])
        # builder.set_initial_pose(sapien.Pose(p=[5.0, -5.0, 0.0]))
        # builder.add_nonconvex_collision_from_file(filename='assets/store_mini_shelf_one_sided.glb', scale=[1, 1, 1])
        # self.mini_shelf = builder.build_static(name='mini_shelf')







def main():
    env_kwargs = dict(
        render_mode='human',
        num_envs=1,
        sim_backend='cpu', # enable GPU sim!
        enable_shadow=True,
        viewer_camera_configs={'shader_pack': 'default'}, 
        human_render_camera_configs={'shader_pack': 'default'},
        parallel_in_single_scene=False,
    )
    env = gym.make(
        'BuggyPush',
        **env_kwargs
    )
    env.reset()
    episode_length = 10000

    for i in tqdm(range(episode_length)):
        _, _, _, _, _ = env.step(None)
        
        env.render_human()

    # wait for exit [HERE YCB ASSET BEGIN TO FLOAT!!!]
    viewer = env.render_human()
    while True:
        if viewer.closed:
            exit()
        if viewer.window.key_down("c"):
            break
        env.render_human()

    env.close()

if __name__ == '__main__':
    main()