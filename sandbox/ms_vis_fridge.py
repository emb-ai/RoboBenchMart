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
import numpy as np
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
        # shelf_asset.trimesh_scene.show(flags={'axis': True})

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
        # shelf_asset.trimesh_scene.show(flags={'axis': True})

        self.two_sided_shelf = shelf_asset.ms_build_actor(
            'two_sided_shelf', 
            self.scene,
            pose=sapien.Pose(p=[5.0, -6.0, 0.0]),
            force_static=True
        )

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[0.01, 0.2, 0.2],
        )
        builder.add_box_visual(
            half_size=[0.01, 0.8, 0.8],
            material=sapien.render.RenderMaterial(
                # RGBA values, this is a red cube
                base_color=[0.9, 0.9, 0.9, 1],
                transmission=1.0,
                transmission_roughness=0.001,
                ior=1.6,
                roughness = 0.001,
                metallic=0.0005,
            ),
        )
        
        self.cube = builder.build_static('cube222')

        builder = self.scene.create_actor_builder()
        glass = '/home/kvsoshin/Work/glass.glb'
        # glass = 'assets/food_showcase/meshes/obj/glass2.glb'
        # glass = 'assets/food_showcase/meshes/obj/model_7.obj'
        builder.add_visual_from_file(filename=glass, scale=[1, 1, 1],
            #                          material=sapien.render.RenderMaterial(
            #     # RGBA values, this is a red cube
            #     base_color=[0.9, 0.9, 0.9, 1],
            #     transmission=1.0,
            #     transmission_roughness=0.001,
            #     ior=1.6,
            #     roughness = 0.001,
            #     metallic=0.0005,
            # )
            )
        builder.set_initial_pose(sapien.Pose(p=[3.0, -3.0, 0.0]))
        builder.add_convex_collision_from_file(filename=glass, scale=[1, 1, 1])
        self.mini_shelf = builder.build_static(name='door')
        print(self.mini_shelf._objs[0].components[0].render_shapes[0].get_parts()[1].get_material())

        # mt = sapien.render.RenderMaterial()
        # mt.diffuse_texture = sapien.render.RenderTexture2D(filename='/home/kvsoshin/Work/glass.glb')





def main():
    from mani_skill.render.shaders import ShaderConfig, rt_texture_names, rt_texture_transforms
    shader_config=ShaderConfig(
                                shader_pack="rt",
                                texture_names=rt_texture_names,
                                shader_pack_config={
                                    "ray_tracing_samples_per_pixel": 8,
                                    "ray_tracing_path_depth": 4,
                                    "ray_tracing_denoiser": "optix",
                                },
                                texture_transforms=rt_texture_transforms,
                            )
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