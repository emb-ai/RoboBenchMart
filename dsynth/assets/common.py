import sapien
from mani_skill.envs.scene import ManiSkillScene

CEILING_LAMP_HEIGHT = 0.4

def ceiling_lamp(name, scene: ManiSkillScene, pose: sapien.Pose, assets_dir_path):
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(filename=f'{assets_dir_path}/lamp1.glb')
    builder.set_initial_pose(pose)
    return builder.build_static(name=name)
