import numpy as np
from pathlib import Path
import itertools
from typing import List, Optional
import re

from transforms3d.euler import euler2quat
import torch
import yaml
import json
from copy import deepcopy
from typing import Dict, List, Optional
import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder, FIXTURES, FIXTURES_INTERIOR
from mani_skill.utils.scene_builder.robocasa.utils import scene_registry, scene_utils
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import (
    Fixture,
    FixtureType,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture_stack import FixtureStack
from mani_skill.utils.scene_builder.robocasa.fixtures.others import Box, Floor, Wall
from mani_skill.utils.structs import Actor
from mani_skill.utils.scene_builder.robocasa.utils.placement_samplers import (
    RandomizationError,
)

from transforms3d.euler import euler2quat
from transforms3d import quaternions

from dsynth.scene_gen.arrangements import CELL_SIZE, DEFAULT_ROOM_HEIGHT
from dsynth.envs.fixtures.robocasaroom import DarkstoreScene, Ceiling, _get_absolute_matrix, _get_pq

def get_arena_data(x_size=4., y_size=5., height = DEFAULT_ROOM_HEIGHT):
    return {
        'meta': {
            'x_size': x_size,
            'y_size': y_size,
            'height': height
        },
        'arena_config': {
            'room': {
                'walls': [
                    {'name': 'wall', 'type': 'wall', 'size': [x_size / 2, height / 2, 0.02], 'pos': [x_size / 2, y_size, height / 2]}, 
                    {'name': 'wall_backing', 'type': 'wall', 'backing': True, 'backing_extended': [True, False], 'size': [x_size / 2, height / 2, 0.1], 'pos': [x_size / 2, y_size, height / 2]}, 
                    
                    {'name': 'wall_front', 'type': 'wall', 'wall_side' : 'front', 'size': [x_size / 2, height / 2, 0.02], 'pos': [x_size / 2, 0, height / 2]}, 
                    {'name': 'wall_front_backing', 'type': 'wall', 'wall_side' : 'front', 'backing': True, 'size': [x_size / 2, height / 2, 0.1], 'pos': [x_size / 2, 0, height / 2]}, 
                    
                    {'name': 'wall_left', 'type': 'wall', 'wall_side': 'left', 'size': [y_size / 2, height / 2, 0.02], 'pos': [0, y_size / 2, height / 2]}, 
                    {'name': 'wall_left_backing', 'type': 'wall', 'wall_side': 'left', 'backing': True, 'size': [y_size / 2, height / 2, 0.1], 'pos': [0, y_size / 2, height / 2]}, 
                    
                    {'name': 'wall_right', 'type': 'wall', 'wall_side': 'right', 'size': [y_size / 2, height / 2, 0.02], 'pos': [x_size, y_size / 2, height / 2]}, 
                    {'name': 'wall_right_backing', 'type': 'wall', 'wall_side': 'right', 'backing': True, 'size': [y_size / 2, height / 2, 0.1], 'pos': [x_size, y_size / 2, height / 2]}
                ], 
                'floor': [
                    {'name': 'floor', 'type': 'floor', 'size': [x_size / 2, y_size / 2, 0.02], 'pos': [x_size / 2, y_size / 2, 0.0]}, 
                    {'name': 'floor_backing', 'type': 'floor', 'backing': True, 'size': [x_size / 2, y_size / 2, 0.1], 'pos': [x_size / 2, y_size / 2, 0.0]},
                ],
                'ceiling': [
                    {'name': 'ceiling', 'type': 'floor', 'size': [x_size / 2, y_size / 2, 0.02], 'pos': [x_size / 2, y_size / 2, height]},
                    {'name': 'ceiling_backing', 'type': 'floor', 'backing': True, 'size': [x_size / 2, y_size / 2, 0.01], 'pos': [x_size / 2, y_size / 2, height + 0.08]}
                ]
            }
        }
    }

class DarkstoreSceneContinuous(DarkstoreScene):
    def __init__(self, *args, config_dir_path=None, **kwargs):
        self.config_dir_path = config_dir_path
        self.scene_config_paths = sorted(list(Path(self.config_dir_path).glob('*layout_data*.json')))
        self.num_generated_scenes = len(self.scene_config_paths)
        
        self.x_size = []
        self.y_size = []
        self.height = []
        self.layout_data = []
        self.ds_names = []

        RoboCasaSceneBuilder.__init__(self, *args, **kwargs)

    def load_arrangement_from_json(self, scene_idx, scene_data):
        active_fixtures_categories = ["active_shelvings",
            "active_wall_shelvings"
            ]
        inactive_fixtures_categories = [
            "inactive_shelvings",
            "inactive_wall_shelvings",
            "scene_fixtures"
            ]
        for fixture_category in inactive_fixtures_categories:
            for i, inactive_fixture in enumerate(scene_data['layout_data'][fixture_category]):
                shelf_id = inactive_fixture['asset_name']
                fixture_name = inactive_fixture['name']
                item_name = f'[ENV#{scene_idx}]_inactive_{fixture_name}_{i}'
                p = np.array([inactive_fixture['x'], inactive_fixture['y'], 0.])
                angle = 0.
                if inactive_fixture['orientation'] == 'vertical':
                    angle = 3.14 / 2.
                q = euler2quat(0, 0, angle)
                pose = sapien.Pose(p=p, q=q)

                asset = self.env.assets_lib[shelf_id]
                asset.ms_is_nonconvex_collision = False
                actor = asset.ms_build_actor(item_name, self.env.scene, pose=pose, scene_idxs=[scene_idx])
                self.env.actors["fixtures"]["shelves"][item_name] = actor
        
        self.env.active_shelves[scene_idx] = []

        for fixture_category in active_fixtures_categories:
            for i, active_fixture in enumerate(scene_data['layout_data'][fixture_category]):
                shelf_id = active_fixture['asset_name']
                fixture_name = active_fixture['name']
                item_name = f'[ENV#{scene_idx}]_active_{fixture_name}_{i}'
                p = np.array([active_fixture['x'], active_fixture['y'], 0.])
                angle = 0.
                if active_fixture['orientation'] == 'vertical':
                    angle = 3.14 / 2.
                q = euler2quat(0, 0, angle)
                pose = sapien.Pose(p=p, q=q)

                asset = self.env.assets_lib[shelf_id]
                asset.ms_is_nonconvex_collision = True
                actor = asset.ms_build_actor(item_name, self.env.scene, pose=pose, scene_idxs=[scene_idx])
                self.env.actors["fixtures"]["shelves"][item_name] = actor
                self.env.active_shelves[scene_idx].append(item_name)

                with open(Path(self.config_dir_path) / f'{fixture_name}.json') as f:
                    shelf_arrangement = json.load(f)
                
                nodes_dict = {}
                for node in shelf_arrangement["graph"]:
                    nodes_dict[node[1]] = node

                for node in shelf_arrangement["graph"]:
                    parent_name, obj_name, props = node
                    if '/' not in obj_name and 'SHELF' not in obj_name:
                        abs_matrix = _get_absolute_matrix(node, nodes_dict)
                        p, q = _get_pq(abs_matrix, [0., 0., 0.])
                        prod_pose = pose * sapien.Pose(p=p, q=q)
                        asset_name = f'products_hierarchy.{obj_name.split(":")[0]}'
                        item_name = f'[ENV#{scene_idx}]_{obj_name}'
                        actor = self.env.assets_lib[asset_name].ms_build_actor(item_name, self.env.scene, pose=prod_pose, scene_idxs=[scene_idx])
                        self.env.actors["products"][item_name] = actor
        # origin = - self.IMPORTED_SS_SCENE_SHIFT

        # nodes_dict = {}
        # for node in scene_data["graph"]:
        #     nodes_dict[node[1]] = node

        # for node in scene_data["graph"]:
        #     parent_name, obj_name, props = node
        #     if '/' not in obj_name:
        #         abs_matrix = _get_absolute_matrix(node, nodes_dict)
        #         p, q = _get_pq(abs_matrix, origin)
        #         pose = sapien.Pose(p=p, q=q)
        #         if 'SHELF' in obj_name:
        #             shelf_name = re.sub(r"SHELF_\d+_", "", obj_name)
        #             zone_id, shelf_id = shelf_name.split('.')
        #             shelf_asset_name = self.env.cfg.ds.zones[zone_id][shelf_id].shelf_asset
        #             if shelf_asset_name is None:
        #                 shelf_asset_name = 'fixtures.shelf'
        #             item_name = f'[ENV#{scene_idx}]_{obj_name}'
        #             actor = self.env.assets_lib[shelf_asset_name].ms_build_actor(item_name, self.env.scene, pose=pose, scene_idxs=[scene_idx])
        #             self.env.actors["fixtures"]["shelves"][item_name] = actor
        #             continue

        #         asset_name = f'products_hierarchy.{obj_name.split(":")[0]}'
        #         item_name = f'[ENV#{scene_idx}]_{obj_name}'
        #         actor = self.env.assets_lib[asset_name].ms_build_actor(
        #             item_name, 
        #             self.env.scene, 
        #             pose=pose, 
        #             scene_idxs=[scene_idx],
        #             force_static=self.env.all_static)
        #         self.env.actors["products"][item_name] = actor
                
        #         zone_id, shelf_id = _get_zone_shelf_ids(node, nodes_dict)
        #         self.env.products2shelves[item_name] = (zone_id, shelf_id)

    def _get_lamps_coords(self, x_size, y_size, num_lamps_x=4, num_lamps_y=4, dist_from_wall=0.0):
        lamps_coords = []
        step_x = (x_size - 2*dist_from_wall)/(num_lamps_x+1)
        step_y = (y_size - 2*dist_from_wall)/(num_lamps_y+1)
        for x in range(1, num_lamps_x+1):
            for y in range(1, num_lamps_y+1):
                lamps_coords.append((dist_from_wall+step_x*x, dist_from_wall+step_y*y))
        return lamps_coords

    def build(self, build_config_idxs: Optional[List[int]] = None):
        if self.env.agent is not None:
            self.robot_poses = self.env.agent.robot.initial_pose
        else:
            self.robot_poses = None

        if build_config_idxs is None:
            build_config_idxs = []
            for i in range(self.env.num_envs):
                # Total number of configs is 10 * 12 = 120
                config_idx = self.env._batched_episode_rng[i].randint(0, self.num_generated_scenes)
                build_config_idxs.append(config_idx)

        # random indexes for walls, floors and ceilings
        num_wall_textures = len(list(Path('assets/textures/walls').iterdir()))
        wall_texture_idxs = [self.env._batched_episode_rng[i].randint(0, num_wall_textures) for i in range(len(build_config_idxs))]

        num_floor_textures = len(list(Path('assets/textures/floors').iterdir()))
        floor_texture_idxs = [self.env._batched_episode_rng[i].randint(0, num_floor_textures) for i in range(len(build_config_idxs))]

        num_ceiling_textures = len(list(Path('assets/textures/ceilings').iterdir()))
        ceiling_texture_idxs = [self.env._batched_episode_rng[i].randint(0, num_ceiling_textures) for i in range(len(build_config_idxs))]

        for scene_idx, build_config_idx in enumerate(build_config_idxs):
            config_path = self.scene_config_paths[build_config_idx]
            
            with open(config_path, "r") as f:
                scene_data = json.load(f)

            arena_data = get_arena_data(x_size=scene_data['size_x'], 
                           y_size=scene_data['size_y'])
            
            self.x_size.append(arena_data['meta']['x_size'])
            self.y_size.append(arena_data['meta']['y_size'])
            self.height.append(arena_data['meta']['height'])
            self.layout_data.append(scene_data['layout_data'])

            # self.ds_names.append(scene_data['meta'].get('ds_names', None))


            arena_config = arena_data['arena_config']


            style_idx = build_config_idx % 12  # Get style index (0-11)
            floor_texture_id = floor_texture_idxs[scene_idx]
            wall_texture_id = wall_texture_idxs[scene_idx]
            ceiling_texture_id = ceiling_texture_idxs[scene_idx]
            # layout_path = scene_registry.get_layout_path(layout_idx)
            style_path = scene_registry.get_style_path(style_idx)
            # load style
            with open(style_path, "r") as f:
                style = yaml.safe_load(f)

            # # load arena
            # if self.arena_config is None:
            #     layout_path = 'layout_warehouse.yaml'
            #     with open(layout_path, "r") as f:
            #         arena_config = yaml.safe_load(f)
            # else:
            #     arena_config = self.arena_config

            # contains all fixtures with updated configs
            arena = list()

            # Update each fixture config. First iterate through groups: subparts of the arena that can be
            # rotated and displaced together. example: island group, right group, room group, etc
            for group_name, group_config in arena_config.items():
                group_fixtures = list()
                # each group is further divded into similar subcollections of fixtures
                # ex: main group counter accessories, main group top cabinets, etc
                for k, fixture_list in group_config.items():
                    # these values are rotations/displacements that are applied to all fixtures in the group
                    if k in ["group_origin", "group_z_rot", "group_pos"]:
                        continue
                    elif type(fixture_list) != list:
                        raise ValueError(
                            '"{}" is not a valid argument for groups'.format(k)
                        )

                    # add suffix to support different groups
                    for fxtr_config in fixture_list:
                        fxtr_config["name"] += "_" + group_name
                        # update fixture names for alignment, interior objects, etc.
                        for k in scene_utils.ATTACH_ARGS + [
                            "align_to",
                            "stack_fixtures",
                            "size",
                        ]:
                            if k in fxtr_config:
                                if isinstance(fxtr_config[k], list):
                                    for i in range(len(fxtr_config[k])):
                                        if isinstance(fxtr_config[k][i], str):
                                            fxtr_config[k][i] += "_" + group_name
                                else:
                                    if isinstance(fxtr_config[k], str):
                                        fxtr_config[k] += "_" + group_name

                    group_fixtures.extend(fixture_list)

                # update group rotation/displacement if necessary
                if "group_origin" in group_config:
                    for fxtr_config in group_fixtures:
                        # do not update the rotation of the walls/floor
                        if fxtr_config["type"] in ["wall", "floor"]:
                            continue
                        fxtr_config["group_origin"] = group_config["group_origin"]
                        fxtr_config["group_pos"] = group_config["group_pos"]
                        fxtr_config["group_z_rot"] = group_config["group_z_rot"]

                # addto overall fixture list
                arena.extend(group_fixtures)

            # maps each fixture name to its object class
            fixtures: Dict[str, Fixture] = dict()
            # maps each fixture name to its configuration
            configs = dict()
            # names of composites, delete from fixtures before returning
            composites = list()

            for fixture_config in arena:
                # scene_registry.check_syntax(fixture_config)
                fixture_name = fixture_config["name"]

                # stack of fixtures, handled separately
                if fixture_config["type"] == "stack":
                    stack = FixtureStack(
                        self.scene,
                        fixture_config,
                        fixtures,
                        configs,
                        style,
                        default_texture=None,
                        rng=self.env._batched_episode_rng[scene_idx],
                    )
                    fixtures[fixture_name] = stack
                    configs[fixture_name] = fixture_config
                    composites.append(fixture_name)
                    continue

                # load style information and update config to include it
                default_config = scene_utils.load_style_config(style, fixture_config)
                if default_config is not None:
                    for k, v in fixture_config.items():
                        default_config[k] = v
                    fixture_config = default_config

                if fixture_config["type"] == "wall":
                    fixture_config['texture'] = str(list(Path('assets/textures/walls').iterdir())[wall_texture_id].resolve())
                elif fixture_config["type"] == "floor":
                    fixture_config['texture'] = str(list(Path('assets/textures/floors').iterdir())[floor_texture_id].resolve())

                # set fixture type
                if fixture_config["type"] not in FIXTURES:
                    continue
                fixture_config["type"] = FIXTURES[fixture_config["type"]]

                # modify type to ceiling
                if fixture_config['name'] == "ceiling_room":
                    fixture_config['type'] = Ceiling
                    fixture_config['texture'] = str(list(Path('assets/textures/ceilings').iterdir())[ceiling_texture_id].resolve())

                # pre-processing for fixture size
                size = fixture_config.get("size", None)
                if isinstance(size, list):
                    for i in range(len(size)):
                        elem = size[i]
                        if isinstance(elem, str):
                            ref_fxtr = fixtures[elem]
                            size[i] = ref_fxtr.size[i]

                # initialize fixture
                # TODO (stao): use batched episode rng later
                fixture = scene_utils.initialize_fixture(
                    self.scene,
                    fixture_config,
                    fixtures,
                    rng=self.env._batched_episode_rng[scene_idx],
                )

                fixtures[fixture_name] = fixture
                configs[fixture_name] = fixture_config
                pos = None
                # update fixture position
                if fixture_config["type"] not in FIXTURES_INTERIOR.values():
                    # relative positioning
                    if "align_to" in fixture_config:
                        pos = scene_utils.get_relative_position(
                            fixture,
                            fixture_config,
                            fixtures[fixture_config["align_to"]],
                            configs[fixture_config["align_to"]],
                        )

                    elif "stack_on" in fixture_config:
                        stack_on = fixtures[fixture_config["stack_on"]]

                        # account for off-centered objects
                        stack_on_center = stack_on.center

                        # infer unspecified axes of position
                        pos = fixture_config["pos"]
                        if pos[0] is None:
                            pos[0] = stack_on.pos[0] + stack_on_center[0]
                        if pos[1] is None:
                            pos[1] = stack_on.pos[1] + stack_on_center[1]

                        # calculate height of fixture
                        pos[2] = (
                            stack_on.pos[2] + stack_on.size[2] / 2 + fixture.size[2] / 2
                        )
                        pos[2] += stack_on_center[2]
                    else:
                        # absolute position
                        pos = fixture_config.get("pos", None)
                if pos is not None and type(fixture) not in [Wall, Floor, Ceiling]:
                    fixture.set_pos(deepcopy(pos))
            # composites are non-MujocoObjects, must remove
            for composite in composites:
                del fixtures[composite]

            # update the rotation and postion of each fixture based on their group
            for name, fixture in fixtures.items():
                # check if updates are necessary
                config = configs[name]
                if "group_origin" not in config:
                    continue

                # TODO: add default for group origin?
                # rotate about this coordinate (around the z-axis)
                origin = config["group_origin"]
                pos = config["group_pos"]
                z_rot = config["group_z_rot"]
                displacement = [pos[0] - origin[0], pos[1] - origin[1]]

                if type(fixture) not in [Wall, Floor, Ceiling]:
                    dx = fixture.pos[0] - origin[0]
                    dy = fixture.pos[1] - origin[1]
                    dx_rot = dx * np.cos(z_rot) - dy * np.sin(z_rot)
                    dy_rot = dx * np.sin(z_rot) + dy * np.cos(z_rot)

                    x_rot = origin[0] + dx_rot
                    y_rot = origin[1] + dy_rot
                    z = fixture.pos[2]
                    pos_new = [x_rot + displacement[0], y_rot + displacement[1], z]

                    # account for previous z-axis rotation
                    rot_prev = fixture.euler
                    if rot_prev is not None:
                        # TODO: switch to quaternion since euler rotations are ambiguous
                        rot_new = rot_prev
                        rot_new[2] += z_rot
                    else:
                        rot_new = [0, 0, z_rot]
                    fixture.pos = np.array(pos_new)
                    fixture.set_euler(rot_new)

            # self.actors = actors
            # fixtures = fixtures
            fixture_cfgs = self.get_fixture_cfgs(fixtures)
            # generate initial poses for objects so that they are spawned in nice places during GPU initialization
            # to be more performant
            (
                fxtr_placements,
                robot_base_pos,
                robot_base_ori,
            ) = self._generate_initial_placements(
                fixtures, fixture_cfgs, rng=self.env._batched_episode_rng[scene_idx]
            )
            self.scene_data.append(
                dict(
                    fixtures=fixtures,
                    fxtr_placements=fxtr_placements,
                    fixture_cfgs=fixture_cfgs,
                )
            )

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in fxtr_placements.values():
                assert isinstance(obj, Fixture)
                obj.pos = obj_pos
                obj.quat = obj_quat

            if self.env.agent is not None:
                self.robot_poses.raw_pose[scene_idx][:3] = torch.from_numpy(
                    robot_base_pos
                ).to(self.robot_poses.device)
                self.robot_poses.raw_pose[scene_idx][3:] = torch.from_numpy(
                    euler2quat(*robot_base_ori)
                ).to(self.robot_poses.device)

            actors: Dict[str, Actor] = {}

            ### collision handling and optimization ###
            # Generally we aim to ensure all articulations in a stack have the same collision bits so they can't collide with each other
            # and with a range of [22, 30] we can generally ensure adjacent articulations can collide with each other.
            # walls and floors cannot collide with anything. Walls can only collide with the robot. They are assigned bits 22 to 30.
            # mobile base robots have their wheels/non base links assigned bit of 30 to not collide with the floor or walls.
            # the base links can optionally be also assigned a bit of 31 to not collide with walls.

            # fixtures that are not articulated are always static and cannot hit other non-articulated fixtures. This scenario is assigned bit 21.
            actor_bit = 21
            # prismatic_drawer_bit = 25

            collision_start_bit = 22
            fixture_idx = 0
            stack_collision_bits = dict()
            for stack_index, stack in enumerate(composites):
                stack_collision_bits[stack] = collision_start_bit + stack_index % 9
            for k, v in fixtures.items():
                fixture_idx += 1
                built = v.build(scene_idxs=[scene_idx])
                if built is not None:
                    actors[k] = built
                    # ensure all rooted articulated objects have collisions ignored with all static objects
                    # ensure all articulations in the same stack have the same collision bits, since by definition for robocasa they cannot
                    # collide with each other
                    if (
                        built.is_articulation
                        and built.articulation.fixed_root_link.all()
                    ):
                        collision_bit = collision_start_bit + fixture_idx % 5
                        if "stack" in v.name:
                            for stack_group in stack_collision_bits.keys():
                                if stack_group in v.name:
                                    collision_bit = stack_collision_bits[stack_group]
                                    break
                        # is_prismatic_cabinet = False
                        # for joint in built.articulation.joints:
                        #     if joint.type[0] == "prismatic":
                        #         is_prismatic_cabinet = True
                        #         break
                        for link in built.articulation.links:
                            # if "object" in link.name:
                            #     import ipdb; ipdb.set_trace()
                            link.set_collision_group(
                                group=2, value=0
                            )  # clear all default ignored collisions
                            if link.joint.type[0] == "fixed":
                                link.set_collision_group_bit(
                                    group=2, bit_idx=actor_bit, bit=1
                                )
                            link.set_collision_group_bit(
                                group=2, bit_idx=collision_bit, bit=1
                            )

                    else:
                        if built.actor.px_body_type == "static":
                            collision_bit = collision_start_bit + fixture_idx % 5
                            if "stack" in v.name:
                                for stack_group in stack_collision_bits.keys():
                                    if stack_group in v.name:
                                        collision_bit = stack_collision_bits[
                                            stack_group
                                        ]
                                        break
                            if isinstance(v, Floor):
                                for bit_idx in range(21, 32):
                                    built.actor.set_collision_group_bit(
                                        group=2, bit_idx=bit_idx, bit=1
                                    )
                            elif isinstance(v, Wall):
                                for bit_idx in range(21, 31):
                                    built.actor.set_collision_group_bit(
                                        group=2, bit_idx=bit_idx, bit=1
                                    )
                            elif isinstance(v, Ceiling):
                                for bit_idx in range(21, 32):
                                    built.actor.set_collision_group_bit(
                                        group=2, bit_idx=bit_idx, bit=1
                                    )
                            else:
                                built.actor.set_collision_group_bit(
                                    group=2,
                                    bit_idx=collision_bit,
                                    bit=1,
                                )
                                built.actor.set_collision_group_bit(
                                    group=2, bit_idx=actor_bit, bit=1
                                )
            # self.actors = actors

            self.load_arrangement_from_json(scene_idx, scene_data)
            lamp_coords = self._get_lamps_coords(
                arena_data['meta']['x_size'],
                arena_data['meta']['y_size'],
            )
            self._load_lamps(scene_idx, lamp_coords, arena_data['meta']['height'])
            self._load_lighting(scene_idx, lamp_coords, arena_data['meta']['height'])
            self._load_door(scene_idx, arena_data['meta']['x_size'], arena_data['meta']['y_size'])

        # disable collisions
        if self.env.robot_uids == "fetch":
            self.env.agent
            for link in [self.env.agent.l_wheel_link, self.env.agent.r_wheel_link]:
                for bit_idx in range(25, 31):
                    link.set_collision_group_bit(group=2, bit_idx=bit_idx, bit=1)
            # for bit_idx in range(25, 31):
            self.env.agent.base_link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

        elif self.env.robot_uids == "unitree_g1_simplified_upper_body":
            # TODO (stao): determine collisions to disable for unitree robot
            pass
