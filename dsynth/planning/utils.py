import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
import numpy as np
from transforms3d.euler import euler2quat
import mplib
from mplib.sapien_utils.conversion import convert_object_name
from mplib.collision_detection.fcl import CollisionGeometry
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib.collision_detection.fcl import Convex, CollisionObject, FCLObject
from mplib.collision_detection import fcl

import sapien
import sapien.physx as physx
from sapien import Entity
from sapien.physx import (
    PhysxArticulation,
    PhysxArticulationLinkComponent,
    PhysxCollisionShapeConvexMesh
)


from typing import Literal, Optional, Sequence, Union
import sys
import trimesh
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube
from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils import common
from mani_skill import Actor


def attach_object(  # type: ignore
    planning_world: SapienPlanningWorld,
    obj: Union[Entity, str],
    articulation: Union[PhysxArticulation, str],
    link: Union[PhysxArticulationLinkComponent, int],
    pose: Optional[mplib.Pose] = None,
    *,
    touch_links: Optional[list[Union[PhysxArticulationLinkComponent, str]]] = None,
    obj_geom: Optional[CollisionGeometry] = None,
) -> None:
    """
    Attaches given non-articulated object to the specified link of articulation.

    Updates ``acm_`` to allow collisions between attached object and touch_links.

    :param obj: the non-articulated object (or its name) to attach
    :param articulation: the planned articulation (or its name) to attach to
    :param link: the link of the planned articulation (or its index) to attach to
    :param pose: attached pose (relative pose from attached link to object).
        If ``None``, attach the object at its current pose.
    :param touch_links: links (or their names) that the attached object touches.
        When ``None``,

        * if the object is not currently attached, touch_links are set to the name
        of articulation links that collide with the object in the current state.

        * if the object is already attached, touch_links of the attached object
        is preserved and ``acm_`` remains unchanged.
    :param obj_geom: a CollisionGeometry object representing the attached object.
        If not ``None``, pose must be not ``None``.

    .. raw:: html

        <details>
        <summary><a>Overloaded
        <code class="docutils literal notranslate">
        <span class="pre">PlanningWorld.attach_object()</span>
        </code>
        methods</a></summary>
    .. automethod:: mplib.PlanningWorld.attach_object
        :no-index:
    .. raw:: html
        </details>
    """
    kwargs = {"name": obj, "art_name": articulation, "link_id": link}
    if pose is not None:
        kwargs["pose"] = pose
    if touch_links is not None:
        kwargs["touch_links"] = [
            l.name if isinstance(l, PhysxArticulationLinkComponent) else l
            for l in touch_links  # noqa: E741
        ]
    if obj_geom is not None:
        kwargs["obj_geom"] = obj_geom

    if isinstance(obj, Entity):
        kwargs["name"] = convert_object_name(obj)
    if isinstance(articulation, PhysxArticulation):
        kwargs["art_name"] = articulation = convert_object_name(articulation)
    if isinstance(link, PhysxArticulationLinkComponent):
        kwargs["link_id"] = (
            planning_world.get_articulation(articulation)
            .get_pinocchio_model()
            .get_link_names()
            .index(link.name)
        )

    planning_world.attach_object(**kwargs)


def get_fcl_object_name(entity):
    component = entity._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
    return convert_object_name(component.entity)


def compute_box_grasp_thin_side_info(
    obb: trimesh.primitives.Box,
    target_closing=None,
    ee_direction=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    inds = np.argsort(extents[:2])
    short_base_side_ind = inds[0]
    long_base_side_ind = inds[1]

    height = extents[2]

    approaching = np.array(T[:3, long_base_side_ind])
    approaching = common.np_normalize_vector(approaching)

    if ee_direction @ approaching < 0:
        approaching = -approaching

    closing = np.array(T[:3, short_base_side_ind])

    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    # Find the origin on the surface
    center = T[:3, 3]
    half_size = extents[long_base_side_ind] / 2
    center = center + approaching * (-half_size + min(depth, half_size))

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info

def convert_actor_convex_mesh_to_fcl(actor: Actor):
    component = actor._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
    assert component is not None, (
        f"No PhysxRigidBaseComponent found in {actor.name}: "
        f"{actor.components=}"
    )
    assert len(component.collision_shapes) == 1
    shape = component.collision_shapes[0]
    assert isinstance(shape, physx.PhysxCollisionShapeConvexMesh)

    # tranform vertices, so that scale == 1.0
    vertices = shape.vertices
    vertices[:, 0] *= shape.scale[0]
    vertices[:, 1] *= shape.scale[1]
    vertices[:, 2] *= shape.scale[2]
    c_geom = Convex(vertices=vertices, faces=shape.triangles)
    collision_shape = CollisionObject(c_geom)

    return FCLObject(
        convert_object_name(component.entity),
        component.entity.pose,
        [collision_shape],
        [mplib.Pose(shape.local_pose)],
    )


class SapienPlanningWorldV2(SapienPlanningWorld):
    """
    Patched version of SapienPlanningWorld for meshes with scale
    """
    @staticmethod
    def convert_physx_component(comp: physx.PhysxRigidBaseComponent) -> FCLObject | None:
        """
        Converts a SAPIEN physx.PhysxRigidBaseComponent to an FCLObject.
        All shapes in the returned FCLObject are already set at their world poses.

        :param comp: a SAPIEN physx.PhysxRigidBaseComponent.
        :return: an FCLObject containing all collision shapes in the Physx component.
            If the component has no collision shapes, return ``None``.
        """
        shapes: list[CollisionObject] = []
        shape_poses: list[mplib.Pose] = []
        for shape in comp.collision_shapes:
            shape_poses.append(mplib.Pose(shape.local_pose))  # type: ignore

            if isinstance(shape, physx.PhysxCollisionShapeBox):
                c_geom = fcl.Box(side=shape.half_size * 2)
            elif isinstance(shape, physx.PhysxCollisionShapeCapsule):
                c_geom = fcl.Capsule(radius=shape.radius, lz=shape.half_length * 2)
                # NOTE: physx Capsule has x-axis along capsule height
                # FCL Capsule has z-axis along capsule height
                shape_poses[-1] *= mplib.Pose(q=euler2quat(0, np.pi / 2, 0))
            elif isinstance(shape, PhysxCollisionShapeConvexMesh):
                # assert np.allclose(
                #     shape.scale, 1.0
                # ), f"Not unit scale {shape.scale}, need to rescale vertices?"

                # Scale vertices!
                vertices = shape.vertices
                vertices[:, 0] *= shape.scale[0]
                vertices[:, 1] *= shape.scale[1]
                vertices[:, 2] *= shape.scale[2]
                c_geom = Convex(vertices=vertices, faces=shape.triangles)
            elif isinstance(shape, physx.PhysxCollisionShapeCylinder):
                c_geom = fcl.Cylinder(radius=shape.radius, lz=shape.half_length * 2)
                # NOTE: physx Cylinder has x-axis along cylinder height
                # FCL Cylinder has z-axis along cylinder height
                shape_poses[-1] *= mplib.Pose(q=euler2quat(0, np.pi / 2, 0))
            elif isinstance(shape, physx.PhysxCollisionShapePlane):
                # PhysxCollisionShapePlane are actually a halfspace
                # https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/Geometry.html#planes
                # PxPlane's Pose determines its normal and offert (normal is +x)
                n = shape_poses[-1].to_transformation_matrix()[:3, 0]
                d = n.dot(shape_poses[-1].p)
                c_geom = fcl.Halfspace(n=n, d=d)
                shape_poses[-1] = mplib.Pose()
            elif isinstance(shape, physx.PhysxCollisionShapeSphere):
                c_geom = fcl.Sphere(radius=shape.radius)
            elif isinstance(shape, physx.PhysxCollisionShapeTriangleMesh):
                c_geom = fcl.BVHModel()
                c_geom.begin_model()
                c_geom.add_sub_model(vertices=shape.vertices, faces=shape.triangles)  # type: ignore
                c_geom.end_model()
            else:
                raise TypeError(f"Unknown shape type: {type(shape)}")
            shapes.append(CollisionObject(c_geom))
            
        if len(shapes) == 0:
            return None

        return FCLObject(
            comp.name
            if isinstance(comp, PhysxArticulationLinkComponent)
            else convert_object_name(comp.entity),
            comp.entity.pose,  # type: ignore
            shapes,
            shape_poses,
        )