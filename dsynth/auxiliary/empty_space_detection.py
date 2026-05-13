import json
import numpy as np
import sapien
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from transforms3d.euler import euler2quat
import json
import sapien
from transforms3d.euler import euler2quat
import cv2
from typing import List, Dict, Any, Optional
from mani_skill.utils.registration import register_env
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb


from dsynth.envs.fixtures.robocasaroom import _get_absolute_matrix, _get_pq
from dsynth.envs.darkstore_cont_base import DarkstoreContinuousBaseEnv
from dsynth.envs.fixtures.robocasaroom_cont import DarkstoreSceneContinuous
from dsynth.assets.ss_assets import WIDTH, DEPTH

def extract_free_spaces(env: DarkstoreContinuousBaseEnv) -> List[int]:
    free_spaces: List[int] = []

    for free_space in env.unwrapped.actors["free_spaces"].values():
        if not free_space.name.endswith("0"):
            continue
        free_spaces.append(1000 + free_space.per_scene_id[0].item())

    return free_spaces

def annotate_image_with_free_spaces(
    image: np.ndarray, segmentation: np.ndarray, env: DarkstoreContinuousBaseEnv
) -> np.ndarray:
    free_spaces = extract_free_spaces(env)
    palette = build_palette(free_spaces)
    output = image.copy()

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    bg_color = (255, 255, 255)

    bboxes = dict()

    for free_space_id in free_spaces:
        bbox = build_bbox(segmentation, free_space_id - 1000)
        if bbox is None:
            continue

        x_min, y_min, x_max, y_max = bbox
        color = tuple(map(int, palette[free_space_id]))

        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), color, 1)

        label = str(free_space_id)
        (label_width, label_height), baseline = cv2.getTextSize(
            label, font_face, font_scale, font_thickness
        )
        label_x, label_y = x_min, max(y_min, label_height)

        cv2.rectangle(
            output,
            (label_x, label_y - label_height - baseline),
            (label_x + label_width, label_y - baseline),
            bg_color,
            -1,
        )

        bboxes[free_space_id] = bbox

        cv2.putText(
            output,
            label,
            (label_x, label_y - baseline),
            font_face,
            font_scale,
            color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return output, bboxes

def detect_free_spaces(env: DarkstoreContinuousBaseEnv) -> Dict[str, Any]:
    obs_wo_free = env.base_env.get_obs()

    build_free_spaces(env.base_env)

    obs_w_free = env.base_env.get_obs()

    result = {}
    raw_images = []
    annotated_images = []
    bboxes = {}
    cameras = ["left_base_camera_link", "fetch_hand", "right_base_camera_link"]
    for camera in cameras:
        # camera_data = obs_wo_free["sensor_data"][camera]
        image = obs_wo_free["sensor_data"][camera]["rgb"][0].cpu().numpy()[:, :, ::-1]
        seg = obs_w_free["sensor_data"][camera]["segmentation"][0].cpu().numpy()[..., 0]

        scale = 768 / image.shape[0]
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        seg = cv2.resize(seg, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        annotated_image, camera_bboxes = annotate_image_with_free_spaces(image, seg, env)
        bboxes[camera] = camera_bboxes
        result[camera] = {
            "image": image,
            "annotated_image": annotated_image,
        }
        raw_images.append(image)
        annotated_images.append(annotated_image)

    result["combined"] = {
        "image": np.hstack(raw_images),
        "annotated_image": np.hstack(annotated_images),
    }
    result["spaces_description"] = get_free_spaces_description(env)
    result["bboxes"] = bboxes
    destroy_free_spaces(env.base_env)

    return result

def get_free_spaces_description(env: DarkstoreContinuousBaseEnv) -> List[Dict[str, Any]]:
    actor_by_free_space_id = {
        1000 + actor.per_scene_id[0].item(): actor
        for actor in env.unwrapped.actors["free_spaces"].values()
    }

    return [
        {
            "free_space": product_id,
            "num_board":  int(actor_by_free_space_id[product_id].name.split("_")[-2])
        }
        for product_id in extract_free_spaces(env)
        if (actor := actor_by_free_space_id.get(product_id)) is not None
    ]

def build_palette(product_ids: List[int], seed: int = 42) -> np.ndarray:
    max_product_id = max(product_ids)
    palette_size = max(128, max_product_id + 1)

    rng = np.random.RandomState(seed)
    palette = rng.randint(20, 235, size=(palette_size, 3), dtype=np.uint8)

    return palette

def build_bbox(segmentation: np.ndarray, product_id: int) -> Optional[List[int]]:
    mask = segmentation == product_id

    if not mask.any():
        return None

    height, width = segmentation.shape
    padding = 3

    ys, xs = np.where(mask)
    x_min = max(0, int(xs.min()) - padding)
    y_min = max(0, int(ys.min()) - padding)
    x_max = min(width - 1, int(xs.max()) + padding)
    y_max = min(height - 1, int(ys.max()) + padding)

    return [x_min, y_min, x_max, y_max]


def build_free_spaces(env):
    for scene_idx in range(env.num_envs):
        for num_board in range(5):
            _build_free_spaces_on_board(env, num_board, scene_idx)

def destroy_free_spaces(env):
    for actor in env.actors['free_spaces'].values():
        env.remove_from_state_dict_registry(actor)
        actor.remove_from_scene()
        env.scene.actors.pop(actor.name)
        # env.scene.remove_actor(actor)
    env.actors['free_spaces'] = {}

def get_bbox(actor):
    obb = get_actor_obb(actor)
    extents = np.array(obb.extents)
    T = np.array(obb.transform)
    center = T[:3, 3]

    for i in range(3):
        if np.abs(T[:3, i] @ [0, 0, 1]) > 1e-1:
            height_extent_ids = i
            break

    inds = np.array([0, 1, 2])
    inds = inds[inds != height_extent_ids]

    h, w = extents[inds]

    return [
        center[0] - h / 2,
        center[1] - w / 2,
        center[0] + h / 2,
        center[1] + w / 2,
    ]

def _get_product_bboxes(env, num_board: int, scene_idx=0):
    scene_prducts_df = env.products_df[env.products_df['scene_idx'] == scene_idx]
    board_products_df = scene_prducts_df[scene_prducts_df['board_idxs'] == str(num_board)]
    bboxes = []
    for actor_name in board_products_df['actor_name']:
        actor = env.actors['products'][actor_name]
        bbox = get_bbox(actor)
        bboxes.append(bbox)
    return bboxes

def _get_shelf_bbox(env, scene_idx=0):
    shelf_name = env.active_shelves[scene_idx][0]
    shelf_pose = env.actors["fixtures"]["shelves"][shelf_name].pose.sp
    direction_to_shelf = env.directions_to_shelf[scene_idx]
    
    x_len = WIDTH
    y_len = DEPTH
    if np.abs(np.dot(direction_to_shelf, [0, 1, 0])) < 1e-2:
        x_len = DEPTH
        y_len = WIDTH
    shelf_center = shelf_pose.p.copy()
    board_bbox = [shelf_center[0] - x_len / 2, 
                shelf_center[1] - y_len / 2,
                shelf_center[0] + x_len / 2, 
                shelf_center[1] + y_len / 2]
    return board_bbox

def _get_free_spaces_on_board(env, num_board, scene_idx=0, shift=0.02):
    item_bboxes = _get_product_bboxes(env, num_board, scene_idx)
    board_bbox = _get_shelf_bbox(env, scene_idx)
    board_bbox_eroded = [board_bbox[0] +shift, board_bbox[1] + shift, 
                    board_bbox[2] -shift, board_bbox[3] - shift]
    start_bbox, direction = get_start_point_and_direction(env.directions_to_shelf[scene_idx], board_bbox_eroded)
    return find_free_spaces(start_bbox, item_bboxes, board_bbox_eroded, marching_direction=direction)

def _build_free_spaces_on_board(env, num_board, scene_idx=0, shift=0.02):
    INTERBOARD_HEIGHT = 0.397
    board_height = 0.178 + num_board * 0.4
    free_spaces = _get_free_spaces_on_board(env, num_board, scene_idx, shift)
    if 'free_spaces' not in env.actors:
        env.actors['free_spaces'] = {}
    for i, free_region in enumerate(free_spaces):
        half_sizes = [(free_region[2] - free_region[0]) / 2, (free_region[3] - free_region[1]) / 2, (INTERBOARD_HEIGHT - 0.1)/ 2]
        pose = sapien.Pose(p=[(free_region[2] + free_region[0]) / 2, (free_region[3] + free_region[1]) / 2, board_height + 0.01 + half_sizes[2]])
        builder = env.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=half_sizes,
            material=sapien.render.RenderMaterial(
                base_color=[0, 1, 0, 0.5],
            ),
        )
        builder.set_scene_idxs([scene_idx])
        builder.set_initial_pose(pose)
        name = f"[ENV#{scene_idx}]_free_space_{num_board}_{i}"
        env.actors["free_spaces"][name] = builder.build_kinematic(name=name)

def find_free_spaces(start_rect, rect_world, shelf_rect, marching_direction=[1, 0], step=0.02):
    free_spaces = []
    if not is_rect_inside(start_rect, shelf_rect):
        print("Start rectangle is not inside the shelf!")
        return None
    current_rect = np.array(start_rect).copy()
    free_region = current_rect.copy()
    rect_growing = False
    while True:
        # print(current_rect, free_region, rect_growing)
        if not is_rect_inside(current_rect, shelf_rect):
            if rect_growing:
                free_spaces.append(free_region)
            break
        if is_in_collision(current_rect, rect_world):
            if rect_growing:
                free_spaces.append(free_region)
                rect_growing = False
        else:
            if rect_growing:
                free_region = merge_bboxes(free_region, current_rect)
            else:
                rect_growing = True
                free_region = current_rect.copy()
                
        current_rect = move_rect(current_rect, marching_direction, step)
    return free_spaces

def is_in_collision(rect, others, allow_touch: bool = False) -> bool:
    """
    Returns True if `rect` collides with any bbox in `others`, else False.
    allow_touch=True  -> touching edges/corners is NOT collision
    allow_touch=False -> touching counts as collision
    """
    rx1, ry1, rx2, ry2 = rect
    if not (rx2 > rx1 and ry2 > ry1):
        raise ValueError("Invalid rect")
    for ox1, oy1, ox2, oy2 in others:
        if not (ox2 > ox1 and oy2 > oy1):
            continue  # skip invalid boxes in others
        if allow_touch:
            hit = (rx1 < ox2) and (rx2 > ox1) and (ry1 < oy2) and (ry2 > oy1)
        else:
            hit = (rx1 <= ox2) and (rx2 >= ox1) and (ry1 <= oy2) and (ry2 >= oy1)
        if hit:
            return True
    return False
    
def is_rect_inside(inner, outer, allow_touch: bool = False) -> bool:
    """
    Returns True if `inner` is inside `outer`.
    allow_touch=True  -> edges may coincide (inside-or-on-boundary)
    allow_touch=False -> strict inside (no touching boundary)
    """
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    if not (ix2 > ix1 and iy2 > iy1 and ox2 > ox1 and oy2 > oy1):
        raise ValueError("Invalid rectangle coordinates")
    if allow_touch:
        return (ox1 <= ix1) and (oy1 <= iy1) and (ix2 <= ox2) and (iy2 <= oy2)
    else:
        return (ox1 < ix1) and (oy1 < iy1) and (ix2 < ox2) and (iy2 < oy2)

def move_rect(rect, marching_direction, step):
    return np.array(rect) + np.concatenate([marching_direction, marching_direction]) * step

def merge_bboxes(a_bbox, b_bbox):
    # print("WTF", a_bbox, b_bbox)
    return np.array([
        min(a_bbox[0], b_bbox[0]),
        min(a_bbox[1], b_bbox[1]),
        max(a_bbox[2], b_bbox[2]),
        max(a_bbox[3], b_bbox[3])
    ])

def get_start_point_and_direction(dir_to_shelf, shelf_bbox, size_h=0.1, size_w=0.1, eps=1e-3):
    shelf_corners = [
        [shelf_bbox[0] + eps + size_w / 2, shelf_bbox[1] + eps + size_h / 2], # bottom left
        [shelf_bbox[0] + eps + size_w / 2, shelf_bbox[3] - eps - size_h / 2], # top left
        [shelf_bbox[2] - eps - size_w / 2, shelf_bbox[3] - eps - size_h / 2], # top right
        [shelf_bbox[2] - eps - size_w / 2, shelf_bbox[1] + eps + size_h / 2] # bottom right
    ]
    if np.abs(np.dot(dir_to_shelf, [1, 0, 0]) - 1) < 1e-2:
        bbox_center = shelf_corners[1] # top left
        direction = [0, -1]
    if np.abs(np.dot(dir_to_shelf, [0, 1, 0]) - 1) < 1e-2:
        bbox_center = shelf_corners[0] # bottom left
        direction = [1, 0]
    if np.abs(np.dot(dir_to_shelf, [-1, 0, 0]) - 1) < 1e-2:
        bbox_center = shelf_corners[3] # bottom right
        direction = [0, 1]
    if np.abs(np.dot(dir_to_shelf, [0, -1, 0]) - 1) < 1e-2:
        bbox_center = shelf_corners[2] # top right
        direction = [-1, 0]
    start_bbox = [bbox_center[0] - size_w / 2, bbox_center[1] - size_h / 2,
                 bbox_center[0] + size_w / 2, bbox_center[1] + size_h / 2]

    return start_bbox, direction