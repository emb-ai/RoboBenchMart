import base64
import inspect
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from dsynth.envs import DarkstoreContinuousBaseEnv

def prepare_observations(env: DarkstoreContinuousBaseEnv) -> Dict[str, Any]:
    obs = env.base_env.get_obs()

    result = {}
    raw_images = []
    annotated_images = []

    cameras = ["left_base_camera_link", "fetch_hand", "right_base_camera_link"]
    for camera in cameras:
        camera_data = obs["sensor_data"][camera]
        image = camera_data["rgb"][0].cpu().numpy()[:, :, ::-1]
        seg = camera_data["segmentation"][0].cpu().numpy()[..., 0]

        scale = 768 / image.shape[0]
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        seg = cv2.resize(seg, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        annotated_image = annotate_image(image, seg, env)

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
    result["scene_description"] = prepare_scene_description(env)

    return result


def image_to_base64(image: np.ndarray) -> Optional[str]:
    success, encoded_image = cv2.imencode(".png", image)
    if not success:
        return None
    encoded_bytes = encoded_image.tobytes()
    return base64.b64encode(encoded_bytes).decode("utf-8")


def prepare_scene_description(env: DarkstoreContinuousBaseEnv) -> Dict[str, Any]:
    return {
        "robot": get_robot_description(env),
        "shelf": get_shelf_description(env),
        "products": get_products_description(env),
    }


def get_robot_description(env: DarkstoreContinuousBaseEnv) -> Dict[str, Any]:
    return {
        "base_position": [round(x, 3) for x in env.unwrapped.agent.base_link.pose.sp.p],
        "ee_position": [round(x, 3) for x in env.unwrapped.agent.tcp.pose.sp.p],
        "joints": [
            {
                "name": joint.name,
                "qpos": round(joint.qpos.item(), 3),
                "limits": [round(x, 3) for x in joint.limits[0].numpy()],
            }
            for joint in env.unwrapped.agent.robot.active_joints
            if "root" not in joint.name
        ],
    }


def get_shelf_description(env: DarkstoreContinuousBaseEnv) -> Dict[str, Any]:
    shelf_name = env.unwrapped.active_shelves[0][0]
    shelf_pos = env.unwrapped.actors["fixtures"]["shelves"][shelf_name].pose.sp.p

    return {
        "position": [round(float(x), 3) for x in shelf_pos],
    }


def get_products_description(env: DarkstoreContinuousBaseEnv) -> List[Dict[str, Any]]:
    actor_by_product_id = {
        actor.per_scene_id[0].item(): actor
        for actor in env.unwrapped.actors["products"].values()
    }

    product_name_by_actor_name = (
        env.unwrapped.products_df.set_index("actor_name")["product_name"].to_dict()
    )

    return [
        {
            "product_id": product_id,
            "product_name": product_name_by_actor_name.get(actor.name),
        }
        for product_id in extract_reachable_products(env)
        if (actor := actor_by_product_id.get(product_id)) is not None
    ]


def extract_reachable_products(env: DarkstoreContinuousBaseEnv) -> List[int]:
    products: List[int] = []

    for product in env.unwrapped.actors["products"].values():
        if not product.name.endswith("0"):
            continue
        products.append(product.per_scene_id[0].item())

    return products


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


def annotate_image(
    image: np.ndarray, segmentation: np.ndarray, env: DarkstoreContinuousBaseEnv
) -> np.ndarray:
    products = extract_reachable_products(env)
    palette = build_palette(products)
    output = image.copy()

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    bg_color = (255, 255, 255)

    for product_id in products:
        bbox = build_bbox(segmentation, product_id)
        if bbox is None:
            continue

        x_min, y_min, x_max, y_max = bbox
        color = tuple(map(int, palette[product_id]))

        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), color, 1)

        label = str(product_id)
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

    return output


def build_palette(product_ids: List[int], seed: int = 42) -> np.ndarray:
    max_product_id = max(product_ids)
    palette_size = max(128, max_product_id + 1)

    rng = np.random.RandomState(seed)
    palette = rng.randint(20, 235, size=(palette_size, 3), dtype=np.uint8)

    return palette


def get_function_description(name: str, function) -> str:
    sig = inspect.signature(function)
    params = [
        f"{n}: {p.annotation.__name__ if p.annotation is not inspect.Parameter.empty else 'Any'}"
        for n, p in sig.parameters.items()
        if n != "self"
    ]
    sig_str = f"{name}({', '.join(params)})"
    doc = inspect.getdoc(function) or ""
    lines = [f"  {line}" for line in doc.split("\n")]
    return f"'{sig_str}'\n" + "\n".join(lines)


def build_skills_description(controller_cls) -> str:
    skills = []
    for name, method in inspect.getmembers(
        controller_cls, predicate=inspect.isfunction
    ):
        if name.startswith("_"):
            continue
        skills.append(get_function_description(name, method))
    return "\n\n".join(f"{i}. {s}" for i, s in enumerate(skills, start=1))


def draw_normalized_bbox(
    image: np.ndarray, bbox: dict, color: tuple = (0, 255, 0), thickness: int = 2
) -> np.ndarray:
    h, w = image.shape[:2]
    pt1 = (int(bbox["x_min"] * w), int(bbox["y_min"] * h))
    pt2 = (int(bbox["x_max"] * w), int(bbox["y_max"] * h))
    output = image.copy()
    cv2.rectangle(output, pt1, pt2, color, thickness)
    return output