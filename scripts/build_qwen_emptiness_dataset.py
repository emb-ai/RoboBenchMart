#!/usr/bin/env python3
"""Build a test Qwen2.5-VL finetuning dataset for shelf free-space detection.

Input layout (produced by scripts/gen_data_emptiness.py):
  <input_dir>/
    descriptions/seed=*.json
    images/seed=*_im.jpg  (preferred) or *_annot.jpg

Output layout:
  <output_dir>/
    train.jsonl
    val.jsonl
    manifest.json
    qa_spotcheck.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


CAMERAS = ("left_base_camera_link", "fetch_hand", "right_base_camera_link")
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}
BOARD_NUMBER_HUMAN_STR = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
}

GLOBAL_TEMPLATES = [
    "Find free space on the shelf.",
    "Locate all free spaces visible on the shelf.",
    "Mark empty regions where an item can be placed.",
    "Identify empty shelf areas in the image.",
    "Using this camera view, find free space on the shelf.",
    "From this image, mark all empty placement regions.",
]

BOARD_TEMPLATES = [
    "Find free space on the {ordinal} board of the shelf.",
    "Locate empty areas on board {board_number_human}.",
    "Locate empty areas on the {board_number_human_str} board.",
    "Show free spaces only on the {ordinal} shelf level.",
    "Which empty spots are on the {ordinal} board?",
    "In the current frame, locate free spaces on board {board_number_human}.",
]

STRICT_TEMPLATES = [
    "Return bounding box coordinates for all free spaces.",
    "Output a list of bboxes for free spaces on the {ordinal} board.",
    "Respond only with free-space bounding boxes.",
]


@dataclass
class Example:
    example_id: str
    source_json: str
    camera: str
    image_path: str
    prompt: str
    prompt_type: str
    board: str
    bboxes: list[list[int]]
    image_exists: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Qwen2.5-VL test dataset for shelf free-space detection."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("generated_envs/cluttered/emptiness_data"),
        help="Path to emptiness_data directory with descriptions/ and images/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/finetune_qwen_emptiness_test"),
        help="Output directory for train/val JSONL and reports.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio in [0, 1).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of examples to export after generation.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Allow exporting samples with missing image files (for dry runs).",
    )
    return parser.parse_args()


def safe_int_bbox(raw_bbox: Any) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in raw_bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def validate_bbox_in_image(bbox: list[int], width: int, height: int) -> bool:
    x1, y1, x2, y2 = bbox
    return 0 <= x1 < x2 <= width - 1 and 0 <= y1 < y2 <= height - 1


def load_image_shape(image_path: Path) -> tuple[int, int] | None:
    if not image_path.exists() or Image is None:
        return None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            return width, height
    except OSError:
        return None


def choose_image_path(input_dir: Path, stem: str) -> tuple[Path, bool]:
    im_path = input_dir / "images" / f"{stem}_im.jpg"
    if im_path.exists():
        return im_path, True
    annot_path = input_dir / "images" / f"{stem}_annot.jpg"
    if annot_path.exists():
        return annot_path, True
    return im_path, False


def build_prompt(template: str, board: int | None) -> str:
    if board is None:
        return template
    return template.format(
        ordinal=ORDINAL[board],
        board_number_human=board,
        board_number_human_str=BOARD_NUMBER_HUMAN_STR[board],
    )


def to_qwen_record(example: Example) -> dict[str, Any]:
    answer_payload = {
        "bboxes": example.bboxes,
        "board": example.board,
    }
    answer_text = json.dumps(answer_payload, ensure_ascii=True)
    return {
        "id": example.example_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example.image_path},
                    {"type": "text", "text": example.prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_text},
                ],
            },
        ],
        "metadata": {
            "source_json": example.source_json,
            "camera": example.camera,
            "prompt_type": example.prompt_type,
            "board": example.board,
            "image_exists": example.image_exists,
        },
    }


def generate_examples(
    input_dir: Path,
    rng: random.Random,
    allow_missing_images: bool,
) -> tuple[list[Example], dict[str, Any]]:
    desc_dir = input_dir / "descriptions"
    json_files = sorted(desc_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No description JSON files found in {desc_dir}")

    stats: dict[str, Any] = {
        "scenes": len(json_files),
        "examples_before_cap": 0,
        "missing_images": 0,
        "bbox_out_of_bounds": 0,
        "bbox_invalid": 0,
        "free_space_id_mismatches": 0,
        "camera_examples": Counter(),
        "prompt_type_examples": Counter(),
        "board_examples": Counter(),
        "negative_examples": 0,
        "source_distribution": Counter(),
    }

    all_examples: list[Example] = []

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            sample = json.load(f)

        stem = json_path.stem
        image_path, image_exists = choose_image_path(input_dir, stem)
        if not image_exists:
            stats["missing_images"] += 1
            if not allow_missing_images:
                raise FileNotFoundError(
                    f"Missing image for {json_path.name}: expected {image_path}"
                )

        image_shape = load_image_shape(image_path) if image_exists else None
        free_space_to_board: dict[str, int] = {}
        for entry in sample.get("free_spaces", []):
            free_space_to_board[str(entry["free_space"])] = int(entry["num_board"])

        camera_to_boxes: dict[str, dict[str, list[int]]] = sample.get("bboxes", {})

        for camera in CAMERAS:
            raw_camera_boxes = camera_to_boxes.get(camera, {})
            board_to_boxes: dict[int, list[list[int]]] = defaultdict(list)
            global_boxes: list[list[int]] = []

            for free_space_id, raw_bbox in raw_camera_boxes.items():
                key = str(free_space_id)
                if key not in free_space_to_board:
                    stats["free_space_id_mismatches"] += 1
                    continue

                bbox = safe_int_bbox(raw_bbox)
                if bbox is None:
                    stats["bbox_invalid"] += 1
                    continue

                if image_shape is not None:
                    width, height = image_shape
                    if not validate_bbox_in_image(bbox, width, height):
                        stats["bbox_out_of_bounds"] += 1
                        continue

                board_human = free_space_to_board[key] + 1
                board_to_boxes[board_human].append(bbox)
                global_boxes.append(bbox)

            global_boxes = sorted(global_boxes, key=lambda b: (b[1], b[0], b[3], b[2]))
            for board in range(1, 6):
                board_to_boxes[board] = sorted(
                    board_to_boxes.get(board, []), key=lambda b: (b[1], b[0], b[3], b[2])
                )

            # Per (scene, camera) build 8 prompts:
            #   3 global + 4 board-specific + 1 strict.
            global_template_pool = GLOBAL_TEMPLATES
            board_template_pool = BOARD_TEMPLATES
            strict_template_pool = STRICT_TEMPLATES

            selected_global_templates = rng.sample(global_template_pool, k=3)
            selected_boards = rng.sample([1, 2, 3, 4, 5], k=4)

            image_path_str = str(image_path.resolve())
            source_json_str = str(json_path.resolve())

            for idx, template in enumerate(selected_global_templates):
                prompt = build_prompt(template, board=None)
                all_examples.append(
                    Example(
                        example_id=f"{stem}:{camera}:global:{idx}",
                        source_json=source_json_str,
                        camera=camera,
                        image_path=image_path_str,
                        prompt=prompt,
                        prompt_type="global",
                        board="all",
                        bboxes=global_boxes,
                        image_exists=image_exists,
                    )
                )

            for board in selected_boards:
                template = rng.choice(board_template_pool)
                prompt = build_prompt(template, board=board)
                all_examples.append(
                    Example(
                        example_id=f"{stem}:{camera}:board:{board}",
                        source_json=source_json_str,
                        camera=camera,
                        image_path=image_path_str,
                        prompt=prompt,
                        prompt_type="board_specific",
                        board=str(board),
                        bboxes=board_to_boxes[board],
                        image_exists=image_exists,
                    )
                )

            strict_template = rng.choice(strict_template_pool)
            strict_board = None
            strict_board_label = "all"
            strict_boxes = global_boxes
            if "{ordinal}" in strict_template:
                strict_board = rng.choice([1, 2, 3, 4, 5])
                strict_board_label = str(strict_board)
                strict_boxes = board_to_boxes[strict_board]
            prompt = build_prompt(strict_template, board=strict_board)
            all_examples.append(
                Example(
                    example_id=f"{stem}:{camera}:strict",
                    source_json=source_json_str,
                    camera=camera,
                    image_path=image_path_str,
                    prompt=prompt,
                    prompt_type="strict",
                    board=strict_board_label,
                    bboxes=strict_boxes,
                    image_exists=image_exists,
                )
            )

    for ex in all_examples:
        stats["camera_examples"][ex.camera] += 1
        stats["prompt_type_examples"][ex.prompt_type] += 1
        stats["board_examples"][ex.board] += 1
        stats["source_distribution"][Path(ex.source_json).name] += 1
        if not ex.bboxes:
            stats["negative_examples"] += 1

    stats["examples_before_cap"] = len(all_examples)
    return all_examples, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_spotcheck(path: Path, examples: list[Example], rng: random.Random) -> None:
    rows: list[dict[str, Any]] = []
    if not examples:
        write_jsonl(path, rows)
        return
    sample_size = min(20, len(examples))
    for ex in rng.sample(examples, k=sample_size):
        rows.append(
            {
                "id": ex.example_id,
                "source_json": ex.source_json,
                "camera": ex.camera,
                "prompt_type": ex.prompt_type,
                "prompt": ex.prompt,
                "board": ex.board,
                "num_bboxes": len(ex.bboxes),
                "first_bbox": ex.bboxes[0] if ex.bboxes else None,
                "image_exists": ex.image_exists,
            }
        )
    write_jsonl(path, rows)


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1).")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    rng = random.Random(args.seed)
    examples, stats = generate_examples(
        input_dir=args.input_dir,
        rng=rng,
        allow_missing_images=args.allow_missing_images,
    )

    rng.shuffle(examples)
    if len(examples) > args.max_samples:
        examples = examples[: args.max_samples]

    split_idx = int(len(examples) * (1.0 - args.val_ratio))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    train_rows = [to_qwen_record(ex) for ex in train_examples]
    val_rows = [to_qwen_record(ex) for ex in val_examples]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_spotcheck(output_dir / "qa_spotcheck.jsonl", examples, rng)

    manifest = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "max_samples": args.max_samples,
        "allow_missing_images": args.allow_missing_images,
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "total_exported": len(train_rows) + len(val_rows),
            "total_generated_before_cap": stats["examples_before_cap"],
        },
        "qa": {
            "missing_images": stats["missing_images"],
            "free_space_id_mismatches": stats["free_space_id_mismatches"],
            "bbox_invalid": stats["bbox_invalid"],
            "bbox_out_of_bounds": stats["bbox_out_of_bounds"],
            "negative_examples": stats["negative_examples"],
        },
        "distribution": {
            "camera_examples": dict(stats["camera_examples"]),
            "prompt_type_examples": dict(stats["prompt_type_examples"]),
            "board_examples": dict(stats["board_examples"]),
            "source_distribution": dict(stats["source_distribution"]),
        },
        "templates": {
            "global": GLOBAL_TEMPLATES,
            "board_specific": BOARD_TEMPLATES,
            "strict": STRICT_TEMPLATES,
        },
        "output_contract": {
            "assistant_json": {"bboxes": [[0, 0, 1, 1]], "board": "all|1|2|3|4|5"},
            "rules": [
                "Coordinates are pixel integers in input image space.",
                "board='all' for global queries; otherwise 1..5.",
                "Use empty list for no free-space result.",
            ],
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(f"Exported {len(train_rows)} train and {len(val_rows)} val examples.")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Spotcheck: {output_dir / 'qa_spotcheck.jsonl'}")


if __name__ == "__main__":
    main()
