#!/usr/bin/env python3
"""Local Qwen-VL evaluation for shelf free-space bbox detection.

This script evaluates `val.jsonl` samples produced by
`scripts/build_qwen_emptiness_dataset.py` and reports detection quality:
precision/recall/F1 with greedy IoU@0.5 matching.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Qwen-VL inference and evaluate on emptiness val subset."
    )
    parser.add_argument(
        "--model-id-or-path",
        type=str,
        required=True,
        help="HF model id or local model path.",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("data/finetune_qwen_emptiness_test/val.jsonl"),
        help="Validation JSONL in Qwen chat format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/finetune_qwen_emptiness_test/eval_qwen_local"),
        help="Directory for metrics and prediction artifacts.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on number of evaluated samples (0 means all).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("float16", "float32"),
        default="float16",
        help="Model dtype for loading.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0 means deterministic).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for positive match.",
    )
    parser.add_argument(
        "--mock-mode",
        type=str,
        choices=("none", "empty", "echo_gt"),
        default="none",
        help=(
            "Debug mode. 'none': real model inference; "
            "'empty': always predict empty boxes; 'echo_gt': copy GT answer."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Validation file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def find_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_json_payload(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_text.strip()
    if not text:
        return None, "empty response"

    candidates = [text]
    if text.startswith("```"):
        stripped = text.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        candidates.append(stripped)

    json_substring = find_first_json_object(text)
    if json_substring is not None:
        candidates.append(json_substring)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload, None
        except json.JSONDecodeError:
            continue

    return None, "no parseable JSON object in response"


def normalize_board(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "all":
            return "all"
        if value.isdigit():
            number = int(value)
            if 1 <= number <= 5:
                return str(number)
        return None
    if isinstance(value, int):
        if 1 <= value <= 5:
            return str(value)
    return None


def normalize_bbox(raw_bbox: Any) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_payload(payload: dict[str, Any]) -> tuple[list[list[int]], str | None]:
    board = normalize_board(payload.get("board"))
    bboxes: list[list[int]] = []
    for item in payload.get("bboxes", []):
        bbox = normalize_bbox(item)
        if bbox is not None:
            bboxes.append(bbox)
    bboxes.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    return bboxes, board


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float((ax2 - ax1) * (ay2 - ay1))
    area_b = float((bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def greedy_match(
    pred_boxes: list[list[int]],
    gt_boxes: list[list[int]],
    iou_threshold: float,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = bbox_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))
    candidates.sort(reverse=True, key=lambda item: item[0])

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[dict[str, Any]] = []

    for iou, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append({"pred_idx": pred_idx, "gt_idx": gt_idx, "iou": iou})

    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matches


def get_user_prompt_and_image(sample: dict[str, Any]) -> tuple[str, str]:
    user_msg = sample["messages"][0]
    content = user_msg["content"]
    image_path = None
    prompt = None
    for item in content:
        if item.get("type") == "image":
            image_path = item.get("image")
        elif item.get("type") == "text":
            prompt = item.get("text")
    if not isinstance(image_path, str) or not isinstance(prompt, str):
        raise ValueError(f"Invalid user content in sample {sample.get('id')}")
    return prompt, image_path


def get_ground_truth(sample: dict[str, Any]) -> tuple[list[list[int]], str]:
    raw_text = sample["messages"][1]["content"][0]["text"]
    payload = json.loads(raw_text)
    gt_boxes, gt_board = normalize_payload(payload)
    if gt_board is None:
        raise ValueError(f"Invalid GT board in sample {sample.get('id')}")
    return gt_boxes, gt_board


def resolve_device(device_arg: str) -> str:
    import torch

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def load_model_and_processor(
    model_id_or_path: str,
    device: str,
    torch_dtype: Any,
) -> tuple[Any, Any]:
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)
    model = ModelClass.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, processor


def run_model_inference(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if hasattr(processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = processor(
                text=[text], images=[image], return_tensors="pt", padding=True
            )
        else:
            model_inputs = processor(
                text=[prompt], images=[image], return_tensors="pt", padding=True
            )

    device = next(model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        generated = model.generate(**model_inputs, **generate_kwargs)

    # Keep only newly generated tokens when possible.
    if "input_ids" in model_inputs:
        generated = generated[:, model_inputs["input_ids"].shape[-1] :]

    response_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return response_text.strip()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if not (0.0 <= args.iou_threshold <= 1.0):
        raise ValueError("--iou-threshold must be in [0, 1]")

    rng = random.Random(args.seed)
    rows = read_jsonl(args.val_jsonl)
    rng.shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    model = None
    processor = None
    if args.mock_mode == "none":
        device = resolve_device(args.device)
        torch_dtype = resolve_torch_dtype(args.dtype)
        model, processor = load_model_and_processor(
            model_id_or_path=args.model_id_or_path,
            device=device,
            torch_dtype=torch_dtype,
        )
    else:
        device = "mock"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tp_total = 0
    fp_total = 0
    fn_total = 0
    board_match_total = 0
    parse_failures = 0
    valid_json = 0
    missing_images = 0
    skipped_samples = 0
    total_latency_s = 0.0

    prediction_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    progress = tqdm(rows, desc="Evaluating", dynamic_ncols=True)
    for row in progress:
        sample_id = row.get("id")
        prompt, image_path_str = get_user_prompt_and_image(row)
        image_path = Path(image_path_str)

        if not image_path.exists():
            missing_images += 1
            skipped_samples += 1
            failure_rows.append(
                {
                    "id": sample_id,
                    "type": "missing_image",
                    "image_path": str(image_path),
                }
            )
            continue

        gt_boxes, gt_board = get_ground_truth(row)

        if args.mock_mode == "echo_gt":
            raw_prediction = row["messages"][1]["content"][0]["text"]
            latency_s = 0.0
        elif args.mock_mode == "empty":
            raw_prediction = json.dumps({"bboxes": [], "board": gt_board})
            latency_s = 0.0
        else:
            t0 = time.perf_counter()
            raw_prediction = run_model_inference(
                model=model,
                processor=processor,
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            latency_s = time.perf_counter() - t0

        total_latency_s += latency_s

        pred_payload, parse_error = parse_json_payload(raw_prediction)
        if pred_payload is None:
            parse_failures += 1
            pred_boxes: list[list[int]] = []
            pred_board = None
        else:
            valid_json += 1
            pred_boxes, pred_board = normalize_payload(pred_payload)

        board_match = pred_board == gt_board
        if board_match:
            board_match_total += 1

        tp, fp, fn, matches = greedy_match(
            pred_boxes=pred_boxes, gt_boxes=gt_boxes, iou_threshold=args.iou_threshold
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn

        sample_precision = safe_div(tp, tp + fp)
        sample_recall = safe_div(tp, tp + fn)
        sample_f1 = safe_div(2 * sample_precision * sample_recall, sample_precision + sample_recall)

        pred_row = {
            "id": sample_id,
            "image_path": str(image_path),
            "prompt": prompt,
            "raw_prediction": raw_prediction,
            "parse_error": parse_error,
            "pred_board": pred_board,
            "gt_board": gt_board,
            "board_match": board_match,
            "pred_bboxes": pred_boxes,
            "gt_bboxes": gt_boxes,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "matches": matches,
            "sample_precision": sample_precision,
            "sample_recall": sample_recall,
            "sample_f1": sample_f1,
            "latency_s": latency_s,
        }
        prediction_rows.append(pred_row)

        if parse_error is not None:
            failure_rows.append(
                {
                    "id": sample_id,
                    "type": "parse_failure",
                    "parse_error": parse_error,
                    "raw_prediction": raw_prediction,
                    "prompt": prompt,
                }
            )
        elif fp + fn > 0:
            failure_rows.append(
                {
                    "id": sample_id,
                    "type": "quality_error",
                    "fp": fp,
                    "fn": fn,
                    "sample_f1": sample_f1,
                    "pred_bboxes": pred_boxes,
                    "gt_bboxes": gt_boxes,
                    "prompt": prompt,
                }
            )

        running_precision = safe_div(tp_total, tp_total + fp_total)
        running_recall = safe_div(tp_total, tp_total + fn_total)
        running_f1 = safe_div(
            2 * running_precision * running_recall, running_precision + running_recall
        )
        progress.set_postfix(
            {
                "f1": f"{running_f1:.3f}",
                "parse_fail": parse_failures,
                "missing_img": missing_images,
            }
        )

    evaluated = len(prediction_rows)
    precision = safe_div(tp_total, tp_total + fp_total)
    recall = safe_div(tp_total, tp_total + fn_total)
    f1 = safe_div(2 * precision * recall, precision + recall)

    metrics = {
        "config": {
            "model_id_or_path": args.model_id_or_path,
            "val_jsonl": str(args.val_jsonl.resolve()),
            "output_dir": str(output_dir.resolve()),
            "max_samples": args.max_samples,
            "seed": args.seed,
            "device": device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "iou_threshold": args.iou_threshold,
            "mock_mode": args.mock_mode,
        },
        "counts": {
            "total_rows_loaded": len(rows),
            "evaluated_samples": evaluated,
            "skipped_samples": skipped_samples,
            "missing_images": missing_images,
            "valid_json_predictions": valid_json,
            "parse_failures": parse_failures,
        },
        "detection": {
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": precision,
            "recall": recall,
            "f1_iou50": f1,
        },
        "aux_metrics": {
            "valid_json_rate": safe_div(valid_json, evaluated),
            "parse_failure_rate": safe_div(parse_failures, evaluated),
            "board_match_rate": safe_div(board_match_total, evaluated),
            "mean_latency_s": safe_div(total_latency_s, evaluated),
        },
    }

    # Keep top difficult quality errors and all parse/missing failures.
    quality_errors = [x for x in failure_rows if x.get("type") == "quality_error"]
    quality_errors.sort(key=lambda x: (x.get("fp", 0) + x.get("fn", 0)), reverse=True)
    major_failures = [x for x in failure_rows if x.get("type") != "quality_error"]
    major_failures.extend(quality_errors[:100])

    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(output_dir / "failures.jsonl", major_failures)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    print(f"Evaluated samples: {evaluated}")
    print(f"F1@IoU{args.iou_threshold:.2f}: {f1:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Metrics written to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
