#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export MapAnything camera poses to JSONL instead of GLB"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images for reconstruction",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        help="Source video id to include in each JSON record",
    )
    parser.add_argument(
        "--segment_start_time",
        type=float,
        default=0.0,
        help="Segment start time in seconds within the original video",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame sampling FPS used before inference",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    return parser


def sorted_frame_names(image_folder: Path) -> list[str]:
    return [
        path.name
        for path in sorted(image_folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def rotmat_to_quat_wxyz(rotation: np.ndarray) -> list[float]:
    r = rotation.astype(np.float64)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero")
    quat /= norm
    return [float(v) for v in quat]


def main() -> int:
    args = get_parser().parse_args()

    image_folder = Path(args.image_folder)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")

    model = MapAnything.from_pretrained(model_name).to(device)

    print(f"Loading images from: {image_folder}")
    views = load_images(str(image_folder))
    frame_names = sorted_frame_names(image_folder)
    print(f"Loaded {len(views)} views")

    print("Running inference...")
    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    print("Inference complete!")

    if len(frame_names) != len(outputs):
        print(
            f"Warning: found {len(frame_names)} frame files but model returned {len(outputs)} outputs. "
            "Continuing with index based timestamps."
        )

    with output_path.open("w", encoding="utf-8") as f:
        for frame_index, pred in enumerate(outputs):
            camera_pose = pred["camera_poses"][0].detach().cpu().numpy()
            translation = [float(v) for v in camera_pose[:3, 3]]
            rotation = rotmat_to_quat_wxyz(camera_pose[:3, :3])
            timestamp_us = int(round((args.segment_start_time + frame_index / args.fps) * 1_000_000))

            record = {
                "video_id": args.video_id,
                "frame_index": frame_index,
                "frame_name": frame_names[frame_index] if frame_index < len(frame_names) else None,
                "timestamp_us": timestamp_us,
                "translation": translation,
                "rotation": rotation,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved pose JSONL: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
