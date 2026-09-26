"""Encode the clips of a manifest into LTX-2 video latents.

Usage:
    python -m tokenization.encode encode MANIFEST OUT_DIR --vae-checkpoint CKPT --resolution 960x544
    python -m tokenization.encode index OUT_DIR

Multi-GPU: start one process per GPU with RANK / WORLD_SIZE set (e.g. torchrun), or pass --rank / --world-size.
Videos are split between processes; run `index` once all processes are finished.
Reruns resume: videos with a done marker are skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from tokenization import frames as frame_io
from tokenization.clips import CLIP_FPS, CLIP_NUM_FRAMES, Clip, group_by_video, load_manifest
from tokenization.shards import VideoShardWriter, build_index, check_or_write_config, is_video_done
from tokenization.vae import VideoEncoder, preprocess_frames, validate_clip_shape

logger = logging.getLogger("tokenization")


def parse_resolution(value: str) -> tuple[int, int]:
    """'960x544' -> (width, height)."""
    try:
        width, height = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"resolution must look like 960x544, got {value!r}")
    return width, height


def encode_video(
    clips: list[Clip],
    encoder: VideoEncoder,
    out_dir: Path,
    width: int,
    height: int,
    batch_size: int,
    max_clips_per_shard: int,
    shard_metadata: dict,
) -> tuple[int, list[str]]:
    """Encode all clips of one video. Returns (number of clips encoded, ids of clips that could not be decoded)."""
    writer = VideoShardWriter(out_dir, clips[0].video_id, max_clips_per_shard, shard_metadata)
    failed: list[str] = []
    batch: list[tuple[Clip, torch.Tensor]] = []
    encoded = 0

    def flush() -> None:
        nonlocal encoded
        if not batch:
            return
        latents = encoder.encode(torch.stack([video for _, video in batch]))
        for (clip, _), clip_latents in zip(batch, latents):
            writer.add(clip.clip_id, clip.start_frame, clip_latents)
        encoded += len(batch)
        batch.clear()

    for clip, clip_frames in frame_io.iter_clip_frames(clips):
        if clip_frames is None:
            failed.append(clip.clip_id)
            continue
        batch.append((clip, preprocess_frames(clip_frames, height, width, device=encoder.device)))
        if len(batch) >= batch_size:
            flush()
    flush()
    writer.finish(failed_clip_ids=failed)
    return encoded, failed


def run_encode(
    manifest_path: Path,
    out_dir: Path,
    encoder: VideoEncoder,
    encoder_name: str,
    width: int,
    height: int,
    batch_size: int = 1,
    max_clips_per_shard: int = 256,
    rank: int = 0,
    world_size: int = 1,
    overwrite: bool = False,
    keep_going: bool = True,
    max_videos: Optional[int] = None,
) -> int:
    clips = load_manifest(manifest_path)
    if not clips:
        logger.error(f"No clips in {manifest_path}")
        return 1

    num_frames = {c.num_frames for c in clips}
    fps_values = {c.fps for c in clips}
    if len(num_frames) != 1 or len(fps_values) != 1:
        logger.error(f"All clips must share num_frames and fps, got {sorted(num_frames)} and {sorted(fps_values)}")
        return 1
    clip_frames, fps = num_frames.pop(), fps_values.pop()
    if (clip_frames, fps) != (CLIP_NUM_FRAMES, CLIP_FPS):
        logger.warning(
            f"Manifest clips are {clip_frames} frames at {fps} fps, expected {CLIP_NUM_FRAMES} at {CLIP_FPS}"
        )
    validate_clip_shape(clip_frames, height, width, encoder.scale_factors)

    config = {
        "encoder": encoder_name,
        "width": width,
        "height": height,
        "num_frames": clip_frames,
        "fps": fps,
        "scale_factors": list(encoder.scale_factors),
        "frame_decoder": frame_io.FRAME_DECODER_VERSION,
    }
    check_or_write_config(out_dir, config, overwrite=overwrite)

    groups = group_by_video(clips)
    video_ids = [vid for i, vid in enumerate(groups) if i % world_size == rank]
    if max_videos is not None:
        video_ids = video_ids[:max_videos]
    logger.info(
        f"rank {rank}/{world_size}: {len(video_ids)} of {len(groups)} videos, "
        f"{sum(len(groups[v]) for v in video_ids)} clips -> {out_dir}"
    )

    failures: list[tuple[str, str]] = []
    total_clips = 0
    start = time.time()
    for video_id in tqdm(video_ids, desc=f"rank {rank}", unit="video"):
        if not overwrite and is_video_done(out_dir, video_id):
            continue
        try:
            encoded, failed = encode_video(
                groups[video_id], encoder, out_dir, width, height, batch_size, max_clips_per_shard, config
            )
            total_clips += encoded
            if failed:
                logger.warning(f"{video_id}: {len(failed)} clips could not be decoded")
        except Exception as exc:  # the video has no done marker, so it is retried on the next run
            logger.error(f"{video_id}: {exc}")
            failures.append((video_id, str(exc)))
            if not keep_going:
                break

    elapsed = time.time() - start
    logger.info(f"rank {rank}: encoded {total_clips} clips in {elapsed:.0f}s")
    if world_size == 1:
        logger.info(f"Indexed {build_index(out_dir)} clips in {out_dir / 'index.jsonl'}")
    else:
        logger.info("Run `python -m tokenization.encode index OUT_DIR` after all ranks have finished.")
    if failures:
        logger.error(f"{len(failures)} videos failed: {', '.join(v for v, _ in failures[:20])}")
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="encode manifest clips into latent shards")
    enc.add_argument("manifest", type=Path, help="JSONL manifest, one clip per line")
    enc.add_argument("out_dir", type=Path)
    enc.add_argument("--vae-checkpoint", required=True, help="LTX-2 checkpoint or split video VAE .safetensors")
    enc.add_argument("--resolution", type=parse_resolution, required=True, help="WIDTHxHEIGHT, e.g. 960x544")
    enc.add_argument("--batch-size", type=int, default=1, help="clips per VAE forward pass")
    enc.add_argument("--max-clips-per-shard", type=int, default=256)
    enc.add_argument("--device", default=None, help="default: cuda:LOCAL_RANK if available, else cpu")
    enc.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    enc.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    enc.add_argument("--max-videos", type=int, default=None, help="only encode the first N videos of this rank")
    enc.add_argument("--overwrite", action="store_true", help="re-encode videos that are already done")
    enc.add_argument("--stop-on-error", action="store_true")

    idx = sub.add_parser("index", help="(re)build index.jsonl from finished videos")
    idx.add_argument("out_dir", type=Path)

    args = parser.parse_args(argv)

    if args.command == "index":
        logger.info(f"Indexed {build_index(args.out_dir)} clips in {args.out_dir / 'index.jsonl'}")
        return 0

    from tokenization.vae import LTXVideoEncoder

    device = args.device
    if device is None:
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    width, height = args.resolution
    encoder = LTXVideoEncoder(args.vae_checkpoint, device=device)
    return run_encode(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        encoder=encoder,
        encoder_name=Path(args.vae_checkpoint).name,
        width=width,
        height=height,
        batch_size=args.batch_size,
        max_clips_per_shard=args.max_clips_per_shard,
        rank=args.rank,
        world_size=args.world_size,
        overwrite=args.overwrite,
        keep_going=not args.stop_on_error,
        max_videos=args.max_videos,
    )


if __name__ == "__main__":
    sys.exit(main())
