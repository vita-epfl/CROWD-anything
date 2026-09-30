"""Encode the clips of a manifest into LTX-2 video latents.

Usage:
    python -m tokenization.encode encode MANIFEST OUT_ROOT --video-root VIDEOS --vae-checkpoint CKPT \
        --resolution 960x544 512x288
    python -m tokenization.encode index OUT_ROOT

MANIFEST is a wm-data-manifest sample manifest (parquet), e.g. processed_datasets/crowd/manifest/samples_24fps.parquet;
VIDEOS is the directory its source_video_path values are relative to.

Each resolution is written to its own directory OUT_ROOT/<width>x<height>. Frames are decoded once per video and
encoded at every requested resolution; a resolution can be added later without touching the others.

Multi-GPU: start one process per GPU with RANK / WORLD_SIZE set (e.g. torchrun), or pass --rank / --world-size.
Videos are split between processes; run `index` once all processes are finished.
Reruns resume: videos already done at a resolution are skipped for that resolution.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from tokenization import frames as frame_io
from tokenization.clips import Clip, group_by_video, load_manifest
from tokenization.shards import ProgressLog, VideoShardWriter, build_index, check_or_write_config, load_done_videos
from tokenization.vae import VideoEncoder, preprocess_frames, validate_clip_shape

logger = logging.getLogger("tokenization")


def parse_resolution(value: str) -> tuple[int, int]:
    """'960x544' -> (width, height)."""
    try:
        width, height = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"resolution must look like 960x544, got {value!r}")
    return width, height


@dataclass
class ResolutionTarget:
    width: int
    height: int
    out_dir: Path
    config: dict
    progress: ProgressLog
    done: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return f"{self.width}x{self.height}"


def encode_video(
    clips: list[Clip],
    encoder: VideoEncoder,
    targets: list[ResolutionTarget],
    batch_size: int,
    max_clips_per_shard: int,
) -> tuple[int, list[str]]:
    """
    Decode the clips of one video once and encode them at every target resolution.
    Returns (number of clips encoded, ids of clips that could not be decoded).
    """
    video_id = clips[0].video_id
    writers = {}
    for target in targets:
        target.progress.append({"video_id": video_id, "status": "started"})
        writers[target.name] = VideoShardWriter(target.out_dir, video_id, max_clips_per_shard, target.config)
    batches: dict[str, list[tuple[Clip, torch.Tensor]]] = {target.name: [] for target in targets}
    failed: list[str] = []
    encoded = 0

    def flush(name: str) -> None:
        batch = batches[name]
        if not batch:
            return
        latents = encoder.encode(torch.stack([video for _, video in batch]))
        for (clip, _), clip_latents in zip(batch, latents):
            writers[name].add(clip.clip_id, clip.start_frame, clip_latents)
        batch.clear()

    for clip, clip_frames in frame_io.iter_clip_frames(clips):
        if clip_frames is None:
            failed.append(clip.clip_id)
            continue
        for target in targets:
            video = preprocess_frames(clip_frames, target.height, target.width, device=encoder.device)
            batches[target.name].append((clip, video))
            if len(batches[target.name]) >= batch_size:
                flush(target.name)
        encoded += 1

    for target in targets:
        flush(target.name)
        target.progress.append(writers[target.name].finish(failed_clip_ids=failed))
        target.done.add(video_id)
    return encoded, failed


def run_encode(
    manifest_path: Path,
    video_root: Path,
    out_root: Path,
    encoder: VideoEncoder,
    encoder_name: str,
    resolutions: list[tuple[int, int]],
    batch_size: int = 1,
    max_clips_per_shard: int = 1024,
    rank: int = 0,
    world_size: int = 1,
    overwrite: bool = False,
    keep_going: bool = True,
    max_videos: Optional[int] = None,
    countries: Optional[list[str]] = None,
) -> int:
    """Encode the manifest clips at each (width, height) in `resolutions` into out_root/<width>x<height>."""
    clips = load_manifest(manifest_path, video_root, filters=[("country", "in", countries)] if countries else None)
    if not clips:
        logger.error(f"No clips in {manifest_path}")
        return 1
    if len(set(resolutions)) != len(resolutions):
        logger.error(f"Duplicate resolutions in {resolutions}")
        return 1

    num_frames = {c.num_frames for c in clips}
    fps_values = {c.fps for c in clips}
    if len(num_frames) != 1 or len(fps_values) != 1:
        logger.error(f"All clips must share num_frames and fps, got {sorted(num_frames)} and {sorted(fps_values)}")
        return 1
    clip_frames, fps = num_frames.pop(), fps_values.pop()

    targets: list[ResolutionTarget] = []
    for width, height in resolutions:
        validate_clip_shape(clip_frames, height, width, encoder.scale_factors)
        out_dir = Path(out_root) / f"{width}x{height}"
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
        done = set() if overwrite else set(load_done_videos(out_dir))
        targets.append(ResolutionTarget(width, height, out_dir, config, ProgressLog(out_dir, rank), done))

    groups = group_by_video(clips)
    video_ids = [vid for i, vid in enumerate(groups) if i % world_size == rank]
    if max_videos is not None:
        video_ids = video_ids[:max_videos]
    logger.info(
        f"rank {rank}/{world_size}: {len(video_ids)} of {len(groups)} videos, "
        f"{sum(len(groups[v]) for v in video_ids)} clips, resolutions {', '.join(t.name for t in targets)} "
        f"-> {out_root}"
    )

    failures: list[tuple[str, str]] = []
    total_clips = 0
    start = time.time()
    for video_id in tqdm(video_ids, desc=f"rank {rank}", unit="video"):
        pending = [t for t in targets if video_id not in t.done]
        if not pending:
            continue
        try:
            encoded, failed = encode_video(groups[video_id], encoder, pending, batch_size, max_clips_per_shard)
            total_clips += encoded
            if failed:
                logger.warning(f"{video_id}: {len(failed)} clips could not be decoded")
        except Exception as exc:  # no "done" line is logged, so the video is retried on the next run
            logger.error(f"{video_id}: {exc}")
            failures.append((video_id, str(exc)))
            if not keep_going:
                break

    elapsed = time.time() - start
    logger.info(f"rank {rank}: encoded {total_clips} clips in {elapsed:.0f}s")
    if world_size == 1:
        for target in targets:
            logger.info(f"Indexed {build_index(target.out_dir)} clips in {target.out_dir / 'index.jsonl'}")
    else:
        logger.info("Run `python -m tokenization.encode index OUT_ROOT` after all ranks have finished.")
    if failures:
        logger.error(f"{len(failures)} videos failed: {', '.join(v for v, _ in failures[:20])}")
        return 1
    return 0


def index_all(out_root: Path) -> int:
    """Rebuild index.jsonl of every resolution directory under out_root."""
    out_dirs = sorted(p.parent for p in Path(out_root).glob("*/encoding_config.json"))
    if not out_dirs:
        logger.error(f"No resolution directories with encoding_config.json in {out_root}")
        return 1
    for out_dir in out_dirs:
        logger.info(f"Indexed {build_index(out_dir)} clips in {out_dir / 'index.jsonl'}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="encode manifest clips into latent shards")
    enc.add_argument("manifest", type=Path, help="wm-data-manifest sample manifest (parquet file or partition dir)")
    enc.add_argument("out_root", type=Path, help="each resolution is written to OUT_ROOT/<width>x<height>")
    enc.add_argument("--video-root", type=Path, required=True, help="directory source_video_path is relative to")
    enc.add_argument("--countries", nargs="+", default=None, help="only encode clips of these countries")
    enc.add_argument("--vae-checkpoint", required=True, help="LTX-2 checkpoint or split video VAE .safetensors")
    enc.add_argument(
        "--resolution", type=parse_resolution, nargs="+", required=True,
        help="one or more WIDTHxHEIGHT values, multiples of 32, e.g. 960x544 512x288",
    )
    enc.add_argument("--batch-size", type=int, default=1, help="clips per VAE forward pass")
    enc.add_argument("--max-clips-per-shard", type=int, default=1024)
    enc.add_argument("--device", default=None, help="default: cuda:LOCAL_RANK if available, else cpu")
    enc.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    enc.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    enc.add_argument("--max-videos", type=int, default=None, help="only encode the first N videos of this rank")
    enc.add_argument("--overwrite", action="store_true", help="re-encode videos that are already done")
    enc.add_argument("--stop-on-error", action="store_true")

    idx = sub.add_parser("index", help="(re)build index.jsonl of every resolution from finished videos")
    idx.add_argument("out_root", type=Path)

    args = parser.parse_args(argv)

    if args.command == "index":
        return index_all(args.out_root)

    from tokenization.vae import LTXVideoEncoder

    device = args.device
    if device is None:
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    encoder = LTXVideoEncoder(args.vae_checkpoint, device=device)
    return run_encode(
        manifest_path=args.manifest,
        video_root=args.video_root,
        out_root=args.out_root,
        encoder=encoder,
        encoder_name=Path(args.vae_checkpoint).name,
        resolutions=args.resolution,
        batch_size=args.batch_size,
        max_clips_per_shard=args.max_clips_per_shard,
        rank=args.rank,
        world_size=args.world_size,
        overwrite=args.overwrite,
        keep_going=not args.stop_on_error,
        max_videos=args.max_videos,
        countries=args.countries,
    )


if __name__ == "__main__":
    sys.exit(main())
