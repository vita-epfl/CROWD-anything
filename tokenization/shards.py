"""Sharded storage of clip latents: usually one file per source video instead of one file per clip.

Layout of an output directory (one per resolution):
    encoding_config.json                       settings used to encode (checkpoint, resolution, fps, clip length)
    shards/<xx>/<video_id>.<part>.safetensors  "latents" [N, C, F', H', W'] and "start_frames" [N] for N clips
    progress/rank<k>.jsonl                     append-only log per process; a "done" line marks a video complete
    index.jsonl                                one line per clip: clip_id, video_id, start_frame, shard, row

<xx> is two hex characters of a hash of the video id, which keeps directories small. Completion is logged instead of
written as one marker file per video, to keep the number of files (inodes) low.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT_VERSION = "1"
CONFIG_FILE = "encoding_config.json"
INDEX_FILE = "index.jsonl"
PROGRESS_DIR = "progress"


def video_shard_dir(out_dir: Path, video_id: str) -> Path:
    prefix = hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:2]
    return Path(out_dir) / "shards" / prefix


def load_done_videos(out_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Read all progress logs and return {video_id: done record} for completed videos.
    The last line per video wins: a "started" line after a "done" line (a re-encode in progress) makes it not done.
    Truncated lines from an interrupted write are ignored.
    """
    state: dict[str, dict[str, Any]] = {}
    for log_path in sorted((Path(out_dir) / PROGRESS_DIR).glob("*.jsonl")):
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    state[record["video_id"]] = record
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    return {vid: rec for vid, rec in state.items() if rec.get("status") == "done"}


class ProgressLog:
    """Append-only log of one process. Each line is flushed to disk before the call returns."""

    def __init__(self, out_dir: Path, rank: int) -> None:
        directory = Path(out_dir) / PROGRESS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"rank{rank}.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def check_or_write_config(out_dir: Path, config: dict[str, Any], overwrite: bool = False) -> None:
    """Refuse to mix latents encoded with different settings in one output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CONFIG_FILE
    config = {"format_version": FORMAT_VERSION, **config}
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text())
        if existing != config:
            diff = {k: (existing.get(k), config.get(k)) for k in set(existing) | set(config)
                    if existing.get(k) != config.get(k)}
            raise ValueError(
                f"{path} was written with different settings {diff} (existing, requested). "
                "Use a new output directory, or --overwrite to re-encode everything."
            )
        return
    _atomic_write_bytes(path, json.dumps(config, indent=2, sort_keys=True).encode("utf-8"))


class VideoShardWriter:
    """Collects the latents of one video and writes them in parts of at most `max_clips_per_shard` clips."""

    def __init__(self, out_dir: Path, video_id: str, max_clips_per_shard: int, metadata: dict[str, Any]) -> None:
        self.out_dir = Path(out_dir)
        self.video_id = video_id
        self.max_clips_per_shard = max_clips_per_shard
        self.metadata = metadata
        self.shard_dir = video_shard_dir(self.out_dir, video_id)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        # remove leftovers of an interrupted earlier attempt at this video
        for pattern in (f"{glob_escape(video_id)}.*", f".{glob_escape(video_id)}.*.tmp"):
            for stale in self.shard_dir.glob(pattern):
                stale.unlink()
        self._latents: list[torch.Tensor] = []
        self._clip_ids: list[str] = []
        self._start_frames: list[int] = []
        self._parts: list[dict[str, Any]] = []

    def add(self, clip_id: str, start_frame: int, latents: torch.Tensor) -> None:
        self._latents.append(latents.detach().to("cpu").contiguous())
        self._clip_ids.append(clip_id)
        self._start_frames.append(start_frame)
        if len(self._latents) >= self.max_clips_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._latents:
            return
        part = len(self._parts)
        name = f"{self.video_id}.{part:04d}.safetensors"
        path = self.shard_dir / name
        tmp = path.with_name(f".{name}.{os.getpid()}.tmp")
        metadata = {
            "format_version": FORMAT_VERSION,
            "video_id": self.video_id,
            "clip_ids": json.dumps(self._clip_ids),
            **{k: json.dumps(v) for k, v in self.metadata.items()},
        }
        save_file(
            {
                "latents": torch.stack(self._latents),
                "start_frames": torch.tensor(self._start_frames, dtype=torch.int64),
            },
            str(tmp),
            metadata=metadata,
        )
        os.replace(tmp, path)
        self._parts.append({"file": name, "clip_ids": self._clip_ids, "start_frames": self._start_frames})
        self._latents, self._clip_ids, self._start_frames = [], [], []

    def finish(self, failed_clip_ids: Optional[list[str]] = None) -> dict[str, Any]:
        """Write the remaining clips and return the "done" record for the progress log."""
        self._flush()
        return {
            "video_id": self.video_id,
            "status": "done",
            "parts": self._parts,
            "failed_clip_ids": failed_clip_ids or [],
        }


def glob_escape(value: str) -> str:
    return "".join(f"[{c}]" if c in "*?[" else c for c in value)


def build_index(out_dir: Path) -> int:
    """Write index.jsonl from the progress logs. Returns the number of clips indexed."""
    out_dir = Path(out_dir)
    count = 0
    tmp = out_dir / f".{INDEX_FILE}.{os.getpid()}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for video_id, done in sorted(load_done_videos(out_dir).items()):
            shard_dir = video_shard_dir(out_dir, video_id)
            for part in done["parts"]:
                shard = str((shard_dir / part["file"]).relative_to(out_dir))
                for row, (clip_id, start_frame) in enumerate(zip(part["clip_ids"], part["start_frames"])):
                    record = {
                        "clip_id": clip_id,
                        "video_id": video_id,
                        "start_frame": start_frame,
                        "shard": shard,
                        "row": row,
                    }
                    f.write(json.dumps(record) + "\n")
                    count += 1
    os.replace(tmp, out_dir / INDEX_FILE)
    return count


class LatentClipDataset(torch.utils.data.Dataset):
    """
    Reads single clips from the shards. Items follow the layout of ltx-trainer's precomputed latents:
    {"latents": [C, F', H', W'], "num_frames": F', "height": H', "width": W', "fps": fps} plus clip identifiers.
    """

    def __init__(self, out_dir: str | Path, max_open_files: int = 32) -> None:
        self.out_dir = Path(out_dir)
        index_path = self.out_dir / INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} not found. Run: python -m tokenization.encode index {out_dir}")
        with index_path.open("r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        self.config = json.loads((self.out_dir / CONFIG_FILE).read_text())
        self.max_open_files = max_open_files
        self._handles: OrderedDict[str, Any] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def _open(self, shard: str):
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.out_dir / shard), framework="pt")
            self._handles[shard] = handle
            if len(self._handles) > self.max_open_files:
                self._handles.popitem(last=False)
        else:
            self._handles.move_to_end(shard)
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        row = record["row"]
        latents = self._open(record["shard"]).get_slice("latents")[row:row + 1][0]
        _, num_frames, height, width = latents.shape
        return {
            "latents": latents,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": self.config["fps"],
            "clip_id": record["clip_id"],
            "video_id": record["video_id"],
            "start_frame": record["start_frame"],
        }

    def __getstate__(self) -> dict[str, Any]:
        # open file handles are not picklable; each DataLoader worker opens its own
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state
