"""Clips to encode, read from a wm-data-manifest sample manifest (parquet, one row per sample).

The manifest (e.g. processed_datasets/crowd/manifest/samples_24fps.parquet) defines every clip: 121 frames on the
canonical 24 fps grid, consecutive clips sharing one frame. The columns used here are `sample_id`,
`source_video_path` (relative to the dataset's video root), `start_frame`, `num_frames`, `fps` and, when present,
`video_id` (CROWD); frames are decoded as the manifest defines them (see frames.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pyarrow.parquet as pq

REQUIRED_COLUMNS = ("sample_id", "source_video_path", "start_frame", "num_frames", "fps")


@dataclass(frozen=True)
class Clip:
    clip_id: str  # the manifest's sample_id
    video_id: str  # groups clips of one source video; the manifest's video_id, else its source_video_path
    video_path: str  # absolute path of the source video
    start_frame: int  # on the canonical grid, counted from the start of the video
    num_frames: int
    fps: float

    @property
    def end_frame(self) -> int:
        """Exclusive end frame index."""
        return self.start_frame + self.num_frames


def load_manifest(path: str | Path, video_root: str | Path, filters: Optional[list] = None) -> list[Clip]:
    """Clips of a parquet manifest (file or directory of partitions). `filters` is passed to pyarrow, e.g.
    [("country", "=", "Canada")]."""
    # partitioning=None: partition directories (country=<Country>/) repeat a column already in the files
    table = pq.read_table(path, filters=filters, partitioning=None)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    video_root = Path(video_root)
    has_video_id = "video_id" in table.column_names
    clips: list[Clip] = []
    seen: set[str] = set()
    for row in table.to_pylist():
        clip = Clip(
            clip_id=row["sample_id"],
            video_id=row["video_id"] if has_video_id else row["source_video_path"],
            video_path=str(video_root / row["source_video_path"]),
            start_frame=int(row["start_frame"]),
            num_frames=int(row["num_frames"]),
            fps=float(row["fps"]),
        )
        if clip.clip_id in seen:
            raise ValueError(f"{path}: duplicate sample_id {clip.clip_id}")
        seen.add(clip.clip_id)
        clips.append(clip)
    return clips


def group_by_video(clips: Iterable[Clip]) -> dict[str, list[Clip]]:
    """Group clips per source video, each group sorted by start frame."""
    groups: dict[str, list[Clip]] = {}
    for clip in clips:
        groups.setdefault(clip.video_id, []).append(clip)
    for video_clips in groups.values():
        video_clips.sort(key=lambda c: c.start_frame)
        paths = {c.video_path for c in video_clips}
        if len(paths) > 1:
            raise ValueError(f"video_id {video_clips[0].video_id} has several video paths: {sorted(paths)}")
    return dict(sorted(groups.items()))
