"""Clip definition and manifest reading.

PROVISIONAL: the split rule and the manifest schema below must be replaced by (or validated against) the manifest
codebase. The current rule is: clips of 121 frames at 24 fps, consecutive clips sharing 1 frame (stride 120), and
frame indices counted at 24 fps from the start of the source video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

CLIP_NUM_FRAMES = 121
CLIP_FPS = 24.0
CLIP_OVERLAP_FRAMES = 1

REQUIRED_FIELDS = ("clip_id", "video_id", "video_path", "start_frame", "num_frames", "fps")


@dataclass(frozen=True)
class Clip:
    clip_id: str
    video_id: str
    video_path: str
    start_frame: int  # index of the first frame, counted at `fps` from the start of the source video
    num_frames: int
    fps: float

    @property
    def end_frame(self) -> int:
        """Exclusive end frame index."""
        return self.start_frame + self.num_frames


def split_range_into_clips(
    first_frame: int,
    end_frame: int,
    num_frames: int = CLIP_NUM_FRAMES,
    overlap: int = CLIP_OVERLAP_FRAMES,
) -> list[int]:
    """Return start frames of all full clips inside [first_frame, end_frame). Trailing partial clips are dropped."""
    stride = num_frames - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than num_frames ({num_frames})")
    starts = []
    start = first_frame
    while start + num_frames <= end_frame:
        starts.append(start)
        start += stride
    return starts


def make_clip_id(video_id: str, start_frame: int) -> str:
    return f"{video_id}_{start_frame:08d}"


def load_manifest(path: str | Path) -> list[Clip]:
    """Read a JSONL manifest with one clip per line (fields: REQUIRED_FIELDS)."""
    path = Path(path)
    clips: list[Clip] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = [key for key in REQUIRED_FIELDS if key not in record]
            if missing:
                raise ValueError(f"{path}:{line_no} is missing fields: {', '.join(missing)}")
            clip = Clip(
                clip_id=str(record["clip_id"]),
                video_id=str(record["video_id"]),
                video_path=str(record["video_path"]),
                start_frame=int(record["start_frame"]),
                num_frames=int(record["num_frames"]),
                fps=float(record["fps"]),
            )
            if clip.clip_id in seen:
                raise ValueError(f"{path}:{line_no} duplicate clip_id {clip.clip_id}")
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


def contiguous_runs(clips: list[Clip]) -> Iterator[list[Clip]]:
    """Split start-sorted clips into runs whose frame ranges touch or overlap, so each run is decoded once."""
    run: list[Clip] = []
    run_end = None
    for clip in clips:
        if run and clip.start_frame > run_end:
            yield run
            run = []
        run.append(clip)
        run_end = clip.end_frame if run_end is None or len(run) == 1 else max(run_end, clip.end_frame)
    if run:
        yield run
