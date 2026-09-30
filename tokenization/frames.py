"""Decode clip frames exactly as the manifest (wm-data-manifest) defines them.

The manifest addresses frames on a canonical grid: the whole source video resampled through ffmpeg's `fps=24`
filter, anchored at t=0, frame i at t = i / 24 s (wm_data_manifest.natix.resample, reused for CROWD). The only
way to get the same frames is to decode the video from its start through that same filter and count frames --
seeking to a clip's start time can land on different frames, especially for variable frame rate videos. So each
video is decoded in one pass from t=0; frames outside the requested clips are read and discarded, and decoding
stops after the last clip.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from typing import Iterator, Optional

import numpy as np

from tokenization.clips import Clip

logger = logging.getLogger(__name__)

# Stored in each output's encoding_config.json. Change it whenever the decoding changes, so latents decoded
# differently are never mixed in one output directory.
FRAME_DECODER_VERSION = "wm-data-manifest-canonical-fps24-t0-v1"


def canonical_decode_command(ffmpeg_bin: str, video_path: str, fps: float) -> list[str]:
    """rgb24 frames of the whole video on the canonical grid, to stdout. Same video stream, filter and sync as
    wm_data_manifest.natix.resample (canonical_resample_command / count_resampled_frames)."""
    return [
        ffmpeg_bin,
        "-nostdin",
        "-v", "error",
        "-i", video_path,
        "-map", "0:v:0",
        "-vf", f"fps={fps:g}",
        "-vsync", "cfr",
        "-an",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]


def probe_frame_size(video_path: str, ffprobe_bin: Optional[str] = None) -> tuple[int, int]:
    """Return (width, height) of decoded frames, accounting for rotation metadata (ffmpeg auto-rotates)."""
    ffprobe_bin = ffprobe_bin or shutil.which("ffprobe")
    if not ffprobe_bin:
        raise RuntimeError("ffprobe was not found on PATH")
    result = subprocess.run(
        [
            ffprobe_bin, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:stream_side_data=rotation",
            "-of", "json", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream in {video_path}")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            rotation = int(side_data["rotation"])
    if rotation % 180 != 0:
        width, height = height, width
    return width, height


def _read_into(stream, buffer: memoryview) -> bool:
    """Fill `buffer` from `stream`; False if the stream ends first."""
    filled = 0
    while filled < len(buffer):
        n = stream.readinto(buffer[filled:])
        if not n:
            return False
        filled += n
    return True


def count_canonical_frames(video_path: str, fps: float, ffmpeg_bin: Optional[str] = None) -> int:
    """Number of frames `iter_clip_frames` sees for this video (for checks against the manifest)."""
    ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg was not found on PATH")
    width, height = probe_frame_size(video_path)
    scratch = memoryview(bytearray(width * height * 3))
    proc = subprocess.Popen(
        canonical_decode_command(ffmpeg_bin, video_path, fps), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    assert proc.stdout is not None  # stdout=PIPE
    count = 0
    try:
        while _read_into(proc.stdout, scratch):
            count += 1
    finally:
        proc.stdout.close()
        proc.wait()
    return count


def iter_clip_frames(
    clips: list[Clip],
    ffmpeg_bin: Optional[str] = None,
) -> Iterator[tuple[Clip, Optional[np.ndarray]]]:
    """
    Yield (clip, frames) for the clips of one video in start order, frames as uint8 [F, H, W, 3].
    Frames is None when the video ends before the clip is complete.
    """
    if not clips:
        return
    ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg was not found on PATH")

    clips = sorted(clips, key=lambda c: c.start_frame)
    video_path = clips[0].video_path
    fps = clips[0].fps
    if any(c.fps != fps or c.video_path != video_path for c in clips):
        raise ValueError(f"Clips of {clips[0].video_id} use different fps values or video paths")

    width, height = probe_frame_size(video_path)
    frame_bytes = width * height * 3
    scratch = memoryview(bytearray(frame_bytes))

    # stderr goes to a file: an unread pipe could fill up and block ffmpeg on corrupt videos
    stderr_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(canonical_decode_command(ffmpeg_bin, video_path, fps), stdout=subprocess.PIPE,
                            stderr=stderr_file)
    try:
        next_frame = 0  # index of the next frame ffmpeg will output
        buffer: list[np.ndarray] = []  # frames [buffer_start, next_frame) kept for the current clip(s)
        buffer_start = 0
        exhausted = False
        for clip in clips:
            # drop kept frames before this clip, then skip frames up to its start
            if clip.start_frame >= next_frame:
                buffer, buffer_start = [], next_frame
            else:
                del buffer[: clip.start_frame - buffer_start]
                buffer_start = clip.start_frame
            while not exhausted and next_frame < clip.start_frame:
                exhausted = not _read_into(proc.stdout, scratch)
                if not exhausted:
                    next_frame += 1
                    buffer_start = next_frame
            # read until the clip is complete
            while not exhausted and next_frame < clip.end_frame:
                frame = np.empty((height, width, 3), dtype=np.uint8)
                exhausted = not _read_into(proc.stdout, memoryview(frame).cast("B"))
                if not exhausted:
                    buffer.append(frame)
                    next_frame += 1
            if next_frame < clip.end_frame:
                stderr_file.seek(0)
                error_tail = stderr_file.read()[-500:].decode(errors="replace").strip()
                message = (
                    f"{clip.video_id}: video ended at frame {next_frame}, "
                    f"clip {clip.clip_id} needs up to {clip.end_frame}"
                )
                logger.warning(message + (f" | ffmpeg: {error_tail}" if error_tail else ""))
                yield clip, None
            else:
                yield clip, np.stack(buffer[: clip.num_frames])
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.kill()
        proc.wait()
        stderr_file.close()
