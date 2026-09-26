"""Decode clip frames from source videos with ffmpeg at a fixed frame rate.

PROVISIONAL: `build_ffmpeg_command` must be replaced by the exact ffmpeg command from the manifest codebase, so every
consumer (ego trajectory extraction and tokenization) decodes exactly the same frames. Everything else in this module
only streams raw RGB frames out of that command.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from typing import Iterator, Optional

import numpy as np

from tokenization.clips import Clip, contiguous_runs

logger = logging.getLogger(__name__)

# Stored in each output's encoding_config.json. Change it whenever the decoding changes, so latents decoded
# differently are never mixed in one output directory.
FRAME_DECODER_VERSION = "provisional-ffmpeg-ss-fps-v1"


def build_ffmpeg_command(ffmpeg_bin: str, video_path: str, start_frame: int, num_frames: int, fps: float) -> list[str]:
    """Command that writes `num_frames` rgb24 frames at `fps`, starting at frame `start_frame`, to stdout."""
    return [
        ffmpeg_bin,
        "-nostdin",
        "-loglevel", "error",
        "-ss", f"{start_frame / fps:.6f}",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-frames:v", str(num_frames),
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


def _read_exact(stream, num_bytes: int) -> Optional[bytes]:
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_clip_frames(
    clips: list[Clip],
    ffmpeg_bin: Optional[str] = None,
) -> Iterator[tuple[Clip, Optional[np.ndarray]]]:
    """
    Yield (clip, frames) for clips of one video, frames as uint8 [F, H, W, 3].
    Frames is None when the video ends before the clip is complete.
    Clips whose frame ranges touch or overlap are decoded in a single ffmpeg pass.
    """
    if not clips:
        return
    ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg was not found on PATH")

    clips = sorted(clips, key=lambda c: c.start_frame)
    video_path = clips[0].video_path
    fps = clips[0].fps
    if any(c.fps != fps for c in clips):
        raise ValueError(f"Clips of {clips[0].video_id} use different fps values")

    width, height = probe_frame_size(video_path)
    frame_bytes = width * height * 3

    for run in contiguous_runs(clips):
        run_start = run[0].start_frame
        run_end = max(c.end_frame for c in run)
        cmd = build_ffmpeg_command(ffmpeg_bin, video_path, run_start, run_end - run_start, fps)
        # stderr goes to a file: an unread pipe could fill up and block ffmpeg on corrupt videos
        stderr_file = tempfile.TemporaryFile()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_file)
        try:
            buffer: list[np.ndarray] = []
            buffer_start = run_start
            exhausted = False
            for clip in run:
                # drop frames before this clip
                drop = clip.start_frame - buffer_start
                if drop > 0:
                    del buffer[:drop]
                    buffer_start = clip.start_frame
                # read until the buffer covers the clip
                while not exhausted and buffer_start + len(buffer) < clip.end_frame:
                    raw = _read_exact(proc.stdout, frame_bytes)
                    if raw is None:
                        exhausted = True
                        break
                    buffer.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3))
                if buffer_start + len(buffer) < clip.end_frame:
                    stderr_file.seek(0)
                    error_tail = stderr_file.read()[-500:].decode(errors="replace").strip()
                    message = (
                        f"{clip.video_id}: video ended at frame {buffer_start + len(buffer)}, "
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
