"""Stream frames of generated and ground-truth videos on the same grid and at the same size.

Both videos go through ffmpeg's `fps` filter from t=0, as the tokenizer and the manifest do (tokenization.frames),
so frame i of a generated video is compared with frame i of its ground truth. When the sizes differ, the ground
truth is scaled to cover the generated size and centre-cropped, so the comparison happens at the model's
resolution and aspect ratio.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from typing import Generator, Iterator, Optional

import numpy as np

from tokenization.frames import _read_into, probe_frame_size


def decode_command(ffmpeg_bin: str, video_path: str, fps: float, size: Optional[tuple[int, int]] = None) -> list[str]:
    """rgb24 frames on the canonical grid, to stdout; with `size` (width, height) scaled to cover it and cropped."""
    filters = [f"fps={fps:g}"]
    if size is not None:
        width, height = size
        filters += [f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=bicubic",
                    f"crop={width}:{height}"]
    return [
        ffmpeg_bin,
        "-nostdin",
        "-v", "error",
        "-i", video_path,
        "-map", "0:v:0",
        "-vf", ",".join(filters),
        "-vsync", "cfr",
        "-an",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]


def iter_frames(
    video_path: str,
    fps: float = 24.0,
    size: Optional[tuple[int, int]] = None,
    max_frames: Optional[int] = None,
    ffmpeg_bin: Optional[str] = None,
) -> Generator[np.ndarray, None, None]:
    """Yield uint8 frames [H, W, 3] one by one, at most `max_frames`."""
    ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg was not found on PATH")
    width, height = size or probe_frame_size(video_path)
    # stderr goes to a file: an unread pipe could fill up and block ffmpeg on corrupt videos
    stderr_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(decode_command(ffmpeg_bin, video_path, fps, size), stdout=subprocess.PIPE,
                            stderr=stderr_file)
    assert proc.stdout is not None  # stdout=PIPE
    count = 0
    try:
        while max_frames is None or count < max_frames:
            frame = np.empty((height, width, 3), dtype=np.uint8)
            if not _read_into(proc.stdout, memoryview(frame).cast("B")):
                break
            count += 1
            yield frame
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
        stderr_file.close()


def iter_frame_pairs(
    generated_path: str,
    reference_path: str,
    fps: float = 24.0,
    max_frames: Optional[int] = None,
    ffmpeg_bin: Optional[str] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (generated, reference) frames at the generated video's size until the shorter video ends."""
    size = probe_frame_size(generated_path)
    generated = iter_frames(generated_path, fps, None, max_frames, ffmpeg_bin)
    reference = iter_frames(reference_path, fps, size, max_frames, ffmpeg_bin)
    try:
        yield from zip(generated, reference)
    finally:
        generated.close()
        reference.close()
