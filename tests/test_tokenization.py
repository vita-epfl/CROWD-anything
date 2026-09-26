import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tokenization import encode as enc
from tokenization.clips import Clip, contiguous_runs, load_manifest, make_clip_id, split_range_into_clips
from tokenization.frames import iter_clip_frames
from tokenization.shards import LatentClipDataset, is_video_done
from tokenization.vae import ScaleFactors, preprocess_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def make_indexed_video(path: Path, seconds: float, fps: int, width: int = 64, height: int = 48) -> None:
    """Lossless video whose red/green channels encode the source frame number."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", f"nullsrc=s={width}x{height}:r={fps}:d={seconds},geq=r='mod(N,256)':g='floor(N/256)':b=0",
            "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path),
        ],
        check=True,
    )


def frame_numbers(frames: np.ndarray) -> list[int]:
    return [int(f[0, 0, 0]) + 256 * int(f[0, 0, 1]) for f in frames]


def clip(video_path: Path, start: int, num_frames: int = 121, video_id: str = "vid") -> Clip:
    return Clip(make_clip_id(video_id, start), video_id, str(video_path), start, num_frames, 24.0)


class FakeEncoder:
    """Deterministic stand-in with the LTX-2 VAE's shapes: 128 channels, 8x temporal and 32x spatial compression."""

    scale_factors = ScaleFactors(time=8, height=32, width=32)
    device = torch.device("cpu")

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        first = torch.nn.functional.avg_pool3d(videos[:, :, :1], (1, 32, 32))
        rest = torch.nn.functional.avg_pool3d(videos[:, :, 1:], (8, 32, 32))
        pooled = torch.cat([first, rest], dim=2).mean(dim=1, keepdim=True)  # [B, 1, F', H', W']
        scale = torch.linspace(-1, 1, 128).view(1, 128, 1, 1, 1)
        return (pooled * scale).to(torch.bfloat16)


def test_split_range_into_clips_overlaps_one_frame():
    assert split_range_into_clips(0, 361) == [0, 120, 240]
    assert split_range_into_clips(0, 360) == [0, 120]
    assert split_range_into_clips(10, 130) == []
    starts = split_range_into_clips(0, 1000)
    assert all(b - a == 120 for a, b in zip(starts, starts[1:]))


def test_contiguous_runs_split_on_gaps(tmp_path):
    clips = [clip(tmp_path, s) for s in (0, 120, 240, 1000, 1120)]
    assert [[c.start_frame for c in run] for run in contiguous_runs(clips)] == [[0, 120, 240], [1000, 1120]]


def test_frames_are_exact_at_native_24fps(tmp_path):
    video = tmp_path / "v.mkv"
    make_indexed_video(video, seconds=30, fps=24)  # 720 frames
    clips = [clip(video, s) for s in split_range_into_clips(0, 720)]
    results = list(iter_clip_frames(clips))
    assert len(results) == 5
    for c, frames in results:
        assert frames.shape == (121, 48, 64, 3)
        assert frame_numbers(frames) == list(range(c.start_frame, c.end_frame))
    # consecutive clips share exactly one frame
    for (_, a), (_, b) in zip(results, results[1:]):
        assert np.array_equal(a[-1], b[0])


@pytest.mark.parametrize("source_fps", [24, 25, 30])
def test_clip_decoded_alone_matches_clip_decoded_in_a_run(tmp_path, source_fps):
    """Everyone must get the same frames for a clip, whether it is decoded alone or as part of a longer run."""
    video = tmp_path / "v.mkv"
    make_indexed_video(video, seconds=30, fps=source_fps)
    in_run = dict((c.start_frame, f) for c, f in iter_clip_frames([clip(video, s) for s in (0, 120, 240, 360)]))
    for start in (120, 240, 360):
        (_, alone), = list(iter_clip_frames([clip(video, start)]))
        assert frame_numbers(alone) == frame_numbers(in_run[start]), f"start={start} source_fps={source_fps}"


def test_clip_past_end_of_video_is_reported(tmp_path):
    video = tmp_path / "v.mkv"
    make_indexed_video(video, seconds=6, fps=24)  # 144 frames
    results = list(iter_clip_frames([clip(video, 0), clip(video, 120)]))
    assert results[0][1] is not None and results[1][1] is None


def test_preprocess_matches_ltx_resize_and_crop():
    frames = np.random.randint(0, 256, size=(9, 1080, 1920, 3), dtype=np.uint8)
    video = preprocess_frames(frames, target_height=544, target_width=960)
    assert video.shape == (3, 9, 544, 960)
    assert video.min() >= -1.0 and video.max() <= 1.0
    tall = preprocess_frames(np.zeros((9, 1920, 1080, 3), dtype=np.uint8), 544, 960)
    assert tall.shape == (3, 9, 544, 960)


def write_manifest(path: Path, clips: list[Clip]) -> None:
    with path.open("w") as f:
        for c in clips:
            f.write(json.dumps(c.__dict__) + "\n")


def test_encode_index_read_and_resume(tmp_path):
    video_a = tmp_path / "-dashid.mkv"
    video_b = tmp_path / "short.mkv"
    make_indexed_video(video_a, seconds=21, fps=24, width=128, height=96)  # 504 frames -> 4 clips
    make_indexed_video(video_b, seconds=6, fps=24, width=128, height=96)  # 144 frames -> 1 full clip, 1 fails
    clips = [clip(video_a, s, video_id="-dashid") for s in split_range_into_clips(0, 504)]
    clips += [clip(video_b, 0, video_id="short"), clip(video_b, 120, video_id="short")]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, clips)
    assert load_manifest(manifest) == clips

    out = tmp_path / "latents"
    kwargs = dict(encoder=FakeEncoder(), encoder_name="fake", width=64, height=64, batch_size=3, max_clips_per_shard=3)
    assert enc.run_encode(manifest, out, **kwargs) == 0
    assert is_video_done(out, "-dashid") and is_video_done(out, "short")
    assert len(list(out.glob("shards/*/*.safetensors"))) == 3  # 4 clips in 2 parts + 1 clip

    dataset = LatentClipDataset(out)
    assert [item["clip_id"] for item in dataset] == [c.clip_id for c in clips[:5]]
    item = dataset[1]
    assert item["latents"].shape == (128, 16, 2, 2) and item["latents"].dtype == torch.bfloat16
    assert (item["num_frames"], item["height"], item["width"], item["fps"]) == (16, 2, 2, 24.0)

    # the stored latents equal a direct encode of the same clip
    (_, frames), = list(iter_clip_frames([clips[1]]))
    expected = FakeEncoder().encode(preprocess_frames(frames, 64, 64)[None])[0]
    assert torch.equal(item["latents"], expected)

    # rerun: everything is done, nothing is re-encoded
    mtimes = {p: p.stat().st_mtime_ns for p in out.glob("shards/*/*")}
    assert enc.run_encode(manifest, out, **kwargs) == 0
    assert {p: p.stat().st_mtime_ns for p in out.glob("shards/*/*")} == mtimes

    # different settings must not be mixed into the same output directory
    with pytest.raises(ValueError, match="different settings"):
        enc.run_encode(manifest, out, **{**kwargs, "width": 96})


def test_multi_rank_split_covers_all_videos(tmp_path):
    clips = []
    for i in range(5):
        video = tmp_path / f"v{i}.mkv"
        make_indexed_video(video, seconds=6, fps=24, width=64, height=64)
        clips.append(clip(video, 0, video_id=f"v{i}"))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, clips)
    out = tmp_path / "latents"
    for rank in range(2):
        assert enc.run_encode(manifest, out, FakeEncoder(), "fake", 64, 64, rank=rank, world_size=2) == 0
    assert enc.main(["index", str(out)]) == 0
    assert sorted(item["video_id"] for item in LatentClipDataset(out)) == [f"v{i}" for i in range(5)]
