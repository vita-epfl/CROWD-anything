import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from tokenization import encode as enc
from tokenization.clips import Clip, load_manifest
from tokenization.frames import count_canonical_frames, iter_clip_frames
from tokenization.shards import LatentClipDataset, ProgressLog, build_index, load_done_videos
from tokenization.vae import ScaleFactors, preprocess_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def make_indexed_video(path: Path, seconds: float, fps: int, width: int = 64, height: int = 48, vfr: bool = False):
    """Lossless video whose red/green channels encode the source frame number. With vfr, frame durations vary
    (the first 2 s at `fps`, then at fps / 3), like the variable frame rate of many uploaded videos."""
    graph = f"nullsrc=s={width}x{height}:r={fps}:d={seconds},geq=r='mod(N,256)':g='floor(N/256)':b=0"
    args = []
    if vfr:
        graph += f",setpts='if(lt(N,{2 * fps}),N/{fps},2+(N-{2 * fps})*3/{fps})/TB'"
        args = ["-fps_mode", "vfr"]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", graph, *args, "-c:v", "ffv1", "-pix_fmt", "bgr0",
         str(path)],
        check=True,
    )


def frame_numbers(frames) -> list[int]:
    return [int(f[0, 0, 0]) + 256 * int(f[0, 0, 1]) for f in frames]


def window_starts(num_frames: int, frames_per_sample: int = 121, stride: int = 120) -> list[int]:
    """Same windows as wm_data_manifest.windowing.samples_for_frame_count."""
    if num_frames < frames_per_sample:
        return []
    return [i * stride for i in range((num_frames - frames_per_sample) // stride + 1)]


def clip(video_path: Path, start: int, num_frames: int = 121, video_id: str = "vid") -> Clip:
    return Clip(f"{video_path.name}#f{start:06d}", video_id, str(video_path), start, num_frames, 24.0)


def write_manifest(path: Path, clips: list[Clip], video_root: Path, country: str = "X") -> None:
    """A manifest with the wm-data-manifest columns the tokenizer reads."""
    rows = [
        {
            "sample_id": c.clip_id,
            "source_video_path": Path(c.video_path).relative_to(video_root).as_posix(),
            "video_id": c.video_id,
            "country": country,
            "start_frame": c.start_frame,
            "end_frame": c.end_frame,
            "num_frames": c.num_frames,
            "fps": c.fps,
        }
        for c in clips
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


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


def test_frames_are_exact_at_native_24fps(tmp_path):
    video = tmp_path / "v.mkv"
    make_indexed_video(video, seconds=30, fps=24)  # 720 frames
    clips = [clip(video, s) for s in window_starts(720)]
    results = list(iter_clip_frames(clips))
    assert len(results) == 5
    for c, frames in results:
        assert frames.shape == (121, 48, 64, 3)
        assert frame_numbers(frames) == list(range(c.start_frame, c.end_frame))
    # consecutive clips share exactly one frame
    for (_, a), (_, b) in zip(results, results[1:]):
        assert np.array_equal(a[-1], b[0])


def test_clips_with_gaps_and_selected_subsets_give_the_same_frames(tmp_path):
    """A clip's frames must not depend on which other clips of the video are decoded with it."""
    video = tmp_path / "v.mkv"
    make_indexed_video(video, seconds=40, fps=30)
    together = [clip(video, s) for s in (0, 120, 400, 520, 800)]
    all_clips = {c.start_frame: f for c, f in iter_clip_frames(together)}
    for start in (120, 400, 800):
        (_, alone), = list(iter_clip_frames([clip(video, start)]))
        assert np.array_equal(alone, all_clips[start])


@pytest.mark.parametrize("source_fps,vfr", [(24, False), (25, False), (30, False), (30, True), (60, True)])
def test_frames_match_the_manifest_canonical_decode(tmp_path, source_fps, vfr):
    """Same frame count as the manifest's count_resampled_frames, and every clip frame identical to the frame at
    that index of the manifest's canonical_resample_command output."""
    resample = pytest.importorskip("wm_data_manifest.natix.resample")
    video = tmp_path / "v.mkv"
    make_indexed_video(video, seconds=26, fps=source_fps, vfr=vfr)

    num_frames = resample.count_resampled_frames(video)
    assert count_canonical_frames(str(video), 24.0) == num_frames

    png_dir = tmp_path / "canonical"
    png_dir.mkdir()
    subprocess.run(resample.canonical_resample_command(video, png_dir / "%06d.png"), check=True, capture_output=True)
    pngs = sorted(png_dir.glob("*.png"))
    assert len(pngs) == num_frames
    canonical = frame_numbers(np.asarray(Image.open(p).convert("RGB")) for p in pngs)

    clips = [clip(video, s) for s in window_starts(num_frames)]
    assert clips, "test video too short for a clip"
    for c, frames in iter_clip_frames(clips):
        assert frame_numbers(frames) == canonical[c.start_frame:c.end_frame]


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


def test_encode_index_read_and_resume(tmp_path):
    video_a = tmp_path / "-dashid.mkv"
    video_b = tmp_path / "short.mkv"
    make_indexed_video(video_a, seconds=21, fps=24, width=128, height=96)  # 504 frames -> 4 clips
    make_indexed_video(video_b, seconds=6, fps=24, width=128, height=96)  # 144 frames -> 1 full clip, 1 fails
    clips = [clip(video_a, s, video_id="-dashid") for s in window_starts(504)]
    clips += [clip(video_b, 0, video_id="short"), clip(video_b, 120, video_id="short")]
    manifest = tmp_path / "manifest.parquet"
    write_manifest(manifest, clips, tmp_path)
    assert load_manifest(manifest, tmp_path) == clips

    root = tmp_path / "latents"
    out = root / "64x64"
    kwargs = dict(encoder=FakeEncoder(), encoder_name="fake", resolutions=[(64, 64)], batch_size=3,
                  max_clips_per_shard=3)
    assert enc.run_encode(manifest, tmp_path, root, **kwargs) == 0
    done = load_done_videos(out)
    assert set(done) == {"-dashid", "short"} and done["short"]["failed_clip_ids"] == [clips[5].clip_id]
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
    assert enc.run_encode(manifest, tmp_path, root, **kwargs) == 0
    assert {p: p.stat().st_mtime_ns for p in out.glob("shards/*/*")} == mtimes

    # different settings must not be mixed into the same output directory
    with pytest.raises(ValueError, match="different settings"):
        enc.run_encode(manifest, tmp_path, root, **{**kwargs, "encoder_name": "other-vae"})


def test_interrupted_reencode_and_truncated_log_line(tmp_path):
    out = tmp_path / "out"
    log = ProgressLog(out, rank=0)
    log.append({"video_id": "a", "status": "done", "parts": [], "failed_clip_ids": []})
    log.append({"video_id": "b", "status": "done", "parts": [], "failed_clip_ids": []})
    log.append({"video_id": "b", "status": "started"})  # re-encode of b interrupted
    with log.path.open("a") as f:
        f.write('{"video_id": "c", "status": "do')  # write interrupted mid-line
    assert set(load_done_videos(out)) == {"a"}


def test_multiple_resolutions_decode_once_and_add_later(tmp_path, monkeypatch):
    video = tmp_path / "v.mkv"
    make_indexed_video(video, seconds=11, fps=24, width=160, height=96)  # 264 frames -> 2 clips
    clips = [clip(video, s) for s in window_starts(264)]
    manifest = tmp_path / "manifest.parquet"
    write_manifest(manifest, clips, tmp_path)
    root = tmp_path / "latents"

    decodes = []
    real_iter = enc.frame_io.iter_clip_frames
    monkeypatch.setattr(enc.frame_io, "iter_clip_frames", lambda c: decodes.append(len(c)) or real_iter(c))

    assert enc.run_encode(manifest, tmp_path, root, FakeEncoder(), "fake", resolutions=[(64, 64), (128, 64)]) == 0
    assert decodes == [2]  # decoded once for both resolutions
    shapes = {res: LatentClipDataset(root / res)[0]["latents"].shape for res in ("64x64", "128x64")}
    assert shapes == {"64x64": (128, 16, 2, 2), "128x64": (128, 16, 2, 4)}

    # adding a resolution later only encodes the new one
    assert enc.run_encode(manifest, tmp_path, root, FakeEncoder(), "fake", resolutions=[(64, 64), (96, 96)]) == 0
    assert decodes == [2, 2]
    assert len(LatentClipDataset(root / "96x96")) == 2

    # files per resolution: config, index, progress log, one shard (plus the directories holding them)
    files = [p for p in (root / "64x64").rglob("*") if p.is_file()]
    assert len(files) == 4


def test_multi_rank_split_covers_all_videos(tmp_path):
    clips = []
    for i in range(5):
        video = tmp_path / f"v{i}.mkv"
        make_indexed_video(video, seconds=6, fps=24, width=64, height=64)
        clips.append(clip(video, 0, video_id=f"v{i}"))
    manifest = tmp_path / "manifest.parquet"
    write_manifest(manifest, clips, tmp_path)
    root = tmp_path / "latents"
    for rank in range(2):
        assert enc.run_encode(manifest, tmp_path, root, FakeEncoder(), "fake", [(64, 64)], rank=rank,
                              world_size=2) == 0
    assert enc.main(["index", str(root)]) == 0
    assert sorted(item["video_id"] for item in LatentClipDataset(root / "64x64")) == [f"v{i}" for i in range(5)]
    assert build_index(root / "64x64") == 5


def test_load_manifest_from_partitions_with_country_filter(tmp_path):
    video = tmp_path / "videos" / "v.mkv"
    video.parent.mkdir()
    make_indexed_video(video, seconds=11, fps=24)
    parts = tmp_path / "manifest" / "samples_24fps"
    for country, start in (("A", 0), ("B", 120)):
        write_manifest(parts / f"country={country}" / "part-000.parquet", [clip(video, start, video_id="v")],
                       tmp_path / "videos", country)
    assert [c.start_frame for c in load_manifest(parts, tmp_path / "videos")] == [0, 120]
    only_b = load_manifest(parts, tmp_path / "videos", filters=[("country", "in", ["B"])])
    assert [(c.start_frame, c.video_path) for c in only_b] == [(120, str(video))]
