import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation import paired
from evaluation.metrics import psnr, ssim, to_tensor
from evaluation.video import iter_frame_pairs, iter_frames
from tokenization.clips import Clip
from tokenization.frames import iter_clip_frames

CPU = torch.device("cpu")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def random_frames(seed: int, n: int = 2, h: int = 40, w: int = 56) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def test_psnr_matches_skimage():
    skm = pytest.importorskip("skimage.metrics")
    a, b = random_frames(0), random_frames(1)
    ours = psnr(to_tensor(a, CPU), to_tensor(b, CPU)).numpy()
    ref = [skm.peak_signal_noise_ratio(x, y, data_range=255) for x, y in zip(a, b)]
    np.testing.assert_allclose(ours, ref, rtol=1e-4)


def test_ssim_matches_skimage():
    skm = pytest.importorskip("skimage.metrics")
    a = random_frames(0)
    # a correlated second image, so SSIM is far from 0 and from 1
    b = np.clip(a.astype(int) + np.random.default_rng(2).integers(-40, 40, a.shape), 0, 255).astype(np.uint8)
    ours = ssim(to_tensor(a, CPU), to_tensor(b, CPU)).numpy()
    ref = [skm.structural_similarity(x, y, data_range=255, channel_axis=2, gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=True) for x, y in zip(a, b)]
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_identical_frames_score_perfectly():
    a = to_tensor(random_frames(3), CPU)
    assert torch.allclose(ssim(a, a), torch.ones(2), atol=1e-5)
    assert (psnr(a, a) > 90).all()


def make_indexed_video(path: Path, seconds: float, fps: int, width: int, height: int):
    """Lossless video whose red channel encodes the frame number (constant over the frame)."""
    graph = f"nullsrc=s={width}x{height}:r={fps}:d={seconds},geq=r='mod(N,256)':g=128:b=64"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", graph, "-c:v", "ffv1",
                    "-pix_fmt", "bgr0", str(path)], check=True)


@needs_ffmpeg
def test_frame_pairs_are_aligned_and_resized(tmp_path):
    # generated: 64x48 at 24 fps, 2 s; reference: 96x54 (other aspect ratio) at 48 fps, 3 s
    make_indexed_video(tmp_path / "gen.mkv", 2, 24, 64, 48)
    make_indexed_video(tmp_path / "ref.mkv", 3, 48, 96, 54)
    pairs = list(iter_frame_pairs(str(tmp_path / "gen.mkv"), str(tmp_path / "ref.mkv"), fps=24))
    assert len(pairs) == 48  # stops with the shorter video
    # the reference gets the frames the tokenizer's canonical decode picks (fps filter from t=0), only resized
    clip = Clip(clip_id="ref", video_id="ref", video_path=str(tmp_path / "ref.mkv"), fps=24, start_frame=0,
                num_frames=48)
    canonical = [int(f[0, 0, 0]) for f in next(iter_clip_frames([clip]))[1]]
    assert canonical[:2] == [0, 2]
    for i, (gen, ref) in enumerate(pairs):
        assert gen.shape == ref.shape == (48, 64, 3)
        assert int(gen[0, 0, 0]) == i
        assert int(ref[0, 0, 0]) == canonical[i]


@needs_ffmpeg
def test_max_frames(tmp_path):
    make_indexed_video(tmp_path / "v.mkv", 2, 24, 32, 32)
    assert len(list(iter_frames(str(tmp_path / "v.mkv"), max_frames=10))) == 10


@needs_ffmpeg
def test_run_writes_horizons(tmp_path):
    gen_dir, ref_dir = tmp_path / "gen", tmp_path / "gt"
    gen_dir.mkdir()
    ref_dir.mkdir()
    make_indexed_video(gen_dir / "long.mkv", 3, 24, 64, 48)
    make_indexed_video(ref_dir / "long.mkv", 3, 24, 64, 48)
    make_indexed_video(gen_dir / "short.mkv", 1.5, 24, 64, 48)
    make_indexed_video(ref_dir / "short.mkv", 3, 24, 64, 48)
    make_indexed_video(gen_dir / "no_gt.mkv", 1, 24, 64, 48)

    pairs = paired.pairs_from_folders(gen_dir, ref_dir)
    assert [p.clip_id for p in pairs] == ["long", "short"]
    summary = paired.run(pairs, tmp_path / "out", ["psnr", "ssim"], [1, 2, 10], device=CPU)

    with open(tmp_path / "out" / "per_clip.csv") as f:
        rows = list(csv.DictReader(f))
    assert [(r["clip_id"], r["horizon_s"], r["num_frames"]) for r in rows] == [
        ("long", "1", "24"), ("long", "2", "48"), ("long", "all", "72"),
        ("short", "1", "24"), ("short", "all", "36"),
    ]
    assert {r["horizon_s"]: r["num_clips"] for r in summary} == {"1": 2, "2": 1, "all": 2}
    # identical videos: perfect SSIM everywhere
    assert all(float(r["ssim"]) > 0.9999 for r in rows)
    with open(tmp_path / "out" / "per_frame.csv") as f:
        assert sum(1 for _ in csv.DictReader(f)) == 72 + 36
