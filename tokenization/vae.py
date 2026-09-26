"""LTX-2 video VAE encoding with the same preprocessing as ltx-trainer (scripts/process_videos.py).

Preprocessing, as in ltx-trainer:
    uint8 frames -> float [0, 1] -> resize keeping aspect ratio (bicubic) -> center crop -> clamp -> [-1, 1]
The LTX packages (ltx-core, ltx-trainer from https://github.com/Lightricks/LTX-2) are imported lazily, so the rest of
the pipeline can be used and tested without them.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize


class ScaleFactors(NamedTuple):
    time: int
    height: int
    width: int


class VideoEncoder(Protocol):
    scale_factors: ScaleFactors
    device: torch.device

    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """[B, 3, F, H, W] in [-1, 1] -> latents [B, C, F', H', W']."""
        ...


def resize_and_crop(frames: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    """Resize [F, C, H, W] so it covers the target, then center crop. Same as ltx-trainer (reshape_mode="center")."""
    current_height, current_width = frames.shape[2], frames.shape[3]
    if current_width / current_height > target_width / target_height:
        new_width = int(current_width * target_height / current_height)
        frames = resize(frames, size=[target_height, new_width], interpolation=InterpolationMode.BICUBIC)
    else:
        new_height = int(current_height * target_width / current_width)
        frames = resize(frames, size=[new_height, target_width], interpolation=InterpolationMode.BICUBIC)
    top = (frames.shape[2] - target_height) // 2
    left = (frames.shape[3] - target_width) // 2
    return crop(frames, top=top, left=left, height=target_height, width=target_width)


def preprocess_frames(
    frames: np.ndarray,
    target_height: int,
    target_width: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """uint8 [F, H, W, 3] -> float32 [3, F, H', W'] in [-1, 1]."""
    video = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div(255.0)  # [F, C, H, W]
    video = resize_and_crop(video, target_height, target_width)
    video = video.clamp(0.0, 1.0).sub(0.5).div(0.5)
    return video.permute(1, 0, 2, 3).contiguous()  # [C, F, H, W]


def validate_clip_shape(num_frames: int, height: int, width: int, scale_factors: ScaleFactors) -> None:
    if num_frames % scale_factors.time != 1:
        raise ValueError(f"num_frames={num_frames} must satisfy num_frames % {scale_factors.time} == 1")
    if height % scale_factors.height or width % scale_factors.width:
        raise ValueError(
            f"resolution {width}x{height} must be a multiple of {scale_factors.width}x{scale_factors.height}"
        )


class LTXVideoEncoder:
    """Video VAE encoder loaded from an LTX-2 checkpoint (unified checkpoint or split video VAE file)."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        try:
            from ltx_trainer.model_loader import load_video_vae_encoder, read_video_scale_factors
        except ImportError as exc:
            raise ImportError(
                "LTX-2 is not installed. Install ltx-core and ltx-trainer from https://github.com/Lightricks/LTX-2 "
                "(packages/ltx-core, packages/ltx-trainer) into this environment."
            ) from exc

        self.device = torch.device(device)
        self.dtype = dtype
        factors = read_video_scale_factors(checkpoint_path)
        self.scale_factors = ScaleFactors(time=factors.time, height=factors.height, width=factors.width)
        self.vae = load_video_vae_encoder(checkpoint_path, device=self.device, dtype=dtype)
        self.vae.eval()

    @torch.inference_mode()
    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        return self.vae(videos.to(device=self.device, dtype=self.dtype))
