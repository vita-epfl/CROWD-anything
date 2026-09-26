"""Encode CROWD video clips into LTX-2 video VAE latents, stored as per-video shards.

Pipeline:
    manifest (clips) -> frames.py (ffmpeg, 24 fps) -> vae.py (LTX preprocessing + encoder) -> shards.py (safetensors)

The clip definition (clips.py) and the ffmpeg command (frames.py) are provisional until they are aligned with
the manifest codebase, so that everyone loads exactly the same frames.
"""
