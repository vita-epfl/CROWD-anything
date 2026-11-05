"""Inputs of the topology evaluation for an RGWM generation export (e.g. rgwm-eval-180-shared-20261103).

The export's generations.csv has one row per generated video (sample_id, profile, mode, file). This writes the
long-format clips.csv of evaluation.topology (clip_id = NATIX sample_id, side = <profile>_<mode>, plus a `reference`
side) and cuts each sample's real clip from its NATIX source video: the sample's frames [start_frame, start_frame +
num_frames) on the canonical 24 fps grid anchored at t=0 (as the manifest and the tokenizer read them), scaled to
cover the generated size and centre-cropped, so the VLM sees real and generated videos alike.

  python -m evaluation.rgwm prepare --export <export dir> --natix-root <NATIX videos root> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from tokenization.frames import probe_frame_size

logger = logging.getLogger(__name__)


def split_sample_id(sample_id: str) -> tuple[str, int]:
    """'<source_video_path>#f000360' -> (source_video_path, 360)."""
    path, _, frame = sample_id.rpartition("#f")
    if not path or not frame.isdigit():
        raise ValueError(f"not a sample id: {sample_id}")
    return path, int(frame)


def reference_command(ffmpeg_bin: str, source: str, start_frame: int, num_frames: int, fps: float,
                      size: tuple[int, int], out: str) -> list[str]:
    width, height = size
    filters = [
        f"fps={fps:g}",  # the canonical grid from t=0, then the sample's frames
        f"trim=start_frame={start_frame}:end_frame={start_frame + num_frames}",
        "setpts=PTS-STARTPTS",
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=bicubic",
        f"crop={width}:{height}",
    ]
    return [ffmpeg_bin, "-y", "-nostdin", "-v", "error", "-i", source, "-map", "0:v:0", "-vf", ",".join(filters),
            "-vsync", "cfr", "-r", f"{fps:g}", "-an", "-c:v", "libx264", "-crf", "12", "-preset", "slow",
            "-pix_fmt", "yuv420p", out]


def read_generations(export: Path) -> list[dict]:
    with open(export / "generations.csv", newline="") as f:
        return list(csv.DictReader(f))


def prepare(export: Path, natix_root: Path, out: Path, num_frames: int = 121, fps: float = 24.0,
            ffmpeg_bin: Optional[str] = None) -> Path:
    """Cut the reference clips into out/reference/ and write out/clips.csv; returns the clips.csv path."""
    ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg was not found on PATH")
    rows = read_generations(export)
    (out / "reference").mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    references: dict[str, str] = {}
    for row in rows:
        sample_id = row["sample_id"]
        video = export / row["file"]
        clips.append({"clip_id": sample_id, "side": f"{row['profile']}_{row['mode']}", "video": str(video)})
        if sample_id in references:
            continue
        source, start_frame = split_sample_id(sample_id)
        target = out / "reference" / f"{row['sample']}.mp4"
        if not target.is_file():
            size = probe_frame_size(str(video))
            tmp = target.with_name(target.stem + ".tmp.mp4")
            subprocess.run(reference_command(ffmpeg_bin, str(natix_root / source), start_frame, num_frames, fps,
                                             size, str(tmp)), check=True)
            tmp.replace(target)
            logger.info("cut %s", target)
        references[sample_id] = str(target)
    clips += [{"clip_id": sid, "side": "reference", "video": path} for sid, path in references.items()]
    clips_csv = out / "clips.csv"
    with open(clips_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "side", "video"])
        writer.writeheader()
        writer.writerows(clips)
    return clips_csv


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="cut reference clips and write clips.csv")
    prep.add_argument("--export", type=Path, required=True, help="generation export with generations.csv")
    prep.add_argument("--natix-root", type=Path, required=True, help="root the sample ids' video paths are under")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--num-frames", type=int, default=121)
    args = parser.parse_args(argv)
    clips_csv = prepare(args.export, args.natix_root, args.out, args.num_frames)
    logger.info("clips written to %s", clips_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
