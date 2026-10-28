"""
Compare generated videos with their ground truth frame by frame (evaluation part 2: how exactly a model
reproduces the real video from its first frame, the reality signals and the ego trajectory).

Pairs come from two folders (files matched by name without extension) or from a CSV with columns
`clip_id,generated,reference`. Both videos are read on the same fps grid from t=0; the ground truth is scaled
and centre-cropped to the generated size. PSNR, SSIM and LPIPS are computed for every frame and averaged over
the first N seconds for each horizon; a clip counts towards a horizon only when both videos are that long.

Outputs in --out:
  per_frame.csv  clip_id, frame, time_s, one column per metric
  per_clip.csv   clip_id, horizon_s, num_frames, mean of each metric over the horizon
  summary.csv    horizon_s, num_clips, mean and std over clips of each metric

Example:
  python -m evaluation.paired --generated gen/ --reference gt/ --out results/model_a --horizons 5,10,30,60
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from evaluation.metrics import Lpips, default_device, psnr, ssim, to_tensor
from evaluation.video import iter_frame_pairs

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
METRICS = ("psnr", "ssim", "lpips")


@dataclass(frozen=True)
class Pair:
    clip_id: str
    generated: str
    reference: str


def pairs_from_folders(generated_dir: Path, reference_dir: Path) -> list[Pair]:
    def videos(folder: Path) -> dict[str, Path]:
        return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_EXTENSIONS}

    generated, reference = videos(generated_dir), videos(reference_dir)
    missing = sorted(set(generated) - set(reference))
    if missing:
        logger.warning("%d generated videos have no ground truth, skipped (e.g. %s)", len(missing), missing[:3])
    return [Pair(stem, str(path), str(reference[stem])) for stem, path in generated.items() if stem in reference]


def pairs_from_csv(path: Path) -> list[Pair]:
    with open(path, newline="") as f:
        return [Pair(row["clip_id"], row["generated"], row["reference"]) for row in csv.DictReader(f)]


def metric_functions(names: list[str], device: torch.device) -> dict[str, Callable]:
    functions: dict[str, Callable] = {"psnr": psnr, "ssim": ssim}
    if "lpips" in names:
        functions["lpips"] = Lpips(device)
    return {name: functions[name] for name in names}


def score_pair(
    pair: Pair,
    metrics: dict[str, Callable],
    device: torch.device,
    fps: float,
    batch_size: int,
    max_frames: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Per-frame values of each metric for one pair, as arrays of the number of frames both videos have."""
    values: dict[str, list[np.ndarray]] = {name: [] for name in metrics}
    batch_gen: list[np.ndarray] = []
    batch_ref: list[np.ndarray] = []

    def flush() -> None:
        if not batch_gen:
            return
        gen = to_tensor(np.stack(batch_gen), device)
        ref = to_tensor(np.stack(batch_ref), device)
        for name, fn in metrics.items():
            values[name].append(fn(gen, ref).float().cpu().numpy())
        batch_gen.clear()
        batch_ref.clear()

    for gen_frame, ref_frame in iter_frame_pairs(pair.generated, pair.reference, fps, max_frames):
        batch_gen.append(gen_frame)
        batch_ref.append(ref_frame)
        if len(batch_gen) == batch_size:
            flush()
    flush()
    return {name: np.concatenate(v) if v else np.empty(0) for name, v in values.items()}


def horizon_means(per_frame: dict[str, np.ndarray], horizons_s: list[float], fps: float) -> list[dict]:
    """One row per horizon the clip is long enough for, plus 'all' (the whole clip)."""
    num_frames = len(next(iter(per_frame.values())))
    rows = []
    for horizon in horizons_s:
        n = int(round(horizon * fps))
        if n <= num_frames:
            rows.append({"horizon_s": f"{horizon:g}", "num_frames": n,
                         **{name: float(v[:n].mean()) for name, v in per_frame.items()}})
    if num_frames:
        rows.append({"horizon_s": "all", "num_frames": num_frames,
                     **{name: float(v.mean()) for name, v in per_frame.items()}})
    return rows


def summarise(per_clip: list[dict], metric_names: list[str]) -> list[dict]:
    summary = []
    horizons = list(dict.fromkeys(row["horizon_s"] for row in per_clip))
    for horizon in horizons:
        rows = [row for row in per_clip if row["horizon_s"] == horizon]
        out = {"horizon_s": horizon, "num_clips": len(rows)}
        for name in metric_names:
            v = np.array([row[name] for row in rows])
            out[f"{name}_mean"] = float(v.mean())
            out[f"{name}_std"] = float(v.std())
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    pairs: list[Pair],
    out_dir: Path,
    metric_names: list[str],
    horizons_s: list[float],
    fps: float = 24.0,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> list[dict]:
    device = device or default_device()
    metrics = metric_functions(metric_names, device)
    max_frames = int(round(max(horizons_s) * fps)) if horizons_s else None
    out_dir.mkdir(parents=True, exist_ok=True)
    per_frame_rows: list[dict] = []
    per_clip_rows: list[dict] = []
    for i, pair in enumerate(pairs, 1):
        per_frame = score_pair(pair, metrics, device, fps, batch_size, max_frames)
        num_frames = len(next(iter(per_frame.values())))
        if not num_frames:
            logger.warning("%s: no frames decoded, skipped", pair.clip_id)
            continue
        for frame in range(num_frames):
            per_frame_rows.append({"clip_id": pair.clip_id, "frame": frame, "time_s": round(frame / fps, 4),
                                   **{name: float(v[frame]) for name, v in per_frame.items()}})
        for row in horizon_means(per_frame, horizons_s, fps):
            per_clip_rows.append({"clip_id": pair.clip_id, **row})
        logger.info("[%d/%d] %s: %d frames", i, len(pairs), pair.clip_id, num_frames)
    summary = summarise(per_clip_rows, metric_names)
    write_csv(out_dir / "per_frame.csv", per_frame_rows)
    write_csv(out_dir / "per_clip.csv", per_clip_rows)
    write_csv(out_dir / "summary.csv", summary)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--generated", type=Path, help="folder of generated videos (with --reference)")
    source.add_argument("--pairs", type=Path, help="CSV with columns clip_id,generated,reference")
    parser.add_argument("--reference", type=Path, help="folder of ground-truth videos, same file names")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", default=",".join(METRICS), help=f"comma-separated subset of {METRICS}")
    parser.add_argument("--horizons", default="5,10,30,60", help="seconds from the first frame, comma-separated")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: the best available)")
    args = parser.parse_args(argv)

    if args.generated and not args.reference:
        parser.error("--generated needs --reference")
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = set(metric_names) - set(METRICS)
    if unknown:
        parser.error(f"unknown metrics: {sorted(unknown)}")
    horizons = [float(h) for h in args.horizons.split(",") if h.strip()]
    pairs = pairs_from_csv(args.pairs) if args.pairs else pairs_from_folders(args.generated, args.reference)
    if not pairs:
        parser.error("no video pairs found")
    device = torch.device(args.device) if args.device else None
    summary = run(pairs, args.out, metric_names, horizons, args.fps, args.batch_size, device)
    for row in summary:
        logger.info("%s", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
