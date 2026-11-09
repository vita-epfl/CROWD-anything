"""
Speed of the other vehicles in generated driving videos (does the traffic drive at speeds that fit the road's OSM
speed limit, as in the real clip?). A VLM cannot read speed off frames, so it is measured in 3D.

Measurement (`measure`, in the ViPE environment): ViPE (`vipe infer <video> -o <dir>`) gives per frame the metric
depth, the camera pose (OpenCV camera-to-world) and intrinsics, and tracked instance masks with a phrase each. For
every vehicle instance and frame, the mask pixels nearest the camera (its visible surface; ``NEAR_QUANTILE``) are
lifted to 3D and put in world coordinates with the camera pose; a vehicle's speed is the robust (Theil-Sen) slope
of its position on the ground plane (x, z of ViPE's first camera) over time. Vehicles cut by the image's left or
right edge, smaller than MIN_AREA_PX or further than MAX_DISTANCE_M are left out of that frame, and tracks shorter
than MIN_TRACK_S are left out: what remains is mostly the traffic ahead and oncoming (parked cars are passed within a
second or two). Checked by eye on the 30 real clips: cars ahead on a motorway at 76 km/h measure 73-86 km/h, a queue
0-6 km/h.

Scale: ViPE's metric scale is off by up to ~1.7x per video even on real clips, so the camera car's speed from ViPE's
poses (over 1 s steps) is compared with the real one (the generated car follows the real trajectory); `scale` = real
/ ViPE ego speed rescales the vehicle speeds (when the car moves at MIN_EGO_KMH or more). Videos whose scale is off by
more than MAX_SCALE_ERROR (a collapsed depth map) are left out.

Scoring (`score`, standard library only): per video, the moving vehicles (speed >= MOVING_KMH) and their median
speed; per side, against the real clip of the same scene (difference of median speeds) and against the OSM speed
limit of the road the car is on (median speed / limit, share of moving vehicles above the limit).

  python -m evaluation.vehicle_speed measure --videos videos.csv --vipe vipe/ --out tracks.csv
      (videos.csv: id,video; ViPE output of video id in vipe/<id>/)
  python -m evaluation.vehicle_speed score --tracks tracks.csv --scenes scenes.csv --out results/
      (scenes.csv: id, scene, side, ego_kmh, limit_kmh; side `reference` = the real clip)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)
REFERENCE = "reference"
VEHICLES = ("car", "bus", "truck", "vehicle", "van", "motorcycle")
NEAR_QUANTILE = 0.3  # mask pixels up to this depth quantile are the vehicle's visible surface
MIN_AREA_PX = 150
MAX_DISTANCE_M = 30.0
EDGE_PX = 2  # a mask within this many pixels of the left or right image edge is cut by it
MIN_TRACK_S = 1.5
MIN_PAIR_S = 0.25  # Theil-Sen uses pairs of samples at least this far apart
MOVING_KMH = 5.0
MIN_EGO_KMH = 10.0  # below this the scale is not estimated
# videos with scale outside [1 / x, x] are left out: a collapsed depth map (17.9x on one real clip); corrections of
# 3-5x still gave the right speeds by eye
MAX_SCALE_ERROR = 10.0


def is_vehicle(phrase: str) -> bool:
    return any(word in phrase.lower() for word in VEHICLES)


def fit_speed(times: Sequence[float], points: Sequence[Sequence[float]]) -> tuple[float, float]:
    """(speed in m/s, median distance in m of the points from the fitted line) of 3D points over time: Theil-Sen
    slopes of x and z (the ground plane of an OpenCV camera frame) over pairs at least MIN_PAIR_S apart."""
    import numpy as np

    t = np.asarray(times, dtype=float)
    p = np.asarray(points, dtype=float)[:, [0, 2]]
    i, j = np.triu_indices(len(t), 1)
    keep = t[j] - t[i] >= MIN_PAIR_S
    if not keep.any():
        return float("nan"), float("nan")
    slope = np.median((p[j] - p[i])[keep] / (t[j] - t[i])[keep, None], axis=0)
    offset = np.median(p - t[:, None] * slope, axis=0)
    residual = np.linalg.norm(p - (t[:, None] * slope + offset), axis=1)
    return float(np.linalg.norm(slope)), float(np.median(residual))


def ego_speed(centres: Sequence[Sequence[float]], fps: float) -> float:
    """Mean speed (m/s) of the camera along its path, over 1 s steps (frame-to-frame steps add up jitter)."""
    import numpy as np

    step = max(1, round(fps))
    c = np.asarray(centres, dtype=float)[::step]
    if len(c) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum() / (len(c) - 1))


def vehicle_points(depth, mask, phrases: dict[int, str], intrinsics: Sequence[float], c2w) -> dict[int, list[float]]:
    """{instance: world position} of the vehicles in one frame (depth [H, W] metres, mask [H, W] instance ids)."""
    import numpy as np

    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    if mask.shape != depth.shape:  # masks and depth come from the same frames; resize the mask if they differ
        rows = np.arange(height) * mask.shape[0] // height
        cols = np.arange(width) * mask.shape[1] // width
        mask = mask[rows][:, cols]
    out = {}
    for instance in np.unique(mask):
        if instance == 0 or not is_vehicle(phrases.get(int(instance), "")):
            continue
        v, u = np.nonzero(mask == instance)
        if len(u) < MIN_AREA_PX or u.min() < EDGE_PX or u.max() >= width - EDGE_PX:
            continue
        z = depth[v, u]
        valid = np.isfinite(z) & (z > 0)
        if valid.sum() < MIN_AREA_PX // 2:
            continue
        u, v, z = u[valid], v[valid], z[valid]
        near = z <= np.quantile(z, NEAR_QUANTILE)
        u, v, z = u[near], v[near], z[near]
        if np.median(z) > MAX_DISTANCE_M:
            continue
        camera = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
        centre = np.median(camera, axis=0)
        out[int(instance)] = (c2w[:3, :3] @ centre + c2w[:3, 3]).tolist()
    return out


def measure_video(vipe_dir: Path, fps: float = 24.0) -> tuple[list[dict], float]:
    """(one row per vehicle track, ViPE ego speed in m/s) from a ViPE output folder."""
    import numpy as np
    from vipe.utils.io import read_depth_artifacts, read_instance_artifacts, read_instance_phrases

    name = next((vipe_dir / "pose").glob("*.npz")).stem
    pose = np.load(vipe_dir / "pose" / f"{name}.npz")
    poses = dict(zip(pose["inds"].tolist(), pose["data"]))
    intr = np.load(vipe_dir / "intrinsics" / f"{name}.npz")
    intrinsics = dict(zip(intr["inds"].tolist(), intr["data"]))
    phrases = read_instance_phrases(vipe_dir / "mask" / f"{name}.txt")
    masks = {i: m.numpy() for i, m in read_instance_artifacts(vipe_dir / "mask" / f"{name}.zip")}
    tracks: dict[int, list[tuple[float, list[float]]]] = defaultdict(list)
    for frame, depth in read_depth_artifacts(vipe_dir / "depth" / f"{name}.zip"):
        if frame not in masks or frame not in poses or frame not in intrinsics:
            continue
        for instance, point in vehicle_points(depth.numpy(), masks[frame], phrases, intrinsics[frame],
                                              poses[frame]).items():
            tracks[instance].append((frame / fps, point))
    rows = []
    for instance, samples in sorted(tracks.items()):
        times = [t for t, _ in samples]
        if len(samples) < 3 or times[-1] - times[0] < MIN_TRACK_S:
            continue
        speed, residual = fit_speed(times, [p for _, p in samples])
        camera = [poses[round(t * fps)][:3, 3] for t in times]
        distance = statistics.median(float(np.linalg.norm(np.asarray(p) - c)) for (_, p), c in zip(samples, camera))
        rows.append({"instance": instance, "phrase": phrases.get(instance, ""), "frames": len(samples),
                     "duration_s": round(times[-1] - times[0], 2), "distance_m": round(distance, 1),
                     "speed_kmh_raw": round(speed * 3.6, 1), "residual_m": round(residual, 2)})
    centres = [poses[i][:3, 3] for i in sorted(poses)]
    return rows, ego_speed(centres, fps)


def summarize(tracks: list[dict], scenes: list[dict]) -> tuple[list[dict], list[dict]]:
    """(per video, per side) rows; tracks: id, speed_kmh; scenes: id, scene, side, limit_kmh."""
    by_video = defaultdict(list)
    for t in tracks:
        by_video[t["id"]].append(float(t["speed_kmh"]))
    videos = []
    for s in scenes:
        moving = [v for v in by_video.get(s["id"], []) if v >= MOVING_KMH]
        limit = float(s["limit_kmh"]) if s.get("limit_kmh") else None
        videos.append({"id": s["id"], "scene": s["scene"], "side": s["side"], "limit_kmh": limit,
                       "vehicles": len(by_video.get(s["id"], [])), "moving": len(moving),
                       "median_moving_kmh": statistics.median(moving) if moving else None,
                       "above_limit": sum(v > limit for v in moving) if limit and moving else None})
    reference = {v["scene"]: v for v in videos if v["side"] == REFERENCE}
    sides = []
    for side in sorted({v["side"] for v in videos}, key=lambda s: (s != REFERENCE, s)):
        rows = [v for v in videos if v["side"] == side]
        speeds = [v["median_moving_kmh"] for v in rows if v["median_moving_kmh"] is not None]
        ratios = [v["median_moving_kmh"] / v["limit_kmh"] for v in rows
                  if v["median_moving_kmh"] is not None and v["limit_kmh"]]
        moving = sum(v["moving"] for v in rows if v["limit_kmh"])
        above = sum(v["above_limit"] or 0 for v in rows if v["limit_kmh"])
        diffs = [v["median_moving_kmh"] - reference[v["scene"]]["median_moving_kmh"] for v in rows
                 if v["median_moving_kmh"] is not None and v["scene"] in reference
                 and reference[v["scene"]]["median_moving_kmh"] is not None]
        sides.append({"side": side, "videos": len(rows), "with_moving": len(speeds),
                      "vehicles_per_video": statistics.mean(v["vehicles"] for v in rows) if rows else None,
                      "median_moving_kmh": statistics.median(speeds) if speeds else None,
                      "speed_over_limit": statistics.median(ratios) if ratios else None,
                      "share_above_limit": above / moving if moving else None,
                      "diff_vs_reference_kmh": statistics.median(diffs) if side != REFERENCE and diffs else None,
                      "abs_diff_vs_reference_kmh": statistics.median(abs(d) for d in diffs)
                      if side != REFERENCE and diffs else None})
    return videos, sides


def _write_csv(rows: list[dict], path: Path) -> None:
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    measure_cmd = sub.add_parser("measure", help="vehicle tracks and their speeds from ViPE outputs")
    measure_cmd.add_argument("--videos", type=Path, required=True, help="CSV id,video")
    measure_cmd.add_argument("--vipe", type=Path, required=True, help="folder with one ViPE output per video id")
    measure_cmd.add_argument("--out", type=Path, required=True, help="tracks CSV to write")
    measure_cmd.add_argument("--fps", type=float, default=24.0)
    score_cmd = sub.add_parser("score", help="compare vehicle speeds with the real clip and the speed limit")
    score_cmd.add_argument("--tracks", type=Path, required=True)
    score_cmd.add_argument("--scenes", type=Path, required=True, help="CSV id,scene,side,ego_kmh,limit_kmh")
    score_cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "measure":
        rows, egos = [], []
        with open(args.videos, newline="") as f:
            ids = [r["id"] for r in csv.DictReader(f)]
        for video_id in ids:
            folder = args.vipe / video_id
            if not (folder / "pose").exists():
                logger.warning("no ViPE output for %s", video_id)
                continue
            tracks, ego = measure_video(folder, args.fps)
            rows += [{"id": video_id, **t} for t in tracks]
            egos.append({"id": video_id, "vipe_ego_kmh": round(ego * 3.6, 1)})
        _write_csv(rows, args.out)
        _write_csv(egos, args.out.with_name(args.out.stem + "_ego.csv"))
        logger.info("%d vehicle tracks in %d videos", len(rows), len(egos))
        return 0

    with open(args.scenes, newline="") as f:
        scenes = list(csv.DictReader(f))
    with open(args.tracks, newline="") as f:
        raw = list(csv.DictReader(f))
    ego_file = args.tracks.with_name(args.tracks.stem + "_ego.csv")
    with open(ego_file, newline="") as f:
        vipe_ego = {r["id"]: float(r["vipe_ego_kmh"]) for r in csv.DictReader(f)}
    real_ego = {s["id"]: float(s["ego_kmh"]) for s in scenes if s.get("ego_kmh")}
    scale = {i: real_ego[i] / v for i, v in vipe_ego.items()
             if i in real_ego and real_ego[i] >= MIN_EGO_KMH and v > 0}
    failed = {i for i, x in scale.items() if not 1 / MAX_SCALE_ERROR <= x <= MAX_SCALE_ERROR}
    for i in sorted(failed):
        logger.warning("ViPE scale of %s off by %.1fx, left out", i, scale[i])
    tracks = [dict(t, scale=scale.get(t["id"], 1.0), speed_kmh=float(t["speed_kmh_raw"]) * scale.get(t["id"], 1.0))
              for t in raw if t["id"] not in failed]
    scenes = [s for s in scenes if s["id"] not in failed]
    videos, sides = summarize(tracks, scenes)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(tracks, args.out / "tracks.csv")
    _write_csv([{"id": i, "real_ego_kmh": real_ego.get(i), "vipe_ego_kmh": v, "scale": scale.get(i)}
                for i, v in vipe_ego.items()], args.out / "ego.csv")
    _write_csv(videos, args.out / "per_video.csv")
    _write_csv(sides, args.out / "summary.csv")
    for row in sides:
        logger.info("%s", json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
