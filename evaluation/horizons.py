"""
Inputs of the long-horizon evaluation: 5 s windows of one-minute (or longer) generations, each with the real 5 s NATIX
sample under it, its caption row, its OSM route record and its anchor images, from Valentin's anchor-covered
selection (e.g. tmp_shared/rgwm-pablo-anchor-covered-20261108/60s: windows_anchors.jsonl, sample_ids.txt).

A window of a video is matched to its sample through windows_anchors.jsonl (the source chunk and 5 s segment of the
window's start), not through the caption rows' `t_s` or `captions[i]`: after a recording join they drift apart. The
real clip of a window is that sample's 121 frames on the canonical 24 fps grid (evaluation.rgwm); a generated
window is frames [start_s * 24, start_s * 24 + 121) of the generated video.

  python -m evaluation.horizons prepare --selection <selection>/60s --inputs one_min_418.jsonl \
      --captions captions_topology_free.csv --records <osm_bev/text_predicted> --windows 0,10,25,55 \
      [--generations generations.csv] [--limit 40] --out run/
      (generations.csv: video_id, side, video; video_id as in the selection's sample_ids.txt)
  python -m evaluation.horizons cut --run run/ --natix-root <NATIX videos> --anchor-images anchor_images.jsonl
  python -m evaluation.horizons merge-statements --statements split.json --out run/statements.json

prepare writes windows.csv, cuts.csv, captions.csv (scene and action caption of each window, for
evaluation.caption_statements), facts.json (evaluation.topology, without `turn`: the ego motion is given), scenes.csv
(evaluation.vehicle_speed: real ego speed and OSM limit) and anchors.csv; cut writes the clips, clips.csv (clip_id =
window id) and anchors.json (evaluation.traffic_elements).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
REFERENCE = "reference"
FPS, NUM_FRAMES = 24, 121
ANCHOR_RANKS = 5  # the generator's loader takes up to 5 current and 5 future anchors per window


def sample_of(chunk: str, segment: int) -> str:
    """NATIX sample id of a 5 s segment of a recording ('<country>/<date>/<session>/<HH-MM-SS>'); the selection
    writes countries and states with underscores ("United_States/New_Jersey"), sample ids with spaces."""
    parts = chunk.split("/")
    parts[:-3] = [p.replace("_", " ") for p in parts[:-3]]
    return "%s/FRONT_FOLDER/FRONT_%s_%s.mp4#f%06d" % ("/".join(parts), parts[-3], parts[-1], 120 * segment)


def window_sample(window: dict) -> Optional[str]:
    """The sample under a window's start (its `current` endpoint, else the one before its `future` endpoint)."""
    if window.get("current_source_chunk"):
        return sample_of(window["current_source_chunk"], window["current_source_segment"])
    if window.get("future_source_chunk") and window.get("future_source_segment", 0) > 0:
        return sample_of(window["future_source_chunk"], window["future_source_segment"] - 1)
    return None


def window_id(clip_id: str, start_s: float) -> str:
    return f"{clip_id}@{start_s:g}"


def merge_statements(split: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """{window: scene statements + action statements} from caption_statements output keyed <window>#scene/#action."""
    out: dict[str, list[dict]] = {}
    for key in sorted(split):
        window, _, kind = key.rpartition("#")
        if kind in ("scene", "action"):
            out.setdefault(window, [])
    for window in out:
        out[window] = split.get(f"{window}#scene", []) + split.get(f"{window}#action", [])
    return out


def scene_row(record: Optional[dict]) -> dict:
    """Real ego speed and the OSM limit of the road the car is on longest, from a route record."""
    if not record:
        return {"ego_kmh": "", "limit_kmh": "", "limit_source": "", "road_class": ""}
    roads = record.get("roads") or []
    road = max(roads, key=lambda r: r["t_end_s"] - r["t_start_s"]) if roads else {}
    return {"ego_kmh": (record.get("motion") or {}).get("mean_speed", ""), "limit_kmh": road.get("speed_limit") or "",
            "limit_source": road.get("speed_limit_source") or "", "road_class": road.get("road_class") or ""}


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> None:
    from evaluation.route_records import facts_from_record, load_records

    starts = [float(s) for s in args.windows.split(",")]
    selected = (args.selection / "sample_ids.txt").read_text().split()[:args.limit or None]
    with open(args.inputs) as f:
        clips = {c["sample_id"]: c for c in map(json.loads, f) if c["sample_id"] in set(selected)}
    windows: list[tuple[dict, str]] = []  # (window, its sample)
    with open(args.selection / "windows_anchors.jsonl") as f:
        for line in f:
            w = json.loads(line)
            sample = window_sample(w)
            if w["video_id"] in clips and float(w["start_s"]) in starts and sample:
                windows.append((w, sample))
    windows.sort(key=lambda ws: (selected.index(ws[0]["video_id"]), ws[0]["start_s"]))
    with open(args.captions, newline="") as f:
        captions = {(r["video_id"], r["row_sample_id"]): r for r in csv.DictReader(f)}
    generations: dict[str, dict[str, str]] = {}
    if args.generations:
        with open(args.generations, newline="") as f:
            for r in csv.DictReader(f):
                generations.setdefault(r["video_id"], {})[r["side"]] = r["video"]
    records = load_records(args.records, [sample for _, sample in windows])

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows, cuts, caption_rows, scenes, anchors, facts = [], [], [], [], [], {}
    for n, (w, sample) in enumerate(windows):
        clip = clips[w["video_id"]]
        wid = window_id(clip["eval_sample_id"], float(w["start_s"]))
        tag = f"w{n:04d}"
        caption = captions.get((w["video_id"], sample))
        rows.append({"window": wid, "tag": tag, "video_id": w["video_id"], "clip_id": clip["eval_sample_id"],
                     "start_s": w["start_s"], "sample_id": sample, "caption": bool(caption),
                     "record": sample in records})
        sides = {REFERENCE: ("sample", sample, 0), **{side: ("generated", video, round(float(w["start_s"]) * FPS))
                                                      for side, video in generations.get(w["video_id"], {}).items()}}
        for side, (kind, source, start_frame) in sides.items():
            cuts.append({"window": wid, "tag": tag, "side": side, "kind": kind, "source": source,
                         "start_frame": start_frame, "out": str(out / "clips" / side / f"{tag}.mp4")})
            scenes.append({"id": f"{tag}__{side}", "scene": wid, "side": side, **scene_row(records.get(sample))})
        if caption:
            caption_rows += [{"sample_id": f"{wid}#scene", "caption": caption["scene_caption"]},
                             {"sample_id": f"{wid}#action", "caption": caption["action_caption"]}]
        if sample in records:
            facts[wid] = {a: v for a, v in facts_from_record(records[sample]).items() if a != "turn"}
        for role in ("current", "future"):
            for a in w[role]:
                if a["rank"] < ANCHOR_RANKS:
                    anchors.append({"window": wid, "candidate_id": a["candidate_id"], "role": role, "rank": a["rank"],
                                    "out": str(out / "anchors" / tag / f"{role}_{a['rank']}.png")})
    _write_csv(rows, out / "windows.csv")
    _write_csv(cuts, out / "cuts.csv")
    _write_csv(caption_rows, out / "captions.csv")
    _write_csv(scenes, out / "scenes.csv")
    _write_csv(anchors, out / "anchors.csv")
    (out / "facts.json").write_text(json.dumps(facts, indent=1))
    logger.info("%d windows of %d clips: %d with a caption row, %d with an OSM record; %d cuts, %d anchor images",
                len(rows), len(clips), sum(r["caption"] for r in rows), len(facts), len(cuts), len(anchors))


def cut(args: argparse.Namespace) -> None:
    from evaluation.rgwm import reference_command

    run = args.run
    with open(run / "cuts.csv", newline="") as f:
        cuts = list(csv.DictReader(f))
    with open(run / "anchors.csv", newline="") as f:
        anchors = list(csv.DictReader(f))
    width, height = (int(x) for x in args.size.split("x"))

    def one(c: dict) -> Optional[dict]:
        target = Path(c["out"])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.stem + ".tmp.mp4")
            start = int(c["start_frame"])
            if c["kind"] == "sample":
                source, _, frame = c["source"].rpartition("#f")
                source, start = str(args.natix_root / source), int(frame)
            else:
                source = c["source"]
            command = reference_command(args.ffmpeg, source, start, NUM_FRAMES, FPS, (width, height), str(tmp))
            if subprocess.run(command, capture_output=True).returncode or not tmp.exists():
                logger.warning("cut failed: %s %s", c["window"], c["side"])
                return None
            tmp.replace(target)
        return {"clip_id": c["window"], "side": c["side"], "video": c["out"]}

    with ThreadPoolExecutor(args.workers) as pool:
        clips = [c for c in pool.map(one, cuts) if c]
    _write_csv(clips, run / "clips.csv")

    wanted = {a["candidate_id"] for a in anchors}
    index = {}
    with open(args.anchor_images) as f:
        for line in f:
            entry = json.loads(line)
            if entry["candidate_id"] in wanted:
                index[entry["candidate_id"]] = entry
    by_tar: dict[str, list[dict]] = {}
    for a in anchors:
        entry = index.get(a["candidate_id"])
        if entry and not Path(a["out"]).exists():
            by_tar.setdefault(entry["image_file"], []).append(dict(a, key=entry["image_key"]))
    for tar_path, items in by_tar.items():
        with tarfile.open(tar_path) as tar:
            names = {m.name: m for m in tar.getmembers()}
            for a in items:
                member = names.get(a["key"]) or names.get(a["key"] + ".png")
                if member is None:
                    logger.warning("anchor %s not in %s", a["key"], tar_path)
                    continue
                Path(a["out"]).parent.mkdir(parents=True, exist_ok=True)
                Path(a["out"]).write_bytes(tar.extractfile(member).read())  # type: ignore[union-attr]
    images: dict[str, list[str]] = {}
    for a in anchors:
        if Path(a["out"]).exists():
            images.setdefault(a["window"], []).append(a["out"])
    (run / "anchors.json").write_text(json.dumps(images, indent=1))
    logger.info("%d of %d clips cut; anchor images for %d windows", len(clips), len(cuts), len(images))


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="windows, captions, facts, scenes and the cuts and anchors to extract")
    prep.add_argument("--selection", type=Path, required=True, help="selection folder (windows_anchors.jsonl)")
    prep.add_argument("--inputs", type=Path, required=True, help="the clips' input JSONL (e.g. one_min_418.jsonl)")
    prep.add_argument("--captions", type=Path, required=True, help="captions_topology_free.csv")
    prep.add_argument("--records", type=Path, required=True, help="root of the OSM route records")
    prep.add_argument("--windows", default="0,10,25,55", help="window starts in seconds")
    prep.add_argument("--generations", type=Path, help="CSV video_id,side,video of the generated videos")
    prep.add_argument("--limit", type=int, default=0, help="only the first n clips of the selection")
    prep.add_argument("--out", type=Path, required=True)
    cut_cmd = sub.add_parser("cut", help="cut the window clips and extract the anchor images")
    cut_cmd.add_argument("--run", type=Path, required=True)
    cut_cmd.add_argument("--natix-root", type=Path, required=True)
    cut_cmd.add_argument("--anchor-images", type=Path, required=True, help="the selection's anchor_images.jsonl")
    cut_cmd.add_argument("--size", default="768x512", help="size of the cut clips (that of the generations)")
    cut_cmd.add_argument("--ffmpeg", default="ffmpeg")
    cut_cmd.add_argument("--workers", type=int, default=16)
    merge = sub.add_parser("merge-statements", help="scene + action statements per window")
    merge.add_argument("--statements", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "cut":
        cut(args)
    else:
        merged = merge_statements(json.loads(args.statements.read_text()))
        args.out.write_text(json.dumps(merged, indent=1))
        logger.info("statements for %d windows", len(merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
