"""
Topology adherence of generated driving videos (evaluation part 1): does the video show the road topology of the
OpenStreetMap data it was conditioned on?

Step 1, facts: closed-set topology facts per clip from OSM along the clip's trajectory (evaluation.osm):
  python -m evaluation.topology facts --trajectories trajs/ --out facts.json
  (one CSV per clip, named <clip_id>.csv, columns lat,lon in time order; OSM cached in --cache)

Step 2, score: each fact asked as a multiple-choice question (evaluation.vqa) on the generated and/or the
ground-truth video of the clip, answers scored against the facts:
  python -m evaluation.topology score --clips clips.csv --facts facts.json --out results/model_a \\
      --vqa-url http://localhost:8000/v1 --model <served VLM>
  clips.csv: clip_id, generated, reference (either video may be empty, e.g. reference only to validate the metric)

Per attribute and video (generated / reference), in --out:
  answers.jsonl       every question and reply (a rerun skips questions already answered)
  per_attribute.csv   accuracy against the facts; `normalized` = generated accuracy / reference accuracy on the
                      clips that have both (the VLM's errors on real video are not the generator's); `agreement` =
                      generated and reference gave the same answer; `control` = accuracy against the facts of
                      another clip (a wrong-location map), which a metric that sees topology must score lower;
                      `chance` = 1 / number of options, `majority` = always the most frequent fact
  summary.json        the same, averaged over attributes (macro)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evaluation.osm import ATTRIBUTES, OVERPASS_URL, facts_for_trajectories, read_trajectory
from evaluation.vqa import OpenAICompatibleVqa, VqaModel, question_for, sample_frames

logger = logging.getLogger(__name__)
SIDES = ("generated", "reference")


@dataclass(frozen=True)
class Clip:
    clip_id: str
    videos: dict[str, str]  # side -> path, only the sides given


def read_clips(path: Path) -> list[Clip]:
    with open(path, newline="") as f:
        return [Clip(row["clip_id"], {s: row[s] for s in SIDES if (row.get(s) or "").strip()})
                for row in csv.DictReader(f)]


def ask_all(
    clips: list[Clip],
    facts: dict[str, dict[str, str]],
    model: VqaModel,
    answers_path: Path,
    num_frames: int = 8,
) -> list[dict]:
    """Ask every fact of every clip on each of its videos; answers already in `answers_path` are kept."""
    done: dict[tuple[str, str, str], dict] = {}
    if answers_path.is_file():
        for line in answers_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("model") == model.name:
                done[(record["clip_id"], record["side"], record["attribute"])] = record
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with answers_path.open("a") as out:
        for clip in clips:
            clip_facts = facts.get(clip.clip_id, {})
            for side, video in clip.videos.items():
                todo = [a for a in clip_facts if a in ATTRIBUTES and (clip.clip_id, side, a) not in done]
                if not todo:
                    continue
                frames = sample_frames(video, num_frames)
                for attribute in todo:
                    question = question_for(attribute)
                    index, reply = model.answer(frames, question, video)
                    record = {
                        "clip_id": clip.clip_id, "side": side, "attribute": attribute, "model": model.name,
                        "truth": clip_facts[attribute], "answer": None if index is None else question.options[index],
                        "reply": reply,
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    done[(clip.clip_id, side, attribute)] = record
    return list(done.values())


def _rate(hits: list[bool]) -> Optional[float]:
    return sum(hits) / len(hits) if hits else None


def score(answers: list[dict]) -> list[dict]:
    """One row per attribute with the metrics described in the module docstring."""
    rows = []
    for attribute in ATTRIBUTES:
        records = [r for r in answers if r["attribute"] == attribute]
        if not records:
            continue
        by_side = {s: {r["clip_id"]: r for r in records if r["side"] == s} for s in SIDES}
        truths = {r["clip_id"]: r["truth"] for r in records}
        clip_ids = sorted(truths)
        # control: each clip's answer against the facts of the next clip (rotation), i.e. another place's map
        control_truth = {c: truths[clip_ids[(i + 1) % len(clip_ids)]] for i, c in enumerate(clip_ids)}
        majority = max(set(truths.values()), key=list(truths.values()).count)
        row: dict = {"attribute": attribute, "clips": len(clip_ids), "chance": 1 / len(ATTRIBUTES[attribute]),
                     "majority": sum(t == majority for t in truths.values()) / len(truths)}
        for side in SIDES:
            side_records = by_side[side]
            row[f"n_{side}"] = len(side_records)
            row[f"accuracy_{side}"] = _rate([r["answer"] == r["truth"] for r in side_records.values()])
            row[f"control_{side}"] = (
                _rate([r["answer"] == control_truth[c] for c, r in side_records.items()])
                if len(clip_ids) > 1 else None
            )
            row[f"unanswered_{side}"] = sum(r["answer"] is None for r in side_records.values())
        both = sorted(set(by_side["generated"]) & set(by_side["reference"]))
        gen_both = _rate([by_side["generated"][c]["answer"] == truths[c] for c in both])
        ref_both = _rate([by_side["reference"][c]["answer"] == truths[c] for c in both])
        row["normalized"] = gen_both / ref_both if gen_both is not None and ref_both else None
        row["agreement"] = _rate([by_side["generated"][c]["answer"] == by_side["reference"][c]["answer"]
                                  for c in both])
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k not in ("attribute", "clips")] if rows else []
    summary: dict = {"attributes": len(rows)}
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
    return summary


def write_results(rows: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_dir / "per_attribute.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    facts_cmd = sub.add_parser("facts", help="topology facts per clip from OSM along its trajectory")
    facts_cmd.add_argument("--trajectories", type=Path, required=True, help="folder of <clip_id>.csv (lat,lon)")
    facts_cmd.add_argument("--out", type=Path, required=True, help="facts JSON to write")
    facts_cmd.add_argument("--cache", type=Path, default=Path("_cache/osm"), help="Overpass response cache")
    facts_cmd.add_argument("--overpass-url", default=OVERPASS_URL)

    score_cmd = sub.add_parser("score", help="ask the facts on the videos and score the answers")
    score_cmd.add_argument("--clips", type=Path, required=True, help="CSV clip_id,generated,reference")
    score_cmd.add_argument("--facts", type=Path, required=True)
    score_cmd.add_argument("--out", type=Path, required=True)
    score_cmd.add_argument("--vqa-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    score_cmd.add_argument("--model", required=True, help="model name as served")
    score_cmd.add_argument("--frames", type=int, default=8, help="frames sampled per video")
    args = parser.parse_args(argv)

    if args.command == "facts":
        trajectories = [(p.stem, read_trajectory(p)) for p in sorted(args.trajectories.glob("*.csv"))]
        facts = facts_for_trajectories(trajectories, args.cache, args.overpass_url)
        args.out.write_text(json.dumps(facts, indent=1, sort_keys=True))
        logger.info("facts of %d clips written to %s", len(facts), args.out)
        return 0

    clips = read_clips(args.clips)
    facts = json.loads(args.facts.read_text())
    model = OpenAICompatibleVqa(args.vqa_url, args.model)
    answers = ask_all(clips, facts, model, args.out / "answers.jsonl", args.frames)
    summary = write_results(score(answers), args.out)
    logger.info("summary: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
