"""
Topology adherence of generated driving videos (evaluation part 1): does a video show the road topology of the
OpenStreetMap data it was conditioned on?

Facts: closed-set topology facts per clip (evaluation.osm.ATTRIBUTES), either from the per-sample OSM route records
(evaluation.route_records, the NATIX ground truth) or from OSM along a trajectory (evaluation.osm):
  python -m evaluation.topology facts --records <text_predicted root> --samples generations.csv --out facts.json
  python -m evaluation.topology facts --trajectories trajs/ --out facts.json   # <clip_id>.csv with lat,lon

Videos: clips.csv in long format, columns clip_id,side,video: one row per video, `side` naming what produced it
(e.g. top_i2v, ego_only_t2v) and `reference` for the real video of the clip. clip_id is the key of facts.json.

Answers: every fact asked as a multiple-choice question on every video of its clip (evaluation.vqa), either
  - directly through an OpenAI-compatible VLM server:  score --vqa-url http://host:8000/v1 --model <name>
  - or in batch through another runner (e.g. Qwen-Drive's run_video_batch.sh): `export` writes its videos.json and
    prompts.json, and `score --batch-answers <its output json>` reads its answers.
With --leaks (evaluation.caption_leaks), facts a clip's caption gives away are not scored for that clip.

Results in --out, per attribute and side:
  answers.jsonl       every question and reply (a rerun skips questions already answered)
  per_attribute.csv   accuracy against the facts; `normalized` = side accuracy / reference accuracy on the clips that
                      have both (the VLM's errors on real video are not the generator's); `agreement` = same answer
                      as the reference video; `control` = accuracy against another clip's facts (a wrong-location
                      map), which must be clearly lower if the measure sees topology; `chance` = 1 / number of options;
                      `majority` = always the most frequent fact
  summary.csv         per side, the same averaged over attributes (macro)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

from evaluation.osm import ATTRIBUTES, OVERPASS_URL, facts_for_trajectories, read_trajectory
from evaluation.vqa import OpenAICompatibleVqa, VqaModel, parse_letter, question_for, sample_frames

logger = logging.getLogger(__name__)
REFERENCE = "reference"

Clips = dict[str, dict[str, str]]  # clip_id -> {side: video path}


def read_clips(path: Path) -> Clips:
    clips: Clips = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            clips.setdefault(row["clip_id"], {})[row["side"]] = row["video"]
    return clips


def apply_leaks(facts: dict[str, dict[str, str]], leaks: dict[str, dict]) -> dict[str, dict[str, str]]:
    """Facts without the attributes each clip's caption leaks (leaks: evaluation.caption_leaks output)."""
    return {cid: {a: v for a, v in f.items() if a not in leaks.get(cid, {}).get("leaked", {})}
            for cid, f in facts.items()}


def _record(clip_id: str, side: str, attribute: str, model: str, truth: str, answer: Optional[str], reply: str):
    return {"clip_id": clip_id, "side": side, "attribute": attribute, "model": model, "truth": truth,
            "answer": answer, "reply": reply}


def ask_all(clips: Clips, facts: dict[str, dict[str, str]], model: VqaModel, answers_path: Path,
            num_frames: int = 8) -> list[dict]:
    """Ask every fact of every clip on each of its videos; answers already in `answers_path` are kept."""
    done: dict[tuple[str, str, str], dict] = {}
    if answers_path.is_file():
        for line in answers_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("model") == model.name:
                done[(record["clip_id"], record["side"], record["attribute"])] = record
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with answers_path.open("a") as out:
        for clip_id, videos in clips.items():
            clip_facts = facts.get(clip_id, {})
            for side, video in videos.items():
                todo = [a for a in clip_facts if a in ATTRIBUTES and (clip_id, side, a) not in done]
                if not todo:
                    continue
                frames = sample_frames(video, num_frames)
                for attribute in todo:
                    question = question_for(attribute)
                    index, reply = model.answer(frames, question, video)
                    answer = None if index is None else question.options[index]
                    record = _record(clip_id, side, attribute, model.name, clip_facts[attribute], answer, reply)
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    done[(clip_id, side, attribute)] = record
    return list(done.values())


def export_batch(clips: Clips, facts: dict[str, dict[str, str]], out_dir: Path, num_frames: int = 8) -> None:
    """videos.json and prompts.json for a batch VQA runner (Qwen-Drive's run_video_batch.sh format): one prompt per
    attribute, limited to the videos whose clip has that fact."""
    videos: list[dict] = []
    by_attribute: dict[str, list[str]] = {a: [] for a in ATTRIBUTES}
    for clip_id, sides in clips.items():
        for side, path in sides.items():
            video_id = f"{clip_id}::{side}"
            videos.append({"id": video_id, "path": str(path)})
            for attribute in facts.get(clip_id, {}):
                if attribute in by_attribute:
                    by_attribute[attribute].append(video_id)
    prompts = [{"id": a, "text": question_for(a).prompt(num_frames), "video_ids": ids}
               for a, ids in by_attribute.items() if ids]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos.json").write_text(json.dumps({"videos": videos}, indent=1))
    (out_dir / "prompts.json").write_text(json.dumps({"prompts": prompts}, indent=1))


def read_batch_answers(path: Path, facts: dict[str, dict[str, str]], model_name: str) -> list[dict]:
    """Answer records from a batch runner's output ({"results": [{video_id, prompt_id, status, answer}]})."""
    records = []
    for result in json.loads(path.read_text())["results"]:
        clip_id, side = result["video_id"].split("::", 1)
        attribute = result["prompt_id"]
        if attribute not in facts.get(clip_id, {}):
            continue  # e.g. left out by --leaks
        question = question_for(attribute)
        reply = result.get("answer") or ""
        index = parse_letter(reply, len(question.options)) if result.get("status") == "ok" else None
        answer = None if index is None else question.options[index]
        records.append(_record(clip_id, side, attribute, model_name, facts[clip_id][attribute], answer, reply))
    return records


def _rate(hits: list[bool]) -> Optional[float]:
    return sum(hits) / len(hits) if hits else None


def score(answers: list[dict]) -> list[dict]:
    """One row per attribute and side with the metrics described in the module docstring."""
    rows = []
    sides = sorted({r["side"] for r in answers}, key=lambda s: (s != REFERENCE, s))
    for attribute in ATTRIBUTES:
        records = [r for r in answers if r["attribute"] == attribute]
        if not records:
            continue
        truths = {r["clip_id"]: r["truth"] for r in records}
        clip_ids = sorted(truths)
        # control: each clip's answer against the facts of the next clip (rotation), i.e. another place's map
        control = {c: truths[clip_ids[(i + 1) % len(clip_ids)]] for i, c in enumerate(clip_ids)}
        majority = max(set(truths.values()), key=list(truths.values()).count)
        reference = {r["clip_id"]: r for r in records if r["side"] == REFERENCE}
        for side in sides:
            by_clip = {r["clip_id"]: r for r in records if r["side"] == side}
            if not by_clip:
                continue
            both = sorted(set(by_clip) & set(reference))
            side_both = _rate([by_clip[c]["answer"] == truths[c] for c in both])
            ref_both = _rate([reference[c]["answer"] == truths[c] for c in both])
            rows.append({
                "attribute": attribute, "side": side, "clips": len(by_clip),
                "accuracy": _rate([r["answer"] == truths[c] for c, r in by_clip.items()]),
                "normalized": side_both / ref_both if side_both is not None and ref_both else None,
                "agreement": _rate([by_clip[c]["answer"] == reference[c]["answer"] for c in both]),
                "control": (_rate([r["answer"] == control[c] for c, r in by_clip.items()])
                            if len(clip_ids) > 1 else None),
                "chance": 1 / len(ATTRIBUTES[attribute]),
                "majority": sum(t == majority for t in truths.values()) / len(truths),
                "unanswered": sum(r["answer"] is None for r in by_clip.values()),
            })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per side, the macro average over attributes of each metric."""
    out = []
    for side in dict.fromkeys(r["side"] for r in rows):
        side_rows = [r for r in rows if r["side"] == side]
        summary: dict = {"side": side, "attributes": len(side_rows)}
        for key in ("accuracy", "normalized", "agreement", "control", "chance", "majority"):
            values = [r[key] for r in side_rows if r[key] is not None]
            summary[key] = sum(values) / len(values) if values else None
        out.append(summary)
    return out


def _write_csv(rows: list[dict], path: Path) -> None:
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_results(rows: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "per_attribute.csv")
    summary = summarize(rows)
    _write_csv(summary, out_dir / "summary.csv")
    return summary


def _load_facts(args) -> dict[str, dict[str, str]]:
    facts = json.loads(args.facts.read_text())
    return apply_leaks(facts, json.loads(args.leaks.read_text())) if args.leaks else facts


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    facts_cmd = sub.add_parser("facts", help="topology facts per clip")
    facts_cmd.add_argument("--out", type=Path, required=True, help="facts JSON to write")
    facts_cmd.add_argument("--records", type=Path, help="root of the per-sample route records (text_predicted)")
    facts_cmd.add_argument("--samples", type=Path, help="CSV with a sample_id column (with --records)")
    facts_cmd.add_argument("--trajectories", type=Path, help="folder of <clip_id>.csv (lat,lon), OSM via Overpass")
    facts_cmd.add_argument("--cache", type=Path, default=Path("_cache/osm"), help="Overpass response cache")
    facts_cmd.add_argument("--overpass-url", default=OVERPASS_URL)

    for name, help_text in (("export", "write videos.json/prompts.json for a batch VQA runner"),
                            ("score", "ask (or read) the answers and score them")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--clips", type=Path, required=True, help="CSV clip_id,side,video")
        cmd.add_argument("--facts", type=Path, required=True)
        cmd.add_argument("--leaks", type=Path, help="caption_leaks output: leave leaked facts out")
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--frames", type=int, default=8, help="frames sampled per video")
    score_cmd = sub.choices["score"]
    score_cmd.add_argument("--vqa-url", help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    score_cmd.add_argument("--model", help="model name (as served, or of the batch runner)")
    score_cmd.add_argument("--batch-answers", type=Path, help="output JSON of the batch runner")
    args = parser.parse_args(argv)

    if args.command == "facts":
        if args.records:
            from evaluation.route_records import facts_from_record, load_records

            with open(args.samples, newline="") as f:
                sample_ids = sorted({row["sample_id"] for row in csv.DictReader(f)})
            records = load_records(args.records, sample_ids)
            missing = [s for s in sample_ids if s not in records]
            if missing:
                logger.warning("%d samples have no route record, e.g. %s", len(missing), missing[:3])
            facts = {sid: facts_from_record(record) for sid, record in records.items()}
        else:
            trajectories = [(p.stem, read_trajectory(p)) for p in sorted(args.trajectories.glob("*.csv"))]
            facts = facts_for_trajectories(trajectories, args.cache, args.overpass_url)
        args.out.write_text(json.dumps(facts, indent=1, sort_keys=True))
        logger.info("facts of %d clips written to %s", len(facts), args.out)
        return 0

    clips, facts = read_clips(args.clips), _load_facts(args)
    if args.command == "export":
        export_batch(clips, facts, args.out, args.frames)
        logger.info("batch inputs written to %s", args.out)
        return 0

    if args.batch_answers:
        answers = read_batch_answers(args.batch_answers, facts, args.model or "batch")
    elif args.vqa_url and args.model:
        answers = ask_all(clips, facts, OpenAICompatibleVqa(args.vqa_url, args.model), args.out / "answers.jsonl",
                          args.frames)
    else:
        parser.error("score needs --batch-answers, or --vqa-url and --model")
    for side in write_results(score(answers), args.out):
        logger.info("%s", json.dumps(side))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
