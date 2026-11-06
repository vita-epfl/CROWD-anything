"""
Prompt following of generated driving videos (evaluation part 2): does a video show what its text caption says?

Statements: each caption split into short statements a viewer can check from the frames, tagged by kind:
  static    the scene: objects, buildings, weather, road surface and markings
  dynamic   what moves: the camera car's motion, other cars and people, the order of events
  topology  lanes, junctions, crossings, signals, road type (scored on their own: evaluation.topology measures them)
A few statements per caption are deliberately false (a changed colour, direction or object): a judge that says yes to
everything confirms the true ones too, and only the false ones show it.
statements.json: {clip_id: [{"text": "...", "kind": "static", "true": true}, ...]}

Videos: clips.csv as in evaluation.topology (clip_id,side,video; `reference` = the real video). `export` also adds a
`control` side per clip: the real video of another clip (rotation) asked this clip's statements, i.e. a caption that
does not describe the video.

Answers: all statements of a clip in one prompt per video (yes or no per numbered line), through a batch runner such
as Qwen-Drive's run_video_batch.sh: `export` writes its videos.json and prompts.json, `score --batch-answers` reads
its output(s).

Results in --out:
  per_statement.csv  every statement and answer per video
  summary.csv        per side: `confirmed_<kind>` = share of the true statements answered yes; `rejected` = share of
                     the false statements answered no; `balanced` = mean of confirmed (static + dynamic) and rejected;
                     `confirmed_on_real` = confirmed, counting only the statements the judge confirms on the
                     reference (what it cannot see on real video is not the generator's fault). The reference is the
                     ceiling, `control` the floor: a usable judge confirms far fewer statements on the control.

  python -m evaluation.prompt_following export --clips clips.csv --statements statements.json --out batch/
  python -m evaluation.prompt_following score --clips clips.csv --statements statements.json \
      --batch-answers batch/answers-*.json --out results/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

from evaluation.topology import REFERENCE, _rate, _write_csv, read_clips

logger = logging.getLogger(__name__)
CONTROL = "control"
KINDS = ("static", "dynamic", "topology")
SCORED = ("static", "dynamic")  # topology is scored by evaluation.topology

Statements = dict[str, list[dict]]  # clip_id -> [{text, kind, true}]


def read_statements(path: Path) -> Statements:
    statements: Statements = json.loads(path.read_text())
    for clip_id, items in statements.items():
        for item in items:
            if item.get("kind") not in KINDS or not isinstance(item.get("true"), bool) or not item.get("text"):
                raise ValueError(f"{clip_id}: bad statement {item}")
    return statements


def prompt(statements: list[dict], num_frames: int) -> str:
    lines = [f"{i}. {s['text']}" for i, s in enumerate(statements, 1)]
    return (
        f"These are {num_frames} frames of a dashcam video from a car, in time order.\n"
        "For each statement, answer yes if the video shows it and no if it does not.\n" + "\n".join(lines) +
        "\nReply with one line per statement and nothing else, like:\n1: yes\n2: no"
    )


def parse_answers(text: str, count: int) -> list[Optional[bool]]:
    """yes/no per statement from lines like "3: yes", "3. No", "**3**: yes"; None where a statement has no answer
    (the first answer to a number counts)."""
    answers: list[Optional[bool]] = [None] * count
    for number, word in re.findall(r"^\W*(\d+)\W+(yes|no)\b", text, re.IGNORECASE | re.MULTILINE):
        index = int(number) - 1
        if 0 <= index < count and answers[index] is None:
            answers[index] = word.lower() == "yes"
    return answers


def with_control(clips: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """The clips with a `control` side: the reference video of the next clip (in clip_id order)."""
    ids = sorted(c for c in clips if REFERENCE in clips[c])
    out = {c: dict(sides) for c, sides in clips.items()}
    if len(ids) > 1:
        for i, clip_id in enumerate(ids):
            out[clip_id][CONTROL] = clips[ids[(i + 1) % len(ids)]][REFERENCE]
    return out


def export_batch(clips: dict[str, dict[str, str]], statements: Statements, out_dir: Path,
                 num_frames: int = 8) -> None:
    """videos.json and prompts.json for a batch VQA runner: one prompt per clip (its statements), on every video of
    the clip, the control included."""
    videos, prompts = [], []
    for clip_id, sides in with_control(clips).items():
        if not statements.get(clip_id):
            continue
        ids = [f"{clip_id}::{side}" for side in sides]
        videos += [{"id": f"{clip_id}::{side}", "path": str(path)} for side, path in sides.items()]
        prompts.append({"id": clip_id, "text": prompt(statements[clip_id], num_frames), "video_ids": ids})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos.json").write_text(json.dumps({"videos": videos}, indent=1))
    (out_dir / "prompts.json").write_text(json.dumps({"prompts": prompts}, indent=1))


def read_batch_answers(paths: list[Path], statements: Statements) -> list[dict]:
    """One row per statement and video from batch runner outputs ({"results": [{video_id, prompt_id, status,
    answer}]})."""
    rows = []
    for result in (r for path in paths for r in json.loads(path.read_text())["results"]):
        clip_id, side = result["video_id"].split("::", 1)
        items = statements.get(result["prompt_id"])
        if not items or result["prompt_id"] != clip_id:
            continue
        reply = (result.get("answer") or "") if result.get("status") == "ok" else ""
        for index, (item, answer) in enumerate(zip(items, parse_answers(reply, len(items)))):
            rows.append({"clip_id": clip_id, "side": side, "index": index, "kind": item["kind"],
                         "true": item["true"], "answer": answer, "text": item["text"]})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per side, the metrics described in the module docstring (pooled over the side's statements)."""
    on_real = {(r["clip_id"], r["index"]) for r in rows if r["side"] == REFERENCE and r["true"] and r["answer"]}
    out = []
    sides = sorted({r["side"] for r in rows}, key=lambda s: (s != REFERENCE, s == CONTROL, s))
    for side in sides:
        side_rows = [r for r in rows if r["side"] == side]
        true_rows = [r for r in side_rows if r["true"] and r["answer"] is not None]
        scored = [r["answer"] for r in true_rows if r["kind"] in SCORED]
        rejected = _rate([not r["answer"] for r in side_rows if not r["true"] and r["answer"] is not None])
        summary: dict = {"side": side, "clips": len({r["clip_id"] for r in side_rows})}
        for kind in KINDS:
            summary[f"confirmed_{kind}"] = _rate([r["answer"] for r in true_rows if r["kind"] == kind])
        summary["confirmed"] = _rate(scored)
        summary["rejected"] = rejected
        summary["balanced"] = (summary["confirmed"] + rejected) / 2 if scored and rejected is not None else None
        summary["confirmed_on_real"] = _rate([r["answer"] for r in true_rows
                                              if r["kind"] in SCORED and (r["clip_id"], r["index"]) in on_real])
        summary["unanswered"] = sum(r["answer"] is None for r in side_rows)
        out.append(summary)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("export", "write videos.json/prompts.json for a batch VQA runner"),
                            ("score", "score the batch runner's answers")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--clips", type=Path, required=True, help="CSV clip_id,side,video")
        cmd.add_argument("--statements", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
    sub.choices["export"].add_argument("--frames", type=int, default=8, help="frames sampled per video")
    sub.choices["score"].add_argument("--batch-answers", type=Path, nargs="+", required=True,
                                      help="output JSON(s) of the batch runner")
    args = parser.parse_args(argv)

    clips, statements = read_clips(args.clips), read_statements(args.statements)
    if args.command == "export":
        export_batch(clips, statements, args.out, args.frames)
        logger.info("batch inputs written to %s", args.out)
        return 0
    rows = [r for r in read_batch_answers(args.batch_answers, statements)
            if r["side"] in with_control(clips).get(r["clip_id"], {})]
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.out / "per_statement.csv")
    summary = summarize(rows)
    _write_csv(summary, args.out / "summary.csv")
    for side in summary:
        logger.info("%s", json.dumps(side))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
