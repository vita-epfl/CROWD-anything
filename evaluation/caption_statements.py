"""
Split generation captions into checkable statements for evaluation.prompt_following, with a local text LLM (e.g.
Qwen3-32B through transformers, thinking off).

Each caption becomes up to 12 true statements (one fact each, tagged static, dynamic or topology) and 3 false ones:
near-misses of true statements with one visible detail changed (a colour, the kind of vehicle or object, the
direction of a turn), so that a judge that does not look closely accepts them. Left/right placement is not changed:
VLM judges confuse it even on real video, so such a false statement would test the judge rather than the generator.
The LLM swaps sides anyway (14 of 30 captions in a test), so it is asked for 5 false statements and the first 3 that
are not a true one with left and right swapped are kept. Replies that are not valid JSON of that form are asked again
once with sampling; captions still without valid statements are left out (and logged).

  python -m evaluation.caption_statements --captions captions.csv --model /path/to/Qwen3-32B --out statements.json
  (CSV with columns sample_id and caption)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)
KINDS = ("static", "dynamic", "topology")  # evaluation.prompt_following.KINDS; not imported: this runs in the
# LLM's environment, which lacks the evaluation dependencies (requests, ...)
MAX_TRUE, NUM_FALSE, FALSE_ASKED = 12, 3, 5  # false statements kept, asked for

INSTRUCTIONS = f"""You split the caption of a 5-second dashcam video into short statements that a viewer can check by
looking at frames of that video.

Rules for the true statements (at most {MAX_TRUE}):
- One fact per statement, in plain words, about what is visible: "A white van is parked on the right."
- Only what the caption says; do not add or infer anything. Skip what frames cannot show (sign text that is
  illegible, "good visibility", intentions) and skip negations ("there are no pedestrians").
- Call the car with the camera "the camera car".
- kind "static": the scene: objects, vehicles and where they are, buildings, vegetation, weather, light, road surface
  and markings, signs.
  kind "dynamic": what moves or changes: the camera car's motion (straight, turning, slowing), other road users'
  motion, the order of events.
  kind "topology": the road layout: number of lanes, intersections, T-junctions, roundabouts, side roads,
  pedestrian crossings, traffic lights, one-way or oncoming traffic, the kind of road (highway, residential street).

Then exactly {FALSE_ASKED} false statements: each takes one true statement and changes one visible detail so that
the caption contradicts it: a colour, the kind of vehicle or object, or the direction of a turn (left turn instead
of right turn), or speeding up instead of slowing down. Do not change on which side or in which lane something is,
or from which side something comes. Prefer details of the most visible objects and motions; do not invent things the
caption does not mention; no negations.

Reply with JSON only:
{{"statements": [{{"text": "...", "kind": "static|dynamic|topology", "true": true}}, ...]}}"""

EXAMPLES = [
    ("The camera vehicle approaches a roundabout and begins to decelerate. A white sedan is positioned ahead in the "
     "same lane, also slowing down as it enters the roundabout. The white sedan turns right, navigating the curve of "
     "the roundabout. The camera vehicle follows the white sedan, maintaining a safe distance as it enters the "
     "circular intersection.",
     [("The camera car approaches a roundabout.", "topology", True),
      ("The camera car slows down.", "dynamic", True),
      ("A white sedan is ahead in the same lane.", "static", True),
      ("The white sedan turns right into the roundabout.", "dynamic", True),
      ("The camera car follows the white sedan.", "dynamic", True),
      ("A black sedan is ahead in the same lane.", "static", False),
      ("The white sedan turns left into the roundabout.", "dynamic", False),
      ("The camera car speeds up.", "dynamic", False),
      ("A white truck is ahead in the same lane.", "static", False),
      ("The camera car approaches a crossroads.", "topology", False)]),
    ("The scene is a daytime urban intersection under clear weather conditions with bright illumination. The road "
     "surface is paved asphalt featuring a large white right-turn arrow painted in the ego vehicle's lane, a white "
     "stop line, and a pedestrian crosswalk. A grey SUV is positioned in the adjacent left lane, stopped or moving "
     "slowly. On the right side, a corrugated metal fence runs parallel to the road, separating the street from a "
     "beige building with large windows and black-and-white posters. A traffic light hangs overhead, currently "
     "displaying a yellow signal.",
     [("It is daytime with clear weather.", "static", True),
      ("A large white arrow is painted on the road in the camera car's lane.", "static", True),
      ("There is a pedestrian crosswalk on the road.", "topology", True),
      ("A grey SUV is in the lane to the left.", "static", True),
      ("A corrugated metal fence runs along the right side of the road.", "static", True),
      ("A beige building with large windows is on the right.", "static", True),
      ("A traffic light hangs over the road.", "topology", True),
      ("A white SUV is in the lane to the left.", "static", False),
      ("A brick wall runs along the right side of the road.", "static", False),
      ("A red building with large windows is on the right.", "static", False),
      ("It is night.", "static", False),
      ("A large yellow arrow is painted on the road in the camera car's lane.", "static", False)]),
]


def messages(caption: str) -> list[dict]:
    chat = [{"role": "system", "content": INSTRUCTIONS}]
    for example, items in EXAMPLES:
        chat.append({"role": "user", "content": f"Caption: {example}"})
        reply = {"statements": [{"text": t, "kind": k, "true": v} for t, k, v in items]}
        chat.append({"role": "assistant", "content": json.dumps(reply)})
    chat.append({"role": "user", "content": f"Caption: {caption}"})
    return chat


SIDES = {"left": "right", "right": "left"}
TURNING = re.compile(r"\b(?:turn|turns|turning|curve|curves|curving|bend|bends|pan|pans|panning)\b")


def swaps_sides(false: str, true: str) -> bool:
    """Whether a false statement is a true one with left and right swapped ("on the left side" for "on the right
    side", "crosses from right to left" for "from left to right"); a turn, curve or pan to the other side is
    allowed."""
    words = re.findall(r"[a-z]+", false.lower())
    if TURNING.search(false.lower()) or not any(w in SIDES for w in words):
        return False
    return [SIDES.get(w, w) for w in words] == re.findall(r"[a-z]+", true.lower())


def parse_statements(text: str) -> Optional[list[dict]]:
    """The statements of a reply (true and false interleaved, so the false ones are not all at the end), or None if
    it is not the JSON asked for: at least 2 true static or dynamic statements and NUM_FALSE false ones that are not
    a true statement with left and right swapped (the first NUM_FALSE of those are kept)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    try:
        items = json.loads(text[start:end + 1])["statements"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(items, list):
        return None
    clean = []
    for item in items:
        if not isinstance(item, dict) or item.get("kind") not in KINDS or not isinstance(item.get("true"), bool):
            return None
        statement = str(item.get("text") or "").strip()
        if statement:
            clean.append({"text": statement, "kind": item["kind"], "true": item["true"]})
    true = [s for s in clean if s["true"]][:MAX_TRUE]
    false = [f for f in clean if not f["true"] and not any(swaps_sides(f["text"], t["text"]) for t in true)]
    false = false[:NUM_FALSE]
    if sum(s["kind"] != "topology" for s in true) < 2 or len(false) != NUM_FALSE:
        return None
    step = len(true) // (NUM_FALSE + 1) or 1
    out = list(true)
    for i, wrong in enumerate(false):
        out.insert(min((i + 1) * step + i, len(out)), wrong)
    return out


class TransformersLlm:
    """A chat model loaded with transformers on one device, generating for a batch of chats at once."""

    def __init__(self, path: str, device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device).eval()
        self.torch = torch

    def generate(self, chats: Sequence[list[dict]], max_new_tokens: int = 1024, sample: bool = False) -> list[str]:
        texts = [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) for c in chats]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        options = {"do_sample": True, "temperature": 0.7, "top_p": 0.9} if sample else {"do_sample": False}
        with self.torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=max_new_tokens, **options)
        return self.tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)


def split_captions(captions: Sequence[str], llm, batch_size: int = 16) -> dict[str, list[dict]]:
    """{caption: statements} for the distinct captions; one retry with sampling for invalid replies."""
    todo, done = list(dict.fromkeys(captions)), {}
    total = len(todo)
    for sample in (False, True):
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            for caption, reply in zip(chunk, llm.generate([messages(c) for c in chunk], sample=sample)):
                statements = parse_statements(reply)
                if statements:
                    done[caption] = statements
            logger.info("%d of %d captions split", len(done), total)
        todo = [c for c in todo if c not in done]
    for caption in todo:
        logger.warning("no valid statements for caption: %s", caption[:120])
    return done


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=Path, required=True, help="CSV with columns sample_id and caption")
    parser.add_argument("--model", required=True, help="local path or hub id of the chat model")
    parser.add_argument("--out", type=Path, required=True, help="statements JSON to write")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    with open(args.captions, newline="") as f:
        by_sample = {row["sample_id"]: row["caption"].strip() for row in csv.DictReader(f) if row["caption"].strip()}
    split = split_captions(list(by_sample.values()), TransformersLlm(args.model, args.device), args.batch_size)
    statements = {sample_id: split[caption] for sample_id, caption in by_sample.items() if caption in split}
    args.out.write_text(json.dumps(statements, indent=1))
    logger.info("statements for %d of %d samples written to %s", len(statements), len(by_sample), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
