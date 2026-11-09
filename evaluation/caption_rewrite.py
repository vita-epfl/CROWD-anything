"""
Rewrite generation captions without road topology, with a local text LLM (e.g. Qwen3-32B through transformers,
thinking off), so that a video conditioned on the rewritten caption can only get its topology from the map.

What is removed: the number of lanes, lane markings and lane positions, the kind of road (highway, residential
street, avenue), intersections, junctions, roundabouts, side streets and corners, pedestrian crossings, traffic lights
and traffic signs, one-way, oncoming or opposing traffic and medians, speed limits, and the geometry of the road
(curves, bends, straight stretches, turns at junctions). Everything else stays in the caption's own words: weather,
light, buildings, vegetation, vehicles and people and their colours, and how the camera car and others move (slowing
down, following, panning). The examples are rewrites from the 30-scene eval set, which Valentin regenerated from.

Each rewrite is checked with evaluation.caption_leaks plus the paraphrases the LLM uses instead (PARAPHRASES: "a
marked area with white stripes" for a crosswalk, "the road changes direction" for a curve, signs and road arrows): a
rewrite that still mentions any of them is sent back with the phrases that leak, up to FEEDBACK_ROUNDS times; if it
still leaks, the sentences that leak are dropped (`method` = "llm+drop"; asked again with sampling instead, in a pilot
of 186 captions, 69 ended that way). Replies that
are not valid JSON are asked again with sampling; captions still without a rewrite are left out (and logged).

  python -m evaluation.caption_rewrite --captions captions.csv --model /path/to/Qwen3-32B --out rewritten.csv
  (CSV with columns sample_id and caption; out: sample_id, original_caption, rewritten_caption, removed, kept_hints,
  leaks_left, method, words_before, words_after)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional, Sequence

from evaluation.caption_leaks import SENTENCE, leaks

logger = logging.getLogger(__name__)

INSTRUCTIONS = """You edit the caption of a dashcam video so that it says nothing about the road layout. A video
model will be conditioned on your caption together with a map; the road layout must come only from the map.

Remove every mention of:
- lanes: their number, lane markings and lines, which lane a vehicle is in ("adjacent left lane" -> "on the left")
- the kind of road: highway, motorway, residential street, avenue, main road, side street, country road
- intersections, junctions, T-junctions, roundabouts, side roads, corners, cross streets
- pedestrian crossings, crosswalks, zebra crossings (a person may still cross the street); do not describe them as
  "a marked area" or "white stripes across the road" either
- traffic lights and traffic signs of any kind, speed limits; drop the sign entirely, do not keep it as "a sign",
  "a signpost" or "a blue sign with arrows"
- road markings: lines, arrows painted on the road
- one-way roads, oncoming or opposing traffic, the opposite direction, medians, divided roads
- the geometry of the road: curves, bends, straight stretches, and turns at junctions, of the camera car or others;
  do not replace them with other words for a curve ("bend", "the road changes direction"): write "continues along
  the road" or leave the motion out

Keep everything else, word for word and in order: weather, light, road surface, buildings, vegetation, poles, fields,
parked and moving vehicles and people with their colours and kinds, and how the camera car and others move (moving
forward, slowing down, stopping, following, panning). Change only the words a removal requires; do not reword
scenery and do not add anything the caption does not say.

Reply with JSON only:
{"rewritten": "...", "removed": ["what you removed", ...], "kept_hints": ["what still hints at the layout", ...]}
kept_hints: phrases you kept because the caption needs them but that still suggest a layout (e.g. "a pedestrian
crosses the street"); empty if none. Nothing from the list above may be kept: it is removed, not a hint."""

# what the LLM writes instead of the words evaluation.caption_leaks looks for (in 6,016 rewrites of the 255 one-minute
# clips: ~190 "marked area"/"white stripes" crosswalks, ~280 road or sign arrows, ~80 curves, signs and lanes)
PARAPHRASES: dict[str, str] = {
    "crossing": r"\bmarked areas?\b|\bstriped areas?\b|\bstripes? (?:span|across|on the road)|\bzebra\b",
    "markings": r"\barrows?\b|\broad markings?\b|\bmarked with lines\b"
                r"|\b(?:white|yellow|painted|dashed|solid) lines?\b",
    "lanes": r"\blanes?\b",
    "signs": r"\bsign(?:s|post|posts|board|boards|age)?\b",
    "geometry": r"\bbends?\b|\bchang(?:e|es|ed|ing) (?:its |their )?direction\b|\bcurv\w*|\bstraight\b",
}
PARAPHRASE_PATTERNS = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in PARAPHRASES.items()}

FEEDBACK_ROUNDS = 2
FEEDBACK = ("Your rewrite still mentions: {}. Remove these too, rephrasing only those sentences, and keep everything "
            "else. Reply with JSON only.")

EXAMPLES = [
    ("The camera begins with a static view of the plaza and storefront. It then initiates a smooth pan to the left, "
     "revealing a pedestrian crossing sign and the adjacent street. As the camera moves, a pedestrian wearing dark "
     "clothing carrying a bag enters the frame from the left sidewalk. The pedestrian walks along the pavement and "
     "steps onto the crosswalk, crossing the street from left to right in front of the camera's path.",
     {"rewritten": "The camera begins with a static view of the plaza and storefront. It then initiates a smooth pan "
                   "to the left, revealing the adjacent street. As the camera moves, a pedestrian wearing dark "
                   "clothing carrying a bag enters the frame from the left sidewalk. The pedestrian walks along the "
                   "pavement and then crosses the street from left to right in front of the camera's path.",
      "removed": ["pedestrian crossing sign", "crosswalk"],
      "kept_hints": ["a pedestrian crosses the street (no marked crossing named)"]}),
    ("The camera vehicle approaches a roundabout and begins to decelerate. A white sedan is positioned ahead in the "
     "same lane, also slowing down as it enters the roundabout. The white sedan turns right, navigating the curve of "
     "the roundabout. The camera vehicle follows the white sedan, maintaining a safe distance as it enters the "
     "circular intersection.",
     {"rewritten": "The camera vehicle moves along the road and begins to decelerate. A white sedan is positioned "
                   "ahead, also slowing down. The white sedan continues along the road, and the camera vehicle "
                   "follows it, maintaining a safe distance.",
      "removed": ["roundabout (3x)", "circular intersection", "the sedan turns right", "curve of the roundabout",
                  "same lane"],
      "kept_hints": []}),
    ("The scene depicts a daytime urban road environment under a bright blue sky with scattered white clouds. The "
     "road surface is paved asphalt with visible lane markings, including a solid white line on the right edge and a "
     "dashed white line separating lanes. On the right side, there is a curb with a row of parked cars along the "
     "sidewalk, and a traffic signpost displaying a speed limit of 30 km/h and a yield sign. Trees line both sides of "
     "the road, providing shade. In the background, multi-story residential buildings are visible on the left. The "
     "overall visibility is clear, and the lighting suggests it is midday.",
     {"rewritten": "The scene depicts a daytime urban environment under a bright blue sky with scattered white "
                   "clouds. The road surface is paved asphalt. On the right side, there is a curb with a row of "
                   "parked cars along the sidewalk. Trees line both sides of the road, providing shade. In the "
                   "background, multi-story apartment buildings are visible on the left. The overall visibility is "
                   "clear, and the lighting suggests it is midday.",
      "removed": ["lane markings separating lanes", "30 km/h speed-limit sign", "yield sign",
                  "residential (buildings)"],
      "kept_hints": []}),
]


def remaining(text: str) -> dict[str, list[str]]:
    """Topology a rewrite still mentions: evaluation.caption_leaks' attributes and PARAPHRASES, with matched words."""
    found = leaks(text)
    for name, pattern in PARAPHRASE_PATTERNS.items():
        matches = sorted({m.group(0).lower() for m in pattern.finditer(text)})
        if matches:
            found[name] = matches
    return found


def drop_leaking_sentences(text: str) -> str:
    return " ".join(s for s in SENTENCE.split(text.strip()) if s and not remaining(s))


def messages(caption: str) -> list[dict]:
    chat = [{"role": "system", "content": INSTRUCTIONS}]
    for example, reply in EXAMPLES:
        chat.append({"role": "user", "content": f"Caption: {example}"})
        chat.append({"role": "assistant", "content": json.dumps(reply)})
    chat.append({"role": "user", "content": f"Caption: {caption}"})
    return chat


def parse_rewrite(text: str) -> Optional[dict]:
    """{rewritten, removed, kept_hints} of a reply, or None if it is not the JSON asked for."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    try:
        reply = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(reply, dict) or not isinstance(reply.get("rewritten"), str) or not reply["rewritten"].strip():
        return None
    lists = [reply.get(key) or [] for key in ("removed", "kept_hints")]
    if not all(isinstance(items, list) for items in lists):
        return None
    return {"rewritten": reply["rewritten"].strip(), "removed": [str(x) for x in lists[0]],
            "kept_hints": [str(x) for x in lists[1]]}


def finish(caption: str, reply: dict, method: str) -> dict:
    rewritten = reply["rewritten"]
    return {"original_caption": caption, "rewritten_caption": rewritten, "removed": "; ".join(reply["removed"]),
            "kept_hints": "; ".join(reply["kept_hints"]), "leaks_left": json.dumps(remaining(rewritten)),
            "method": method, "words_before": len(caption.split()), "words_after": len(rewritten.split())}


def feedback(caption: str, reply_text: str, reply: dict) -> list[dict]:
    """The chat of a rewrite that still leaks, with the leaking phrases sent back."""
    phrases = sorted({p for found in remaining(reply["rewritten"]).values() for p in found})
    return messages(caption) + [{"role": "assistant", "content": reply_text},
                                {"role": "user", "content": FEEDBACK.format(", ".join(f'"{p}"' for p in phrases))}]


def rewrite_captions(captions: Sequence[str], llm, batch_size: int = 16) -> dict[str, dict]:
    """{caption: row} for the distinct captions: an LLM rewrite without leaks (`llm`, possibly after the leaking
    phrases were sent back), else the last valid rewrite with its leaking sentences dropped (`llm+drop`)."""
    todo, done = list(dict.fromkeys(captions)), {}
    leaking: dict[str, tuple[str, dict]] = {}  # caption -> (reply text, parsed reply) of its last leaking rewrite
    total = len(todo)
    for round_ in range(1 + FEEDBACK_ROUNDS):
        # leaking rewrites go back with their leaks (greedy); invalid replies are asked again with sampling
        groups = [([c for c in todo if c in leaking], False), ([c for c in todo if c not in leaking], round_ > 0)]
        for group, sample in groups:
            for i in range(0, len(group), batch_size):
                chunk = group[i:i + batch_size]
                chats = [feedback(c, *leaking[c]) if c in leaking else messages(c) for c in chunk]
                for caption, text in zip(chunk, llm.generate(chats, sample=sample)):
                    reply = parse_rewrite(text)
                    if reply is None:
                        continue
                    if remaining(reply["rewritten"]):
                        leaking[caption] = (text, reply)
                    else:
                        done[caption] = finish(caption, reply, "llm")
                        leaking.pop(caption, None)
        logger.info("round %d: %d of %d captions rewritten without leaks", round_, len(done), total)
        todo = [c for c in todo if c not in done]
    for caption in todo:
        if caption not in leaking:
            logger.warning("no valid rewrite for caption: %s", caption[:120])
            continue
        reply = leaking[caption][1]
        reply = dict(reply, rewritten=drop_leaking_sentences(reply["rewritten"]))
        if reply["rewritten"]:
            done[caption] = finish(caption, reply, "llm+drop")
    return done


def main(argv: Optional[list[str]] = None) -> int:
    from evaluation.caption_statements import TransformersLlm

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=Path, required=True, help="CSV with columns sample_id and caption")
    parser.add_argument("--model", required=True, help="local path or hub id of the chat model")
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    with open(args.captions, newline="") as f:
        by_sample = {row["sample_id"]: row["caption"].strip() for row in csv.DictReader(f) if row["caption"].strip()}
    done = rewrite_captions(list(by_sample.values()), TransformersLlm(args.model, args.device), args.batch_size)
    rows = [{"sample_id": sample_id, **done[caption]} for sample_id, caption in by_sample.items() if caption in done]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", *finish("", {"rewritten": "", "removed": [],
                                                                         "kept_hints": []}, "")])
        writer.writeheader()
        writer.writerows(rows)
    methods = [r["method"] for r in rows]
    logger.info("%d of %d samples rewritten (%d with leaking sentences dropped) to %s", len(rows), len(by_sample),
                methods.count("llm+drop"), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
