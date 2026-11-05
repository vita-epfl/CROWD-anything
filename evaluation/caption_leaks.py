"""Find topology facts that a generation's text caption already gives away, and suggest captions without them.

If the caption says "a roundabout ahead", a video that shows a roundabout may follow the text, not the map, so that
fact cannot measure topology adherence for that video. For each caption this lists the attributes it mentions
(evaluation.osm.ATTRIBUTES) with the matched words, and proposes a rewritten caption that drops every sentence
mentioning one, for regenerating the video. Attributes leaked by a clip's caption are left out of its scoring.

  python -m evaluation.caption_leaks --captions generations.csv --out leaks.json
  (CSV with columns sample_id and caption; one entry per distinct caption)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PATTERNS: dict[str, str] = {
    "lanes": r"\b(?:one|two|three|four|single|double|multi|\d+)[- ]lanes?\b|\blanes\b|\blane markings?\b"
             r"|\b(?:adjacent|left|right|opposite|opposing|travel|ego|left-hand|right-hand) lane\b"
             r"|\bcent(?:er|re) ?lines?\b|\b(?:dashed|solid|double) (?:white|yellow) lines?\b",
    "intersection": r"\bintersections?\b|\bjunctions?\b|\bcrossroads?\b|\broundabouts?\b|\btraffic circles?\b"
                    r"|\bside (?:road|street)s?\b|\bcross(?:ing)? street\b|\bcentral island\b"
                    r"|\b(?:stop|yield|give way) (?:signs?|lines?)\b|\b(?:on|at) the corner\b",
    "pedestrian_crossing": r"\bcross ?walks?\b|\bzebra\b|\bpedestrian crossings?\b|\bcrossing sign\b",
    "traffic_signals": r"\btraffic (?:lights?|signals?)\b|\b(?:red|green|amber|yellow) (?:lights?|signal)\b"
                       r"|\bstop ?lights?\b",
    "oneway": r"\bone[- ]way\b|\boncoming\b|\bopposite direction\b|\bopposing (?:traffic|lanes?|direction)\b"
              r"|\b(?:divided|dual) (?:highway|carriageway|road)\b|\bmedian\b",
    "road_type": r"\b(?:highway|motorway|expressway|freeway|residential|boulevard|avenue|main road|side street"
                 r"|narrow street|country road|dual carriageway)\b",
    "turn": r"\bturn(?:s|ed|ing)? (?:to the )?(?:left|right)\b|\b(?:left|right)[- ]turn\b|\bgoes straight\b"
            r"|\bstraight ahead\b|\bcontinues straight\b|\btravels straight\b|\bstraight (?:stretch|road)\b"
            r"|\bcurv(?:e|es|ing|ed|ature)\b(?:\s+\w+){0,3}\s+(?:to the )?(?:left|right)\b"
            r"|\bbends? (?:to the )?(?:left|right)\b",
    "speed_limit": r"\bspeed limits?\b|\b\d+ ?(?:km/h|mph)\b",  # not asked yet; for a speed-limit check
}
COMPILED = {attribute: re.compile(pattern, re.IGNORECASE) for attribute, pattern in PATTERNS.items()}
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def leaks(caption: str) -> dict[str, list[str]]:
    """{attribute: matched phrases} for every attribute the caption mentions."""
    found: dict[str, list[str]] = {}
    for attribute, pattern in COMPILED.items():
        matches = sorted({m.group(0).lower() for m in pattern.finditer(caption)})
        if matches:
            found[attribute] = matches
    return found


def rewrite(caption: str) -> str:
    """The caption without the sentences that mention a topology attribute."""
    sentences = [s for s in SENTENCE.split(caption.strip()) if s]
    kept = [s for s in sentences if not leaks(s)]
    return " ".join(kept)


def check_captions(rows: list[dict]) -> dict[str, dict]:
    """{sample_id: {caption, leaked, rewritten}} for rows with sample_id and caption (first caption per sample)."""
    out: dict[str, dict] = {}
    for row in rows:
        sample_id, caption = row["sample_id"], row["caption"]
        if sample_id in out:
            if out[sample_id]["caption"] != caption:
                raise ValueError(f"{sample_id} has different captions; check them separately")
            continue
        found = leaks(caption)
        out[sample_id] = {"caption": caption, "leaked": found, "rewritten": rewrite(caption) if found else caption}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=Path, required=True, help="CSV with sample_id and caption columns")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with open(args.captions, newline="") as f:
        result = check_captions(list(csv.DictReader(f)))
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    leaked = {sid: sorted(r["leaked"]) for sid, r in result.items() if r["leaked"]}
    print(f"{len(leaked)} of {len(result)} captions leak topology facts")
    for sid, attributes in leaked.items():
        print(f"  {sid}: {', '.join(attributes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
