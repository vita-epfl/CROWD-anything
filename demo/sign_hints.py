"""
Map hints for pinning a drive's GPS by hand, from the text read in its frames (read_signs.py): street names (boxes on
one line are joined, so "Rue" + "de Rivoli" is one name) and shop or place names read in 2+ frames (in 720p video
street plates are mostly too small; shop fronts are not), each kept once per 20 s, looked up on OSM
(Nominatim, inside the city, one request a second, cached); a shop or place counts only when OSM has exactly one of
that name in the city (not a chain). Writes <id>_hints.csv (video time, name, OSM position,
Street View link) and <id>_candidate_pins.json for gps_annotator.html's "Import pins": only names OSM puts in a small
area (a square, a short street, a shop, a landmark), since a long street's centre says little about where the car is.

  python demo/sign_hints.py --text Qxn7g5k6d20_text.csv --city "Geneva, Switzerland" --start 415 --end 6599
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

NOMINATIM = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "CROWD-anything demo GPS hints (TU/e)"
STREET = re.compile(r"\b(rue|av\.?|avenue|bd|boulevard|quai|place|pl\.|pont|route|rte|chemin|ch\.|all[ée]e|cours|"
                    r"square|impasse|passage|rond-point|esplanade|promenade|parc|gare|porte|pont|jardin|mus[ée]e)\b",
                    re.IGNORECASE)
PLACE_CLASSES = {"shop", "amenity", "tourism", "historic", "building", "leisure", "office", "railway", "craft"}
NAME = re.compile(r"[^\W\d_][^\W\d_' -]{3,}[^\W\d_]", re.UNICODE)  # letters only: not a number plate
GENERIC = {"maison", "caisse", "louer", "radio", "jardin", "parking", "musee", "hotel", "pharmacie", "banque", "cafe",
           "restaurant", "boulangerie", "tabac", "presse", "pizza", "garage", "sortie", "centre", "ville", "poste",
           "entree", "ouvert", "soldes", "location", "taxi", "stop", "police"}  # shop-front words, not names
MIN_SEEN = 2  # a shop or place name must be read in this many frames (single reads are mostly OCR noise)
GROUP_S = 20.0  # the same name read again within this time is the same sign
SMALL_M = 300.0  # an OSM result whose box diagonal is below this becomes a candidate pin


def lines(boxes: list[dict]) -> list[tuple[str, float]]:
    """Texts of one frame with the boxes on one line joined (left to right): (text, lowest confidence)."""
    boxes = sorted(boxes, key=lambda b: (b["y"] + b["h"] / 2, b["x"]))
    out, used = [], set()
    for i, b in enumerate(boxes):
        if i in used:
            continue
        line, right = [b], b["x"] + b["w"]
        for j in range(i + 1, len(boxes)):
            c = boxes[j]
            if j in used or abs((c["y"] + c["h"] / 2) - (b["y"] + b["h"] / 2)) > b["h"] / 2:
                continue
            if 0 <= c["x"] - right < 1.5 * b["h"]:
                line.append(c)
                used.add(j)
                right = c["x"] + c["w"]
        out.append((" ".join(c["text"] for c in line), min(c["confidence"] for c in line)))
    return out


def names(rows: list[dict], min_confidence: float) -> list[dict]:
    """Street and place names, once per GROUP_S: first time, times seen, best confidence."""
    frames = collections.defaultdict(list)
    for r in rows:
        frames[float(r["t_s"])].append({k: (r[k] if k == "text" else float(r[k]))
                                        for k in ("text", "confidence", "x", "y", "w", "h")})
    out: list[dict] = []
    last: dict[str, dict] = {}
    for t in sorted(frames):
        for text, confidence in lines(frames[t]):
            text = re.sub(r"\s+", " ", text).strip(" .,-'\"")
            street = bool(STREET.search(text)) and len(text.split()) >= 2  # "Rue de Rivoli", not "JARDIN" alone
            if confidence < min_confidence or plain(text) in GENERIC or not (street or NAME.fullmatch(text)):
                continue
            key = text.lower()
            if key in last and t - last[key]["last_t"] <= GROUP_S:
                last[key].update(last_t=t, seen=last[key]["seen"] + 1,
                                 confidence=max(last[key]["confidence"], confidence))
                continue
            last[key] = {"t_s": t, "last_t": t, "text": text, "seen": 1, "confidence": confidence, "street": street}
            out.append(last[key])
    return [n for n in out if n["street"] or n["seen"] >= MIN_SEEN]


def plain(text: str) -> str:
    """Lower case without accents, for comparing a name read in the video with an OSM name."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c)).strip()


def nominatim(query: str, viewbox: Optional[str], cache: dict) -> list[dict]:
    """Up to 5 OSM results (inside the viewbox when given), cached."""
    key = f"{query}|{viewbox}|5"
    if key not in cache:
        params = {"q": query, "format": "json", "limit": 5}
        if viewbox:
            params.update(viewbox=viewbox, bounded=1)
        request = urllib.request.Request(f"{NOMINATIM}?{urllib.parse.urlencode(params)}",
                                         headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            cache[key] = json.load(response)
        time.sleep(1.1)
    return cache[key]


def place(text: str, hits: list[dict]) -> Optional[dict]:
    """The one shop or place of that name, or None: chains (Migros, Indigo) have several in a city."""
    same = [h for h in hits if h.get("class") in PLACE_CLASSES
            and plain(h["display_name"].split(",")[0]) == plain(text)]
    return same[0] if len(same) == 1 else None


def diagonal_m(hit: dict) -> float:
    south, north, west, east = (float(v) for v in hit["boundingbox"])
    dy = (north - south) * 111_320
    dx = (east - west) * 111_320 * math.cos(math.radians((north + south) / 2))
    return math.hypot(dx, dy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", type=Path, required=True, help="read_signs.py output, <video id>_text.csv")
    parser.add_argument("--city", required=True, help='e.g. "Geneva, Switzerland"')
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--cache", type=Path, default=Path(__file__).with_name("nominatim_cache.json"))
    args = parser.parse_args()

    video_id = args.text.name.split("_text")[0]
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    city = next(iter(nominatim(args.city, None, cache)), None)
    if not city:
        raise SystemExit(f"{args.city} not found on OSM")
    south, north, west, east = city["boundingbox"]
    viewbox = f"{west},{north},{east},{south}"
    with open(args.text, newline="") as f:
        found = names(list(csv.DictReader(f)), args.min_confidence)
    hints, pins = [], []
    for n in found:
        hits = nominatim(n["text"], viewbox, cache)
        # a shop or place only by its own name ("Japon" is not "Ambassade du Japon") and only if the city has one
        hit = next(iter(hits), None) if n["street"] else place(n["text"], hits)
        row = {"t_s": n["t_s"], "kind": "street" if n["street"] else "place", "time": f"{int(n['t_s'] // 60)}:{n['t_s'] % 60:04.1f}", "text": n["text"],
               "seen": n["seen"], "confidence": round(n["confidence"], 2), "osm_name": "", "lat": "", "lon": "",
               "area_m": "", "street_view": ""}
        if hit:
            row.update(osm_name=hit["display_name"].split(",")[0], lat=hit["lat"], lon=hit["lon"],
                       area_m=round(diagonal_m(hit)),
                       street_view="https://www.google.com/maps/@?api=1&map_action=pano&viewpoint="
                                   f"{hit['lat']},{hit['lon']}")
            if row["area_m"] < SMALL_M:
                pins.append({"t": n["t_s"], "lat": float(hit["lat"]), "lon": float(hit["lon"]), "straight": False})
        hints.append(row)
    args.cache.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    out = args.text.with_name(f"{video_id}_hints.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, list(hints[0]) if hints else ["t_s"])
        writer.writeheader()
        writer.writerows(hints)
    pins_out = args.text.with_name(f"{video_id}_candidate_pins.json")
    pins_out.write_text(json.dumps({"video_id": video_id, "t0": args.start, "t1": args.end or None,
                                    "pins": pins}, indent=1))
    print(f"{len(hints)} names, {sum(bool(h['lat']) for h in hints)} found on OSM, {len(pins)} candidate pins: "
          f"{out}, {pins_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
