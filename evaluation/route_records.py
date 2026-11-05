"""Topology facts of NATIX samples from the per-sample OSM route records of wm-data-manifest's map step (David's
`osm_bev/text_predicted`), the ground truth of the topology evaluation.

A record describes the 5 s sample from its start: the road(s) the car is on (class, one-way, lanes with their source,
speed limit), its manoeuvre, and the events along and beyond its path (intersections, T-junctions, roundabouts,
forks, side roads, crossings, signals, ...) with their distance from the start. It is built from the same OSM data
the models were conditioned on and stays around the car's own trajectory.

Facts (the closed sets of evaluation.osm.ATTRIBUTES) are read as what a forward camera can see in the sample: events
up to `visible_m` metres ahead count, and uncertain values are left out rather than guessed: lanes not from the map
("likely"), signals whose direction is not known ("possibly"), a slight bend, a first junction that is a fork.

Layout: <root>/index.jsonl (one line per sample: sample_id, image = "<session zip>::<entry>"), and in each session
zip a text.jsonl with one record per sample (keyed by sample_id).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable, Optional

from evaluation.osm import MAJOR, MOTORWAY

VISIBLE_M = 100.0  # events up to this far ahead of the sample's start count as seen by the camera

JUNCTIONS = {"intersection": "crossroads", "t_junction": "T-junction", "roundabout": "roundabout"}


def load_records(root: Path, sample_ids: Iterable[str]) -> dict[str, dict]:
    """{sample_id: record} for the given samples (missing ones are left out)."""
    wanted = set(sample_ids)
    zips: dict[str, set[str]] = {}
    with open(root / "index.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            if entry["sample_id"] in wanted:
                zips.setdefault(entry["image"].split("::")[0], set()).add(entry["sample_id"])
    records: dict[str, dict] = {}
    for zip_rel, ids in zips.items():
        with zipfile.ZipFile(root / zip_rel) as z:
            for line in z.read("text.jsonl").decode("utf-8").splitlines():
                record = json.loads(line)
                if record.get("sample_id") in ids:
                    records[record["sample_id"]] = record
    return records


def _road_type(road_class: str) -> str:
    return "motorway" if road_class in MOTORWAY else "major road" if road_class in MAJOR else "minor road"


def _visible(events: list[dict], visible_m: float) -> list[dict]:
    ahead = [e for e in events if e.get("dist_from_start_m") is not None and e["dist_from_start_m"] <= visible_m]
    return sorted(ahead, key=lambda e: e["dist_from_start_m"])


def facts_from_record(record: dict, visible_m: float = VISIBLE_M) -> dict[str, str]:
    facts: dict[str, str] = {}
    roads = record.get("roads") or []
    if roads:
        road = roads[0]  # the road the car starts on
        if road.get("road_class"):
            facts["road_type"] = _road_type(road["road_class"])
        if road.get("one_way") is not None:
            facts["oneway"] = "yes" if road["one_way"] else "no"
        lanes: Optional[int] = road.get("lanes")
        if lanes is not None and road.get("lanes_source") == "map":
            facts["lanes"] = "1" if lanes <= 1 else "2" if lanes == 2 else "3 or more"

    events = _visible(record.get("events") or [], visible_m)
    junctions = [e for e in events if e["type"] in JUNCTIONS or e["type"] == "fork"]
    if not junctions:
        facts["intersection"] = "none"
    elif junctions[0]["type"] in JUNCTIONS:  # a fork first: no fact (not one of the options)
        facts["intersection"] = JUNCTIONS[junctions[0]["type"]]

    facts["pedestrian_crossing"] = "yes" if any(e["type"] == "crossing" for e in events) else "no"
    signals = [e for e in events if e["type"] == "signals"]
    if not signals:
        facts["traffic_signals"] = "no"
    elif any(e.get("direction_known") is not False for e in signals):
        facts["traffic_signals"] = "yes"  # only "possibly" signals (direction not known): no fact

    maneuver = record.get("maneuver") or {}
    if maneuver.get("kind") == "straight":
        facts["turn"] = "straight"
    elif maneuver.get("kind") == "turn" and maneuver.get("direction") in ("left", "right"):
        facts["turn"] = maneuver["direction"]
    return facts
