"""Road topology facts along a trajectory, from OpenStreetMap (evaluation part 1: does a generated video follow the
topology of the map it was given).

The drivable roads, crossings and traffic signals around a trajectory are fetched once from the Overpass API and
cached. The trajectory (WGS84 positions in time order) is projected to a local metric plane and matched to the
nearest drivable road, and closed-set facts are read off the map along it:

  road_type            motorway | major road | minor road        (`highway` of the road the car starts on)
  lanes                1 | 2 | 3 or more                         (`lanes` / `lanes:forward` of that road)
  oneway               yes | no                                  (`oneway`; motorways are one-way)
  intersection         none | T-junction | crossroads | roundabout   (first junction the car passes)
  pedestrian_crossing  yes | no                                  (crossing node next to the path)
  traffic_signals      yes | no                                  (signals node next to the path)
  turn                 straight | left | right                   (heading change of the trajectory itself)

A fact the map does not support (no `lanes` tag, no road within reach) is left out, never guessed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "CROWD-anything topology evaluation (github.com/vita-epfl/CROWD-anything)"

MOTORWAY = {"motorway", "trunk", "motorway_link", "trunk_link"}
MAJOR = {"primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link"}
MINOR = {"unclassified", "residential", "living_street", "road"}
DRIVABLE = MOTORWAY | MAJOR | MINOR  # service roads and tracks left out: driveways would count as junctions

MATCH_RADIUS_M = 25.0  # a trajectory point farther than this from every drivable road is not matched
JUNCTION_RADIUS_M = 12.0  # a junction node this close to the path is passed through
FEATURE_RADIUS_M = 15.0  # crossings and signals this close to the path count
TURN_DEG = 30.0  # heading change between the first and the last fifth of the path that counts as a turn
MIN_TURN_PATH_M = 10.0  # shorter paths (standing still) give no turn fact

ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "road_type": ("motorway", "major road", "minor road"),
    "lanes": ("1", "2", "3 or more"),
    "oneway": ("yes", "no"),
    "intersection": ("none", "T-junction", "crossroads", "roundabout"),
    "pedestrian_crossing": ("yes", "no"),
    "traffic_signals": ("yes", "no"),
    "turn": ("straight", "left", "right"),
}


@dataclass
class OsmData:
    nodes: dict[int, tuple[float, float]]  # id -> (lat, lon)
    node_tags: dict[int, dict[str, str]]
    ways: dict[int, tuple[list[int], dict[str, str]]]  # id -> (node ids, tags), drivable roads only


@dataclass
class Plane:
    """Local equirectangular projection (metres east, north) around a reference point; good for a few km."""

    lat0: float
    lon0: float
    kx: float = field(init=False)
    ky: float = 111_320.0

    def __post_init__(self) -> None:
        self.kx = 111_320.0 * math.cos(math.radians(self.lat0))

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky


def overpass_query(south: float, west: float, north: float, east: float) -> str:
    box = f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"
    return (
        "[out:json][timeout:90];("
        f'way["highway"~"^({"|".join(sorted(DRIVABLE))})$"]{box};'
        f'node["highway"~"^(crossing|traffic_signals)$"]{box};'
        f'node["crossing"]{box};'
        ");out body;>;out skel qt;"
    )


def parse_overpass(payload: dict[str, Any]) -> OsmData:
    nodes: dict[int, tuple[float, float]] = {}
    node_tags: dict[int, dict[str, str]] = {}
    ways: dict[int, tuple[list[int], dict[str, str]]] = {}
    for element in payload.get("elements", []):
        if element["type"] == "node":
            nodes[element["id"]] = (element["lat"], element["lon"])
            if element.get("tags"):
                node_tags[element["id"]] = element["tags"]
        elif element["type"] == "way" and element.get("tags", {}).get("highway") in DRIVABLE:
            ways[element["id"]] = (element["nodes"], element["tags"])
    return OsmData(nodes, node_tags, ways)


def fetch_osm(
    latlons: list[tuple[float, float]],
    cache_dir: Path,
    margin_m: float = 60.0,
    url: str = OVERPASS_URL,
    min_interval_s: float = 1.0,
) -> OsmData:
    """OSM data of the trajectory's bounding box plus `margin_m`, from the cache or the Overpass API."""
    lats, lons = [p[0] for p in latlons], [p[1] for p in latlons]
    dlat = margin_m / 111_320.0
    dlon = margin_m / (111_320.0 * max(0.1, math.cos(math.radians(sum(lats) / len(lats)))))
    query = overpass_query(min(lats) - dlat, min(lons) - dlon, max(lats) + dlat, max(lons) + dlon)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{hashlib.sha1(query.encode()).hexdigest()}.json"
    if not path.is_file():
        time.sleep(min_interval_s)  # the public Overpass servers ask for at most about one request a second
        response = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=180)
        response.raise_for_status()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(response.text)
        tmp.replace(path)
    return parse_overpass(json.loads(path.read_text()))


def _segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _distance_to_path(p: tuple[float, float], path: list[tuple[float, float]]) -> float:
    if len(path) == 1:
        return math.hypot(p[0] - path[0][0], p[1] - path[0][1])
    return min(_segment_distance(p, a, b) for a, b in zip(path, path[1:]))


def _path_length(path: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def _node_degrees(osm: OsmData) -> dict[int, int]:
    """Number of road branches meeting at each node: 2 per interior use of a node in a way, 1 per way end."""
    degree: dict[int, int] = {}
    for node_ids, _ in osm.ways.values():
        for i, node in enumerate(node_ids):
            degree[node] = degree.get(node, 0) + (1 if i in (0, len(node_ids) - 1) else 2)
    return degree


def _matched_way(p: tuple[float, float], osm: OsmData, xy: dict[int, tuple[float, float]]) -> Optional[int]:
    best, best_d = None, MATCH_RADIUS_M
    for way_id, (node_ids, _) in osm.ways.items():
        points = [xy[n] for n in node_ids if n in xy]
        if len(points) >= 2:
            d = _distance_to_path(p, points)
            if d < best_d:
                best, best_d = way_id, d
    return best


def _road_type(highway: str) -> str:
    return "motorway" if highway in MOTORWAY else "major road" if highway in MAJOR else "minor road"


def _lanes(tags: dict[str, str], oneway: bool) -> Optional[str]:
    raw = tags.get("lanes:forward") or tags.get("lanes")
    try:
        lanes = int(str(raw).split(";")[0])
    except ValueError:
        return None
    if not tags.get("lanes:forward") and not oneway:
        lanes = max(1, lanes // 2)  # `lanes` counts both directions on a two-way road
    return "1" if lanes <= 1 else "2" if lanes == 2 else "3 or more"


def _is_oneway(tags: dict[str, str]) -> bool:
    return tags.get("oneway") in ("yes", "1", "-1") or tags.get("highway") in ("motorway", "motorway_link") \
        or tags.get("junction") == "roundabout"


def _turn(path: list[tuple[float, float]]) -> Optional[str]:
    if len(path) < 5 or _path_length(path) < MIN_TURN_PATH_M:
        return None
    k = max(1, len(path) // 5)
    (x0, y0), (x1, y1) = path[0], path[k]
    (x2, y2), (x3, y3) = path[-1 - k], path[-1]
    change = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y0, x1 - x0))
    change = (change + 180.0) % 360.0 - 180.0  # counter-clockwise (left) is positive
    return "straight" if abs(change) < TURN_DEG else "left" if change > 0 else "right"


def topology_facts(latlons: list[tuple[float, float]], osm: OsmData) -> dict[str, str]:
    """Closed-set facts (see ATTRIBUTES) for a trajectory in time order; attributes the map cannot answer are left
    out."""
    if not latlons:
        return {}
    plane = Plane(*latlons[0])
    path = [plane.xy(lat, lon) for lat, lon in latlons]
    xy = {n: plane.xy(*ll) for n, ll in osm.nodes.items()}
    facts: dict[str, str] = {}

    turn = _turn(path)
    if turn:
        facts["turn"] = turn

    start_way = next((w for w in (_matched_way(p, osm, xy) for p in path[:3]) if w is not None), None)
    if start_way is None:
        return facts  # off the drivable road network: no road facts
    tags = osm.ways[start_way][1]
    oneway = _is_oneway(tags)
    facts["road_type"] = _road_type(tags["highway"])
    facts["oneway"] = "yes" if oneway else "no"
    lanes = _lanes(tags, oneway)
    if lanes:
        facts["lanes"] = lanes

    near_path = [n for n in xy if _distance_to_path(xy[n], path) <= JUNCTION_RADIUS_M]
    degree = _node_degrees(osm)
    roundabout_nodes = {n for node_ids, t in osm.ways.values() if t.get("junction") == "roundabout" for n in node_ids}
    passed = sorted((n for n in near_path if degree.get(n, 0) >= 3 or n in roundabout_nodes),
                    key=lambda n: min(range(len(path)), key=lambda i: math.dist(path[i], xy[n])))
    if not passed:
        facts["intersection"] = "none"
    elif passed[0] in roundabout_nodes:
        facts["intersection"] = "roundabout"
    else:
        facts["intersection"] = "T-junction" if degree[passed[0]] == 3 else "crossroads"

    def any_feature(match) -> str:
        found = any(match(t) and _distance_to_path(xy[n], path) <= FEATURE_RADIUS_M
                    for n, t in osm.node_tags.items() if n in xy)
        return "yes" if found else "no"

    facts["pedestrian_crossing"] = any_feature(lambda t: t.get("highway") == "crossing" or "crossing" in t)
    facts["traffic_signals"] = any_feature(lambda t: t.get("highway") == "traffic_signals")
    return facts


def read_trajectory(path: Path) -> list[tuple[float, float]]:
    """A trajectory CSV with columns `lat,lon` (more columns allowed), one row per position in time order."""
    import csv

    with open(path, newline="") as f:
        return [(float(row["lat"]), float(row["lon"])) for row in csv.DictReader(f)]


def facts_for_trajectories(
    trajectories: Iterable[tuple[str, list[tuple[float, float]]]], cache_dir: Path, url: str = OVERPASS_URL
) -> dict[str, dict[str, str]]:
    return {clip_id: topology_facts(latlons, fetch_osm(latlons, cache_dir, url=url))
            for clip_id, latlons in trajectories}
