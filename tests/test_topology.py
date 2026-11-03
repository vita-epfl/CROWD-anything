import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from evaluation import osm, topology, vqa
from evaluation.osm import OsmData, parse_overpass, topology_facts
from evaluation.topology import Clip, ask_all, score, summarize
from evaluation.vqa import OpenAICompatibleVqa, ScriptedVqa, parse_letter, question_for

LAT, LON = 46.52, 6.63
D = 0.0005  # ~55 m north-south, ~38 m east-west at this latitude


def node(i, lat, lon, **tags):
    element = {"type": "node", "id": i, "lat": lat, "lon": lon}
    if tags:
        element["tags"] = tags
    return element


def way(i, nodes, **tags):
    return {"type": "way", "id": i, "nodes": nodes, "tags": tags}


def crossroads(side_road_both_ways=True, **main_tags) -> OsmData:
    """East-west main road through node 2; a north-south side road meets it there."""
    elements = [
        node(1, LAT, LON - 2 * D), node(2, LAT, LON), node(3, LAT, LON + 2 * D),
        node(4, LAT + 2 * D, LON), node(5, LAT - 2 * D, LON),
        node(9, LAT, LON + D, highway="crossing"),
        way(10, [1, 2, 3], highway="residential", lanes="2", **main_tags),
        way(11, [4, 2, 5] if side_road_both_ways else [4, 2], highway="residential"),
    ]
    return parse_overpass({"elements": elements})


def east(n=9):
    return [(LAT, LON - 1.5 * D + i * 3 * D / (n - 1)) for i in range(n)]


def test_crossroads_facts():
    facts = topology_facts(east(), crossroads())
    assert facts == {
        "road_type": "minor road", "oneway": "no", "lanes": "1", "intersection": "crossroads",
        "pedestrian_crossing": "yes", "traffic_signals": "no", "turn": "straight",
    }


def test_t_junction_oneway_and_signals():
    data = crossroads(side_road_both_ways=False, oneway="yes")
    data.node_tags[2] = {"highway": "traffic_signals"}
    facts = topology_facts(east(), data)
    assert (facts["intersection"], facts["oneway"], facts["lanes"], facts["traffic_signals"]) == (
        "T-junction", "yes", "2", "yes")


def test_roundabout_and_road_type():
    elements = [node(1, LAT, LON - 2 * D), node(2, LAT, LON), node(3, LAT + D, LON + D), node(4, LAT, LON + 2 * D),
                way(10, [1, 2], highway="primary"), way(11, [2, 3, 4, 2], highway="primary", junction="roundabout")]
    facts = topology_facts(east(), parse_overpass({"elements": elements}))
    assert (facts["road_type"], facts["intersection"]) == ("major road", "roundabout")


def test_right_turn_and_no_road_nearby():
    path = [(LAT, LON - 2 * D + i * D / 2) for i in range(5)] + [(LAT - i * D / 2, LON) for i in range(1, 5)]
    assert topology_facts(path, crossroads())["turn"] == "right"
    far = [(LAT + 1.0, LON + i * 3 * D / 8) for i in range(9)]  # ~110 km away: no road matched
    assert topology_facts(far, crossroads()) == {"turn": "straight"}


def test_fetch_osm_uses_cache(tmp_path, monkeypatch):
    calls = []

    class Response:
        text = json.dumps({"elements": [node(1, LAT, LON), node(2, LAT, LON + D), way(10, [1, 2], highway="motorway"),
                                        way(11, [1, 2], highway="footway")]})

        def raise_for_status(self):
            pass

    monkeypatch.setattr(osm.requests, "post", lambda *a, **k: calls.append(k) or Response())
    first = osm.fetch_osm(east(), tmp_path, min_interval_s=0)
    second = osm.fetch_osm(east(), tmp_path, min_interval_s=0)
    assert len(calls) == 1 and "highway" in calls[0]["data"]["data"]
    assert list(first.ways) == list(second.ways) == [10]  # footways are not drivable


def test_parse_letter_and_prompt():
    assert parse_letter("B", 3) == 1
    assert parse_letter("The answer is C.", 4) == 2
    assert parse_letter("D", 3) is None and parse_letter("", 2) is None
    prompt = question_for("intersection").prompt(8)
    assert "8 frames" in prompt and "D. roundabout" in prompt and prompt.endswith("letter of one option only.")


def test_openai_compatible_backend_request(monkeypatch):
    sent = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "B"}}]}

    def post(url, json, headers, timeout):
        sent.update(url=url, body=json)
        return Response()

    monkeypatch.setattr(vqa.requests, "post", post)
    model = OpenAICompatibleVqa("http://host:8000/v1/", "qwen-vl")
    frames = [np.zeros((16, 16, 3), dtype=np.uint8)] * 3
    assert model.answer(frames, question_for("oneway"), "v.mp4") == (1, "B")
    content = sent["body"]["messages"][0]["content"]
    assert sent["url"] == "http://host:8000/v1/chat/completions" and sent["body"]["temperature"] == 0.0
    assert [c["type"] for c in content] == ["image_url"] * 3 + ["text"]


def _record(clip, side, attribute, truth, answer):
    return {"clip_id": clip, "side": side, "attribute": attribute, "truth": truth, "answer": answer}


def test_score_metrics():
    truths = {"a": "yes", "b": "no", "c": "yes", "d": "yes"}
    reference = {"a": "yes", "b": "no", "c": "yes", "d": "no"}  # 3 of 4 right
    generated = {"a": "yes", "b": "yes", "c": "no", "d": "yes"}  # 2 of 4 right
    answers = [_record(c, "reference", "oneway", t, reference[c]) for c, t in truths.items()]
    answers += [_record(c, "generated", "oneway", t, generated[c]) for c, t in truths.items()]
    (row,) = score(answers)
    assert (row["accuracy_reference"], row["accuracy_generated"]) == (0.75, 0.5)
    assert row["normalized"] == pytest.approx(0.5 / 0.75)
    assert row["agreement"] == 0.25  # only clip a
    # control truths (next clip's): a->no, b->yes, c->yes, d->yes
    assert row["control_generated"] == 0.5 and row["control_reference"] == 0.25
    assert (row["chance"], row["majority"]) == (0.5, 0.75)
    assert summarize([row])["normalized"] == pytest.approx(2 / 3)


def _make_video(path: Path, seconds: int = 3):
    source = f"testsrc=size=64x48:rate=24:duration={seconds}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", source, "-pix_fmt", "yuv420p", str(path)],
                   check=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_ask_all_end_to_end_and_resume(tmp_path):
    gen, ref = tmp_path / "gen.mp4", tmp_path / "ref.mp4"
    _make_video(gen)
    _make_video(ref)
    assert len(vqa.sample_frames(str(ref), 4)) == 4
    facts = {"c1": {"oneway": "yes", "turn": "left", "unknown_attr": "x"}}
    # the generated video gets no answer for "turn"
    table = {(str(ref), "oneway"): "yes", (str(ref), "turn"): "left", (str(gen), "oneway"): "no"}
    model = ScriptedVqa(table)
    clips = [Clip("c1", {"generated": str(gen), "reference": str(ref)})]
    answers_path = tmp_path / "out" / "answers.jsonl"
    answers = ask_all(clips, facts, model, answers_path, num_frames=4)
    assert len(answers) == 4  # 2 known attributes x 2 sides; unknown attributes are skipped
    rows = {r["attribute"]: r for r in score(answers)}
    assert rows["oneway"]["accuracy_reference"] == 1.0 and rows["oneway"]["accuracy_generated"] == 0.0
    assert rows["turn"]["unanswered_generated"] == 1
    # a rerun asks nothing again
    ask_all(clips, facts, model, answers_path, num_frames=4)
    assert len(answers_path.read_text().splitlines()) == 4
    summary = topology.write_results(score(answers), tmp_path / "out")
    assert (tmp_path / "out" / "per_attribute.csv").is_file() and summary["attributes"] == 2
