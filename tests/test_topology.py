import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from evaluation import osm, topology, vqa
from evaluation.caption_leaks import check_captions, leaks, rewrite
from evaluation.osm import OsmData, parse_overpass, topology_facts
from evaluation.route_records import facts_from_record, load_records
from evaluation.topology import apply_leaks, ask_all, export_batch, read_batch_answers, score, summarize
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
    assert parse_letter("(B) major road", 3) == 1 and parse_letter("A.", 3) == 0
    assert parse_letter("A car drives on a road with two lanes. Answer: (C)", 3) == 2
    assert parse_letter("Based on the visual evidence in the frames, particularly", 3) is None
    assert parse_letter("The road looks wide, so **B**.", 3) == 1
    assert parse_letter("The road has a median and no oncoming traffic.  C", 3) == 2
    assert parse_letter("There is one lane in the camera car's direction of travel. A", 3) == 0
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
    answers += [_record(c, "top_i2v", "oneway", t, generated[c]) for c, t in truths.items()]
    rows = {r["side"]: r for r in score(answers)}
    assert list(rows) == ["reference", "top_i2v"]  # reference first
    ref, gen = rows["reference"], rows["top_i2v"]
    assert (ref["accuracy"], gen["accuracy"]) == (0.75, 0.5)
    assert gen["normalized"] == pytest.approx(0.5 / 0.75) and ref["normalized"] == 1.0
    assert gen["agreement"] == 0.25  # only clip a
    # control truths (next clip's): a->no, b->yes, c->yes, d->yes
    assert gen["control"] == 0.5 and ref["control"] == 0.25
    assert (gen["chance"], gen["majority"]) == (0.5, 0.75)
    summary = {s["side"]: s for s in summarize(list(rows.values()))}
    assert summary["top_i2v"]["normalized"] == pytest.approx(2 / 3)


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
    clips = {"c1": {"top_t2v": str(gen), "reference": str(ref)}}
    answers_path = tmp_path / "out" / "answers.jsonl"
    answers = ask_all(clips, facts, model, answers_path, num_frames=4)
    assert len(answers) == 4  # 2 known attributes x 2 videos; unknown attributes are skipped
    rows = {(r["attribute"], r["side"]): r for r in score(answers)}
    assert rows[("oneway", "reference")]["accuracy"] == 1.0 and rows[("oneway", "top_t2v")]["accuracy"] == 0.0
    assert rows[("turn", "top_t2v")]["unanswered"] == 1
    # a rerun asks nothing again
    ask_all(clips, facts, model, answers_path, num_frames=4)
    assert len(answers_path.read_text().splitlines()) == 4
    summary = topology.write_results(score(answers), tmp_path / "out")
    assert (tmp_path / "out" / "per_attribute.csv").is_file()
    assert [s["side"] for s in summary] == ["reference", "top_t2v"]


def test_export_and_read_batch_answers(tmp_path):
    clips = {"s1": {"reference": "/v/s1_gt.mp4", "top_i2v": "/v/s1_top.mp4"}, "s2": {"reference": "/v/s2_gt.mp4"}}
    facts = {"s1": {"oneway": "yes", "turn": "left"}, "s2": {"oneway": "no"}}
    export_batch(clips, facts, tmp_path)
    videos = json.loads((tmp_path / "videos.json").read_text())["videos"]
    prompts = {p["id"]: p for p in json.loads((tmp_path / "prompts.json").read_text())["prompts"]}
    assert {v["id"] for v in videos} == {"s1::reference", "s1::top_i2v", "s2::reference"}
    assert set(prompts["oneway"]["video_ids"]) == {"s1::reference", "s1::top_i2v", "s2::reference"}
    assert prompts["turn"]["video_ids"] == ["s1::reference", "s1::top_i2v"] and "A. yes" in prompts["oneway"]["text"]

    results = [
        {"video_id": "s1::reference", "prompt_id": "oneway", "status": "ok", "answer": "A"},
        {"video_id": "s1::top_i2v", "prompt_id": "oneway", "status": "ok", "answer": "The answer is B."},
        {"video_id": "s2::reference", "prompt_id": "oneway", "status": "error", "answer": None},
        {"video_id": "s1::reference", "prompt_id": "turn", "status": "ok", "answer": "B"},
    ]
    (tmp_path / "answers.json").write_text(json.dumps({"results": results}))
    records = {(r["clip_id"], r["side"], r["attribute"]): r for r in
               read_batch_answers([tmp_path / "answers.json"], apply_leaks(facts, {"s1": {"leaked": {"turn": ["x"]}}}),
                                  "qwen-drive")}
    assert records[("s1", "reference", "oneway")]["answer"] == "yes"
    assert records[("s1", "top_i2v", "oneway")]["answer"] == "no"
    assert records[("s2", "reference", "oneway")]["answer"] is None
    assert ("s1", "reference", "turn") not in records  # leaked by the caption: not scored


def record(roads=None, events=(), maneuver=None):
    road = {"road_class": "secondary", "one_way": False, "lanes": 2, "lanes_source": "map"}
    road.update(roads or {})
    return {"roads": [road], "events": list(events), "maneuver": maneuver or {"kind": "straight"}}


def event(kind, dist, **extra):
    return {"type": kind, "dist_from_start_m": dist, **extra}


def test_facts_from_route_record():
    rec = record(events=[event("side_road", 20), event("crossing", 40, subtype="marked"), event("roundabout", 55),
                         event("signals", 150)])
    assert facts_from_record(rec) == {
        "road_type": "major road", "oneway": "no", "lanes": "2", "intersection": "roundabout",
        "pedestrian_crossing": "yes", "traffic_signals": "no", "turn": "straight",  # signals beyond 100 m: not seen
    }
    rec = record(roads={"road_class": "trunk", "one_way": True, "lanes": 3, "lanes_source": None},
                 events=[event("fork", 30), event("t_junction", 60), event("signals", 50, direction_known=False)],
                 maneuver={"kind": "slight", "direction": "left"})
    facts = facts_from_record(rec)
    # lanes not from the map, a fork first, "possibly" signals and a slight bend give no fact
    assert facts == {"road_type": "motorway", "oneway": "yes", "pedestrian_crossing": "no"}
    assert facts_from_record(record(events=[event("intersection", 80, arms=4)],
                                    maneuver={"kind": "turn", "direction": "right"}))["intersection"] == "crossroads"
    assert facts_from_record(record(maneuver={"kind": "turn", "direction": "right"}))["turn"] == "right"
    assert facts_from_record(record())["intersection"] == "none"


def test_load_records(tmp_path):
    import zipfile

    sid = "Croatia/x/FRONT.mp4#f000360"
    with zipfile.ZipFile(tmp_path / "s.zip", "w") as z:
        z.writestr("text.jsonl", json.dumps({"sample_id": sid, **record()}) + "\n"
                   + json.dumps({"sample_id": "other#f000000", **record()}) + "\n")
    (tmp_path / "index.jsonl").write_text(json.dumps({"sample_id": sid, "image": "s.zip::a.webp"}) + "\n")
    records = load_records(tmp_path, [sid, "missing#f000000"])
    assert list(records) == [sid] and records[sid]["roads"][0]["road_class"] == "secondary"


def test_caption_leaks():
    caption = ("The camera begins with a static view of the plaza and storefront. It then initiates a smooth pan to "
               "the left, revealing a pedestrian crossing sign and the adjacent street. The pedestrian walks along "
               "the pavement and steps onto the crosswalk, crossing the street from left to right.")
    assert leaks(caption) == {"pedestrian_crossing": ["crosswalk", "pedestrian crossing"]}
    assert rewrite(caption) == "The camera begins with a static view of the plaza and storefront."
    assert leaks("The car turns left at the roundabout on a two-lane road.") == {
        "intersection": ["roundabout"], "lanes": ["two-lane"], "turn": ["turns left"]}
    result = check_captions([{"sample_id": "s1", "caption": caption}, {"sample_id": "s1", "caption": caption},
                             {"sample_id": "s2", "caption": "A sunny day."}])
    assert result["s2"]["leaked"] == {} and result["s2"]["rewritten"] == "A sunny day."
    with pytest.raises(ValueError, match="different captions"):
        check_captions([{"sample_id": "s1", "caption": "a"}, {"sample_id": "s1", "caption": "b"}])


@pytest.mark.parametrize("text,attribute", [
    ("A van approaches from the opposite direction.", "oneway"),
    ("Oncoming traffic is visible.", "oneway"),
    ("The road curves gently to the right.", "turn"),
    ("A stop sign stands on the corner.", "intersection"),
    ("A dashed white center line runs ahead.", "lanes"),
    ("A sign shows a speed limit of 40 km/h.", "speed_limit"),
])
def test_indirect_leaks(text, attribute):
    assert attribute in leaks(text)


def test_harmless_phrases_are_not_leaks():
    assert leaks("The car follows a silver SUV in the same lane past parked cars and trees.") == {}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_rgwm_prepare_cuts_the_sample_window(tmp_path):
    import csv

    from evaluation import rgwm
    from tests.test_tokenization import frame_numbers, make_indexed_video

    natix = tmp_path / "natix"
    (natix / "Croatia").mkdir(parents=True)
    make_indexed_video(natix / "Croatia" / "FRONT.mkv", seconds=4, fps=30)  # 120 source frames, 96 on the 24 fps grid
    export = tmp_path / "export"
    (export / "videos" / "sample-01").mkdir(parents=True)
    _make_video(export / "videos" / "sample-01" / "top_i2v.mp4", seconds=1)  # 64x48, like the source
    sample_id = "Croatia/FRONT.mkv#f000024"
    with open(export / "generations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "sample_id", "profile", "mode", "file"])
        writer.writeheader()
        writer.writerow({"sample": "sample-01", "sample_id": sample_id, "profile": "top", "mode": "i2v",
                         "file": "videos/sample-01/top_i2v.mp4"})
    clips_csv = rgwm.prepare(export, natix, tmp_path / "out", num_frames=9)
    clips = topology.read_clips(clips_csv)
    assert set(clips[sample_id]) == {"top_i2v", "reference"}
    from evaluation.video import iter_frames

    reference = list(iter_frames(clips[sample_id]["reference"]))
    assert len(reference) == 9
    # the grid frames 24..32 of the source (frame i at i/24 s); h264 may shift the encoded values by a level or two
    expected = [round(i * 30 / 24) for i in range(24, 33)]
    assert all(abs(a - b) <= 2 for a, b in zip(frame_numbers(reference), expected))
    assert rgwm.split_sample_id(sample_id) == ("Croatia/FRONT.mkv", 24)
