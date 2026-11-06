import json

from evaluation.prompt_following import (CONTROL, export_batch, parse_answers, read_batch_answers, summarize,
                                         with_control)

STATEMENTS = {
    "a": [{"text": "A red car is ahead.", "kind": "static", "true": True},
          {"text": "It is night.", "kind": "static", "true": False},
          {"text": "The camera car turns left.", "kind": "dynamic", "true": True}],
    "b": [{"text": "A bus is ahead.", "kind": "static", "true": True}],
}
CLIPS = {"a": {"reference": "ra.mp4", "gen": "ga.mp4"}, "b": {"reference": "rb.mp4"}}


def test_parse_answers():
    assert parse_answers("1: yes\n2. No\n**3**: yes", 3) == [True, False, True]
    assert parse_answers("Sure.\n1: yes\n1: no\n4: yes", 2) == [True, None]  # first answer counts, 4 is out of range
    assert parse_answers("Yes, the car turns left.", 1) == [None]


def test_control_is_another_clips_reference():
    clips = with_control(CLIPS)
    assert clips["a"][CONTROL] == "rb.mp4" and clips["b"][CONTROL] == "ra.mp4"
    assert CONTROL not in CLIPS["a"]


def test_export_and_score(tmp_path):
    export_batch(CLIPS, STATEMENTS, tmp_path)
    prompts = json.loads((tmp_path / "prompts.json").read_text())["prompts"]
    assert {p["id"] for p in prompts} == {"a", "b"}
    assert sorted(next(p for p in prompts if p["id"] == "a")["video_ids"]) == ["a::control", "a::gen", "a::reference"]
    assert "3. The camera car turns left." in prompts[0]["text"]

    results = [
        {"video_id": "a::reference", "prompt_id": "a", "status": "ok", "answer": "1: yes\n2: no\n3: yes"},
        {"video_id": "a::gen", "prompt_id": "a", "status": "ok", "answer": "1: yes\n2: yes\n3: no"},
        {"video_id": "a::control", "prompt_id": "a", "status": "ok", "answer": "1: no\n2: no\n3: no"},
        {"video_id": "b::reference", "prompt_id": "b", "status": "error", "answer": "1: yes"},
    ]
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"results": results}))
    rows = read_batch_answers([path], STATEMENTS)
    summary = {s["side"]: s for s in summarize(rows)}
    assert list(summary) == ["reference", "gen", CONTROL]
    assert summary["reference"]["confirmed"] == 1.0 and summary["reference"]["rejected"] == 1.0
    assert summary["gen"]["confirmed_static"] == 1.0 and summary["gen"]["confirmed_dynamic"] == 0.0
    assert summary["gen"]["confirmed"] == 0.5 and summary["gen"]["rejected"] == 0.0
    assert summary["gen"]["balanced"] == 0.25 and summary["gen"]["confirmed_on_real"] == 0.5
    assert summary[CONTROL]["confirmed"] == 0.0
    assert summary["reference"]["unanswered"] == 1  # clip b: the run failed
