import json

from evaluation import prompt_following
from evaluation.caption_statements import KINDS, NUM_FALSE, messages, parse_statements, split_captions, swaps_sides


def reply(true=4, false=NUM_FALSE, kind="static"):
    items = [{"text": f"true {i}", "kind": kind, "true": True} for i in range(true)]
    items += [{"text": f"false {i}", "kind": "dynamic", "true": False} for i in range(false)]
    return json.dumps({"statements": items})


def test_parse_statements_interleaves_false_ones():
    statements = parse_statements("<think>hm</think>Here: " + reply(false=5) + " done")
    assert [s["true"] for s in statements] == [True, False, True, False, True, False, True]  # 3 of the false kept
    assert statements[0] == {"text": "true 0", "kind": "static", "true": True}


def test_parse_statements_rejects_bad_replies():
    assert parse_statements("not json") is None
    assert parse_statements(reply(false=2)) is None  # fewer than NUM_FALSE false statements
    assert parse_statements(reply(kind="topology")) is None  # nothing to score besides topology
    assert parse_statements(json.dumps({"statements": [{"text": "x", "kind": "weather", "true": True}]})) is None


def test_messages_end_with_the_caption():
    chat = messages("A red car.")
    assert chat[0]["role"] == "system" and chat[-1] == {"role": "user", "content": "Caption: A red car."}


class ScriptedLlm:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def generate(self, chats, sample=False):
        self.calls.append(sample)
        return [self.replies[(chat[-1]["content"], sample)] for chat in chats]


def test_split_captions_retries_invalid_replies_with_sampling():
    llm = ScriptedLlm({("Caption: a", False): reply(), ("Caption: b", False): "oops",
                       ("Caption: b", True): reply(true=3)})
    split = split_captions(["a", "b", "a"], llm, batch_size=8)
    assert set(split) == {"a", "b"} and len(split["b"]) == 6
    assert llm.calls == [False, True]


def test_kinds_match_prompt_following():
    assert KINDS == prompt_following.KINDS


def test_swaps_sides():
    assert swaps_sides("A white van is parked on the left.", "A white van is parked on the right.")
    assert swaps_sides("The pedestrian crosses from right to left.", "The pedestrian crosses from left to right.")
    assert not swaps_sides("The camera car turns left.", "The camera car turns right.")  # turn direction: allowed
    assert not swaps_sides("A red van is parked on the right.", "A white van is parked on the right.")


def test_parse_statements_drops_side_swaps():
    items = [{"text": "A white van is parked on the right.", "kind": "static", "true": True},
             {"text": "A grey car is ahead.", "kind": "static", "true": True},
             {"text": "A white van is parked on the left.", "kind": "static", "true": False},
             {"text": "A red car is ahead.", "kind": "static", "true": False},
             {"text": "A grey truck is ahead.", "kind": "static", "true": False}]
    assert parse_statements(json.dumps({"statements": items})) is None  # 2 false ones left
    items.append({"text": "A grey bus is ahead.", "kind": "static", "true": False})
    kept = [s["text"] for s in parse_statements(json.dumps({"statements": items})) if not s["true"]]
    assert kept == ["A red car is ahead.", "A grey truck is ahead.", "A grey bus is ahead."]
