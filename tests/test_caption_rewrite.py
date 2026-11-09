import json

from evaluation.caption_leaks import leaks
from evaluation.caption_rewrite import (EXAMPLES, FEEDBACK, feedback, messages, parse_rewrite, remaining,
                                        rewrite_captions)


def reply(text, removed=("roundabout",), kept=()):
    return json.dumps({"rewritten": text, "removed": list(removed), "kept_hints": list(kept)})


class ScriptedLlm:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def generate(self, chats, sample=False):
        self.calls.append(sample)
        return [self.replies[(chat[-1]["content"], sample)] for chat in chats]


def test_examples_remove_every_leak():
    for caption, example in EXAMPLES:
        assert leaks(caption) and not remaining(example["rewritten"])
    chat = messages("A car.")
    assert chat[0]["role"] == "system" and chat[-1]["content"] == "Caption: A car." and len(chat) == 8


def test_parse_rewrite():
    assert parse_rewrite("<think>x</think> " + reply("A car.", kept=["a hint"])) == {
        "rewritten": "A car.", "removed": ["roundabout"], "kept_hints": ["a hint"]}
    assert parse_rewrite('{"rewritten": "A car."}')["removed"] == []
    for bad in ("oops", '{"rewritten": ""}', '{"rewritten": "A car.", "removed": "x"}', "[1]"):
        assert parse_rewrite(bad) is None


def test_rewrite_captions_sends_leaks_back_then_drops_leaking_sentences():
    leaky_c = reply("A bus stops at the intersection. A tree stands.")
    llm = ScriptedLlm({
        ("Caption: a", False): reply("A white car drives ahead."),
        ("Caption: b", False): "oops",
        ("Caption: b", True): reply("A van parks."),
        ("Caption: c", False): reply("A bus stops. It enters the roundabout."),
        ("Caption: d", False): leaky_c,
        (FEEDBACK.format('"roundabout"'), False): reply("A bus stops."),
        (FEEDBACK.format('"intersection"'), False): leaky_c,
    })
    done = rewrite_captions(["a", "b", "c", "d", "a"], llm, batch_size=8)
    assert llm.calls == [False, False, True, False]  # round 0; feedback + resample in round 1; feedback in round 2
    assert done["a"]["rewritten_caption"] == "A white car drives ahead." and done["a"]["method"] == "llm"
    assert done["b"]["method"] == "llm"
    assert done["c"]["rewritten_caption"] == "A bus stops." and done["c"]["method"] == "llm"
    assert done["d"]["rewritten_caption"] == "A tree stands." and done["d"]["method"] == "llm+drop"
    assert done["d"]["leaks_left"] == "{}" and done["d"]["words_after"] == 3


def test_feedback_chat_lists_the_leaking_phrases():
    text = reply("A crosswalk and a traffic light.")
    chat = feedback("x", text, parse_rewrite(text))
    assert chat[-2] == {"role": "assistant", "content": text}
    assert chat[-1]["content"] == FEEDBACK.format('"crosswalk", "traffic light"')


def test_remaining_catches_paraphrases_but_not_turn_signals():
    found = remaining("A marked area with white stripes spans the road. The road changes direction. A blue sign.")
    assert found == {"crossing": ["marked area", "stripes span"], "signs": ["sign"],
                     "geometry": ["changes direction"]}
    assert remaining("The truck activates its turn signal. A red and white striped tower stands.") == {}
