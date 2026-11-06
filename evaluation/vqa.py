"""Multiple-choice questions about a driving video, answered by a vision-language model.

Frames are sampled evenly over the clip on the same fps grid as the rest of the evaluation (evaluation.video) and
sent in time order with one closed question per topology attribute (evaluation.osm.ATTRIBUTES); the model answers
with an option letter. The backend is any OpenAI-compatible chat API that accepts images, e.g. a Qwen-VL model
served by vLLM (`vllm serve <model>` -> http://localhost:8000/v1).
"""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
import requests
from PIL import Image

from evaluation.osm import ATTRIBUTES
from evaluation.video import iter_frames

LETTERS = "ABCDEFGH"

QUESTIONS = {
    "road_type": "What kind of road is the camera car driving on at the start of the video?",
    "lanes": "How many lanes are there in the camera car's direction of travel at the start of the video?",
    "oneway": "Is the road the camera car drives on at the start a one-way road?",
    "intersection": "What is the first intersection ahead of the camera car in this video?",
    "pedestrian_crossing": "Is there a pedestrian crossing (zebra crossing) ahead of the camera car in this video?",
    "traffic_signals": "Are there traffic lights ahead of the camera car in this video?",
    "turn": "Over the whole video, does the camera car go straight, turn left or turn right?",
}

OPTION_TEXT = {  # what the model reads for each option value; the scoring uses the values
    "road_type": {"motorway": "motorway or expressway", "major road": "major road (main street, arterial)",
                  "minor road": "minor road (residential or side street)"},
    "intersection": {"none": "none: no intersection ahead", "T-junction": "T-junction (three-way)",
                     "crossroads": "crossroads (four-way or more)", "roundabout": "roundabout"},
}


@dataclass(frozen=True)
class Question:
    attribute: str
    text: str
    options: tuple[str, ...]  # option values, in letter order

    def prompt(self, num_frames: int) -> str:
        shown = OPTION_TEXT.get(self.attribute, {})
        lines = [f"{LETTERS[i]}. {shown.get(value, value)}" for i, value in enumerate(self.options)]
        return (
            f"These are {num_frames} frames of a dashcam video from a car, in time order.\n"
            f"{self.text}\n" + "\n".join(lines) + "\nAnswer with the letter of one option only."
        )


def question_for(attribute: str) -> Question:
    return Question(attribute, QUESTIONS[attribute], ATTRIBUTES[attribute])


def parse_letter(text: str, num_options: int) -> Optional[int]:
    """Index of the option the reply chooses, or None: a reply that is the letter ("B", "B.", "(B) major road"), or
    a longer reply's last stated choice ("the answer is B", "Answer: (B)", "**B**", or the letter alone at its end:
    "... a minor road.  C"). A capital letter elsewhere in a sentence ("A car is ...") is not a choice."""
    letters = LETTERS[:num_options]
    text = text.strip()
    match = re.match(rf"\(?([{letters}])(?:[.):]|$)", text)
    if not match:
        stated = re.findall(rf"(?i:answer|option|choice)\W{{0,3}}(?:is\W{{1,3}})?\(?([{letters}])\b"
                            rf"|\*\*\(?([{letters}])[.)]?\*\*|(?<=[.:\s])\(?([{letters}])[.)]?$", text)
        return LETTERS.index("".join(stated[-1])) if stated else None
    return LETTERS.index(match.group(1))


def video_duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def sample_frames(path: str, num_frames: int, max_side: int = 768) -> list[np.ndarray]:
    """`num_frames` frames spread evenly over the video (at its middle times), each scaled to at most `max_side`."""
    duration = video_duration(path)
    frames = list(iter_frames(path, fps=num_frames / duration))
    if len(frames) > num_frames:
        frames = [frames[round(i * (len(frames) - 1) / (num_frames - 1))] for i in range(num_frames)] \
            if num_frames > 1 else frames[:1]
    out = []
    for frame in frames:
        image = Image.fromarray(frame)
        image.thumbnail((max_side, max_side))
        out.append(np.asarray(image))
    return out


class VqaModel(Protocol):
    name: str

    def answer(self, frames: Sequence[np.ndarray], question: Question, video: str) -> tuple[Optional[int], str]:
        """(index of the chosen option or None, raw reply) for the frames of `video`."""
        ...


def _jpeg_data_url(frame: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


class OpenAICompatibleVqa:
    """A VLM behind an OpenAI-compatible /chat/completions endpoint (vLLM, SGLang, ...), greedy decoding."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 300.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.name = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout_s = timeout_s

    def answer(self, frames: Sequence[np.ndarray], question: Question, video: str) -> tuple[Optional[int], str]:
        content: list[dict] = [{"type": "image_url", "image_url": {"url": _jpeg_data_url(f)}} for f in frames]
        content.append({"type": "text", "text": question.prompt(len(frames))})
        body = {"model": self.name, "messages": [{"role": "user", "content": content}],
                "temperature": 0.0, "max_tokens": 8}
        response = requests.post(self.url, json=body, headers=self.headers, timeout=self.timeout_s)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"] or ""
        return parse_letter(reply, len(question.options)), reply


class ScriptedVqa:
    """Answers from a table {(video_path, attribute): option value} (tests and dry runs); unknown -> no answer."""

    name = "scripted"

    def __init__(self, table: dict[tuple[str, str], str]):
        self.table = table

    def answer(self, frames: Sequence[np.ndarray], question: Question, video: str) -> tuple[Optional[int], str]:
        value = self.table.get((video, question.attribute))
        if value is None:
            return None, ""
        return question.options.index(value), json.dumps(value)
