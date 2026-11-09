"""Split a batch runner's videos.json / prompts.json (evaluation.prompt_following or evaluation.topology export) into
n shards, each prompt keeping only its shard's videos: <dir>/videos-<i>.json, <dir>/prompts-<i>.json. All videos of a
clip (the part of the video id before "::") go to the same shard, so every shard holds all kinds of side; the batch
runner does not resume a shard whose inputs changed, so a shard of only one side (e.g. all controls) is lost whole
when that side is redone."""
import json
import sys
from pathlib import Path

folder, n = Path(sys.argv[1]), int(sys.argv[2])
videos = json.loads((folder / "videos.json").read_text())["videos"]
prompts = json.loads((folder / "prompts.json").read_text())["prompts"]
clips = sorted({v["id"].split("::")[0] for v in videos})
shard_of = {clip: i % n for i, clip in enumerate(clips)}
for i in range(n):
    shard = [v for v in videos if shard_of[v["id"].split("::")[0]] == i]
    ids = {v["id"] for v in shard}
    kept = [dict(p, video_ids=[v for v in p["video_ids"] if v in ids]) for p in prompts]
    (folder / f"videos-{i}.json").write_text(json.dumps({"videos": shard}, indent=1))
    (folder / f"prompts-{i}.json").write_text(json.dumps({"prompts": [p for p in kept if p["video_ids"]]}, indent=1))
print(folder, len(videos), "videos,", len(prompts), "prompts in", n, "shards")
