"""The questions of a batch export not answered yet, as a new batch of n shards (python3 pending.py <export dir>
<answers root> <batch dir> n): a (prompt, video) pair is answered when any answers-*.json under the answers root
holds it with status ok. For generated videos arriving in batches: each job asks only what is new or failed, so no
shard has to resume with changed inputs. Prints the number of pairs left; writes nothing when none are."""
import json
import subprocess
import sys
from pathlib import Path

export, root, batch, n = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
done = set()
for path in root.rglob("answers-*.json"):
    try:
        results = json.loads(path.read_text())["results"]
    except (ValueError, KeyError):  # a file being written by a killed job
        continue
    done |= {(r["prompt_id"], r["video_id"]) for r in results if r.get("status") == "ok"}

videos = json.loads((export / "videos.json").read_text())["videos"]
prompts = json.loads((export / "prompts.json").read_text())["prompts"]
left = [dict(p, video_ids=[v for v in p["video_ids"] if (p["id"], v) not in done]) for p in prompts]
left = [p for p in left if p["video_ids"]]
wanted = {v for p in left for v in p["video_ids"]}
print(sum(len(p["video_ids"]) for p in left))
if left:
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "videos.json").write_text(json.dumps({"videos": [v for v in videos if v["id"] in wanted]}, indent=1))
    (batch / "prompts.json").write_text(json.dumps({"prompts": left}, indent=1))
    subprocess.run([sys.executable, str(Path(__file__).with_name("shard.py")), str(batch), n], check=True,
                   stdout=sys.stderr)
