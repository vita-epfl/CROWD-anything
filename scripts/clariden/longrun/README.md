# Long-horizon evaluation on Clariden

Stage jobs for a run folder made by `python -m evaluation.horizons prepare` (windows of one-minute generations, each
with its real 5 s sample, edited caption row, OSM route record and anchor images). Each job takes the run folder as
`RUN` (`sbatch --export=ALL,RUN=<dir> <job>`); paths are those of malam's scratch on Clariden.

| Step | What it does |
|---|---|
| `stage1.sbatch` | cut the window clips and anchor images, split the captions into statements (Qwen3-32B), export the Qwen-Drive questions of prompt following and topology in 4 shards (`shard.py`) |
| `vqa.sbatch` | Qwen-Drive answers (prompt following 512 new tokens, topology 768); resumes, so resubmit until done |
| `vision.sbatch` | traffic signs and lights (Grounding DINO + DINOv2), ViPE on every clip, vehicle speeds |
| `score.py` | scores per window start: prompt following, topology kappa, vehicle speed against the limit |
| `watch.sh` | from a laptop: resubmits the Qwen-Drive job, submits the vision job when a debug slot is free, scores at the end |
| `generated.sbatch` | generated videos arriving in batches (normal partition, 12 h): reuses a finished real-clip run (`REAL`), takes the videos in the generator's manifest (`MANIFEST`) when it starts, runs vision, then only the unanswered questions (`pending.py`), then scores; resubmit as more videos arrive |

Debug takes two jobs per user and runs one at a time; the topology questions dominate the cost (~2.7 GPU-hours per
155 windows on real video only), so a full run of all windows and models belongs on the normal partition.
