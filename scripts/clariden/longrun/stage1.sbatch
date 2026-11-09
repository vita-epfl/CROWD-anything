#!/bin/bash
#SBATCH --job-name=long-prep
#SBATCH --account=ab036
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/malam/rgwm-eval/logs/%x-%j.out
# Long-horizon evaluation, stage 1 for a run folder made by `evaluation.horizons prepare` (sbatch --export=RUN=<dir>):
# cut the window clips and anchor images, split the window captions into statements (Qwen3-32B, 4 GPUs), and
# export the Qwen-Drive questions of prompt following (statements, with a control) and topology (OSM facts) in 4
# shards each.
set -euo pipefail
R=/iopsstor/scratch/cscs/malam/rgwm-eval
SEL=/capstor/store/cscs/swissai/ab036/tmp_shared/rgwm-pablo-anchor-covered-20261108
srun --environment=wm_data_manifest_env bash -c "set -e
cd $R/CROWD-anything
PYTHONPATH=. python -m evaluation.horizons cut --run $RUN --natix-root /capstor/store/cscs/swissai/ab036/datasets/natix \
  --anchor-images $SEL/anchor_images.jsonl
mkdir -p $RUN/split
python - <<'PY'
import csv
rows = list(csv.DictReader(open('$RUN/captions.csv')))
for g in range(4):
    with open('$RUN/split/captions-%d.csv' % g, 'w', newline='') as f:
        w = csv.DictWriter(f, ['sample_id', 'caption']); w.writeheader(); w.writerows(rows[g::4])
PY
(source $R/preprocessin-wm/caption/qwen/env.sh
 for g in 0 1 2 3; do
   [ -s $RUN/split/statements-\$g.json ] || PYTHONPATH=. python -m evaluation.caption_statements \
     --captions $RUN/split/captions-\$g.csv --model /iopsstor/scratch/cscs/malam/models/Qwen3-32B --out $RUN/split/statements-\$g.json \
     --device cuda:\$g > $RUN/split/log-\$g.txt 2>&1 &
 done; wait)
python - <<'PY'
import glob, json
merged = {}
for p in sorted(glob.glob('$RUN/split/statements-*.json')):
    merged.update(json.load(open(p)))
json.dump(merged, open('$RUN/split.json', 'w'))
PY
PYTHONPATH=. python -m evaluation.horizons merge-statements --statements $RUN/split.json --out $RUN/statements.json
PYTHONPATH=. python -m evaluation.prompt_following export --clips $RUN/clips.csv --statements $RUN/statements.json \
  --out $RUN/batch_pf --frames 8
PYTHONPATH=. python -m evaluation.topology export --clips $RUN/clips.csv --facts $RUN/facts.json --out $RUN/batch_topo \
  --frames 8
python $R/CROWD-anything/scripts/clariden/longrun/shard.py $RUN/batch_pf 4
python $R/CROWD-anything/scripts/clariden/longrun/shard.py $RUN/batch_topo 4
"
