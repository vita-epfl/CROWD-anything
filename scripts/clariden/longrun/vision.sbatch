#!/bin/bash
#SBATCH --job-name=long-vision
#SBATCH --account=ab036
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/malam/rgwm-eval/logs/%x-%j.out
# Long-horizon evaluation, vision (sbatch --export=RUN=<dir>): traffic signs and lights (Grounding DINO + DINOv2,
# 4 GPUs on clip shards), then ViPE on every window clip (4 workers) and vehicle speeds. Finished outputs are
# skipped, so a resubmit resumes.
set -euo pipefail
R=/iopsstor/scratch/cscs/malam/rgwm-eval
M=/iopsstor/scratch/cscs/malam/models
srun --environment=wm_data_manifest_env bash -c "set -e
source $R/preprocessin-wm/caption/qwen/env.sh
cd $R/CROWD-anything
python - <<'PY'
import csv
rows = list(csv.DictReader(open('$RUN/clips.csv')))
ids = sorted({r['clip_id'] for r in rows})
for g in range(4):
    keep = set(ids[g::4])
    with open('$RUN/clips-%d.csv' % g, 'w', newline='') as f:
        w = csv.DictWriter(f, ['clip_id', 'side', 'video']); w.writeheader(); w.writerows(r for r in rows if r['clip_id'] in keep)
with open('$RUN/videos.csv', 'w') as f:
    f.write('id,video\n')
    for r in csv.DictReader(open('$RUN/cuts.csv')):
        f.write('%s__%s,%s\n' % (r['tag'], r['side'], r['out']))
PY
for g in 0 1 2 3; do
  PYTHONPATH=. python -m evaluation.traffic_elements detect --clips $RUN/clips-\$g.csv --anchors $RUN/anchors.json \
    --detector $M/grounding-dino-base --features $M/dinov2-base --out $RUN/detections --device cuda:\$g \
    > $RUN/detect-\$g.log 2>&1 &
done; wait
PYTHONPATH=. python -m evaluation.traffic_elements score --clips $RUN/clips.csv --anchors $RUN/anchors.json \
  --detections $RUN/detections --out $RUN/results_traffic --sheets 0
"
srun --environment=vipe_env bash -c "
export HF_HOME=$R/speed30/hf TORCH_HOME=$R/speed30/hf/torch
mkdir -p $RUN/vipe $RUN/vipe_logs && cd $RUN
for g in 0 1 2 3; do
  (tail -n +2 videos.csv | awk -F, -v g=\$g 'NR % 4 == g' | while IFS=, read -r id video; do
     ls vipe/\$id/pose/*.npz >/dev/null 2>&1 && continue
     CUDA_VISIBLE_DEVICES=\$g vipe infer \"\$video\" -o vipe/\$id > vipe_logs/\$id.log 2>&1 || echo \"failed \$id\"
   done) &
done; wait
cd $R/CROWD-anything && PYTHONPATH=. python -m evaluation.vehicle_speed measure --videos $RUN/videos.csv \
  --vipe $RUN/vipe --out $RUN/tracks.csv
"
cd $R/CROWD-anything && PYTHONPATH=. python3.11 -m evaluation.vehicle_speed score --tracks $RUN/tracks.csv \
  --scenes $RUN/scenes.csv --out $RUN/results_speed
