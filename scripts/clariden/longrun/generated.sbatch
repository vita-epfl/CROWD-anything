#!/bin/bash
#SBATCH --job-name=long-gen
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/malam/rgwm-eval/logs/%x-%j.out
# Long-horizon evaluation of generated videos that arrive in batches (sbatch --export=RUN=<dir>,REAL=<dir>,
# MANIFEST=<evaluation_manifest.csv>,WINDOWS=0): REAL is the finished run of the real clips (same windows); RUN
# reuses its clips, anchors, statements, detections, ViPE outputs and answers and only works on the generated
# videos listed in the manifest when the job starts (side = the manifest's model). Vision first (traffic elements,
# ViPE, speeds), then Qwen-Drive on the questions not answered yet, then scores. Submit again when more videos are
# in the manifest, or when a job ran out of time: each job does what is left.
set -euo pipefail
R=/iopsstor/scratch/cscs/malam/rgwm-eval
M=/iopsstor/scratch/cscs/malam/models
SEL=/capstor/store/cscs/swissai/ab036/tmp_shared/rgwm-pablo-anchor-covered-20261108
J=$R/CROWD-anything/scripts/clariden/longrun
mkdir -p $RUN
srun --environment=wm_data_manifest_env bash -c "set -e
cd $R/CROWD-anything
python - <<'PY'
import csv, os
rows = [r for r in csv.DictReader(open('$MANIFEST')) if os.path.exists(r['video'])]
with open('$RUN/generations.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['video_id', 'side', 'video'])
    w.writerows([r['video_id'], r['model'], r['video']] for r in rows)
print(len(rows), 'generated videos')
PY
PYTHONPATH=. python -m evaluation.horizons prepare --selection $SEL/60s \
  --inputs /capstor/store/cscs/swissai/ab036/pacuaviva/dwm-baselines/inputs/one_min_418.jsonl \
  --captions $R/horizons255/captions_topology_free.csv \
  --records /capstor/store/cscs/swissai/ab036/processed_datasets/natix/osm_bev/text_predicted \
  --windows $WINDOWS --generations $RUN/generations.csv --out $RUN
cmp $RUN/windows.csv $REAL/windows.csv  # same windows, same tags: the real run's outputs apply
mkdir -p $RUN/clips $RUN/vipe $RUN/batch_pf $RUN/batch_topo
[ -e $RUN/clips/reference ] || ln -s $REAL/clips/reference $RUN/clips/reference
[ -e $RUN/anchors ] || ln -s $REAL/anchors $RUN/anchors
[ -e $RUN/detections ] || cp -r $REAL/detections $RUN/detections
[ -e $RUN/statements.json ] || cp $REAL/statements.json $RUN/statements.json
for b in batch_pf batch_topo; do  # copies: rglob does not enter linked folders
  [ -e $RUN/\$b/real ] || (mkdir $RUN/\$b/real && cp $REAL/\$b/answers-*.json $RUN/\$b/real/)
done
for d in $REAL/vipe/*__reference; do [ -e $RUN/vipe/\$(basename \$d) ] || ln -s \$d $RUN/vipe/; done
PYTHONPATH=. python -m evaluation.horizons cut --run $RUN --natix-root /capstor/store/cscs/swissai/ab036/datasets/natix \
  --anchor-images $SEL/anchor_images.jsonl
"
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
cd $RUN
for g in 0 1 2 3; do
  (tail -n +2 videos.csv | awk -F, -v g=\$g 'NR % 4 == g' | while IFS=, read -r id video; do
     case \$id in *__reference) continue;; esac  # done (or failed) in the real run
     ls vipe/\$id/pose/*.npz >/dev/null 2>&1 && continue
     mkdir -p vipe_logs; CUDA_VISIBLE_DEVICES=\$g vipe infer \"\$video\" -o vipe/\$id > vipe_logs/\$id.log 2>&1 || echo \"failed \$id\"
   done) &
done; wait
cd $R/CROWD-anything && PYTHONPATH=. python -m evaluation.vehicle_speed measure --videos $RUN/videos.csv \
  --vipe $RUN/vipe --out $RUN/tracks.csv
"
cd $R/CROWD-anything && PYTHONPATH=. python3.11 -m evaluation.vehicle_speed score --tracks $RUN/tracks.csv \
  --scenes $RUN/scenes.csv --out $RUN/results_speed
B=b$SLURM_JOB_ID
srun --environment=wm_data_manifest_env bash -c "set -e
cd $R/CROWD-anything
PYTHONPATH=. python -m evaluation.prompt_following export --clips $RUN/clips.csv --statements $RUN/statements.json \
  --out $RUN/export_pf --frames 8
PYTHONPATH=. python -m evaluation.topology export --clips $RUN/clips.csv --facts $RUN/facts.json --out $RUN/export_topo \
  --frames 8
python $J/pending.py $RUN/export_pf $RUN/batch_pf $RUN/batch_pf/$B 4
python $J/pending.py $RUN/export_topo $RUN/batch_topo $RUN/batch_topo/$B 4
cd $R/preprocessin-wm/caption && for g in 0 1 2 3; do
  (if grep -qs video_ids $RUN/batch_pf/$B/prompts-\$g.json; then bash qwen/run_video_batch.sh --videos $RUN/batch_pf/$B/videos-\$g.json \
     --prompts $RUN/batch_pf/$B/prompts-\$g.json --output $RUN/batch_pf/$B/answers-\$g.json --device cuda:\$g \
     --num-frames 8 --max-new-tokens 512 > $RUN/batch_pf/$B/log-\$g.txt 2>&1; fi
   if grep -qs video_ids $RUN/batch_topo/$B/prompts-\$g.json; then bash qwen/run_video_batch.sh --videos $RUN/batch_topo/$B/videos-\$g.json \
     --prompts $RUN/batch_topo/$B/prompts-\$g.json --output $RUN/batch_topo/$B/answers-\$g.json --device cuda:\$g \
     --num-frames 8 --max-new-tokens 768 > $RUN/batch_topo/$B/log-\$g.txt 2>&1; fi) &
done; wait
python $J/score.py $RUN
"
