#!/bin/bash
# Watch a long-horizon run: resubmit the Qwen-Drive stage until every answer is in, submit the vision stage when a
# debug slot is free, and score at the end (in the container: numpy). Runs on a laptop with SSH to Clariden.
# Usage: bash watch.sh <run dir on Clariden>
X=(-o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=30 -o "ProxyCommand=ssh -o BatchMode=yes -W %h:%p malam@ela.cscs.ch")
R=/iopsstor/scratch/cscs/malam/rgwm-eval
JOBS=$R/CROWD-anything/scripts/clariden/longrun
RUN=$1
c() { ssh "${X[@]}" malam@clariden.cscs.ch "$@"; }
for i in $(seq 1 600); do
  out=$(c "q=\$(squeue -u malam -h -o %j | grep '^long-' | tr '\n' ,); left=\$(python3.11 -c \"
import json, glob, os
n = 0
for p in glob.glob('$RUN/batch_*/prompts-*.json'):
    want = sum(len(x['video_ids']) for x in json.load(open(p))['prompts'])
    a = p.replace('prompts-', 'answers-')
    n += want - (len(json.load(open(a))['results']) if os.path.exists(a) else 0)
print(n if glob.glob('$RUN/batch_pf/prompts-*.json') else -1)\"); speed=\$([ -s $RUN/results_speed/summary.csv ] && echo 1 || echo 0); echo \"\$left \$speed \$q\"" 2>/dev/null)
  read -r left speed queued <<< "$out"
  echo "$(date +%H:%M) vqa_left=$left speed_done=$speed queued=$queued"
  # -1: stage 1 has not exported the questions yet (nor cut the clips)
  if [ -z "$left" ] || [ "$left" = -1 ]; then sleep 120; continue; fi
  n=$(tr ',' '\n' <<< "$queued" | grep -c long-)
  if [ "$left" -gt 0 ] && [[ "$queued" != *long-vqa* ]] && [ "$n" -lt 2 ]; then
    c "cd $JOBS && sbatch --export=ALL,RUN=$RUN vqa.sbatch"; n=$((n + 1)); queued="$queued,long-vqa"
  fi
  if [ "$speed" = 0 ] && [[ "$queued" != *long-vision* ]] && [ "$n" -lt 2 ]; then
    c "cd $JOBS && sbatch --export=ALL,RUN=$RUN vision.sbatch"
  fi
  if [ "$left" = 0 ] && [ "$speed" = 1 ] && [ -z "$queued" ]; then
    c "cd $JOBS && srun -A ab036 -p debug -t 00:10:00 -N1 --environment=wm_data_manifest_env python $JOBS/score.py $RUN"
    exit 0
  fi
  sleep 180
done
