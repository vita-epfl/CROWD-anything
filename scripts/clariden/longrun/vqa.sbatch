#!/bin/bash
#SBATCH --job-name=long-vqa
#SBATCH --account=ab036
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/malam/rgwm-eval/logs/%x-%j.out
# Long-horizon evaluation, Qwen-Drive answers (sbatch --export=RUN=<dir>): GPU g answers prompt-following shard g
# (512 new tokens), then topology shard g (768: shorter cuts off replies). The runner resumes, so resubmit until done.
set -euo pipefail
R=/iopsstor/scratch/cscs/malam/rgwm-eval
srun --environment=wm_data_manifest_env bash -c "cd $R/preprocessin-wm/caption && for g in 0 1 2 3; do
  (bash qwen/run_video_batch.sh --videos $RUN/batch_pf/videos-\$g.json --prompts $RUN/batch_pf/prompts-\$g.json \
     --output $RUN/batch_pf/answers-\$g.json --device cuda:\$g --num-frames 8 --max-new-tokens 512 --resume \
     > $RUN/batch_pf/log-\$g.txt 2>&1
   bash qwen/run_video_batch.sh --videos $RUN/batch_topo/videos-\$g.json --prompts $RUN/batch_topo/prompts-\$g.json \
     --output $RUN/batch_topo/answers-\$g.json --device cuda:\$g --num-frames 8 --max-new-tokens 768 --resume \
     > $RUN/batch_topo/log-\$g.txt 2>&1) &
done; wait"
