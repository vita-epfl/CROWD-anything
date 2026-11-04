#!/bin/bash
# Download CROWD videos from the CROWD file server into Capstor store, on Clariden's xfer partition.
#   LIMIT=40 PAR=8 sbatch fetch_crowd_videos.sh   # test: the first 40 ids of $IDS
#   CHAIN=20 PAR=8 sbatch fetch_crowd_videos.sh   # full run: each job queues its successor (at most CHAIN jobs)
# Each video: its folder on the server (tue4, tue5 or tue3) is found with a HEAD request (only a 200 counts: the
# server's 404 page also has a Content-Length), it is downloaded to <id>.mp4.part (resumed if a part exists) and
# renamed to <id>.mp4 only when its size equals the server's. Finished videos are skipped, so a later job continues
# where one stopped. Login: ~/.netrc-crowd (mode 600). Run from $DATA (sbatch copies the script, so the chain
# resubmits $DATA/fetch_crowd_videos.sh).
#SBATCH --job-name=crowd-fetch
#SBATCH --partition=xfer
#SBATCH --account=ab036
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/iopsstor/scratch/cscs/%u/crowd-fetch/logs/%x-%j.out
set -euo pipefail
umask 0027  # group ab036 can read

DATA=/capstor/store/cscs/swissai/ab036/datasets/crowd
export DEST=$DATA/videos
IDS=${IDS:-$DATA/video_ids_car.txt}
LOGDIR=/iopsstor/scratch/cscs/$USER/crowd-fetch
PAR=${PAR:-8}
mkdir -p "$DEST" "$LOGDIR/logs"

fetch_one() {
    local id=$1 f="$DEST/$1.mp4" url="" size="" alias
    [ -s "$f" ] && return 0
    for alias in tue4 tue5 tue3; do
        size=$(curl -sS -I --netrc-file ~/.netrc-crowd --max-time 60 "https://files.mobility-squad.com/v/$alias/files/$id.mp4" 2>/dev/null \
            | tr -d '\r' | awk 'NR==1{ok=($2=="200")} ok && tolower($1)=="content-length:"{print $2}') || true
        if [ -n "$size" ]; then url="https://files.mobility-squad.com/v/$alias/files/$id.mp4"; break; fi
    done
    if [ -z "$url" ]; then echo "missing $id"; return 0; fi
    local rc=0
    curl -sS -f --netrc-file ~/.netrc-crowd --retry 5 --retry-delay 10 --connect-timeout 60 -C - -o "$f.part" "$url" 2>/dev/null || rc=$?
    if [ "$rc" != 0 ]; then echo "failed $id $alias (curl exit $rc)"; return 0; fi
    if [ "$(stat -c %s "$f.part")" = "$size" ]; then mv "$f.part" "$f"; echo "ok $id $alias $size"
    else echo "size_mismatch $id $alias $(stat -c %s "$f.part") $size"; rm -f "$f.part"; fi
}
export -f fetch_one

# Chain: queue the next job now (it waits for this one), so a job stopped by the 1-day limit still continues.
# A job that ends normally cancels it when nothing is left or when it made no progress (only failures left).
CHAIN=${CHAIN:-0}
next=""
if [ "$CHAIN" -gt 1 ]; then
    next=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,CHAIN=$((CHAIN - 1)),PAR="$PAR" \
        "$DATA/fetch_crowd_videos.sh")
    echo "queued next job $next (chain $((CHAIN - 1)) left)"
fi

todo=$(mktemp)
if [ -n "${LIMIT:-}" ]; then head -n "$LIMIT" "$IDS"; else cat "$IDS"; fi > "$todo"
echo "$(date -Is) start: $(wc -l < "$todo") ids, $PAR in parallel, $(find "$DEST" -maxdepth 1 -name '*.mp4' | wc -l) already in $DEST"
run=$(mktemp)
start=$(date +%s)
xargs -a "$todo" -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {} | tee -a "$LOGDIR/fetch.log" > "$run"
secs=$(( $(date +%s) - start ))
awk -v s="$secs" '$1=="ok"{n++; b+=$4} END{printf "this run: %d videos, %.1f GB in %d s, %.0f MB/s\n", n, b/1e9, s, (s>0 ? b/1e6/s : 0)}' "$run"
grep -v '^ok' "$run" || true
done_now=$(grep -c '^ok' "$run" || true)
left=$(( $(wc -l < "$IDS") - $(find "$DEST" -maxdepth 1 -name '*.mp4' | wc -l) ))
if [ -n "$next" ] && { [ "$left" -le 0 ] || [ "$done_now" -eq 0 ]; }; then
    scancel "$next" && echo "cancelled next job $next: $left left, $done_now downloaded in this run"
fi
rm -f "$todo" "$run"
echo "$(date -Is) end: $(find "$DEST" -maxdepth 1 -name '*.mp4' | wc -l) of $(wc -l < "$IDS") videos in $DEST (all runs: $LOGDIR/fetch.log)"
