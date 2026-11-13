#!/bin/bash
#SBATCH --job-name=read-signs
#SBATCH --account=ab036
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/iopsstor/scratch/cscs/malam/demo-signs/%x-%j.out
# Text in demo drives (demo/read_signs.py), one GPU per drive (sbatch --export=ALL,DRIVES="<id>:<start>:<end> ...").
# A drive not yet in datasets/crowd/videos is fetched from the CROWD file server first (~/.netrc-crowd). EasyOCR and
# the dependencies the Qwen venv lacks go to $OUT/pylib (once); its models to $OUT/easyocr.
set -euo pipefail
R=/iopsstor/scratch/cscs/malam/rgwm-eval
OUT=/iopsstor/scratch/cscs/malam/demo-signs
VIDEOS=/capstor/store/cscs/swissai/ab036/datasets/crowd/videos
mkdir -p $OUT/videos $OUT/easyocr
for drive in $DRIVES; do
  id=${drive%%:*}
  [ -s $VIDEOS/$id.mp4 ] || [ -s $OUT/videos/$id.mp4 ] && continue
  for alias in tue4 tue5 tue3 others; do
    url=https://files.mobility-squad.com/v/$alias/files/$id.mp4
    if curl -sS -f -I --netrc-file ~/.netrc-crowd --max-time 60 "$url" >/dev/null 2>&1; then
      curl -sS -f --netrc-file ~/.netrc-crowd --retry 5 -o $OUT/videos/$id.mp4.part "$url" && mv $OUT/videos/$id.mp4.part $OUT/videos/$id.mp4
      break
    fi
  done
  [ -s $OUT/videos/$id.mp4 ] || echo "not on the file server: $id"
done
srun --environment=wm_data_manifest_env bash -c "set -e
source $R/preprocessin-wm/caption/qwen/env.sh
export PYTHONPATH=$OUT/pylib:\$PYTHONPATH
python -c 'import easyocr' 2>/dev/null || {
  python -m pip --version >/dev/null 2>&1 && PIP='python -m pip' || PIP='uv pip --python python'
  # --no-deps: the venv's torch, numpy, opencv and pillow must stay the ones used
  \$PIP install -q --target $OUT/pylib --no-deps easyocr python-bidi pyclipper shapely scikit-image scipy imageio \\
    tifffile lazy_loader networkx PyYAML ninja
}
# download the OCR models once: parallel first runs collide on easyocr's temp.zip
python -c \"import easyocr; easyocr.Reader(['fr', 'en'], gpu=False, model_storage_directory='$OUT/easyocr')\"
g=0
for drive in $DRIVES; do
  IFS=: read -r id start end <<< \"\$drive\"
  [ -s $OUT/\${id}_text.csv ] && continue
  video=$VIDEOS/\$id.mp4; [ -s \$video ] || video=$OUT/videos/\$id.mp4
  [ -s \$video ] || continue
  CUDA_VISIBLE_DEVICES=\$g python $R/CROWD-anything/demo/read_signs.py --video \$video --start \$start --end \$end \
    --models $OUT/easyocr --out $OUT/\${id}_text.csv > $OUT/\$id.log 2>&1 &
  g=\$((g + 1))
done; wait
"
