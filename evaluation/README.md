# Evaluation of generated driving videos

Evaluation task for the world-model paper (CVPR, 16 Nov 2026; first part on NATIX). Valentin's four parts:

1. **Reality-signal adherence**: how well the video follows the street view, OSM topology and aerial view it is given.
2. **Long-video reproduction**: from the first frame, the reality signals and the ego trajectory, how exactly each
   baseline reproduces the ground-truth video, for several street-view sparsity levels and lengths (5 s to 1 min+).
3. **Control adherence**: ego control and text caption (mostly done by the team).
4. **Global driving-video quality**: our own WorldLens on our data and reality signals.

## What is here

`evaluation.paired` (part 2): compares generated videos with their ground truth frame by frame. Both are read on
the same 24 fps grid from t=0 as the tokenizer (`tokenization.frames`); the ground truth is scaled and centre-cropped
to the generated size. PSNR, SSIM and LPIPS (AlexNet) are computed per frame and averaged over horizons.

```bash
uv sync --extra eval
python -m evaluation.paired --generated gen/ --reference gt/ --out results/model_a --horizons 5,10,30,60
```

Pairs can also come from a CSV (`--pairs`, columns `clip_id,generated,reference`). Videos are read only up to the
longest horizon, so the `all` row covers at most that length. `per_frame.csv` shows how the error grows over time.

## WorldLens (CVPR 2026, github.com/worldbench/WorldLens) and what applies to us

WorldLens has 24 dimensions in 5 aspects. Its toolkit is built for nuScenes (6 cameras, 3D boxes, maps) and an old
stack (Python 3.9, CUDA 11.8, mmcv 1.6), so we reimplement the metrics we need rather than run it as is.
Assuming NATIX is a single front camera with no 3D labels:

| WorldLens dimension | Model / metric | Needs | For us |
|---|---|---|---|
| G.3 Subject consistency | DINO ViT-B/16 | generated + GT | yes |
| G.4 Depth discrepancy | Depth Anything V2 + DINOv2 | generated | yes |
| G.5 Temporal consistency | CLIP | generated + GT | yes |
| G.6 Semantic consistency | SegFormer | generated | yes |
| G.7 Perceptual discrepancy | FVD (I3D) | generated + GT sets | yes |
| G.1, G.2 Subject fidelity / coherence | classifiers, ReID | object boxes, tracks | with a detector/tracker |
| G.8 Cross-view consistency | LoFTR | multi-camera | no (one camera) |
| R.1-R.4 Reconstruction | 4D Gaussians; PSNR/SSIM/LPIPS, AbsRel, MUSIQ, FVD | generated (+ GT) | possible, heavy |
| A.1 Displacement error | UniAD/VAD planner | generated + GT | overlaps Lan's planning work |
| A.2-A.4 Open/closed loop | NAVSIM, DriveArena | simulator | no |
| D.1-D.4 Downstream perception | BEVFusion, SparseOcc | 3D labels, maps | no labels |
| H.1-H.4 Human preference | human raters / WorldLens-Agent | generated | optional |

Next steps, once Valentin confirms the data, models and baselines: FVD and the G.3-G.7 metrics as further modules,
then part 1 (for example segmentation or depth of generated frames against OSM / aerial layout).
