"""
Traffic elements given by anchors (evaluation part 3): does a generated video show the traffic signs and lights of the
real scene that the model saw in its anchor images, and in the same place?

Detection: an open-vocabulary detector (Grounding DINO, "traffic sign. traffic light.") on every `stride`-th frame of
each video of a clip and on its anchor images, on the whole image and on --tiles x --tiles overlapping tiles (distant
signs are a few pixels wide; the detector sees each tile at full input size), merged by class; and a DINOv2 feature
per detected patch for appearance.
Tracks: detections of one video linked over frames (same class, overlapping or nearby boxes, similar appearance).

Elements: the tracks of the real video (`reference`). An element is *anchored* when an anchor image of the clip shows
it (same class, patch features similar above --anchor-similarity): the model was given it as input. The other
elements are the control: the model could only produce them from the first frame (i2v), the map or by chance.
To choose --anchor-similarity, per_element.csv also has each element's best similarity to the anchors of another clip
(`other_anchor_similarity`, the next clip by rotation): what an element that is not in the anchors scores.

Scores per side (videos of the same clips; the camera follows the real trajectory, so an element should appear at
the same image position at the same time): for each element, the generated track of the same class whose boxes'
centres are closest on their common frames counts as the element when that mean distance is <= --max-centre-distance
of the image diagonal (IoU alone is near 0 for small, distant signs even when they are in the right place). Only
frames from --start-frame on count, in every video: an i2v generation starts from the real first frame.
  presence    share of elements found
  centre_distance  mean distance of the box centres (image diagonals), over found elements
  iou         mean box IoU on common frames, over found elements
  similarity  cosine of the patch features of element and generated track, over found elements
  extra       generated tracks per video that match no element (signs the model invents)
reported for anchored and non-anchored elements separately (summary.csv), per element in per_element.csv.

Inputs: clips.csv (clip_id,side,video; `reference` = real video) and anchors.json ({clip_id: [image paths]}).
  python -m evaluation.traffic_elements detect --clips clips.csv --anchors anchors.json \
      --detector models/grounding-dino-base --features models/dinov2-base --out detections/
  python -m evaluation.traffic_elements score --clips clips.csv --detections detections/ --out results/ [--sheets 6]
detect needs torch, transformers and OpenCV with video support; score needs numpy (and OpenCV for --sheets). This
module imports nothing else from `evaluation`, so it runs in a detector's environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)
REFERENCE = "reference"
ANCHORS = "anchors"  # pseudo-side of the detections in the anchor images
PROMPT = "traffic sign. traffic light."
LABELS = ("traffic sign", "traffic light")


def label_of(text: str) -> Optional[str]:
    """The class of a detector label ("traffic sign", "sign", "traffic light", "light"), or None."""
    text = text.lower()
    return "traffic light" if "light" in text else "traffic sign" if "sign" in text else None


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two boxes (x0, y0, x1, y1)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Distance of two boxes' centres, in units of the image diagonal (boxes normalized to [0, 1])."""
    dx = (a[0] + a[2] - b[0] - b[2]) / 2
    dy = (a[1] + a[3] - b[1] - b[3]) / 2
    return float(np.hypot(dx, dy) / np.sqrt(2))


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


@dataclass
class Track:
    label: str
    boxes: dict[int, list[float]] = field(default_factory=dict)  # frame -> normalized box
    features: list[np.ndarray] = field(default_factory=list)

    @property
    def feature(self) -> np.ndarray:
        return _unit(np.mean([_unit(f) for f in self.features], axis=0))


def link_tracks(frames: dict[int, list[dict]], max_gap: int = 2, min_link_iou: float = 0.05,
                max_link_distance: float = 0.08, min_link_similarity: float = 0.5) -> list[Track]:
    """Tracks from per-frame detections ({frame: [{box, label, feature}]}, frames in sampling order): each detection
    joins the most similar live track of its class whose last box overlaps it or is near it, else starts a track.
    A track ends after `max_gap` sampled frames without a detection."""
    order = sorted(frames)
    tracks: list[Track] = []
    last_seen: list[int] = []
    for step, frame in enumerate(order):
        taken: set[int] = set()
        for det in sorted(frames[frame], key=lambda d: -d.get("score", 0)):
            feature = np.asarray(det["feature"], dtype=np.float32)
            best, best_sim = None, min_link_similarity
            for t, track in enumerate(tracks):
                if t in taken or track.label != det["label"] or step - last_seen[t] > max_gap:
                    continue
                box = track.boxes[max(track.boxes)]
                if iou(box, det["box"]) < min_link_iou and centre_distance(box, det["box"]) > max_link_distance:
                    continue
                sim = float(_unit(track.features[-1]) @ _unit(feature))
                if sim >= best_sim:
                    best, best_sim = t, sim
            if best is None:
                tracks.append(Track(det["label"]))
                last_seen.append(step)
                best = len(tracks) - 1
            tracks[best].boxes[frame] = list(det["box"])
            tracks[best].features.append(feature)
            last_seen[best] = step
            taken.add(best)
    return tracks


def anchor_similarity(track: Track, anchor_detections: list[dict]) -> Optional[float]:
    """The best patch similarity of a track to an anchor detection of its class (None without one)."""
    sims = [float(track.feature @ _unit(np.asarray(d["feature"], dtype=np.float32)))
            for d in anchor_detections if d["label"] == track.label]
    return max(sims, default=None)


def anchored(track: Track, anchor_detections: list[dict], min_similarity: float) -> bool:
    sim = anchor_similarity(track, anchor_detections)
    return sim is not None and sim >= min_similarity


def match(element: Track, candidates: list[Track], max_centre_distance: float) -> Optional[dict]:
    """The candidate track that shows the element: same class, common frames, box centres closest on average (and
    within max_centre_distance)."""
    best = None
    for i, track in enumerate(candidates):
        common = sorted(set(element.boxes) & set(track.boxes))
        if track.label != element.label or not common:
            continue
        distance = float(np.mean([centre_distance(element.boxes[f], track.boxes[f]) for f in common]))
        if distance > max_centre_distance or (best is not None and distance >= best["centre_distance"]):
            continue
        best = {"index": i, "iou": float(np.mean([iou(element.boxes[f], track.boxes[f]) for f in common])),
                "centre_distance": distance, "common_frames": len(common),
                "similarity": float(element.feature @ track.feature)}
    return best


@dataclass
class Settings:
    min_track_frames: int = 2  # tracks seen in fewer sampled frames are dropped (flicker)
    anchor_similarity: float = 0.6
    max_centre_distance: float = 0.1
    start_frame: int = 24  # frames before it are not scored (1 s at 24 fps: i2v starts from the real frame)


def scored_tracks(frames: dict[int, list[dict]], settings: Settings) -> list[Track]:
    kept = {f: dets for f, dets in frames.items() if f >= settings.start_frame}
    return [t for t in link_tracks(kept) if len(t.boxes) >= settings.min_track_frames]


def anchor_detections(detections: dict[str, dict[int, list[dict]]]) -> list[dict]:
    return [d for frame in detections.get(ANCHORS, {}).values() for d in frame]


def score_clip(clip_id: str, detections: dict[str, dict[int, list[dict]]], settings: Settings,
               other_anchors: Sequence[dict] = ()) -> tuple[list[dict], list[dict]]:
    """(per-element rows, per-side rows with the count of generated tracks matching no element) for one clip;
    `other_anchors`: anchor detections of another clip, for other_anchor_similarity."""
    tracks = {side: scored_tracks(frames, settings) for side, frames in detections.items() if side != ANCHORS}
    elements = tracks.get(REFERENCE, [])
    anchor_dets = anchor_detections(detections)
    given = [anchor_similarity(e, anchor_dets) for e in elements]
    is_anchored = [sim is not None and sim >= settings.anchor_similarity for sim in given]
    other = [anchor_similarity(e, list(other_anchors)) for e in elements]
    rows, extra = [], []
    for side, side_tracks in tracks.items():
        if side == REFERENCE:
            continue
        used = set()
        for e, element in enumerate(elements):
            found = match(element, side_tracks, settings.max_centre_distance)
            if found:
                used.add(found["index"])
            rows.append({"clip_id": clip_id, "side": side, "element": e, "label": element.label,
                         "frames": len(element.boxes),
                         "anchored": is_anchored[e],
                         "anchor_similarity": given[e], "other_anchor_similarity": other[e],
                         "found": found is not None, "centre_distance": found and found["centre_distance"],
                         "iou": found and found["iou"], "similarity": found and found["similarity"]})
        extra.append({"clip_id": clip_id, "side": side, "tracks": len(side_tracks),
                      "extra": len(side_tracks) - len(used)})
    return rows, extra


def summarize(rows: list[dict], extra: list[dict]) -> list[dict]:
    out = []
    for side in dict.fromkeys(r["side"] for r in rows + extra):
        side_extra = [x["extra"] for x in extra if x["side"] == side]
        for group, is_anchored in (("anchored", True), ("not_anchored", False)):
            group_rows = [r for r in rows if r["side"] == side and r["anchored"] == is_anchored]
            found = [r for r in group_rows if r["found"]]
            out.append({
                "side": side, "elements": group, "n": len(group_rows),
                "presence": len(found) / len(group_rows) if group_rows else None,
                "centre_distance": float(np.mean([r["centre_distance"] for r in found])) if found else None,
                "iou": float(np.mean([r["iou"] for r in found])) if found else None,
                "similarity": float(np.mean([r["similarity"] for r in found])) if found else None,
                "extra_per_video": float(np.mean(side_extra)) if side_extra else None,
            })
    return out


# --- detection (torch, transformers, OpenCV) ---------------------------------------------------------------------

class Detector:
    """Grounding DINO boxes for PROMPT and a DINOv2 feature per box (its patch with a margin)."""

    def __init__(self, detector: str, features: str, device: str = "cuda:0", box_threshold: float = 0.3,
                 text_threshold: float = 0.25, min_side_px: int = 6, tiles: int = 2, overlap: float = 0.2):
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(detector)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(device).eval()
        self.feature_processor = AutoImageProcessor.from_pretrained(features)
        self.feature_model = AutoModel.from_pretrained(features).to(device).eval()
        self.box_threshold, self.text_threshold, self.min_side_px = box_threshold, text_threshold, min_side_px
        self.tiles, self.overlap = tiles, overlap

    def _boxes(self, images: list) -> list[list[tuple[list[float], float, str]]]:
        """(pixel box, score, label) per image: the whole image and its tiles, merged."""
        windows = [tile_windows(image.size[0], image.size[1], self.tiles, self.overlap) for image in images]
        crops = [image.crop(w) for image, image_windows in zip(images, windows) for w in image_windows]
        results = []
        for i in range(0, len(crops), 16):
            chunk = crops[i:i + 16]
            with self.torch.no_grad():
                inputs = self.processor(images=chunk, text=[PROMPT] * len(chunk), return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                results += self.processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"], threshold=self.box_threshold, text_threshold=self.text_threshold,
                    target_sizes=[c.size[::-1] for c in chunk])
        out, k = [], 0
        for image_windows in windows:
            found = []
            for x, y, _, _ in image_windows:
                result = results[k]
                k += 1
                texts = result.get("text_labels", result.get("labels"))
                for box, score, text in zip(result["boxes"].tolist(), result["scores"].tolist(), texts):
                    label = label_of(str(text))
                    if label is not None:
                        found.append(([box[0] + x, box[1] + y, box[2] + x, box[3] + y], score, label))
            out.append(merge_boxes(found))
        return out

    def __call__(self, images: list) -> list[list[dict]]:
        """Detections per PIL image: {box (normalized x0, y0, x1, y1), label, score, feature}."""
        torch = self.torch
        out: list[list[dict]] = []
        for image, boxes in zip(images, self._boxes(images)):
            width, height = image.size
            dets, crops = [], []
            for (x0, y0, x1, y1), score, label in boxes:
                if min(x1 - x0, y1 - y0) < self.min_side_px:
                    continue
                mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
                crops.append(image.crop((max(0, x0 - mx), max(0, y0 - my), min(width, x1 + mx), min(height, y1 + my))))
                dets.append({"box": [x0 / width, y0 / height, x1 / width, y1 / height], "label": label,
                             "score": round(score, 3)})
            if crops:
                with torch.no_grad():
                    batch = self.feature_processor(images=crops, return_tensors="pt").to(self.device)
                    features = self.feature_model(**batch).pooler_output.float().cpu().numpy()
                for det, feature in zip(dets, features):
                    det["feature"] = feature.astype(np.float16).tolist()
            out.append(dets)
        return out


def tile_windows(width: int, height: int, tiles: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """The whole image and a tiles x tiles grid of windows overlapping by `overlap` of a window (pixel boxes)."""
    windows = [(0, 0, width, height)]
    if tiles > 1:
        tw, th = width / (tiles - (tiles - 1) * overlap), height / (tiles - (tiles - 1) * overlap)
        for row in range(tiles):
            for col in range(tiles):
                x, y = round(col * tw * (1 - overlap)), round(row * th * (1 - overlap))
                windows.append((x, y, min(width, round(x + tw)), min(height, round(y + th))))
    return windows


def merge_boxes(found: list[tuple[list[float], float, str]],
                max_cover: float = 0.6) -> list[tuple[list[float], float, str]]:
    """Greedy suppression by score within a class: a box is dropped when it covers, or is covered by, more than
    `max_cover` of a kept box's or its own area (the same sign found whole and cut by a tile edge)."""
    kept: list[tuple[list[float], float, str]] = []
    for box, score, label in sorted(found, key=lambda f: -f[1]):
        area = (box[2] - box[0]) * (box[3] - box[1])
        duplicate = False
        for other, _, other_label in kept:
            if other_label != label:
                continue
            w = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
            h = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
            smaller = min(area, (other[2] - other[0]) * (other[3] - other[1]))
            if smaller > 0 and w * h / smaller > max_cover:
                duplicate = True
                break
        if not duplicate:
            kept.append((box, score, label))
    return kept


def read_frames(path: str, stride: int) -> dict[int, object]:
    """Every `stride`-th frame of a video (and its last one) as PIL images."""
    import cv2
    from PIL import Image

    capture, frames, index, last = cv2.VideoCapture(path), {}, 0, None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        last = (index, frame)
        if index % stride == 0:
            frames[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if last is not None and last[0] not in frames:
        frames[last[0]] = Image.fromarray(cv2.cvtColor(last[1], cv2.COLOR_BGR2RGB))
    if not frames:
        raise ValueError(f"no frames read from {path}")
    return frames


def detect_clip(sides: dict[str, str], anchors: list[str], detector: Detector, stride: int,
                batch_size: int = 8) -> dict[str, dict[int, list[dict]]]:
    from PIL import Image

    out: dict[str, dict[int, list[dict]]] = {}
    sources = [(side, read_frames(path, stride)) for side, path in sides.items()]
    if anchors:
        sources.append((ANCHORS, {i: Image.open(p).convert("RGB") for i, p in enumerate(anchors)}))
    for side, frames in sources:
        keys = sorted(frames)
        out[side] = {}
        for i in range(0, len(keys), batch_size):
            chunk = keys[i:i + batch_size]
            for key, dets in zip(chunk, detector([frames[k] for k in chunk])):
                out[side][key] = dets
    return out


def read_clips(path: Path) -> dict[str, dict[str, str]]:
    clips: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            clips.setdefault(row["clip_id"], {})[row["side"]] = row["video"]
    return clips


def detections_path(out_dir: Path, clip_id: str) -> Path:
    return out_dir / (clip_id.replace("/", "_").replace("#", "_") + ".json")


def load_detections(path: Path) -> dict[str, dict[int, list[dict]]]:
    return {side: {int(k): v for k, v in frames.items()} for side, frames in json.loads(path.read_text()).items()}


# --- contact sheets (OpenCV) --------------------------------------------------------------------------------------

def write_sheet(clip_id: str, sides: dict[str, str], anchors: list[str], detections: dict, settings: Settings,
                path: Path, columns: int = 4, width: int = 384) -> None:
    """One row per video (4 frames, boxes of its tracks; green = anchored element in the reference) and a row of
    anchor images with their boxes, to check detections and matches by eye."""
    import cv2

    anchor_dets = anchor_detections(detections)

    def draw(image: np.ndarray, boxes: list[tuple[list[float], tuple[int, int, int]]]) -> np.ndarray:
        image = cv2.resize(image, (width, int(image.shape[0] * width / image.shape[1])))
        h, w = image.shape[:2]
        for box, colour in boxes:
            cv2.rectangle(image, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)), colour, 2)
        return image

    rows = []
    for side, video in sides.items():
        tracks = scored_tracks(detections.get(side, {}), settings)
        frames = read_frames(video, 1)
        keys = sorted(k for k in detections.get(side, {}) or frames if k >= settings.start_frame)
        picks = [keys[round(i * (len(keys) - 1) / max(1, columns - 1))] for i in range(columns)]
        tiles = []
        for key in picks:
            image = cv2.cvtColor(np.asarray(frames[key]), cv2.COLOR_RGB2BGR)
            boxes = [(t.boxes[key], (0, 200, 0) if side == REFERENCE and anchored(
                t, anchor_dets, settings.anchor_similarity) else (0, 140, 255))
                for t in tracks if key in t.boxes]
            tile = draw(image, boxes)
            cv2.putText(tile, f"{side} f{key}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    anchor_tiles = []
    for i, image_path in enumerate(anchors[:columns]):
        image = cv2.imread(image_path)
        boxes = [(d["box"], (255, 0, 255)) for d in detections.get(ANCHORS, {}).get(i, [])]
        anchor_tiles.append(draw(image, boxes))
    if anchor_tiles:
        blank = np.zeros_like(anchor_tiles[0])
        rows.append(np.hstack(anchor_tiles + [blank] * (columns - len(anchor_tiles))))
    width_max = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width_max - r.shape[1]), (0, 0))) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


def _write_csv(rows: list[dict], path: Path) -> None:
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    detect_cmd = sub.add_parser("detect", help="detect traffic signs and lights in every video and anchor image")
    detect_cmd.add_argument("--clips", type=Path, required=True, help="CSV clip_id,side,video")
    detect_cmd.add_argument("--anchors", type=Path, required=True, help="JSON {clip_id: [anchor image paths]}")
    detect_cmd.add_argument("--detector", required=True, help="Grounding DINO model (path or hub id)")
    detect_cmd.add_argument("--features", required=True, help="DINOv2 model (path or hub id)")
    detect_cmd.add_argument("--out", type=Path, required=True, help="folder of per-clip detection JSONs")
    detect_cmd.add_argument("--stride", type=int, default=4, help="detect on every n-th frame")
    detect_cmd.add_argument("--tiles", type=int, default=2, help="also detect on tiles x tiles tiles (1: off)")
    detect_cmd.add_argument("--box-threshold", type=float, default=0.3)
    detect_cmd.add_argument("--device", default="cuda:0")
    score_cmd = sub.add_parser("score", help="score generated videos against the reference's elements")
    score_cmd.add_argument("--clips", type=Path, required=True)
    score_cmd.add_argument("--anchors", type=Path, help="anchors JSON, for the contact sheets")
    score_cmd.add_argument("--detections", type=Path, required=True)
    score_cmd.add_argument("--out", type=Path, required=True)
    score_cmd.add_argument("--anchor-similarity", type=float, default=Settings.anchor_similarity)
    score_cmd.add_argument("--max-centre-distance", type=float, default=Settings.max_centre_distance)
    score_cmd.add_argument("--start-frame", type=int, default=Settings.start_frame, help="first frame scored")
    score_cmd.add_argument("--sheets", type=int, default=0, help="contact sheets for the first n clips")
    args = parser.parse_args(argv)

    clips = read_clips(args.clips)
    anchors = json.loads(args.anchors.read_text()) if args.anchors else {}
    if args.command == "detect":
        detector = Detector(args.detector, args.features, args.device, box_threshold=args.box_threshold,
                            tiles=args.tiles)
        args.out.mkdir(parents=True, exist_ok=True)
        for n, (clip_id, sides) in enumerate(clips.items(), 1):
            path = detections_path(args.out, clip_id)
            done = json.loads(path.read_text()) if path.exists() else {}
            missing = {side: video for side, video in sides.items() if side not in done}
            if not missing:
                continue
            # sides added since the last run (generated videos arriving in batches) are detected on their own
            new = detect_clip(missing, [] if done else anchors.get(clip_id, []), detector, args.stride)
            path.write_text(json.dumps({**done, **new}))
            logger.info("%d/%d %s", n, len(clips), clip_id)
        return 0

    settings = Settings(anchor_similarity=args.anchor_similarity, max_centre_distance=args.max_centre_distance,
                        start_frame=args.start_frame)
    ids = [c for c in clips if detections_path(args.detections, c).exists()]
    for clip_id in clips.keys() - set(ids):
        logger.warning("no detections for %s", clip_id)
    rows, extra = [], []
    for n, clip_id in enumerate(ids):
        sides, detections = clips[clip_id], load_detections(detections_path(args.detections, clip_id))
        other = anchor_detections(load_detections(detections_path(args.detections, ids[(n + 1) % len(ids)])))
        clip_rows, clip_extra = score_clip(clip_id, detections, settings, other if len(ids) > 1 else ())
        rows += clip_rows
        extra += clip_extra
        if n < args.sheets:
            write_sheet(clip_id, sides, anchors.get(clip_id, []), detections, settings,
                        args.out / "sheets" / (detections_path(args.detections, clip_id).stem + ".jpg"))
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.out / "per_element.csv")
    summary = summarize(rows, extra)
    _write_csv(summary, args.out / "summary.csv")
    for line in summary:
        logger.info("%s", json.dumps(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
