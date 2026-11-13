"""
Text in the frames of a driving video (street-name plates, shop and place names), to find where the car is when
pinning its GPS by hand (gps_annotator.html). EasyOCR on one frame every `--every` seconds between --start and --end;
one row per text box. Runs on a GPU node (scripts/clariden/read_signs.sbatch), not on a laptop.

  python demo/read_signs.py --video Qxn7g5k6d20.mp4 --start 415 --end 6599 --out Qxn7g5k6d20_text.csv

Then `python demo/sign_hints.py` keeps the street and place names and looks them up on OSM.
"""

from __future__ import annotations

import argparse
import csv
import logging

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, default=0.0, help="first second")
    parser.add_argument("--end", type=float, default=0.0, help="last second (0: the end of the video)")
    parser.add_argument("--every", type=float, default=1.0, help="seconds between frames read")
    parser.add_argument("--langs", default="fr,en")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--models", help="EasyOCR model folder")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    import easyocr

    reader = easyocr.Reader(args.langs.split(","), gpu=True, model_storage_directory=args.models)
    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS)
    last = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    end = min(last, int(args.end * fps)) if args.end else last
    count = int((end / fps - args.start) / args.every) + 1
    wanted = {round((args.start + i * args.every) * fps) for i in range(count)}
    video.set(cv2.CAP_PROP_POS_FRAMES, min(wanted))
    frame_index, rows = min(wanted), 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "text", "confidence", "x", "y", "w", "h"])
        while frame_index <= end:
            if not video.grab():
                break
            if frame_index in wanted:
                ok, frame = video.retrieve()
                if ok:
                    for box, text, confidence in reader.readtext(frame):
                        if confidence < args.min_confidence:
                            continue
                        xs, ys = [p[0] for p in box], [p[1] for p in box]
                        writer.writerow([round(frame_index / fps, 2), text, round(confidence, 3), int(min(xs)),
                                         int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))])
                        rows += 1
                if len(wanted) > 50 and frame_index % int(600 * fps) < 1:
                    logger.info("%.0f s, %d texts", frame_index / fps, rows)
            frame_index += 1
    logger.info("%d texts in %s", rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
