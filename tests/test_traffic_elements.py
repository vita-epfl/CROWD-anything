import numpy as np

from evaluation.traffic_elements import (ANCHORS, REFERENCE, Settings, centre_distance, iou, label_of, link_tracks,
                                         merge_boxes, score_clip, summarize, tile_windows)

STOP = np.eye(8)[0].tolist()  # patch features: one direction per kind of sign
YIELD = np.eye(8)[1].tolist()
LIGHT = np.eye(8)[2].tolist()


def det(box, feature, label="traffic sign", score=0.5):
    return {"box": box, "label": label, "feature": feature, "score": score}


def moving(feature, start, frames=(0, 4, 8), step=0.02, label="traffic sign"):
    """A sign drifting right over the sampled frames."""
    return {f: [det([start + i * step, 0.3, start + 0.05 + i * step, 0.4], feature, label)]
            for i, f in enumerate(frames)}


def merge(*per_frame):
    out = {}
    for frames in per_frame:
        for f, dets in frames.items():
            out.setdefault(f, []).extend(dets)
    return out


def test_geometry_and_labels():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0 and iou([0, 0, 0.5, 1], [0.5, 0, 1, 1]) == 0.0
    assert abs(centre_distance([0, 0, 0, 0], [1, 1, 1, 1]) - 1.0) < 1e-9
    assert label_of("traffic light") == "traffic light" and label_of("sign") == "traffic sign"
    assert label_of("car") is None


def test_link_tracks_follows_a_moving_sign_and_keeps_classes_apart():
    frames = merge(moving(STOP, 0.1), moving(LIGHT, 0.1, label="traffic light"), moving(YIELD, 0.7))
    tracks = link_tracks(frames)
    expected = [("traffic light", 3), ("traffic sign", 3), ("traffic sign", 3)]
    assert sorted((t.label, len(t.boxes)) for t in tracks) == expected


def test_score_clip_anchored_found_and_extra():
    detections = {
        REFERENCE: merge(moving(STOP, 0.1), moving(YIELD, 0.7)),
        ANCHORS: {0: [det([0.4, 0.2, 0.5, 0.4], STOP)]},  # the anchor shows the stop sign, not the yield sign
        "anchors_t2v": merge(moving(STOP, 0.11), moving(LIGHT, 0.4, label="traffic light")),  # + invented light
        "ego_t2v": moving(STOP, 0.4),  # a stop sign, but in the wrong place
    }
    rows, extra = score_clip("c", detections, Settings(start_frame=0), [det([0, 0, 0.1, 0.1], YIELD)])
    by = {(r["side"], r["element"]): r for r in rows}
    stop, give_way = (0, 1) if by[("anchors_t2v", 0)]["anchored"] else (1, 0)
    assert by[("anchors_t2v", stop)]["anchored"] and not by[("anchors_t2v", give_way)]["anchored"]
    assert by[("anchors_t2v", give_way)]["other_anchor_similarity"] > 0.99  # the other clip's anchor is a yield sign
    assert by[("anchors_t2v", stop)]["found"] and by[("anchors_t2v", stop)]["iou"] > 0.5
    assert by[("anchors_t2v", stop)]["centre_distance"] < 0.02
    assert by[("anchors_t2v", stop)]["similarity"] > 0.99
    assert not by[("anchors_t2v", give_way)]["found"] and not by[("ego_t2v", stop)]["found"]
    assert {x["side"]: x["extra"] for x in extra} == {"anchors_t2v": 1, "ego_t2v": 1}

    summary = {(s["side"], s["elements"]): s for s in summarize(rows, extra)}
    assert summary[("anchors_t2v", "anchored")]["presence"] == 1.0
    assert summary[("anchors_t2v", "not_anchored")]["presence"] == 0.0
    assert summary[("ego_t2v", "anchored")]["presence"] == 0.0 and summary[("ego_t2v", "anchored")]["iou"] is None


def test_start_frame_drops_early_frames():
    detections = {REFERENCE: moving(STOP, 0.1, frames=(0, 4, 24, 28)), "i2v": moving(STOP, 0.1, frames=(0, 4))}
    rows, extra = score_clip("c", detections, Settings(start_frame=24))
    assert [r["frames"] for r in rows] == [2] and not rows[0]["found"] and extra[0]["tracks"] == 0


def test_tiles_cover_the_image_and_merge_drops_cut_duplicates():
    windows = tile_windows(800, 400, 2, 0.2)
    assert windows[0] == (0, 0, 800, 400) and len(windows) == 5
    assert windows[-1][2:] == (800, 400) and windows[1][2] > windows[2][0]  # last tile at the corner, overlapping
    found = [([100, 100, 120, 140], 0.6, "traffic sign"), ([100, 100, 110, 140], 0.4, "traffic sign"),
             ([100, 100, 120, 140], 0.5, "traffic light"), ([300, 100, 320, 140], 0.3, "traffic sign")]
    assert [(b[0], s) for b, s, _ in merge_boxes(found)] == [(100, 0.6), (100, 0.5), (300, 0.3)]
