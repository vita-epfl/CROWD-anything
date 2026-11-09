from evaluation.horizons import merge_statements, sample_of, scene_row, window_id, window_sample

CHUNK = "United_States/New_Jersey/2025-08-29/2025-08-29_76ab_3/15-57-16"
SAMPLE = "United States/New Jersey/2025-08-29/2025-08-29_76ab_3/15-57-16/FRONT_FOLDER/FRONT_2025-08-29_15-57-16.mp4"


def test_sample_of_and_window_sample():
    assert sample_of(CHUNK, 10) == SAMPLE + "#f001200"
    assert window_sample({"current_source_chunk": CHUNK, "current_source_segment": 3}) == SAMPLE + "#f000360"
    assert window_sample({"current_source_chunk": None, "future_source_chunk": CHUNK,
                          "future_source_segment": 4}) == SAMPLE + "#f000360"
    assert window_sample({"current_source_chunk": None, "future_source_chunk": None}) is None
    assert window_id("a.mp4#f000360", 25.0) == "a.mp4#f000360@25"


def test_merge_statements_puts_scene_before_action():
    s = [{"text": "A tree.", "kind": "static", "true": True}]
    a = [{"text": "The car slows.", "kind": "dynamic", "true": True}]
    assert merge_statements({"w@0#action": a, "w@0#scene": s, "v@10#scene": s}) == {"w@0": s + a, "v@10": s}


def test_scene_row_takes_the_longest_road():
    record = {"motion": {"mean_speed": 31.5}, "roads": [
        {"t_start_s": 0, "t_end_s": 1, "speed_limit": 30, "speed_limit_source": "map", "road_class": "residential"},
        {"t_start_s": 1, "t_end_s": 5, "speed_limit": 50, "speed_limit_source": "inferred", "road_class": "secondary"}]}
    assert scene_row(record) == {"ego_kmh": 31.5, "limit_kmh": 50, "limit_source": "inferred",
                                 "road_class": "secondary"}
    assert scene_row(None)["limit_kmh"] == ""
