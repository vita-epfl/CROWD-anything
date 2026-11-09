import numpy as np

from evaluation.vehicle_speed import ego_speed, fit_speed, is_vehicle, summarize, vehicle_points


def test_fit_and_ego_speed():
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    points = [[0, 3 * t, 10 + 5 * t] for t in times]  # 5 m/s along the optical axis; height ignored
    points[2] = [4, 0, 15]  # one outlier
    speed, residual = fit_speed(times, points)
    assert abs(speed - 5) < 1e-9 and residual < 1e-9
    jitter = [[0.3 * (-1) ** i, 0, i] for i in range(21)]  # 1 m per frame, 10 fps, with lateral jitter
    assert abs(ego_speed(jitter, fps=10) - 10) < 0.01
    assert is_vehicle("car") and is_vehicle("Bus") and not is_vehicle("person")


def test_vehicle_points_lifts_the_near_surface_to_world():
    depth = np.full((40, 60), 30.0)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:30, 20:40] = 2  # a car, 20 x 20 px, at 10 m
    depth[10:30, 20:40] = 10.0
    mask[5:25, 0:15] = 3  # a car cut by the left edge
    mask[30:40, 45:55] = 4  # a person
    c2w = np.eye(4)
    c2w[:3, 3] = [1, 0, 100]  # camera 100 m along z
    points = vehicle_points(depth, mask, {2: "car", 3: "car", 4: "person"}, (50, 50, 30, 20), c2w)
    assert set(points) == {2}
    x, y, z = points[2]
    assert abs(z - 110) < 1e-6 and abs(x - 1) < 0.2 and abs(y) < 0.2


def test_summarize_against_reference_and_limit():
    scenes = [{"id": "s1_ref", "scene": "s1", "side": "reference", "limit_kmh": "50"},
              {"id": "s1_gen", "scene": "s1", "side": "gen", "limit_kmh": "50"}]
    tracks = [{"id": "s1_ref", "speed_kmh": v} for v in (40, 44, 1)] + \
             [{"id": "s1_gen", "speed_kmh": v} for v in (60, 70)]
    videos, sides = summarize(tracks, scenes)
    ref, gen = sides
    assert ref["median_moving_kmh"] == 42 and ref["share_above_limit"] == 0
    assert gen["median_moving_kmh"] == 65 and gen["share_above_limit"] == 1.0
    assert gen["diff_vs_reference_kmh"] == 23 and gen["speed_over_limit"] == 1.3
    assert videos[0]["vehicles"] == 3 and videos[0]["moving"] == 2
