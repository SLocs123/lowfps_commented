import shapely
import numpy as np


# These points describe two hand-drawn expected traffic paths in image space.
# They are used to estimate a likely motion direction for each track.
# They have been manually defined, easy to build a tool for if you want, contact me and i can help, (x,y) pixel coordinates
_RAW_expected_points_LtR = [(69, 324), (347, 243), (431, 217), (473, 204), (494, 199), (520, 192)]
_RAW_expected_points_RtL = [(529, 200), (517, 208), (500, 218), (470, 236), (373, 289), (89, 434)]


def calculate_deltas(expected_points):
    """Turn a sequence of points into frame-to-frame movement vectors."""
    if len(expected_points) < 2:
        raise ValueError("expected_points must contain at least two points")

    deltas = []
    for i in range(1, len(expected_points)):
        prev_point = expected_points[i - 1]
        curr_point = expected_points[i]
        delta_x = (curr_point[0] - prev_point[0])
        delta_y = (curr_point[1] - prev_point[1])
        deltas.append((delta_x, delta_y))
    
    
    return deltas


def calculate_delta_deltas(deltas):
    """Measure how the movement vectors themselves are changing.
    Used to get deltaV for kalman filter, not too effective so positional importance remains low in association code
    """
    if len(deltas) < 2:
        return []

    delta_deltas = []
    for i in range(1, len(deltas)):
        prev_delta = deltas[i - 1]
        curr_delta = deltas[i]
        delta_deltas.append(((curr_delta[0] - prev_delta[0]), (curr_delta[1] - prev_delta[1])))

    return delta_deltas


def predict_final_change(deltas):
    """Extrapolate one extra movement vector beyond the last hand-drawn point."""
    if not deltas:
        raise ValueError("deltas must contain at least one delta")

    if len(deltas) == 1:
        return deltas[0]

    final_delta_delta = calculate_delta_deltas(deltas)[-1]
    final_delta = deltas[-1]
    return (
        final_delta[0] + final_delta_delta[0],
        final_delta[1] + final_delta_delta[1],
    )


def _build_direction_model(raw_points):
    """Precompute the vectors and line geometry for one expected lane."""
    deltas = calculate_deltas(raw_points)
    deltas.append(predict_final_change(deltas))
    points = raw_points
    if len(points) != len(deltas):
        raise ValueError("Direction model points and deltas length mismatch")
    line = shapely.geometry.LineString(points)
    return points, deltas, line


def _nearest_index(bottom_middle, expected_points):
    """Return the closest hand-drawn guide point to the track position."""
    if not expected_points:
        raise ValueError("expected_points is empty")
    return int(
        min(
            range(len(expected_points)),
            key=lambda i: (bottom_middle[0] - expected_points[i][0]) ** 2
            + (bottom_middle[1] - expected_points[i][1]) ** 2,
        )
    )


def _interpolated_delta(bottom_middle, expected_points, deltas, line):
    """Interpolate a smooth expected movement vector along a guide line. Smooths manual points"""
    if len(expected_points) < 2:
        return deltas[0], 0

    # Project the current point onto the guide line so we can locate which
    # line segment it is closest to.
    point = shapely.geometry.Point(bottom_middle)
    s = float(line.project(point))

    # Compute the cumulative length along the polyline segment by segment.
    seg_lengths = []
    cumulative = [0.0]
    for i in range(len(expected_points) - 1):
        p0 = np.asarray(expected_points[i], dtype=float)
        p1 = np.asarray(expected_points[i + 1], dtype=float)
        seg_len = float(np.linalg.norm(p1 - p0))
        seg_lengths.append(seg_len)
        cumulative.append(cumulative[-1] + seg_len)

    total_len = cumulative[-1]
    if total_len <= 0.0:
        return deltas[0], 0

    s = max(0.0, min(s, total_len))

    # Find which segment contains the projected point.
    seg_idx = len(seg_lengths) - 1
    for i, seg_end in enumerate(cumulative[1:]):
        if s <= seg_end:
            seg_idx = i
            break

    seg_start_s = cumulative[seg_idx]
    seg_len = seg_lengths[seg_idx]
    if seg_len <= 1e-9:
        t = 0.0
    else:
        t = (s - seg_start_s) / seg_len

    # Blend the local speeds from neighboring guide vectors.
    d0 = np.asarray(deltas[seg_idx], dtype=float)
    d1 = np.asarray(deltas[min(seg_idx + 1, len(deltas) - 1)], dtype=float)
    speed0 = float(np.linalg.norm(d0))
    speed1 = float(np.linalg.norm(d1))
    speed = (1.0 - t) * speed0 + t * speed1

    p0 = np.asarray(expected_points[seg_idx], dtype=float)
    p1 = np.asarray(expected_points[seg_idx + 1], dtype=float)
    tangent = p1 - p0
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-9:
        delta = d0
    else:
        delta = (tangent / tangent_norm) * speed

    return (float(delta[0]), float(delta[1])), seg_idx


expected_points_LtR, LtR_deltas, LtR_line = _build_direction_model(_RAW_expected_points_LtR)
expected_points_RtL, RtL_deltas, RtL_line = _build_direction_model(_RAW_expected_points_RtL)


def assign_expected_direction(bottom_middle):
    """Choose the nearer lane guide and return its expected movement vector."""
    point = shapely.geometry.Point(bottom_middle)
    ltr_dist = float(point.distance(LtR_line))
    rtl_dist = float(point.distance(RtL_line))

    if ltr_dist < rtl_dist:
        delta, index = _interpolated_delta(bottom_middle, expected_points_LtR, LtR_deltas, LtR_line)
        direction_line = "LtR"
    else:
        delta, index = _interpolated_delta(bottom_middle, expected_points_RtL, RtL_deltas, RtL_line)
        direction_line = "RtL"

    return delta, direction_line, index, ltr_dist, rtl_dist
